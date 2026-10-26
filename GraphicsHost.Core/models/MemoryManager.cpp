#include "models/MemoryManager.h"

#include "engine/FitEstimator.h"

#include <algorithm>

namespace gh::models {

namespace {
constexpr uint64_t kMiB = 1ull << 20;
// RAM that must stay free after moving weights into it, per mode
constexpr uint64_t kHighRamMargin = 512 * kMiB;
constexpr uint64_t kNormalRamMargin = 4096 * kMiB;
}  // namespace

const char* toString(MemoryMode mode) {
    switch (mode) {
        case MemoryMode::GpuOnly: return "gpu-only";
        case MemoryMode::High: return "high";
        case MemoryMode::Normal: return "normal";
        case MemoryMode::Low: return "low";
        case MemoryMode::Stream: return "stream";
    }
    return "normal";
}

std::optional<MemoryMode> parseMemoryMode(const std::string& text) {
    for (const MemoryMode m : {MemoryMode::GpuOnly, MemoryMode::High, MemoryMode::Normal, MemoryMode::Low, MemoryMode::Stream})
        if (text == toString(m)) return m;
    return std::nullopt;
}

const char* toString(Residency residency) {
    switch (residency) {
        case Residency::Vram: return "vram";
        case Residency::Ram: return "ram";
        case Residency::Closed: return "closed";
    }
    return "closed";
}

MemoryManager::MemoryManager(Config config, BudgetFn budget) : config_(config), budget_(std::move(budget)) {}

void MemoryManager::setConfig(Config config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

MemoryManager::Config MemoryManager::config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

MemoryBudget MemoryManager::budget() const {
    BudgetFn fn;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        fn = budget_;
    }
    return fn ? fn() : MemoryBudget{};
}

uint64_t MemoryManager::activationBytes(const std::string& family, int width, int height, int batch) {
    return engine::estimateActivationBytes(family, width, height) * static_cast<uint64_t>(std::max(batch, 1));
}

MemoryPlan MemoryManager::plan(const MemoryRequest& request) const {
    const Config cfg = config();
    const MemoryBudget budget = this->budget();
    MemoryPlan plan;

    uint64_t available = budget.vramFree > cfg.reserveVramBytes ? budget.vramFree - cfg.reserveVramBytes : 0;
    const uint64_t ramMargin = cfg.mode == MemoryMode::High ? kHighRamMargin : kNormalRamMargin;
    uint64_t ramLeft = budget.ramFree > ramMargin ? budget.ramFree - ramMargin : 0;

    const bool streams = cfg.mode == MemoryMode::Stream;
    const uint64_t weights = streams ? 0 : request.weightsBytes;   // streamed weights are not in VRAM
    const uint64_t need = request.activationBytes + weights;
    plan.neededBytes = need;

    // components that may leave VRAM, least recently used first
    std::vector<const ComponentInfo*> candidates;
    for (const ComponentInfo& c : request.components)
        if (!c.pinned && c.residency == Residency::Vram && c.id != request.target) candidates.push_back(&c);
    std::sort(candidates.begin(), candidates.end(), [](const ComponentInfo* a, const ComponentInfo* b) { return a->lastUse < b->lastUse; });

    const auto leave = [&](const ComponentInfo& c) {
        MemoryAction action;
        action.id = c.id;
        if (cfg.mode != MemoryMode::GpuOnly && ramLeft >= c.bytes) {
            action.kind = MemoryAction::Kind::Offload;
            ramLeft -= c.bytes;
        } else {
            action.kind = MemoryAction::Kind::Close;
        }
        plan.actions.push_back(action);
        available += c.bytes;
    };

    if (cfg.mode == MemoryMode::Low || streams) {
        for (const ComponentInfo* c : candidates) leave(*c);   // keep the whole device for the component in use
    } else {
        for (const ComponentInfo* c : candidates) {
            if (available >= need) break;
            leave(*c);
        }
    }
    plan.availableBytes = available;

    if (streams) {
        plan.residentFraction = 0.0;
        plan.fits = available >= request.activationBytes;
        plan.note = plan.fits ? "weights stay in RAM and stream to the GPU during the call" : "the activations alone do not fit in video memory";
        return plan;
    }
    if (available >= need) {
        plan.fits = true;
        return plan;
    }
    if (request.targetSupportsPartial && cfg.mode == MemoryMode::Low && available >= request.activationBytes && request.weightsBytes > 0) {
        const uint64_t resident = available - request.activationBytes;
        plan.residentFraction = std::min(1.0, static_cast<double>(resident) / static_cast<double>(request.weightsBytes));
        plan.fits = true;
        plan.note = "part of the weights stay in RAM and stream to the GPU during the call";
        return plan;
    }
    plan.fits = false;
    plan.note = "needs " + std::to_string(need / kMiB) + " MiB of video memory, " + std::to_string(available / kMiB) + " MiB available";
    if (cfg.mode == MemoryMode::GpuOnly || cfg.mode == MemoryMode::High || cfg.mode == MemoryMode::Normal) plan.note += " (memory mode low or stream can use system RAM)";
    return plan;
}

}  // namespace gh::models
