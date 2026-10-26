#include "engine/ModelManager.h"

#include "core/Log.h"
#include "engine/FitEstimator.h"
#include "models/ComponentCache.h"

#include <filesystem>

namespace gh::engine {

namespace {
// What the model occupies once loaded, from its file (the same estimate the fit check uses).
uint64_t estimatedBytes(const ModelInfo& info) {
    std::error_code ec;
    const uint64_t size = std::filesystem::file_size(info.modelPath, ec);
    return ec ? 0 : estimateWeightsBytes(size, info.family);
}
models::ComponentKey keyOf(const ModelInfo& info) { return {info.modelPath, "model", ""}; }
}  // namespace

ModelManager::ModelManager(EngineProvider engine, Clock clock, std::chrono::seconds sweepInterval)
    : engine_(std::move(engine)), clock_(std::move(clock)) {
    if (sweepInterval.count() > 0) {
        sweeper_ = std::thread([this, sweepInterval] { sweeperLoop(sweepInterval); });
    }
}

ModelManager::~ModelManager() {
    {
        std::lock_guard<std::mutex> lock(sweepMutex_);
        stopping_ = true;
    }
    sweepWake_.notify_all();
    if (sweeper_.joinable()) {
        sweeper_.join();
    }
    // Members go in reverse order: the manager's model reference first, then the queue, which cancels and joins.
    // Running jobs hold their own reference to the model, so it outlives them whichever order is used.
}

std::chrono::steady_clock::time_point ModelManager::now() const {
    return clock_ ? clock_() : std::chrono::steady_clock::now();
}

std::shared_ptr<ILoadedModel> ModelManager::load(const LoadOptions& options) {
    // Hold the lock for the whole load: loading takes seconds, and two callers must not load the same model twice.
    std::lock_guard<std::mutex> lock(mutex_);
    if (model_ && info_ && info_->options == options) {
        info_->lastUsed = now();
        return model_;
    }
    if (model_ && queue_.active() > 0) {
        throw EngineError(EngineErrorCode::Failed, "cannot switch models while generations are queued or running");
    }
    const std::shared_ptr<IImageEngine> engine = engine_ ? engine_() : nullptr;
    if (!engine) {
        throw EngineError(EngineErrorCode::LoadFailed, "no inference runtime is available");
    }
    if (model_) {
        log::info("unloading " + info_->info.modelPath + " to load another model");
        if (components_) components_->forget(keyOf(info_->info));
        model_.reset();
        info_.reset();
    }
    if (components_) components_->closeFile(options.modelPath);   // a file is never held twice
    std::shared_ptr<ILoadedModel> loaded = engine->load(options);
    model_ = loaded;
    LoadedModelInfo info;
    info.info = loaded->info();
    info.options = options;
    info.loadedAt = info.lastUsed = now();
    info_ = std::move(info);
    if (components_) components_->adopt(keyOf(info_->info), estimatedBytes(info_->info));
    return model_;
}

void ModelManager::unload() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (model_) {
        log::info("unloading " + info_->info.modelPath);
        if (components_) components_->forget(keyOf(info_->info));
    }
    model_.reset();
    info_.reset();
}

std::optional<LoadedModelInfo> ModelManager::loaded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return info_;
}

std::shared_ptr<ILoadedModel> ModelManager::current() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (model_) {
        info_->lastUsed = now();
    }
    return model_;
}

std::shared_ptr<Job> ModelManager::submit(GenerationRequest request) {
    std::shared_ptr<ILoadedModel> model = current();
    if (!model) {
        throw EngineError(EngineErrorCode::NotFound, "no model is loaded");
    }
    return queue_.submit(std::move(model), std::move(request));
}

bool ModelManager::sweepIdle() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!model_ || !info_ || info_->options.ttlMinutes <= 0) {
        return false;
    }
    if (queue_.active() > 0) {
        info_->lastUsed = now();  // being used
        return false;
    }
    if (now() - info_->lastUsed < std::chrono::minutes(info_->options.ttlMinutes)) {
        return false;
    }
    log::info("unloading " + info_->info.modelPath + " after " + std::to_string(info_->options.ttlMinutes) +
              " idle minutes");
    if (components_) components_->forget(keyOf(info_->info));
    model_.reset();
    info_.reset();
    return true;
}

void ModelManager::sweeperLoop(std::chrono::seconds interval) {
    std::unique_lock<std::mutex> lock(sweepMutex_);
    while (!stopping_) {
        if (sweepWake_.wait_for(lock, interval, [this] { return stopping_; })) {
            break;
        }
        lock.unlock();
        sweepIdle();
        lock.lock();
    }
}

}  // namespace gh::engine
