#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace gh::models {

// How models use video memory (setting memory.mode). Every mode computes on the Vulkan GPU.
//   gpu-only  weights live only in VRAM; room is made by closing components, never by moving them to RAM
//   high      everything stays loaded while it fits; room is made by moving the least recently used components to RAM
//             (as long as RAM keeps a small reserve)
//   normal    as high, with a larger RAM reserve so the system keeps room; closes components instead when RAM is short
//   low       before a call every component that is not needed moves to RAM, and the weights of the one in use may be only
//             partly resident (the rest streams from RAM)
//   stream    all weights stay in RAM and stream to the GPU during the call; VRAM holds the activations
enum class MemoryMode { GpuOnly, High, Normal, Low, Stream };
const char* toString(MemoryMode mode);
std::optional<MemoryMode> parseMemoryMode(const std::string& text);   // "gpu-only", "high", "normal", "low", "stream"

enum class Residency { Vram, Ram, Closed };
const char* toString(Residency residency);

struct MemoryBudget {
    uint64_t vramTotal = 0;
    uint64_t vramFree = 0;    // free on the device now, whoever uses the rest
    uint64_t ramFree = 0;
};

struct ComponentInfo {
    std::string id;
    uint64_t bytes = 0;                  // weights
    Residency residency = Residency::Vram;
    std::chrono::steady_clock::time_point lastUse{};
    bool pinned = false;                 // a node is using it, or something else owns it: never moved by the plan
};

struct MemoryRequest {
    std::string target;                  // the component to bring into VRAM; empty when only activations need room
    uint64_t weightsBytes = 0;           // of the target, 0 when it is already in VRAM
    uint64_t activationBytes = 0;        // of the call that follows
    bool targetSupportsPartial = false;  // the component can keep only part of its weights in VRAM
    std::vector<ComponentInfo> components;   // everything the cache holds (the target may be among them)
};

struct MemoryAction {
    enum class Kind { Offload, Close } kind = Kind::Offload;
    std::string id;
};

struct MemoryPlan {
    std::vector<MemoryAction> actions;   // in the order to do them
    bool fits = true;
    double residentFraction = 1.0;       // of the target's weights that stay in VRAM; the rest streams from RAM during the call
    uint64_t neededBytes = 0;            // VRAM the call needs next to what stays resident
    uint64_t availableBytes = 0;         // VRAM available after the actions (the reserve already taken off)
    std::string note;                    // why it does not fit, or what was decided
};

// Decides what has to leave VRAM to make room for a component and a call. It only plans: the ComponentCache carries the
// plan out. Estimates of activation memory come from the measured constants in engine/FitEstimator.
class MemoryManager {
public:
    struct Config {
        MemoryMode mode = MemoryMode::Normal;
        uint64_t reserveVramBytes = 512ull << 20;   // left free for the desktop and other programs (memory.reserveVramMb)
        bool asyncOffload = true;                   // move weights in the background while the GPU works (memory.asyncOffload)
    };
    using BudgetFn = std::function<MemoryBudget()>;

    explicit MemoryManager(Config config = {}, BudgetFn budget = {});

    void setConfig(Config config);
    Config config() const;
    MemoryBudget budget() const;

    MemoryPlan plan(const MemoryRequest& request) const;

    // VRAM a call needs besides the weights, from the measured constants (SD 1.x and SDXL; other families use the closest
    // one), scaled by the batch size.
    static uint64_t activationBytes(const std::string& family, int width, int height, int batch = 1);

private:
    mutable std::mutex mutex_;
    Config config_;
    BudgetFn budget_;
};

}  // namespace gh::models
