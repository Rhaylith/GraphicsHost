#pragma once

#include "engine/FitEstimator.h"
#include "engine/GpuDevices.h"

#include <cstdint>
#include <string>

namespace gh::models {

// What this machine can offer to a model.
struct HardwareBudget {
    uint64_t gpuMemoryBytes = 0;        // best supported GPU, 0 = none usable
    uint64_t gpuFreeBytes = 0;
    bool gpuIsIntegrated = false;
    uint64_t systemRamFreeBytes = 0;
};

// useGpu: false when the selected runtime is a CPU one (a GPU would not be used then).
HardwareBudget budgetFrom(const engine::GpuProbe& gpu, bool useGpu, const engine::MemorySnapshot& memory = engine::systemMemory());

struct FitHint {
    engine::FitStatus status = engine::FitStatus::Fits;
    std::string label;      // short, for a column: "fits the GPU", "needs offload", "CPU only", "too big", "small"
    std::string detail;     // one sentence with the numbers
    std::string device;     // where it would run: "vulkan" or "cpu"
};

// For a catalog entry or a search result: its file size and family, at the family's default image size.
// kind "vae" and "lora" are small add-ons and always fit.
FitHint fitHintFor(uint64_t fileSizeBytes, const std::string& family, const std::string& kind, const HardwareBudget& budget);

}  // namespace gh::models
