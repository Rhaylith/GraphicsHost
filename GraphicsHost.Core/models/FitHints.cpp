#include "models/FitHints.h"

#include "engine/GenerationRequest.h"

namespace gh::models {

HardwareBudget budgetFrom(const engine::GpuProbe& gpu, bool useGpu, const engine::MemorySnapshot& memory) {
    HardwareBudget b;
    b.systemRamFreeBytes = memory.freeBytes;
    if (!useGpu) return b;
    for (const auto& d : gpu.devices) {
        if (d.supported && d.vramBytes > b.gpuMemoryBytes) {
            b.gpuMemoryBytes = d.vramBytes;
            b.gpuIsIntegrated = !d.discrete;
        }
    }
    return b;
}

FitHint fitHintFor(uint64_t fileSizeBytes, const std::string& family, const std::string& kind, const HardwareBudget& budget) {
    FitHint hint;
    if (kind == "vae" || kind == "lora") {
        hint.label = "small";
        hint.detail = "an add-on of a few hundred MB";
        hint.device = budget.gpuMemoryBytes > 0 ? "vulkan" : "cpu";
        return hint;
    }
    const engine::FamilyDefaults sizes = engine::familyDefaults(family);
    engine::FitInput in;
    in.family = family;
    in.weightsBytes = engine::estimateWeightsBytes(fileSizeBytes, family);
    in.width = sizes.width;
    in.height = sizes.height;
    in.device = "auto";
    in.gpuMemoryBytes = budget.gpuMemoryBytes;
    in.gpuFreeBytes = budget.gpuFreeBytes;
    in.gpuIsIntegrated = budget.gpuIsIntegrated;
    in.systemRamFreeBytes = budget.systemRamFreeBytes;
    const engine::FitReport r = engine::estimateFit(in);
    hint.status = r.status;
    hint.device = r.device;
    hint.detail = r.summary;
    switch (r.status) {
        case engine::FitStatus::Fits: hint.label = r.device == "vulkan" ? "fits the GPU" : "runs on the CPU"; break;
        case engine::FitStatus::FitsWithOffload: hint.label = "needs offload"; break;
        case engine::FitStatus::CpuOnly: hint.label = "CPU only"; break;
        case engine::FitStatus::WontFit: hint.label = "too big"; break;
    }
    return hint;
}

}  // namespace gh::models
