#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gh::engine {

// Will this model, at this size, fit on the chosen device? Constants come from measurements on the RX 7900 XTX
// (docs/vram-measurements.md). It is an estimate with a 10 % safety margin, meant for warnings and suggestions before a
// load, not a guarantee.

struct FitInput {
    std::string family;                 // "SD 1.x" or "SDXL" as the runtime reports it; empty = guess from weightsBytes
    uint64_t weightsBytes = 0;          // size of the weights in memory (see estimateWeightsBytes)
    int width = 512;
    int height = 512;
    std::string device = "auto";        // auto | cpu | vulkan<N>
    bool flashAttention = false;
    bool offloadParamsToCpu = false;
    uint64_t gpuMemoryBytes = 0;        // dedicated memory of the target GPU, 0 = no usable GPU
    uint64_t gpuFreeBytes = 0;          // free now, 0 = unknown (90 % of the total is assumed)
    bool gpuIsIntegrated = false;       // shares system RAM: the GPU budget is a fraction of the free RAM
    uint64_t systemRamFreeBytes = 0;
};

enum class FitStatus {
    Fits,             // runs as asked
    FitsWithOffload,  // runs on the GPU only with offloadParamsToCpu
    CpuOnly,          // does not fit the GPU, fits in system RAM: expect minutes per image
    WontFit,
};
const char* toString(FitStatus status);

struct FitReport {
    FitStatus status = FitStatus::Fits;
    std::string device;                 // where it would run: "cpu" or "vulkan"
    uint64_t requiredBytes = 0;         // on that device, including the safety margin
    uint64_t availableBytes = 0;
    uint64_t activationBytes = 0;
    bool needsOffload = false;          // apply offloadParamsToCpu
    bool suggestFlashAttention = false;
    std::vector<std::string> suggestions;
    std::string summary;
};

// What the weights occupy once loaded (the engine converts fp32 checkpoints to fp16).
uint64_t estimateWeightsBytes(uint64_t fileSizeBytes, const std::string& family);
// "SDXL" for big checkpoints, "SD 1.x" otherwise: a stand-in until the file header is read (phase 3).
std::string guessFamilyFromFileSize(uint64_t fileSizeBytes);

uint64_t estimateActivationBytes(const std::string& family, int width, int height);

FitReport estimateFit(const FitInput& input);

struct MemorySnapshot {
    uint64_t totalBytes = 0;
    uint64_t freeBytes = 0;
};
MemorySnapshot systemMemory();  // physical RAM

}  // namespace gh::engine
