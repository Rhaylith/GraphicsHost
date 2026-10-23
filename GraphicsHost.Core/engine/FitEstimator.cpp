#include "engine/FitEstimator.h"

#include <windows.h>

#include <algorithm>
#include <cmath>

namespace gh::engine {

namespace {

constexpr double kMiB = 1024.0 * 1024.0;
constexpr double kSafetyMargin = 1.10;
constexpr double kMaxMegapixels = 1.3;  // beyond this the engine segments its compute and memory stops growing

struct Constants {
    double baseMiB;          // activation memory at zero pixels
    double perMegapixelMiB;
    double offloadShare;     // fraction of the weights that stays in VRAM when the rest streams from RAM
    double fileToMemory;     // weights in memory / checkpoint file size
};

Constants constantsFor(const std::string& family) {
    if (family.find("XL") != std::string::npos) {
        return {1100.0, 2540.0, 0.51, 0.955};
    }
    return {1450.0, 3130.0, 0.82, 0.653};  // SD 1.x and anything unknown
}

double megapixels(int width, int height) {
    return static_cast<double>(width) * height / (1024.0 * 1024.0);
}

uint64_t mib(double value) {
    return static_cast<uint64_t>(value * kMiB);
}

std::string gb(uint64_t bytes) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.1f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return buffer;
}

}  // namespace

const char* toString(FitStatus status) {
    switch (status) {
        case FitStatus::Fits: return "fits";
        case FitStatus::FitsWithOffload: return "fits-with-offload";
        case FitStatus::CpuOnly: return "cpu-only";
        case FitStatus::WontFit: return "wont-fit";
    }
    return "unknown";
}

std::string guessFamilyFromFileSize(uint64_t fileSizeBytes) {
    return fileSizeBytes > 5500ull * 1024 * 1024 ? "SDXL" : "SD 1.x";
}

uint64_t estimateWeightsBytes(uint64_t fileSizeBytes, const std::string& family) {
    const std::string f = family.empty() ? guessFamilyFromFileSize(fileSizeBytes) : family;
    return static_cast<uint64_t>(static_cast<double>(fileSizeBytes) * constantsFor(f).fileToMemory);
}

uint64_t estimateActivationBytes(const std::string& family, int width, int height) {
    const Constants c = constantsFor(family);
    const double mp = std::min(megapixels(width, height), kMaxMegapixels);
    return mib(c.baseMiB + c.perMegapixelMiB * mp);
}

FitReport estimateFit(const FitInput& in) {
    FitReport report;
    const std::string family = in.family.empty() ? guessFamilyFromFileSize(in.weightsBytes) : in.family;
    const Constants c = constantsFor(family);
    const uint64_t weights = in.weightsBytes;
    report.activationBytes = estimateActivationBytes(family, in.width, in.height);
    const bool isSd1 = family.find("XL") == std::string::npos;
    const double mp = megapixels(in.width, in.height);

    auto margin = [](uint64_t bytes) { return static_cast<uint64_t>(static_cast<double>(bytes) * kSafetyMargin); };
    const uint64_t onCpu = margin(weights + report.activationBytes);
    const uint64_t onGpuFull = margin(weights + report.activationBytes);
    const uint64_t onGpuOffload = margin(report.activationBytes + static_cast<uint64_t>(static_cast<double>(weights) * c.offloadShare));

    uint64_t gpuBudget = 0;
    if (in.gpuMemoryBytes > 0) {
        const uint64_t base = in.gpuFreeBytes > 0 ? in.gpuFreeBytes : static_cast<uint64_t>(static_cast<double>(in.gpuMemoryBytes) * 0.9);
        gpuBudget = in.gpuIsIntegrated ? std::min(base, in.systemRamFreeBytes / 2) : base;
    }
    const bool wantsGpu = in.device != "cpu" && gpuBudget > 0;

    if (isSd1 && mp > 0.35 && wantsGpu && !in.flashAttention) {
        report.suggestFlashAttention = true;
        report.suggestions.push_back("enable flash attention: SD 1.x above 512x512 is about 5 times faster with it on the GPU");
    }

    const bool explicitGpu = in.device.rfind("vulkan", 0) == 0;
    if (explicitGpu && gpuBudget == 0) {
        report.status = FitStatus::WontFit;
        report.device = "vulkan";
        report.summary = "device " + in.device + " was requested, but there is no usable GPU";
        report.suggestions.push_back("use device cpu or auto");
        return report;
    }

    if (wantsGpu) {
        report.device = "vulkan";
        report.availableBytes = gpuBudget;
        const uint64_t need = in.offloadParamsToCpu ? onGpuOffload : onGpuFull;
        report.requiredBytes = need;
        if (need <= gpuBudget) {
            report.status = in.offloadParamsToCpu ? FitStatus::FitsWithOffload : FitStatus::Fits;
            report.needsOffload = in.offloadParamsToCpu;
            report.summary = "needs about " + gb(need) + " of " + gb(gpuBudget) + " GPU memory";
            return report;
        }
        if (!in.offloadParamsToCpu && onGpuOffload <= gpuBudget && weights <= in.systemRamFreeBytes) {
            report.status = FitStatus::FitsWithOffload;
            report.needsOffload = true;
            report.requiredBytes = onGpuOffload;
            report.suggestions.push_back("turn on offloadParamsToCpu: the weights stay in system RAM and only what is needed is on the GPU");
            report.summary = "needs " + gb(onGpuFull) + " of GPU memory (" + gb(gpuBudget) + " available), but fits with weights offloaded (" + gb(onGpuOffload) + ")";
            return report;
        }
        report.suggestions.push_back("use a smaller image size to fit the GPU");
        if (explicitGpu) {  // the user asked for this GPU: never move to the CPU behind their back
            report.status = FitStatus::WontFit;
            report.summary = "needs about " + gb(onGpuFull) + " but only " + gb(gpuBudget) + " of GPU memory is available";
            return report;
        }
        // device auto: the CPU may still work, if the RAM allows
    }

    report.device = "cpu";
    report.availableBytes = in.systemRamFreeBytes;
    report.requiredBytes = onCpu;
    if (in.systemRamFreeBytes == 0 || onCpu <= in.systemRamFreeBytes) {
        const bool fallback = in.device != "cpu";
        report.status = (fallback && in.gpuMemoryBytes > 0) ? FitStatus::CpuOnly : FitStatus::Fits;
        if (in.device == "auto" && in.gpuMemoryBytes == 0) {
            report.status = FitStatus::Fits;  // no GPU at all: CPU is simply the only choice
        }
        if (report.status == FitStatus::CpuOnly) {
            report.suggestions.push_back("it does not fit the GPU: running on the CPU works but takes minutes per image");
            report.summary = "does not fit the GPU (" + gb(onGpuFull) + " needed); would run on the CPU with " + gb(onCpu) + " of RAM";
        } else {
            report.summary = "needs about " + gb(onCpu) + " of system RAM";
        }
        return report;
    }
    report.status = FitStatus::WontFit;
    report.summary = "needs about " + gb(onCpu) + " but only " + gb(in.systemRamFreeBytes) + " of RAM is free";
    report.suggestions.push_back("close other programs or use a smaller image size");
    return report;
}

MemorySnapshot systemMemory() {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    MemorySnapshot snapshot;
    if (GlobalMemoryStatusEx(&status)) {
        snapshot.totalBytes = status.ullTotalPhys;
        snapshot.freeBytes = status.ullAvailPhys;
    }
    return snapshot;
}

}  // namespace gh::engine
