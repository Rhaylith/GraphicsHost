#include "engine/FitEstimator.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace gh::engine;

namespace {
constexpr uint64_t MiB = 1024ull * 1024;
constexpr uint64_t GiB = 1024ull * MiB;

FitInput sdxl1024(uint64_t gpuGiB, uint64_t freeGiB = 0) {
    FitInput in;
    in.family = "SDXL";
    in.weightsBytes = 6624110000ull;
    in.width = in.height = 1024;
    in.gpuMemoryBytes = gpuGiB * GiB;
    in.gpuFreeBytes = freeGiB * GiB;
    in.systemRamFreeBytes = 64 * GiB;
    return in;
}

FitInput sd15(int size, uint64_t gpuGiB = 24) {
    FitInput in;
    in.family = "SD 1.x";
    in.weightsBytes = 2784450000ull;
    in.width = in.height = size;
    in.gpuMemoryBytes = gpuGiB * GiB;
    in.systemRamFreeBytes = 64 * GiB;
    return in;
}

bool within(double estimate, double measured, double tolerance) {
    return std::abs(estimate - measured) <= measured * tolerance;
}
}  // namespace

TEST_CASE("fit: activation estimates match the measurements on the RX 7900 XTX", "[unit][fit]") {
    // docs/vram-measurements.md (MiB): activations = growth over idle minus weights
    CHECK(within(estimateActivationBytes("SD 1.x", 512, 512) / double(MiB), 2238, 0.08));
    CHECK(within(estimateActivationBytes("SD 1.x", 768, 768) / double(MiB), 3217, 0.08));
    CHECK(within(estimateActivationBytes("SDXL", 768, 768) / double(MiB), 2525, 0.08));
    CHECK(within(estimateActivationBytes("SDXL", 1024, 1024) / double(MiB), 3635, 0.08));
}

TEST_CASE("fit: memory stops growing beyond 1.3 megapixels", "[unit][fit]") {
    CHECK(estimateActivationBytes("SDXL", 2048, 2048) == estimateActivationBytes("SDXL", 1536, 1024));
    CHECK(estimateActivationBytes("SDXL", 1024, 1024) < estimateActivationBytes("SDXL", 1280, 1280));
    CHECK(estimateActivationBytes("SD 1.x", 512, 512) < estimateActivationBytes("SD 1.x", 768, 768));
}

TEST_CASE("fit: weight sizes from checkpoint file sizes", "[unit][fit]") {
    // the two checkpoints used in the measurements
    CHECK(within(estimateWeightsBytes(4265146304ull, "") / 1e6, 2784.45, 0.01));   // SD1.5, fp32 on disk
    CHECK(within(estimateWeightsBytes(6938078334ull, "") / 1e6, 6624.11, 0.01));   // SDXL
    CHECK(guessFamilyFromFileSize(4265146304ull) == "SD 1.x");
    CHECK(guessFamilyFromFileSize(6938078334ull) == "SDXL");
}

TEST_CASE("fit: plenty of GPU memory fits", "[unit][fit]") {
    const auto r = estimateFit(sdxl1024(24));
    CHECK(r.status == FitStatus::Fits);
    CHECK(r.device == "vulkan");
    CHECK_FALSE(r.needsOffload);
    // measured: 9.95 GB growth; the estimate is that plus the 10 % margin
    CHECK(within(r.requiredBytes / double(MiB), 9952 * 1.1, 0.06));
    CHECK(r.requiredBytes < r.availableBytes);
}

TEST_CASE("fit: a tighter GPU needs the weights offloaded", "[unit][fit]") {
    const auto r = estimateFit(sdxl1024(10, 9));
    CHECK(r.status == FitStatus::FitsWithOffload);
    CHECK(r.needsOffload);
    CHECK(r.device == "vulkan");
    CHECK(r.requiredBytes < 9 * GiB);
    REQUIRE_FALSE(r.suggestions.empty());
    CHECK(r.suggestions[0].find("offloadParamsToCpu") != std::string::npos);

    // when the caller already turned it on, the report just says it fits
    auto in = sdxl1024(10, 9);
    in.offloadParamsToCpu = true;
    CHECK(estimateFit(in).status == FitStatus::FitsWithOffload);
}

TEST_CASE("fit: auto falls back to the CPU, an explicit GPU never does", "[unit][fit]") {
    auto small = sdxl1024(6);  // 6 GiB card: not even offloaded
    small.device = "auto";
    const auto automatic = estimateFit(small);
    CHECK(automatic.status == FitStatus::CpuOnly);
    CHECK(automatic.device == "cpu");

    small.device = "vulkan0";
    const auto explicitGpu = estimateFit(small);
    CHECK(explicitGpu.status == FitStatus::WontFit);
    CHECK(explicitGpu.summary.find("GPU memory") != std::string::npos);
}

TEST_CASE("fit: the CPU path depends on RAM only", "[unit][fit]") {
    FitInput in = sdxl1024(0);
    in.device = "cpu";
    in.gpuMemoryBytes = 0;
    CHECK(estimateFit(in).status == FitStatus::Fits);
    CHECK(estimateFit(in).device == "cpu");

    in.systemRamFreeBytes = 4 * GiB;
    const auto r = estimateFit(in);
    CHECK(r.status == FitStatus::WontFit);
    CHECK(r.summary.find("RAM") != std::string::npos);
}

TEST_CASE("fit: no GPU at all with auto simply runs on the CPU", "[unit][fit]") {
    FitInput in = sd15(512, 0);
    in.device = "auto";
    const auto r = estimateFit(in);
    CHECK(r.status == FitStatus::Fits);
    CHECK(r.device == "cpu");
}

TEST_CASE("fit: asking for a GPU that is not there is refused", "[unit][fit]") {
    FitInput in = sd15(512, 0);
    in.device = "vulkan0";
    const auto r = estimateFit(in);
    CHECK(r.status == FitStatus::WontFit);
    CHECK(r.summary.find("no usable GPU") != std::string::npos);
}

TEST_CASE("fit: flash attention is suggested for SD 1.x above 512 on the GPU only", "[unit][fit]") {
    CHECK(estimateFit(sd15(768)).suggestFlashAttention);
    CHECK_FALSE(estimateFit(sd15(512)).suggestFlashAttention);
    CHECK_FALSE(estimateFit(sdxl1024(24)).suggestFlashAttention);  // SDXL does not need it
    auto on = sd15(768);
    on.flashAttention = true;
    CHECK_FALSE(estimateFit(on).suggestFlashAttention);
    auto cpu = sd15(768);
    cpu.device = "cpu";
    CHECK_FALSE(estimateFit(cpu).suggestFlashAttention);  // only the GPU benefits
}

TEST_CASE("fit: an integrated GPU is limited by system RAM", "[unit][fit]") {
    // The same card description fits as a discrete GPU, but not when it shares only 8 GiB of free RAM with the system
    // (at most half of the free RAM is usable as GPU memory).
    FitInput in = sd15(512);  // 24 GiB "GPU"
    in.systemRamFreeBytes = 8 * GiB;
    CHECK(estimateFit(in).status == FitStatus::Fits);
    in.gpuIsIntegrated = true;
    const auto r = estimateFit(in);
    CHECK(r.status == FitStatus::CpuOnly);  // 4 GiB budget < the 5 GiB this needs
}

TEST_CASE("fit: system memory is reported", "[unit][fit]") {
    const auto m = systemMemory();
    CHECK(m.totalBytes > 0);
    CHECK(m.freeBytes > 0);
    CHECK(m.freeBytes <= m.totalBytes);
    CHECK(std::string(toString(FitStatus::FitsWithOffload)) == "fits-with-offload");
}
