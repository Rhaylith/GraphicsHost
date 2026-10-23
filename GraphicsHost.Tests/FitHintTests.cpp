#include "models/FitHints.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::models;
using gh::engine::FitStatus;

namespace {
constexpr uint64_t GiB = 1024ull * 1024 * 1024;
constexpr uint64_t kSd15File = 4265146304ull, kSdxlFile = 6938078334ull;

HardwareBudget gpu(uint64_t gib, uint64_t ramGib = 64) {
    HardwareBudget b;
    b.gpuMemoryBytes = gib * GiB;
    b.systemRamFreeBytes = ramGib * GiB;
    return b;
}
}  // namespace

TEST_CASE("hints: a big GPU fits both reference models", "[unit][hints]") {
    const auto a = fitHintFor(kSd15File, "SD 1.x", "checkpoint", gpu(24));
    CHECK(a.status == FitStatus::Fits);
    CHECK(a.label == "fits the GPU");
    CHECK(a.device == "vulkan");
    CHECK(fitHintFor(kSdxlFile, "SDXL", "checkpoint", gpu(24)).label == "fits the GPU");
}

TEST_CASE("hints: smaller GPUs need offload, then fall back to the CPU", "[unit][hints]") {
    CHECK(fitHintFor(kSdxlFile, "SDXL", "checkpoint", gpu(10)).label == "needs offload");
    const auto small = fitHintFor(kSdxlFile, "SDXL", "checkpoint", gpu(4));
    CHECK(small.label == "CPU only");
    CHECK(small.device == "cpu");
    CHECK(small.detail.find("does not fit the GPU") != std::string::npos);
    CHECK(fitHintFor(kSd15File, "SD 1.x", "checkpoint", gpu(8)).label == "fits the GPU");   // SD 1.5 is small
}

TEST_CASE("hints: no GPU means the CPU, and too little RAM means too big", "[unit][hints]") {
    const auto cpu = fitHintFor(kSd15File, "SD 1.x", "checkpoint", gpu(0, 16));
    CHECK(cpu.label == "runs on the CPU");
    CHECK(cpu.device == "cpu");
    const auto tight = fitHintFor(kSdxlFile, "SDXL", "checkpoint", gpu(0, 4));
    CHECK(tight.label == "too big");
    CHECK(tight.status == FitStatus::WontFit);
    CHECK(tight.detail.find("RAM") != std::string::npos);
}

TEST_CASE("hints: add-ons are always small", "[unit][hints]") {
    CHECK(fitHintFor(334000000ull, "SDXL", "vae", gpu(0, 1)).label == "small");
    CHECK(fitHintFor(100000000ull, "SD 1.x", "lora", gpu(24)).label == "small");
}

TEST_CASE("hints: the budget comes from the best supported GPU and only when the GPU is used", "[unit][hints]") {
    gh::engine::GpuProbe probe;
    probe.vulkanAvailable = true;
    gh::engine::GpuDevice nvidia;
    nvidia.vramBytes = 48 * GiB;
    nvidia.supported = false;  // not supported in v1
    gh::engine::GpuDevice amd;
    amd.vramBytes = 24 * GiB;
    amd.supported = true;
    amd.discrete = true;
    gh::engine::GpuDevice igpu;
    igpu.vramBytes = 2 * GiB;
    igpu.supported = true;
    igpu.discrete = false;
    probe.devices = {nvidia, igpu, amd};
    const gh::engine::MemorySnapshot ram{128 * GiB, 100 * GiB};

    const auto withGpu = budgetFrom(probe, true, ram);
    CHECK(withGpu.gpuMemoryBytes == 24 * GiB);      // the supported discrete card, not the unsupported bigger one
    CHECK_FALSE(withGpu.gpuIsIntegrated);
    CHECK(withGpu.systemRamFreeBytes == 100 * GiB);

    const auto cpuRuntime = budgetFrom(probe, false, ram);
    CHECK(cpuRuntime.gpuMemoryBytes == 0);
    CHECK(cpuRuntime.systemRamFreeBytes == 100 * GiB);

    CHECK(budgetFrom(gh::engine::GpuProbe{}, true, ram).gpuMemoryBytes == 0);
}

#include "models/Recommendation.h"

TEST_CASE("recommend: SDXL when the GPU holds it, otherwise the light model", "[unit][hints][recommend]") {
    const auto catalog = Catalog::fromJson(nlohmann::json::parse(R"({"schemaVersion":1,"entries":[
      {"key":"a","name":"Small","kind":"checkpoint","family":"SD 1.x","repo":"x/a","file":"a.safetensors","sizeBytes":4265146304,"sha256":"0000000000000000000000000000000000000000000000000000000000000000","recommended":true},
      {"key":"b","name":"Big","kind":"checkpoint","family":"SDXL","repo":"x/b","file":"b.safetensors","sizeBytes":6938078334,"sha256":"0000000000000000000000000000000000000000000000000000000000000000","recommended":true},
      {"key":"c","name":"Gated","kind":"checkpoint","family":"SDXL","repo":"x/c","file":"c.safetensors","sizeBytes":6938078334,"sha256":"0000000000000000000000000000000000000000000000000000000000000000","recommended":true,"gated":true},
      {"key":"v","name":"Vae","kind":"vae","family":"SDXL","repo":"x/v","file":"v.safetensors","sizeBytes":300000000,"sha256":"0000000000000000000000000000000000000000000000000000000000000000","recommended":true}]})"));
    REQUIRE(catalog.entries().size() == 4);

    const auto big = recommendCheckpoint(catalog, gpu(24));
    REQUIRE(big.entry);
    CHECK(big.entry->key == "b");
    CHECK(big.why.find("graphics card") != std::string::npos);

    const auto small = recommendCheckpoint(catalog, gpu(8));    // SDXL would need offload
    REQUIRE(small.entry);
    CHECK(small.entry->key == "a");

    const auto cpu = recommendCheckpoint(catalog, gpu(0, 64));
    REQUIRE(cpu.entry);
    CHECK(cpu.entry->key == "a");
    CHECK(cpu.why.find("CPU") != std::string::npos);

    CHECK(recommendCheckpoint(Catalog{}, gpu(24)).entry == nullptr);
}
