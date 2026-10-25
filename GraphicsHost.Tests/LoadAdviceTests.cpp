#include "cli/Cli.h"
#include "engine/LoadAdvice.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::engine;

namespace {
LoadOptions onGpu() {
    LoadOptions o;
    o.modelPath = "m.safetensors";
    o.device = "vulkan0";
    return o;
}
}  // namespace

TEST_CASE("advice: a failed GPU load offers offload first, then the CPU", "[unit][advice]") {
    const auto s = suggestLoadFallbacks(onGpu(), EngineErrorCode::OutOfMemory, "vulkan");
    REQUIRE(s.size() == 2);
    CHECK(s[0].options.offloadParamsToCpu);
    CHECK(s[0].options.device == "vulkan0");  // still the GPU
    CHECK(s[0].cliFlags == "--offload");
    CHECK(s[1].options.device == "cpu");
    CHECK_FALSE(s[1].options.offloadParamsToCpu);
    CHECK(s[1].cliFlags == "--device cpu");
    CHECK(s[1].options.modelPath == "m.safetensors");  // everything else is kept
}

TEST_CASE("advice: offload is not offered when it is already on", "[unit][advice]") {
    auto o = onGpu();
    o.offloadParamsToCpu = true;
    const auto s = suggestLoadFallbacks(o, EngineErrorCode::LoadFailed, "vulkan");
    REQUIRE(s.size() == 1);
    CHECK(s[0].options.device == "cpu");
}

TEST_CASE("advice: device auto on the Vulkan runtime counts as the GPU", "[unit][advice]") {
    auto o = onGpu();
    o.device = "auto";
    CHECK(suggestLoadFallbacks(o, EngineErrorCode::LoadFailed, "vulkan").size() == 2);
}

TEST_CASE("advice: nothing to offer when the CPU already failed or the cause is not the device", "[unit][advice]") {
    auto cpu = onGpu();
    cpu.device = "cpu";
    CHECK(suggestLoadFallbacks(cpu, EngineErrorCode::OutOfMemory, "vulkan").empty());
    CHECK(suggestLoadFallbacks(onGpu(), EngineErrorCode::OutOfMemory, "cpu").empty());
    CHECK(suggestLoadFallbacks(onGpu(), EngineErrorCode::NotFound, "vulkan").empty());
    CHECK(suggestLoadFallbacks(onGpu(), EngineErrorCode::InvalidRequest, "vulkan").empty());
    CHECK(suggestLoadFallbacks(onGpu(), EngineErrorCode::Cancelled, "vulkan").empty());
}

TEST_CASE("exit codes follow the shared table", "[unit][advice]") {
    using gh::cli::exitCodeFor;
    CHECK(exitCodeFor(EngineErrorCode::InvalidRequest) == 2);
    CHECK(exitCodeFor(EngineErrorCode::NotFound) == 3);
    CHECK(exitCodeFor(EngineErrorCode::OutOfMemory) == 6);
    CHECK(exitCodeFor(EngineErrorCode::Cancelled) == 7);
    CHECK(exitCodeFor(EngineErrorCode::LoadFailed) == 1);
    CHECK(exitCodeFor(EngineErrorCode::Failed) == 1);
}
