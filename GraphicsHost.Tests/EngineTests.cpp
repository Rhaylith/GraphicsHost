#include "TestUtil.h"

#include "core/Paths.h"
#include "engine/CpuFeatures.h"
#include "engine/GpuDevices.h"
#include "engine/RuntimeLoader.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::engine;

namespace {

CpuFeatures modernCpu() {
    CpuFeatures c;
    c.avx = c.avx2 = c.fma = c.f16c = true;
    return c;
}

CpuFeatures oldCpu() {
    return CpuFeatures{};
}

GpuProbe amdGpu() {
    GpuProbe p;
    p.vulkanAvailable = true;
    GpuDevice d;
    d.name = "AMD Radeon RX 7900 XTX";
    d.vendorId = 0x1002;
    d.vendor = "AMD";
    d.supported = true;
    p.devices.push_back(d);
    return p;
}

GpuProbe nvidiaOnly() {
    GpuProbe p;
    p.vulkanAvailable = true;
    GpuDevice d;
    d.vendor = "NVIDIA";
    d.note = "unsupported in this version (AMD GPUs only)";
    p.devices.push_back(d);
    return p;
}

GpuProbe noVulkan() {
    GpuProbe p;
    p.error = "vkCreateInstance failed: VK_ERROR_INCOMPATIBLE_DRIVER";
    return p;
}

std::vector<RuntimeCandidate> allThree(const CpuFeatures& cpu, const GpuProbe& gpu) {
    std::vector<RuntimeCandidate> list;
    for (const char* id : {"cpu-avx2", "cpu-noavx", "vulkan"}) {
        RuntimeCandidate c;
        c.id = id;
        c.backend = backendOfRuntimeId(id);
        c.path = std::string("sd-") + id + ".dll";
        c.compatible = isRuntimeCompatible(id, cpu, gpu, &c.reason);
        list.push_back(c);
    }
    return list;
}

}  // namespace

TEST_CASE("cpu detection reports something sensible", "[unit][engine]") {
    const CpuFeatures c = detectCpuFeatures();
    CHECK(c.logicalCores > 0);
    CHECK_FALSE(c.vendor.empty());
    if (c.avx2) {
        CHECK(c.avx);  // AVX2 implies AVX
    }
    if (c.fma || c.f16c) {
        CHECK(c.avx);
    }
}

TEST_CASE("gpu probe never throws and is self-consistent", "[unit][engine]") {
    const GpuProbe p = probeGpus();
    if (p.vulkanAvailable && !p.devices.empty()) {
        for (const auto& d : p.devices) {
            CHECK_FALSE(d.name.empty());
            CHECK(d.supported == (d.vendorId == 0x1002 && d.note != "software renderer"));
        }
    } else {
        CHECK_FALSE(p.error.empty());  // an unavailable Vulkan always says why
    }
}

TEST_CASE("runtime ids map to backends", "[unit][engine]") {
    CHECK(backendOfRuntimeId("cpu-avx2") == "cpu");
    CHECK(backendOfRuntimeId("cpu-noavx") == "cpu");
    CHECK(backendOfRuntimeId("vulkan") == "vulkan");
    CHECK(backendOfRuntimeId("cuda").empty());
}

TEST_CASE("compatibility rules", "[unit][engine]") {
    std::string reason;
    CHECK(isRuntimeCompatible("cpu-avx2", modernCpu(), noVulkan(), &reason));
    CHECK_FALSE(isRuntimeCompatible("cpu-avx2", oldCpu(), noVulkan(), &reason));
    CHECK(reason.find("AVX2") != std::string::npos);
    CHECK(isRuntimeCompatible("cpu-noavx", oldCpu(), noVulkan(), nullptr));

    CHECK(isRuntimeCompatible("vulkan", modernCpu(), amdGpu(), nullptr));
    CHECK_FALSE(isRuntimeCompatible("vulkan", modernCpu(), nvidiaOnly(), &reason));
    CHECK(reason.find("AMD") != std::string::npos);
    CHECK_FALSE(isRuntimeCompatible("vulkan", modernCpu(), noVulkan(), &reason));
    CHECK(reason.find("INCOMPATIBLE_DRIVER") != std::string::npos);
    CHECK_FALSE(isRuntimeCompatible("something", modernCpu(), amdGpu(), nullptr));
}

TEST_CASE("auto prefers Vulkan, then avx2, then noavx", "[unit][engine]") {
    CHECK(chooseRuntime(allThree(modernCpu(), amdGpu()), "auto")->id == "vulkan");
    CHECK(chooseRuntime(allThree(modernCpu(), noVulkan()), "auto")->id == "cpu-avx2");
    CHECK(chooseRuntime(allThree(oldCpu(), noVulkan()), "auto")->id == "cpu-noavx");
    CHECK(chooseRuntime(allThree(modernCpu(), nvidiaOnly()), "auto")->id == "cpu-avx2");  // NVIDIA is not v1
}

TEST_CASE("explicit backend settings are honoured or refused", "[unit][engine]") {
    CHECK(chooseRuntime(allThree(modernCpu(), amdGpu()), "cpu")->id == "cpu-avx2");
    CHECK(chooseRuntime(allThree(oldCpu(), amdGpu()), "cpu")->id == "cpu-noavx");
    CHECK(chooseRuntime(allThree(modernCpu(), amdGpu()), "vulkan")->id == "vulkan");
    CHECK(chooseRuntime(allThree(modernCpu(), noVulkan()), "vulkan") == nullptr);  // never silently falls back
    CHECK(chooseRuntime({}, "auto") == nullptr);
}

TEST_CASE("a user-installed runtime wins over the bundled one with the same id", "[unit][engine]") {
    auto list = allThree(modernCpu(), noVulkan());
    RuntimeCandidate user = list[0];  // cpu-avx2
    user.bundled = false;
    user.path = "user/sd-cpu-avx2.dll";
    list.push_back(user);
    const auto* chosen = chooseRuntime(list, "auto");
    REQUIRE(chosen != nullptr);
    CHECK_FALSE(chosen->bundled);
    CHECK(chosen->path == std::filesystem::path("user/sd-cpu-avx2.dll"));
}

TEST_CASE("discovery finds sd-*.dll in the bundled and the user folder", "[unit][engine]") {
    gh::test::TempDir dir;
    const auto bundled = dir.path() / "bin" / "runtimes";
    const auto user = dir.path() / "home" / "runtimes";
    gh::test::writeText(bundled / "sd-cpu-noavx.dll", "x");
    gh::test::writeText(bundled / "readme.txt", "x");
    gh::test::writeText(bundled / "other.dll", "x");
    gh::test::writeText(user / "nightly" / "sd-vulkan.dll", "x");   // one sub-folder level is allowed
    gh::test::writeText(user / "sd-mystery.dll", "x");

    const auto found = discoverRuntimesIn(bundled, user, modernCpu(), amdGpu());
    REQUIRE(found.size() == 3);
    CHECK(found[0].id == "cpu-noavx");
    CHECK(found[0].bundled);
    CHECK(found[0].compatible);

    const auto vulkan = std::find_if(found.begin(), found.end(), [](const auto& c) { return c.id == "vulkan"; });
    REQUIRE(vulkan != found.end());
    CHECK_FALSE(vulkan->bundled);
    CHECK(vulkan->compatible);

    const auto mystery = std::find_if(found.begin(), found.end(), [](const auto& c) { return c.id == "mystery"; });
    REQUIRE(mystery != found.end());
    CHECK_FALSE(mystery->compatible);
    CHECK(mystery->backend.empty());
}

TEST_CASE("discovery of missing folders is empty, not an error", "[unit][engine]") {
    gh::test::TempDir dir;
    CHECK(discoverRuntimesIn(dir.path() / "nope", dir.path() / "nada", modernCpu(), amdGpu()).empty());
}

TEST_CASE("loading something that is not a runtime fails with a message", "[unit][engine]") {
    gh::test::TempDir dir;
    gh::test::writeText(dir.path() / "sd-fake.dll", "this is not a DLL");
    std::string error;
    CHECK(Runtime::load(dir.path() / "sd-fake.dll", error) == nullptr);
    CHECK_FALSE(error.empty());
}

TEST_CASE("the bundled CPU runtime loads and reports itself", "[integration][engine]") {
    const auto dll = gh::core::AppPaths::executableDir() / "runtimes" / "sd-cpu-noavx.dll";
    if (!std::filesystem::exists(dll)) {
        SKIP("sd-cpu-noavx.dll is not next to the test executable (build the whole solution)");
    }
    std::string error;
    auto runtime = Runtime::load(dll, error);
    if (!runtime) {
        // Windows Application Control can refuse freshly built unsigned DLLs (docs/environment-notes.md)
        WARN("could not load the runtime: " << error);
        SKIP("runtime DLL could not be loaded on this machine");
    }
    CHECK(runtime->id() == "cpu-noavx");
    CHECK(runtime->backend() == "cpu");
    CHECK_FALSE(runtime->engineVersion().empty());
    CHECK(runtime->listDevices().find("CPU") != std::string::npos);
}
