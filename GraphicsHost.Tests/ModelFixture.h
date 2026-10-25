#pragma once

#include "core/Paths.h"
#include "engine/ImageEngine.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace gh::test {

// Finds a real checkpoint for the [model] tests: GRAPHICSHOST_TEST_MODEL, otherwise models/<name> in the repository
// (searched upwards from the test executable).
inline std::optional<std::filesystem::path> findTestModel(const char* name = "v1-5-pruned-emaonly.safetensors") {
    char* env = nullptr;
    size_t size = 0;
    if (_dupenv_s(&env, &size, "GRAPHICSHOST_TEST_MODEL") == 0 && env != nullptr) {
        std::filesystem::path p(env);
        std::free(env);
        if (std::filesystem::exists(p)) {
            return p;
        }
    }
    for (auto dir = core::AppPaths::executableDir(); !dir.empty(); dir = dir.parent_path()) {
        const auto candidate = dir / "models" / name;
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
        if (dir == dir.parent_path()) {
            break;
        }
    }
    return std::nullopt;
}

// The CPU runtime next to the test executable, loaded; or nullptr (and a message) when it cannot be.
inline std::shared_ptr<engine::Runtime> loadCpuRuntime(std::string& problem) {
    const auto dll = core::AppPaths::executableDir() / "runtimes" / "sd-cpu-avx2.dll";
    if (!std::filesystem::exists(dll)) {
        problem = "sd-cpu-avx2.dll is not next to the test executable";
        return nullptr;
    }
    std::string error;
    std::shared_ptr<engine::Runtime> runtime = engine::Runtime::load(dll, error);
    if (!runtime) {
        problem = error;
    }
    return runtime;
}

}  // namespace gh::test
