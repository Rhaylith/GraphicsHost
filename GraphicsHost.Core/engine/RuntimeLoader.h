#pragma once

#include "core/Paths.h"
#include "engine/CpuFeatures.h"
#include "engine/GpuDevices.h"

#include "gh_runtime.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gh::engine {

// One sd-<id>.dll the application found, with a verdict on whether this machine can run it.
struct RuntimeCandidate {
    std::string id;       // "cpu-avx2", "cpu-noavx", "vulkan"
    std::string backend;  // "cpu" or "vulkan"
    std::filesystem::path path;
    bool bundled = true;  // next to the exe, as opposed to the user's runtimes folder
    bool compatible = false;
    std::string reason;   // why not, when incompatible
};

// "cpu" for ids starting with "cpu", "vulkan" for "vulkan", empty for anything else.
std::string backendOfRuntimeId(std::string_view id);

// Can this machine run the runtime with this id? Fills reason when not.
bool isRuntimeCompatible(std::string_view id, const CpuFeatures& cpu, const GpuProbe& gpu, std::string* reason);

// Looks for sd-*.dll in <exe folder>/runtimes (bundled) and <home>/runtimes (user installed, one level of
// sub-folders allowed). Bundled runtimes come first.
std::vector<RuntimeCandidate> discoverRuntimes(const core::AppPaths& paths, const CpuFeatures& cpu,
                                               const GpuProbe& gpu);
std::vector<RuntimeCandidate> discoverRuntimesIn(const std::filesystem::path& bundledDir,
                                                 const std::filesystem::path& userDir, const CpuFeatures& cpu,
                                                 const GpuProbe& gpu);

// The runtime to use for the backend setting ("auto", "vulkan" or "cpu"), or nullptr if none fits.
//   auto   : Vulkan if compatible, otherwise the best CPU runtime (avx2 before noavx)
//   vulkan : the Vulkan runtime only
//   cpu    : the best CPU runtime
// A user-installed runtime wins over a bundled one with the same id.
const RuntimeCandidate* chooseRuntime(const std::vector<RuntimeCandidate>& candidates, std::string_view backendSetting);

// A loaded runtime DLL. Create with Runtime::load; the DLL is unloaded in the destructor, so free all models first.
class Runtime {
public:
    struct Api {
        decltype(&gh_abi_version) abiVersion = nullptr;
        decltype(&gh_runtime_get_info) getInfo = nullptr;
        decltype(&gh_last_error) lastError = nullptr;
        decltype(&gh_list_devices) listDevices = nullptr;
        decltype(&gh_set_log_callback) setLogCallback = nullptr;
        decltype(&gh_load) load = nullptr;
        decltype(&gh_free) free = nullptr;
        decltype(&gh_model_family) modelFamily = nullptr;
        decltype(&gh_generate) generate = nullptr;
        decltype(&gh_cancel) cancel = nullptr;
        decltype(&gh_free_result) freeResult = nullptr;
    };

    // Returns nullptr and fills error (with a hint for Windows error 4551, Application Control) on failure.
    static std::unique_ptr<Runtime> load(const std::filesystem::path& dll, std::string& error);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    const std::string& id() const { return id_; }
    const std::string& backend() const { return backend_; }
    const std::string& engineVersion() const { return engineVersion_; }
    const std::string& systemInfo() const { return systemInfo_; }
    const Api& api() const { return api_; }

    // "name<TAB>description" lines of the ggml devices this runtime sees (the CPU, and GPUs for the Vulkan runtime).
    std::string listDevices() const;

private:
    Runtime() = default;

    void* module_ = nullptr;  // HMODULE
    Api api_;
    std::string id_, backend_, engineVersion_, systemInfo_;
};

}  // namespace gh::engine
