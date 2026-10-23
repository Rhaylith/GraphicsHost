#include "engine/RuntimeLoader.h"

#include <windows.h>

#include <algorithm>
#include <system_error>

namespace gh::engine {

namespace fs = std::filesystem;

namespace {

constexpr int kRuntimePriorityCount = 2;

// Lower rank = preferred.
int cpuRank(std::string_view id) {
    if (id == "cpu-avx2") return 0;
    if (id == "cpu-noavx") return 1;
    return kRuntimePriorityCount;
}

std::string windowsErrorText(DWORD error) {
    if (error == 4551) {
        return "blocked by an Application Control policy (Windows error 4551); see docs/environment-notes.md";
    }
    char* buffer = nullptr;
    const DWORD length = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                            FORMAT_MESSAGE_IGNORE_INSERTS,
                                        nullptr, error, 0, reinterpret_cast<char*>(&buffer), 0, nullptr);
    std::string text = length ? std::string(buffer, length) : std::string();
    if (buffer) {
        LocalFree(buffer);
    }
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) {
        text.pop_back();
    }
    return text + " (Windows error " + std::to_string(error) + ")";
}

void scanFolder(const fs::path& folder, bool bundled, int depth, const CpuFeatures& cpu, const GpuProbe& gpu,
                std::vector<RuntimeCandidate>& out) {
    std::error_code ec;
    if (!fs::is_directory(folder, ec)) {
        return;
    }
    std::vector<fs::path> files;
    std::vector<fs::path> subfolders;
    for (const auto& entry : fs::directory_iterator(folder, ec)) {
        if (entry.is_directory(ec)) {
            subfolders.push_back(entry.path());
        } else if (entry.is_regular_file(ec)) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    for (const auto& file : files) {
        const std::string name = file.filename().string();
        if (name.size() <= 7 || name.rfind("sd-", 0) != 0 || file.extension() != ".dll") {
            continue;
        }
        RuntimeCandidate candidate;
        candidate.id = name.substr(3, name.size() - 3 - 4);
        candidate.backend = backendOfRuntimeId(candidate.id);
        candidate.path = file;
        candidate.bundled = bundled;
        if (candidate.backend.empty()) {
            candidate.reason = "unknown runtime id";
        } else {
            candidate.compatible = isRuntimeCompatible(candidate.id, cpu, gpu, &candidate.reason);
        }
        out.push_back(std::move(candidate));
    }
    if (depth > 0) {
        std::sort(subfolders.begin(), subfolders.end());
        for (const auto& sub : subfolders) {
            scanFolder(sub, bundled, depth - 1, cpu, gpu, out);
        }
    }
}

}  // namespace

std::string backendOfRuntimeId(std::string_view id) {
    if (id.rfind("cpu", 0) == 0) return "cpu";
    if (id == "vulkan") return "vulkan";
    return {};
}

bool isRuntimeCompatible(std::string_view id, const CpuFeatures& cpu, const GpuProbe& gpu, std::string* reason) {
    auto fail = [&](const char* text) {
        if (reason) *reason = text;
        return false;
    };
    if (id == "cpu-avx2") {
        return cpu.canRunAvx2Runtime() ? true : fail("this CPU lacks AVX2, FMA or F16C");
    }
    if (id == "cpu-noavx") {
        return true;
    }
    if (id == "vulkan") {
        if (!gpu.vulkanAvailable) {
            return fail(gpu.error.empty() ? "Vulkan is not available" : gpu.error.c_str());
        }
        return gpu.hasSupportedDevice() ? true : fail("no supported (AMD) GPU found");
    }
    return fail("unknown runtime id");
}

std::vector<RuntimeCandidate> discoverRuntimesIn(const fs::path& bundledDir, const fs::path& userDir,
                                                 const CpuFeatures& cpu, const GpuProbe& gpu) {
    std::vector<RuntimeCandidate> out;
    scanFolder(bundledDir, true, 0, cpu, gpu, out);
    scanFolder(userDir, false, 1, cpu, gpu, out);
    return out;
}

std::vector<RuntimeCandidate> discoverRuntimes(const core::AppPaths& paths, const CpuFeatures& cpu,
                                               const GpuProbe& gpu) {
    return discoverRuntimesIn(core::AppPaths::executableDir() / "runtimes", paths.runtimesDir(), cpu, gpu);
}

const RuntimeCandidate* chooseRuntime(const std::vector<RuntimeCandidate>& candidates, std::string_view backendSetting) {
    // best candidate for an id: a user-installed one wins over the bundled one
    auto find = [&](std::string_view id) -> const RuntimeCandidate* {
        const RuntimeCandidate* best = nullptr;
        for (const auto& c : candidates) {
            if (c.id == id && c.compatible && (!best || (best->bundled && !c.bundled))) {
                best = &c;
            }
        }
        return best;
    };
    auto bestCpu = [&]() -> const RuntimeCandidate* {
        const RuntimeCandidate* best = nullptr;
        for (const auto& c : candidates) {
            if (c.backend != "cpu" || !c.compatible) continue;
            if (!best || cpuRank(c.id) < cpuRank(best->id) || (c.id == best->id && best->bundled && !c.bundled)) {
                best = &c;
            }
        }
        return best;
    };
    if (backendSetting == "vulkan") {
        return find("vulkan");
    }
    if (backendSetting == "cpu") {
        return bestCpu();
    }
    if (const auto* vulkan = find("vulkan")) {
        return vulkan;
    }
    return bestCpu();
}

std::unique_ptr<Runtime> Runtime::load(const fs::path& dll, std::string& error) {
    HMODULE module = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        error = "cannot load " + dll.string() + ": " + windowsErrorText(GetLastError());
        return nullptr;
    }
    std::unique_ptr<Runtime> runtime(new Runtime());
    runtime->module_ = module;

    auto resolve = [&](auto& slot, const char* name) {
        slot = reinterpret_cast<std::remove_reference_t<decltype(slot)>>(GetProcAddress(module, name));
        return slot != nullptr;
    };
    Api& a = runtime->api_;
    const bool ok = resolve(a.abiVersion, "gh_abi_version") && resolve(a.getInfo, "gh_runtime_get_info") &&
                    resolve(a.lastError, "gh_last_error") && resolve(a.listDevices, "gh_list_devices") &&
                    resolve(a.setLogCallback, "gh_set_log_callback") && resolve(a.load, "gh_load") &&
                    resolve(a.free, "gh_free") && resolve(a.modelFamily, "gh_model_family") &&
                    resolve(a.generate, "gh_generate") && resolve(a.cancel, "gh_cancel") &&
                    resolve(a.freeResult, "gh_free_result");
    if (!ok) {
        error = dll.string() + " does not export the GraphicsHost runtime interface";
        return nullptr;  // ~Runtime unloads the module
    }
    if (a.abiVersion() != GH_ABI_VERSION) {
        error = dll.string() + " has ABI version " + std::to_string(a.abiVersion()) + ", this application needs " +
                std::to_string(GH_ABI_VERSION);
        return nullptr;
    }
    gh_runtime_info info{};
    info.struct_size = sizeof(info);
    if (a.getInfo(&info) != GH_OK) {
        error = "gh_runtime_get_info failed: " + std::string(a.lastError());
        return nullptr;
    }
    runtime->id_ = info.id ? info.id : "";
    runtime->backend_ = info.backend ? info.backend : "";
    runtime->engineVersion_ = info.engine_version ? info.engine_version : "";
    runtime->systemInfo_ = info.system_info ? info.system_info : "";
    return runtime;
}

Runtime::~Runtime() {
    if (module_) {
        FreeLibrary(static_cast<HMODULE>(module_));
    }
}

std::string Runtime::listDevices() const {
    const size_t needed = api_.listDevices(nullptr, 0);
    std::string text(needed + 1, '\0');
    api_.listDevices(text.data(), text.size());
    text.resize(std::char_traits<char>::length(text.c_str()));
    return text;
}

}  // namespace gh::engine
