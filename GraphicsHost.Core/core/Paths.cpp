#include "core/Paths.h"

#include <windows.h>

#include <cstdlib>
#include <string>
#include <utility>

namespace gh::core {

namespace {

std::string environmentVariable(const char* name) {
    char* value = nullptr;
    size_t size = 0;
    if (_dupenv_s(&value, &size, name) != 0 || value == nullptr) {
        return {};
    }
    std::string result(value);
    std::free(value);
    return result;
}

}  // namespace

AppPaths::AppPaths(std::filesystem::path root) : root_(std::move(root)) {}

AppPaths AppPaths::fromEnvironment() {
    if (const std::string home = environmentVariable("GRAPHICSHOST_HOME"); !home.empty()) {
        return AppPaths(std::filesystem::path(home));   // an explicit home keeps everything inside it, models too
    }
    std::string profile = environmentVariable("USERPROFILE");
    if (profile.empty()) {
        profile = ".";
    }
    AppPaths paths(std::filesystem::path(profile) / ".graphicshost");
    std::error_code ec;
    paths.setDefaultModelsDir(std::filesystem::current_path(ec) / "DownloadedModels");
    return paths;
}

std::filesystem::path AppPaths::executableDir() {
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) {
            return std::filesystem::current_path();
        }
        if (length < buffer.size()) {
            buffer.resize(length);
            break;
        }
        buffer.resize(buffer.size() * 2);
    }
    return std::filesystem::path(buffer).parent_path();
}

void AppPaths::ensureDirectories() const {
    for (const auto& dir : {root_, presetsDir(), runtimesDir(), logsDir(), outputsDir(), defaultModelsDir()}) {
        std::filesystem::create_directories(dir);
    }
}

}  // namespace gh::core
