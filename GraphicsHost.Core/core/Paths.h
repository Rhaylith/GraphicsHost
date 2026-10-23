#pragma once

#include <filesystem>

namespace gh::core {

// Where the application keeps its files. Everything lives below one root folder:
//   settings.json  presets/  runtimes/  logs/  outputs/  models/
class AppPaths {
public:
    // GRAPHICSHOST_HOME if set (tests use this), otherwise %USERPROFILE%\.graphicshost.
    static AppPaths fromEnvironment();
    // Folder that contains the running executable (the bundled "runtimes" folder lives next to it).
    static std::filesystem::path executableDir();

    explicit AppPaths(std::filesystem::path root);

    const std::filesystem::path& root() const { return root_; }
    std::filesystem::path settingsFile() const { return root_ / "settings.json"; }
    std::filesystem::path presetsDir() const { return root_ / "presets"; }
    std::filesystem::path runtimesDir() const { return root_ / "runtimes"; }
    std::filesystem::path logsDir() const { return root_ / "logs"; }
    std::filesystem::path outputsDir() const { return root_ / "outputs"; }
    std::filesystem::path defaultModelsDir() const { return root_ / "models"; }

    // Creates the root and all sub-folders; throws std::filesystem::filesystem_error on failure.
    void ensureDirectories() const;

private:
    std::filesystem::path root_;
};

}  // namespace gh::core
