#pragma once

#include "core/Paths.h"
#include "core/Settings.h"
#include "engine/CpuFeatures.h"
#include "engine/GpuDevices.h"
#include "engine/RuntimeLoader.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gh::host {

// Composition root: the one place that wires paths, settings, logging, hardware detection and runtime discovery
// together. The desktop UI, the CLI and (later) the HTTP server all talk to the same AppHost; nothing UI specific lives
// below it.
class AppHost {
public:
    // home: settings/data folder (defaults to AppPaths::fromEnvironment()). Never throws for bad settings files;
    // see settingsWarning().
    explicit AppHost(std::optional<std::filesystem::path> home = std::nullopt);
    ~AppHost();
    AppHost(const AppHost&) = delete;
    AppHost& operator=(const AppHost&) = delete;

    const core::AppPaths& paths() const { return paths_; }
    const core::Settings& settings() const { return settings_; }
    const std::optional<std::string>& settingsWarning() const { return settingsWarning_; }

    const engine::CpuFeatures& cpu() const { return cpu_; }
    const engine::GpuProbe& gpu() const { return gpu_; }
    const std::vector<engine::RuntimeCandidate>& runtimes() const { return runtimes_; }

    // The runtime the current backend setting selects, or nullptr if none fits (e.g. "vulkan" without a usable GPU).
    const engine::RuntimeCandidate* selectedRuntime() const;

    // Validates, saves and applies new settings (log level, runtime selection). Throws std::invalid_argument when the
    // settings do not validate.
    void updateSettings(core::Settings settings);

    // Loads the DLL of selectedRuntime() and routes its log into the application log. Returns nullptr and fills error
    // on failure.
    std::unique_ptr<engine::Runtime> loadSelectedRuntime(std::string& error) const;

private:
    core::AppPaths paths_;
    core::SettingsStore store_;
    core::Settings settings_;
    std::optional<std::string> settingsWarning_;
    engine::CpuFeatures cpu_;
    engine::GpuProbe gpu_;
    std::vector<engine::RuntimeCandidate> runtimes_;
};

}  // namespace gh::host
