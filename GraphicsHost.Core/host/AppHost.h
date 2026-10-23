#pragma once

#include "core/Paths.h"
#include "core/Settings.h"
#include "engine/CpuFeatures.h"
#include "engine/GpuDevices.h"
#include "engine/ImageEngine.h"
#include "engine/ModelManager.h"
#include "engine/RuntimeLoader.h"
#include "gallery/GalleryStore.h"
#include "models/ModelLibrary.h"
#include "presets/PresetStore.h"

#include <memory>
#include <mutex>
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

    // The folder checkpoints are looked for in: settings.modelsDir, or <data folder>/models.
    std::filesystem::path modelsDir() const;

    // The inference engine over the selected runtime, loaded on first use and then kept. nullptr (and a message in error)
    // when no runtime fits or it cannot be loaded.
    std::shared_ptr<engine::IImageEngine> engine(std::string* error = nullptr);

    // Services; created on first use except the stores, which are cheap.
    engine::ModelManager& models();      // one loaded model, the job queue
    gallery::GalleryStore& gallery();     // <data folder>/outputs
    presets::PresetStore& presets();      // <data folder>/presets
    presets::ModelDefaultsStore& modelDefaults();  // <data folder>/model-defaults.json
    models::ModelLibrary& library();               // the checkpoints, VAEs and LoRAs in modelsDir()

private:
    core::AppPaths paths_;
    core::SettingsStore store_;
    core::Settings settings_;
    std::optional<std::string> settingsWarning_;
    engine::CpuFeatures cpu_;
    engine::GpuProbe gpu_;
    std::vector<engine::RuntimeCandidate> runtimes_;

    std::mutex serviceMutex_;
    std::shared_ptr<engine::IImageEngine> engine_;
    std::string engineError_;
    std::unique_ptr<gallery::GalleryStore> gallery_;
    presets::PresetStore presets_;
    presets::ModelDefaultsStore modelDefaults_;
    models::ModelLibrary library_;
    std::unique_ptr<engine::ModelManager> models_;
};

}  // namespace gh::host
