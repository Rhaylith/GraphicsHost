#include "host/AppHost.h"

#include "core/Log.h"

#include <utility>

namespace gh::host {

AppHost::AppHost(std::optional<std::filesystem::path> home)
    : paths_(home ? core::AppPaths(*home) : core::AppPaths::fromEnvironment()),
      store_(paths_),
      presets_(paths_.presetsDir()),
      modelDefaults_(paths_.root() / "model-defaults.json") {
    paths_.ensureDirectories();
    core::LoadResult loaded = store_.load();
    settings_ = std::move(loaded.settings);
    settingsWarning_ = std::move(loaded.warning);

    library_.setFolders({modelsDir()});
    log::init(paths_.logsDir(), settings_.logging.level);
    log::info("GraphicsHost starting, data folder " + paths_.root().string());
    if (loaded.migrated) {
        log::info("settings were upgraded to schema version " + std::to_string(core::kSettingsSchemaVersion));
    }
    if (settingsWarning_) {
        log::warn(*settingsWarning_);
    }

    cpu_ = engine::detectCpuFeatures();
    gpu_ = engine::probeGpus();
    runtimes_ = engine::discoverRuntimes(paths_, cpu_, gpu_);

    log::info("CPU: " + cpu_.brand + " (" + std::to_string(cpu_.logicalCores) + " logical cores)");
    if (gpu_.vulkanAvailable) {
        for (const auto& d : gpu_.devices) {
            log::info("GPU " + std::to_string(d.index) + ": " + d.name + " (" + d.vendor + ")" +
                      (d.supported ? "" : " - " + d.note));
        }
    } else {
        log::warn("Vulkan unavailable: " + gpu_.error);
    }
    if (const auto* chosen = selectedRuntime()) {
        log::info("runtime selected: " + chosen->id + " (backend setting '" + settings_.backend + "')");
    } else {
        log::warn("no runtime fits backend setting '" + settings_.backend + "'");
    }
}

AppHost::~AppHost() {
    models_.reset();  // cancels running jobs, then frees the model
    engine_.reset();
    gallery_.reset();
    log::info("GraphicsHost stopping");
    log::shutdown();
}

const engine::RuntimeCandidate* AppHost::selectedRuntime() const {
    return engine::chooseRuntime(runtimes_, settings_.backend);
}

void AppHost::updateSettings(core::Settings settings) {
    store_.save(settings);  // validates first
    settings_ = std::move(settings);
    library_.setFolders({modelsDir()});
    log::setLevel(settings_.logging.level);
    log::info("settings saved");
}

std::unique_ptr<engine::Runtime> AppHost::loadSelectedRuntime(std::string& error) const {
    const engine::RuntimeCandidate* chosen = selectedRuntime();
    if (!chosen) {
        error = "no runtime fits the backend setting '" + settings_.backend + "'";
        return nullptr;
    }
    auto runtime = engine::Runtime::load(chosen->path, error);
    if (runtime) {
        runtime->api().setLogCallback([](int32_t level, const char* text, void*) { log::nativeLog(level, text); },
                                      nullptr);
        log::info("runtime loaded: " + runtime->id() + " (engine " + runtime->engineVersion() + ")");
    } else {
        log::error(error);
    }
    return runtime;
}

}  // namespace gh::host

namespace gh::host {

std::filesystem::path AppHost::modelsDir() const {
    return settings_.modelsDir.empty() ? paths_.defaultModelsDir() : std::filesystem::path(settings_.modelsDir);
}

std::shared_ptr<engine::IImageEngine> AppHost::engine(std::string* error) {
    std::lock_guard<std::mutex> lock(serviceMutex_);
    if (!engine_) {
        std::string problem;
        std::shared_ptr<engine::Runtime> runtime = loadSelectedRuntime(problem);
        if (!runtime) {
            engineError_ = problem;
        } else {
            engine_ = engine::makeRuntimeEngine(std::move(runtime));
            engineError_.clear();
        }
    }
    if (!engine_ && error) {
        *error = engineError_;
    }
    return engine_;
}

engine::ModelManager& AppHost::models() {
    std::lock_guard<std::mutex> lock(serviceMutex_);
    if (!models_) {
        models_ = std::make_unique<engine::ModelManager>([this] { return engine(); });
    }
    return *models_;
}

gallery::GalleryStore& AppHost::gallery() {
    std::lock_guard<std::mutex> lock(serviceMutex_);
    if (!gallery_) {
        gallery_ = std::make_unique<gallery::GalleryStore>(paths_.outputsDir());
    }
    return *gallery_;
}

presets::PresetStore& AppHost::presets() {
    return presets_;
}

presets::ModelDefaultsStore& AppHost::modelDefaults() {
    return modelDefaults_;
}

}  // namespace gh::host

namespace gh::host {

models::ModelLibrary& AppHost::library() {
    return library_;
}

}  // namespace gh::host
