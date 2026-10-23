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
    modelService_.reset();
    downloads_.reset();  // keeps partial files for the next start
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

#include "core/SecretStore.h"

namespace gh::host {

const models::Catalog& AppHost::catalog() {
    std::lock_guard<std::mutex> lock(serviceMutex_);
    if (!catalog_) {
        catalog_ = std::make_unique<models::Catalog>(models::Catalog::load(
            {core::AppPaths::executableDir() / "data" / "catalog.json", paths_.root() / "catalog.json"}));
        for (const auto& w : catalog_->warnings()) log::warn("catalog: " + w);
    }
    return *catalog_;
}

models::DownloadManager& AppHost::downloads() {
    std::lock_guard<std::mutex> lock(serviceMutex_);
    if (!downloads_) {
        downloads_ = std::make_unique<models::DownloadManager>(2);
    }
    return *downloads_;
}

models::ModelService& AppHost::modelService() {
    const models::Catalog& cat = catalog();
    auto& dl = downloads();
    std::lock_guard<std::mutex> lock(serviceMutex_);
    if (!modelService_) {
        models::ModelServiceConfig config;
        config.modelsDir = modelsDir();
        // GRAPHICSHOST_HUB_URL points the application at another Hugging Face compatible server (the tests use it)
        if (const char* hub = std::getenv("GRAPHICSHOST_HUB_URL"); hub && *hub) config.hubBaseUrl = hub;
        config.token = [] { return core::huggingFaceToken().value_or(std::string()); };
        modelService_ = std::make_unique<models::ModelService>(cat, library_, dl, modelDefaults_, std::move(config));
    }
    return *modelService_;
}

}  // namespace gh::host
