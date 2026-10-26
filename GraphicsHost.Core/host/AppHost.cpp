#include "host/AppHost.h"

#include "engine/FitEstimator.h"

#include "core/Log.h"

#include <utility>

namespace gh::host {

AppHost::AppHost(std::optional<std::filesystem::path> home)
    : paths_(home ? core::AppPaths(*home) : core::AppPaths::fromEnvironment()),
      store_(paths_),
      presets_(paths_.presetsDir()),
      modelDefaults_(paths_.root() / "model-defaults.json"),
      tokens_(paths_.root() / "tokens.json") {
    paths_.ensureDirectories();
    core::LoadResult loaded = store_.load();
    settings_ = std::move(loaded.settings);
    settingsWarning_ = std::move(loaded.warning);

    library_.setFolders({modelsDir()});
    library_.setHashCache(paths_.root() / "model-hashes.json");
    log::init(paths_.logsDir(), settings_.logging.level);
    log::info("GraphicsHost starting, data folder " + paths_.root().string());
    reloadModelPaths();
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
    generation_.reset();  // stops observing the queue
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

namespace {
models::MemoryManager::Config memoryConfig(const core::MemorySettings& s) {
    models::MemoryManager::Config c;
    c.mode = models::parseMemoryMode(s.mode).value_or(models::MemoryMode::Normal);
    c.reserveVramBytes = static_cast<uint64_t>(std::max(s.reserveVramMb, 0)) << 20;
    c.asyncOffload = s.asyncOffload;
    return c;
}
}  // namespace

AppHost::FreeMemoryReport AppHost::freeMemory(FreeMemoryRequest request) {
    FreeMemoryReport report;
    if (request.unloadModels) {
        models::ComponentCache& cache = components();
        uint64_t bytes = 0;
        size_t idle = 0;
        for (const auto& c : cache.stats()) {
            if (c.external) continue;
            if (c.users == 0) {
                bytes += c.bytes;
                ++idle;
            } else {
                report.kept.push_back(c.key.file + " is in use by a running node");
            }
        }
        report.componentsClosed = cache.closeAll();
        report.componentBytes = report.componentsClosed == idle ? bytes : 0;
        engine::ModelManager& manager = models();
        if (manager.queue().active() > 0) {
            if (manager.loaded()) report.kept.push_back("the model of the Generate path stays loaded while a generation is queued or running");
        } else if (manager.loaded()) {
            manager.unload();
            report.generateModelUnloaded = true;
        }
    }
    if (request.freeCaches) {
        const GraphService::Freed freed = graph().freeCaches();
        report.resultsDropped = freed.results;
        report.resultBytes = freed.bytes;
    }
    return report;
}

void AppHost::reloadModelPaths() {
    std::vector<std::string> warnings;
    library_.setExtraRoots(models::loadModelPaths(paths_.root() / "model-paths.json", &warnings));
    for (const auto& w : warnings) log::warn(w);
}

void AppHost::updateSettings(core::Settings settings) {
    store_.save(settings);  // validates first
    const engine::RuntimeCandidate* before = selectedRuntime();
    const std::string runtimeBefore = before ? before->id : std::string();
    settings_ = std::move(settings);
    library_.setFolders({modelsDir()});
    log::setLevel(settings_.logging.level);
    log::info("settings saved");
    {
        std::lock_guard<std::mutex> lock(graphMutex_);
        if (graph_) graph_->applySettings(settings_.graph);
    }
    {
        std::lock_guard<std::mutex> lock(serviceMutex_);
        if (memory_) memory_->setConfig(memoryConfig(settings_.memory));
    }

    // another runtime (backend setting) needs another engine: drop the loaded one; it is created again on next use
    const engine::RuntimeCandidate* after = selectedRuntime();
    if ((after ? after->id : std::string()) != runtimeBefore) {
        std::lock_guard<std::mutex> lock(serviceMutex_);
        if (models_ && models_->queue().active() == 0) {
            models_->unload();
            engine_.reset();
            log::info("runtime changed to " + (after ? after->id : std::string("(none)")) + ": the loaded model was unloaded");
        } else if (models_) {
            log::warn("the runtime setting changed while a generation is queued: it applies after a restart");
        } else {
            engine_.reset();
        }
    }
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

models::MemoryManager& AppHost::memory() {
    std::lock_guard<std::mutex> lock(serviceMutex_);
    if (!memory_) {
        // the device the models run on: its size, and what the system says is in use (the Generate path's model included)
        memory_ = std::make_unique<models::MemoryManager>(memoryConfig(settings_.memory), [this] {
            models::MemoryBudget b;
            const int index = settings_.gpuDevice;
            for (const auto& d : gpu_.devices)
                if (d.index == index) b.vramTotal = d.vramBytes;
            const auto used = monitor_.sample().vramUsedBytes;
            b.vramFree = used && *used < b.vramTotal ? b.vramTotal - *used : b.vramTotal;
            b.ramFree = engine::systemMemory().freeBytes;
            return b;
        });
    }
    return *memory_;
}

models::ComponentCache& AppHost::components() {
    memory();   // takes serviceMutex_ itself
    std::lock_guard<std::mutex> lock(serviceMutex_);
    if (!components_) components_ = std::make_unique<models::ComponentCache>(*memory_);
    return *components_;
}

engine::ModelManager& AppHost::models() {
    models::ComponentCache& cache = components();   // takes serviceMutex_ itself
    std::lock_guard<std::mutex> lock(serviceMutex_);
    if (!models_) {
        models_ = std::make_unique<engine::ModelManager>([this] { return engine(); });
        models_->setComponentCache(&cache);
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

namespace gh::host {

server::TokenStore& AppHost::tokens() {
    return tokens_;
}

}  // namespace gh::host

namespace gh::host {

void AppHost::overrideEngine(std::shared_ptr<engine::IImageEngine> engine) {
    std::lock_guard<std::mutex> lock(serviceMutex_);
    engine_ = std::move(engine);
    engineError_.clear();
}

}  // namespace gh::host

namespace gh::host {

GenerationService& AppHost::generation() {
    models();  // GenerationService observes the queue; create that first (it takes serviceMutex_ itself)
    std::lock_guard<std::mutex> lock(generationMutex_);
    if (!generation_) generation_ = std::make_unique<GenerationService>(*this);
    return *generation_;
}

}  // namespace gh::host

namespace gh::host {

GraphService& AppHost::graph() {
    models();     // the graph service queues on the model manager's queue and writes to the gallery; create both first
    gallery();
    std::lock_guard<std::mutex> lock(graphMutex_);
    if (!graph_) graph_ = std::make_unique<GraphService>(*this);
    return *graph_;
}

}  // namespace gh::host

namespace gh::host {

HardwareMonitor& AppHost::monitor() {
    return monitor_;
}

}  // namespace gh::host

namespace gh::host {

engine::SafetyVerdict AppHost::checkSafety(const engine::GenerationRequest& request) const {
    if (!settings_.safety.enabled && !request.safetyFilter) return {};   // off unless the host or the request turns it on
    const engine::BlocklistFilter filter(engine::BlocklistFilter::readTerms(
        {core::AppPaths::executableDir() / "data" / "safety-terms.txt", paths_.root() / "safety-terms.txt"}));
    return filter.checkPrompt(request.prompt);
}

}  // namespace gh::host
