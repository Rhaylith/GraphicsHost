#include "host/AppHost.h"

#include "core/Log.h"

#include <utility>

namespace gh::host {

AppHost::AppHost(std::optional<std::filesystem::path> home)
    : paths_(home ? core::AppPaths(*home) : core::AppPaths::fromEnvironment()), store_(paths_) {
    paths_.ensureDirectories();
    core::LoadResult loaded = store_.load();
    settings_ = std::move(loaded.settings);
    settingsWarning_ = std::move(loaded.warning);

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
    log::info("GraphicsHost stopping");
    log::shutdown();
}

const engine::RuntimeCandidate* AppHost::selectedRuntime() const {
    return engine::chooseRuntime(runtimes_, settings_.backend);
}

void AppHost::updateSettings(core::Settings settings) {
    store_.save(settings);  // validates first
    settings_ = std::move(settings);
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
