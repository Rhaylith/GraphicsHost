#pragma once

#include "core/Paths.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace gh::core {

// Bump when the settings file layout changes and add a step to the list in Migrations.cpp.
inline constexpr int kSettingsSchemaVersion = 1;

struct ServerSettings {
    bool enabled = true;
    int port = 1234;
    bool lan = false;  // bind to all interfaces instead of localhost only
    bool requireToken = false;  // also local clients must send an access token (remote clients always must)
};

struct LoggingSettings {
    std::string level = "info";  // trace|debug|info|warn|error
};

struct SafetySettings {
    bool enabled = false;  // optional safety filter, off by default
};

struct Settings {
    int schemaVersion = kSettingsSchemaVersion;
    std::string backend = "auto";        // auto|vulkan|cpu
    int gpuDevice = 0;                   // index among the usable Vulkan devices
    int threads = 0;                     // CPU threads, 0 = number of physical cores
    std::string modelsDir;               // empty = AppPaths::defaultModelsDir()
    std::string interfaceMode = "user";  // user|power|developer
    ServerSettings server;
    LoggingSettings logging;
    SafetySettings safety;
    nlohmann::json unknown = nlohmann::json::object();  // keys this version does not know, written back unchanged
};

struct FieldError {
    std::string field;
    std::string message;
    bool operator==(const FieldError&) const = default;
};

nlohmann::json toJson(const Settings& settings);
// Missing keys keep their defaults; throws nlohmann::json::exception when a known key has the wrong type.
Settings settingsFromJson(const nlohmann::json& json);

// Field-level validation; empty means valid.
std::vector<FieldError> validate(const Settings& settings);

// Brings an older settings document up to kSettingsSchemaVersion in place. Returns the number of steps applied.
// Throws std::runtime_error if the document was written by a newer version.
int migrate(nlohmann::json& document);

struct LoadResult {
    Settings settings;
    bool migrated = false;                 // an older file was upgraded (a .bak-vN copy was kept)
    std::optional<std::string> warning;    // e.g. the file was unreadable and defaults were used
};

class SettingsStore {
public:
    explicit SettingsStore(AppPaths paths) : paths_(std::move(paths)) {}

    // Never throws for bad content: an unreadable file is moved to settings.json.invalid and defaults are returned
    // with a warning.
    LoadResult load() const;
    // Throws std::invalid_argument (message lists the field errors) if the settings do not validate.
    void save(const Settings& settings) const;

private:
    AppPaths paths_;
};

}  // namespace gh::core
