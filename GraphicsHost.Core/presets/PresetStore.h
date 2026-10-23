#pragma once

#include "engine/GenerationRequest.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gh::presets {

// A named, partial set of parameters: only the keys that are present change anything.
//   request: generation parameters (sampler, steps, size, a prompt prefix, ...)
//   load:    load options (device, flashAttention, offloadParamsToCpu, ...); modelPath is not allowed here
struct Preset {
    std::string name;
    std::string description;
    nlohmann::json request = nlohmann::json::object();
    nlohmann::json load = nlohmann::json::object();
};

constexpr int kPresetSchemaVersion = 1;

// True for names that are safe as a file name: letters, digits, space, '-', '_' and '.', 1 to 64 characters, not
// starting with a dot.
bool validPresetName(const std::string& name);

// The problems with a preset's content: unknown values, wrong types, a name that is not allowed.
std::vector<core::FieldError> validate(const Preset& preset);

// Presets as files <dir>/<name>.json. Every call reads the folder again, so a preset edited or dropped into the folder
// by hand is seen at once (live reload); there is no cache to invalidate.
class PresetStore {
public:
    explicit PresetStore(std::filesystem::path dir) : dir_(std::move(dir)) {}

    std::vector<std::string> list() const;                  // sorted by name
    std::optional<Preset> get(const std::string& name) const;  // nullopt if missing or unreadable
    // Throws std::invalid_argument when the preset does not validate.
    void save(const Preset& preset) const;
    bool remove(const std::string& name) const;

    // Copies a preset file into the store (under the file's own name or newName) after validating it; returns the name.
    // Throws std::invalid_argument / std::runtime_error.
    std::string importFile(const std::filesystem::path& file, const std::string& newName = {}) const;
    void exportFile(const std::string& name, const std::filesystem::path& file) const;

private:
    std::filesystem::path pathFor(const std::string& name) const { return dir_ / (name + ".json"); }
    std::filesystem::path dir_;
};

// Defaults per checkpoint, keyed by the checkpoint's file name (so they survive moving the models folder):
// {"v1-5-pruned-emaonly.safetensors": {"request": {...}, "load": {...}}}. One file, read on every call.
class ModelDefaultsStore {
public:
    explicit ModelDefaultsStore(std::filesystem::path file) : file_(std::move(file)) {}

    struct Defaults {
        nlohmann::json request = nlohmann::json::object();
        nlohmann::json load = nlohmann::json::object();
    };
    std::optional<Defaults> get(const std::string& modelFileName) const;
    void set(const std::string& modelFileName, const Defaults& defaults) const;  // throws std::invalid_argument
    bool remove(const std::string& modelFileName) const;
    std::vector<std::string> models() const;

private:
    std::filesystem::path file_;
};

// The layers in order of increasing priority: built-in defaults, model defaults, preset, explicit values.
// Each layer is a partial JSON document; the result is not validated here.
engine::GenerationRequest layerRequest(const std::vector<nlohmann::json>& layers);
engine::LoadOptions layerLoadOptions(const std::vector<nlohmann::json>& layers);

}  // namespace gh::presets
