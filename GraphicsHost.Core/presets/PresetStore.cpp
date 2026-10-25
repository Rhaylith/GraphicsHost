#include "presets/PresetStore.h"

#include "core/AtomicFile.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace gh::presets {

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

// Validates a partial request document: applied on top of a request with a placeholder prompt, so a missing prompt is
// not an error (a preset may well leave it to the user).
void validateRequestPart(const Json& request, const std::string& field, std::vector<core::FieldError>& errors) {
    if (!request.is_object()) {
        errors.push_back({field, "must be a JSON object"});
        return;
    }
    try {
        engine::GenerationRequest base;
        base.prompt = "placeholder";
        for (const auto& e : engine::validate(engine::applyRequestJson(base, request))) {
            errors.push_back({field + "." + e.field, e.message});
        }
    } catch (const std::exception& e) {
        errors.push_back({field, e.what()});
    }
}

void validateLoadPart(const Json& load, const std::string& field, std::vector<core::FieldError>& errors) {
    if (!load.is_object()) {
        errors.push_back({field, "must be a JSON object"});
        return;
    }
    if (load.contains("modelPath")) {
        errors.push_back({field + ".modelPath", "a preset cannot choose the model; pick the model separately"});
    }
    try {
        engine::LoadOptions base;
        base.modelPath = "placeholder";
        for (const auto& e : engine::validate(engine::applyLoadOptionsJson(base, load))) {
            errors.push_back({field + "." + e.field, e.message});
        }
    } catch (const std::exception& e) {
        errors.push_back({field, e.what()});
    }
}

Json toDocument(const Preset& p) {
    return {{"schemaVersion", kPresetSchemaVersion},
            {"name", p.name},
            {"description", p.description},
            {"request", p.request},
            {"load", p.load}};
}

std::optional<Preset> fromDocument(const Json& doc, const std::string& fallbackName) {
    if (!doc.is_object()) {
        return std::nullopt;
    }
    Preset p;
    p.name = doc.value("name", fallbackName);
    if (p.name.empty()) {
        p.name = fallbackName;
    }
    p.description = doc.value("description", "");
    if (doc.contains("request")) p.request = doc["request"];
    if (doc.contains("load")) p.load = doc["load"];
    return p;
}

}  // namespace

bool validPresetName(const std::string& name) {
    if (name.empty() || name.size() > 64 || name.front() == '.') {
        return false;
    }
    return std::all_of(name.begin(), name.end(), [](unsigned char c) {
        return std::isalnum(c) || c == ' ' || c == '-' || c == '_' || c == '.';
    });
}

std::vector<core::FieldError> validate(const Preset& preset) {
    std::vector<core::FieldError> errors;
    if (!validPresetName(preset.name)) {
        errors.push_back({"name", "must be 1 to 64 letters, digits, spaces, '-', '_' or '.', and not start with a dot"});
    }
    validateRequestPart(preset.request, "request", errors);
    validateLoadPart(preset.load, "load", errors);
    return errors;
}

std::vector<std::string> PresetStore::list() const {
    std::vector<std::string> names;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir_, ec)) {
        if (entry.is_regular_file(ec) && entry.path().extension() == ".json") {
            const std::string stem = entry.path().stem().string();
            if (validPresetName(stem)) {
                names.push_back(stem);
            }
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

std::optional<Preset> PresetStore::get(const std::string& name) const {
    if (!validPresetName(name)) {
        return std::nullopt;
    }
    const auto text = core::readFile(pathFor(name));
    if (!text) {
        return std::nullopt;
    }
    const Json doc = Json::parse(*text, nullptr, false);
    auto preset = fromDocument(doc, name);
    if (preset) {
        preset->name = name;  // the file name wins, so a renamed file is found under its new name
    }
    return preset;
}

void PresetStore::save(const Preset& preset) const {
    const auto errors = validate(preset);
    if (!errors.empty()) {
        std::string message = "invalid preset:";
        for (const auto& e : errors) {
            message += " " + e.field + " " + e.message + ";";
        }
        throw std::invalid_argument(message);
    }
    core::writeFileAtomic(pathFor(preset.name), toDocument(preset).dump(2));
}

bool PresetStore::remove(const std::string& name) const {
    if (!validPresetName(name)) {
        return false;
    }
    std::error_code ec;
    return fs::remove(pathFor(name), ec);
}

std::string PresetStore::importFile(const fs::path& file, const std::string& newName) const {
    const auto text = core::readFile(file);
    if (!text) {
        throw std::runtime_error("cannot read " + file.string());
    }
    const Json doc = Json::parse(*text, nullptr, false);
    auto preset = fromDocument(doc, file.stem().string());
    if (!preset) {
        throw std::invalid_argument(file.string() + " is not a preset file");
    }
    if (!newName.empty()) {
        preset->name = newName;
    } else {
        preset->name = file.stem().string();
    }
    save(*preset);
    return preset->name;
}

void PresetStore::exportFile(const std::string& name, const fs::path& file) const {
    const auto preset = get(name);
    if (!preset) {
        throw std::runtime_error("no preset named " + name);
    }
    core::writeFileAtomic(file, toDocument(*preset).dump(2));
}

std::optional<ModelDefaultsStore::Defaults> ModelDefaultsStore::get(const std::string& modelFileName) const {
    const auto text = core::readFile(file_);
    if (!text) {
        return std::nullopt;
    }
    const Json doc = Json::parse(*text, nullptr, false);
    if (!doc.is_object() || !doc.contains("models") || !doc["models"].contains(modelFileName)) {
        return std::nullopt;
    }
    const Json& entry = doc["models"][modelFileName];
    Defaults d;
    if (entry.contains("request")) d.request = entry["request"];
    if (entry.contains("load")) d.load = entry["load"];
    return d;
}

void ModelDefaultsStore::set(const std::string& modelFileName, const Defaults& defaults) const {
    if (modelFileName.empty()) {
        throw std::invalid_argument("the model file name must not be empty");
    }
    std::vector<core::FieldError> errors;
    validateRequestPart(defaults.request, "request", errors);
    validateLoadPart(defaults.load, "load", errors);
    if (!errors.empty()) {
        std::string message = "invalid model defaults:";
        for (const auto& e : errors) {
            message += " " + e.field + " " + e.message + ";";
        }
        throw std::invalid_argument(message);
    }
    Json doc = Json::object();
    if (const auto text = core::readFile(file_)) {
        const Json existing = Json::parse(*text, nullptr, false);
        if (existing.is_object()) doc = existing;
    }
    doc["schemaVersion"] = kPresetSchemaVersion;
    doc["models"][modelFileName] = {{"request", defaults.request}, {"load", defaults.load}};
    core::writeFileAtomic(file_, doc.dump(2));
}

bool ModelDefaultsStore::remove(const std::string& modelFileName) const {
    const auto text = core::readFile(file_);
    if (!text) {
        return false;
    }
    Json doc = Json::parse(*text, nullptr, false);
    if (!doc.is_object() || !doc.contains("models") || !doc["models"].contains(modelFileName)) {
        return false;
    }
    doc["models"].erase(modelFileName);
    core::writeFileAtomic(file_, doc.dump(2));
    return true;
}

std::vector<std::string> ModelDefaultsStore::models() const {
    std::vector<std::string> names;
    if (const auto text = core::readFile(file_)) {
        const Json doc = Json::parse(*text, nullptr, false);
        if (doc.is_object() && doc.contains("models") && doc["models"].is_object()) {
            for (const auto& [key, value] : doc["models"].items()) {
                names.push_back(key);
            }
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

engine::GenerationRequest layerRequest(const std::vector<Json>& layers) {
    engine::GenerationRequest request;
    for (const auto& layer : layers) {
        if (layer.is_object() && !layer.empty()) {
            request = engine::applyRequestJson(std::move(request), layer);
        }
    }
    return request;
}

engine::LoadOptions layerLoadOptions(const std::vector<Json>& layers) {
    engine::LoadOptions options;
    for (const auto& layer : layers) {
        if (layer.is_object() && !layer.empty()) {
            options = engine::applyLoadOptionsJson(std::move(options), layer);
        }
    }
    return options;
}

}  // namespace gh::presets
