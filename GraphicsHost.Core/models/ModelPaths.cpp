#include "models/ModelPaths.h"

#include "core/AtomicFile.h"

#include <nlohmann/json.hpp>

namespace gh::models {

ExtraRoots loadModelPaths(const std::filesystem::path& file, std::vector<std::string>* warnings) {
    ExtraRoots roots;
    const auto warn = [&](const std::string& text) {
        if (warnings) warnings->push_back(file.filename().string() + ": " + text);
    };
    const auto text = core::readFile(file);
    if (!text) return roots;
    const nlohmann::json doc = nlohmann::json::parse(*text, nullptr, false);
    if (!doc.is_object()) {
        warn("not a JSON object");
        return roots;
    }
    for (const auto& [name, value] : doc.items()) {
        const auto kind = kindFromFolder(name);
        if (!kind) {
            warn("unknown kind '" + name + "'");
            continue;
        }
        if (!value.is_array()) {
            warn("'" + name + "' must be a list of folders");
            continue;
        }
        for (const auto& folder : value) {
            if (folder.is_string() && !folder.get<std::string>().empty()) roots[*kind].emplace_back(folder.get<std::string>());
            else warn("'" + name + "' has an entry that is not a folder name");
        }
    }
    return roots;
}

}  // namespace gh::models
