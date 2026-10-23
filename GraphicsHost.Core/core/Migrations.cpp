#include "core/Settings.h"

#include <functional>
#include <stdexcept>

namespace gh::core {

namespace {

using Json = nlohmann::json;

struct Migration {
    int from;  // schemaVersion this step upgrades from; the result is from + 1
    std::function<void(Json&)> apply;
};

// Version 0 had no schemaVersion and kept the server options at the top level.
void migrate0to1(Json& doc) {
    Json server = doc.value("server", Json::object());
    for (const char* key : {"port", "lan"}) {
        if (doc.contains(key)) {
            server[key] = doc[key];
            doc.erase(key);
        }
    }
    if (!server.empty()) {
        doc["server"] = server;
    }
}

const std::vector<Migration>& migrations() {
    static const std::vector<Migration> list = {
        {0, migrate0to1},
    };
    return list;
}

}  // namespace

int migrate(Json& document) {
    if (!document.is_object()) {
        throw std::runtime_error("settings document is not a JSON object");
    }
    int version = document.value("schemaVersion", 0);
    if (version > kSettingsSchemaVersion) {
        throw std::runtime_error("settings were written by a newer version (schemaVersion " + std::to_string(version) +
                                 ", this version understands up to " + std::to_string(kSettingsSchemaVersion) + ")");
    }
    int applied = 0;
    for (const Migration& step : migrations()) {
        if (step.from == version) {
            step.apply(document);
            ++version;
            ++applied;
        }
    }
    document["schemaVersion"] = version;
    return applied;
}

}  // namespace gh::core
