#include "core/Settings.h"

#include "core/AtomicFile.h"
#include "server/ClientLimits.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <stdexcept>

namespace gh::core {

namespace {

using Json = nlohmann::json;

constexpr std::array<const char*, 12> kKnownKeys = {"schemaVersion", "backend", "gpuDevice", "threads", "modelsDir",
                                                  "interfaceMode",  "server",  "logging",   "safety", "firstRunDone", "theme", "accent"};

bool validAccent(const std::string& c) {
    return c.size() == 7 && c[0] == '#' && std::all_of(c.begin() + 1, c.end(), [](unsigned char ch) { return std::isxdigit(ch) != 0; });
}

bool oneOf(const std::string& value, std::initializer_list<const char*> allowed) {
    return std::any_of(allowed.begin(), allowed.end(), [&](const char* a) { return value == a; });
}

}  // namespace

Json toJson(const Settings& s) {
    Json json = s.unknown.is_object() ? s.unknown : Json::object();
    json["schemaVersion"] = s.schemaVersion;
    json["backend"] = s.backend;
    json["gpuDevice"] = s.gpuDevice;
    json["threads"] = s.threads;
    json["modelsDir"] = s.modelsDir;
    json["interfaceMode"] = s.interfaceMode;
    json["firstRunDone"] = s.firstRunDone;
    json["theme"] = s.theme;
    json["accent"] = s.accent;
    json["server"] = {{"enabled", s.server.enabled}, {"port", s.server.port}, {"lan", s.server.lan}, {"requireToken", s.server.requireToken},
                      {"https", s.server.https}, {"tlsCertFile", s.server.tlsCertFile}, {"tlsKeyFile", s.server.tlsKeyFile},
                      {"allowedAddresses", s.server.allowedAddresses}, {"rateLimitPerMinute", s.server.rateLimitPerMinute}};
    json["logging"] = {{"level", s.logging.level}};
    json["safety"] = {{"enabled", s.safety.enabled}};
    return json;
}

Settings settingsFromJson(const Json& json) {
    Settings s;
    if (!json.is_object()) {
        throw std::runtime_error("settings document is not a JSON object");
    }
    s.schemaVersion = json.value("schemaVersion", s.schemaVersion);
    s.backend = json.value("backend", s.backend);
    s.gpuDevice = json.value("gpuDevice", s.gpuDevice);
    s.threads = json.value("threads", s.threads);
    s.modelsDir = json.value("modelsDir", s.modelsDir);
    s.interfaceMode = json.value("interfaceMode", s.interfaceMode);
    s.firstRunDone = json.value("firstRunDone", s.firstRunDone);
    s.theme = json.value("theme", s.theme);
    s.accent = json.value("accent", s.accent);
    if (const auto it = json.find("server"); it != json.end() && it->is_object()) {
        s.server.enabled = it->value("enabled", s.server.enabled);
        s.server.port = it->value("port", s.server.port);
        s.server.lan = it->value("lan", s.server.lan);
        s.server.requireToken = it->value("requireToken", s.server.requireToken);
        s.server.https = it->value("https", s.server.https);
        s.server.tlsCertFile = it->value("tlsCertFile", s.server.tlsCertFile);
        s.server.tlsKeyFile = it->value("tlsKeyFile", s.server.tlsKeyFile);
        s.server.rateLimitPerMinute = it->value("rateLimitPerMinute", s.server.rateLimitPerMinute);
        if (const auto a = it->find("allowedAddresses"); a != it->end() && a->is_array()) s.server.allowedAddresses = a->get<std::vector<std::string>>();
    }
    if (const auto it = json.find("logging"); it != json.end() && it->is_object()) {
        s.logging.level = it->value("level", s.logging.level);
    }
    if (const auto it = json.find("safety"); it != json.end() && it->is_object()) {
        s.safety.enabled = it->value("enabled", s.safety.enabled);
    }
    for (const auto& [key, value] : json.items()) {
        const bool known = std::any_of(kKnownKeys.begin(), kKnownKeys.end(), [&](const char* k) { return key == k; });
        if (!known) {
            s.unknown[key] = value;
        }
    }
    return s;
}

std::vector<FieldError> validate(const Settings& s) {
    std::vector<FieldError> errors;
    if (!oneOf(s.backend, {"auto", "vulkan", "cpu"})) {
        errors.push_back({"backend", "must be auto, vulkan or cpu"});
    }
    if (s.gpuDevice < 0) {
        errors.push_back({"gpuDevice", "must be 0 or greater"});
    }
    if (s.threads < 0 || s.threads > 1024) {
        errors.push_back({"threads", "must be between 0 (automatic) and 1024"});
    }
    if (!oneOf(s.interfaceMode, {"user", "power", "developer"})) {
        errors.push_back({"interfaceMode", "must be user, power or developer"});
    }
    if (!oneOf(s.theme, {"system", "light", "dark"})) {
        errors.push_back({"theme", "must be system, light or dark"});
    }
    if (!s.accent.empty() && !validAccent(s.accent)) {
        errors.push_back({"accent", "must be empty or a colour like #2b6cb0"});
    }
    for (const auto& a : s.server.allowedAddresses) {
        if (!server::IpAllowList::valid(a)) errors.push_back({"server.allowedAddresses", "not an address or range: " + a});
    }
    if (s.server.rateLimitPerMinute < 0 || s.server.rateLimitPerMinute > 100000) {
        errors.push_back({"server.rateLimitPerMinute", "must be between 0 (no limit) and 100000"});
    }
    if (s.server.https && (s.server.tlsCertFile.empty() != s.server.tlsKeyFile.empty())) {
        errors.push_back({"server.tlsCertFile", "give both a certificate and a key file, or neither (a certificate is made for you)"});
    }
    if (s.server.port < 1 || s.server.port > 65535) {
        errors.push_back({"server.port", "must be between 1 and 65535"});
    }
    if (!oneOf(s.logging.level, {"trace", "debug", "info", "warn", "error"})) {
        errors.push_back({"logging.level", "must be trace, debug, info, warn or error"});
    }
    return errors;
}

LoadResult SettingsStore::load() const {
    LoadResult result;
    const auto path = paths_.settingsFile();
    const auto text = readFile(path);
    if (!text) {
        return result;  // first run: defaults
    }
    try {
        Json document = Json::parse(*text);
        const int from = document.is_object() ? document.value("schemaVersion", 0) : 0;
        const int applied = migrate(document);
        if (applied > 0) {
            auto backup = path;
            backup += ".bak-v" + std::to_string(from);
            writeFileAtomic(backup, *text);
            result.migrated = true;
        }
        result.settings = settingsFromJson(document);
        if (applied > 0) {
            writeFileAtomic(path, toJson(result.settings).dump(2));
        }
    } catch (const std::exception& e) {
        auto invalid = path;
        invalid += ".invalid";
        std::error_code ignored;
        std::filesystem::rename(path, invalid, ignored);
        result.settings = Settings{};
        result.warning = std::string("settings file could not be used (") + e.what() +
                         "); it was moved to settings.json.invalid and defaults are in effect";
    }
    return result;
}

void SettingsStore::save(const Settings& settings) const {
    const auto errors = validate(settings);
    if (!errors.empty()) {
        std::string message = "invalid settings:";
        for (const auto& e : errors) {
            message += " " + e.field + " " + e.message + ";";
        }
        throw std::invalid_argument(message);
    }
    writeFileAtomic(paths_.settingsFile(), toJson(settings).dump(2));
}

}  // namespace gh::core
