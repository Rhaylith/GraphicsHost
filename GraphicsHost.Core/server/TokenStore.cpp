#include "server/TokenStore.h"

#include "core/AtomicFile.h"
#include "core/Sha256.h"

#include <windows.h>
#include <bcrypt.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <stdexcept>

#pragma comment(lib, "bcrypt.lib")

namespace gh::server {

using Json = nlohmann::json;

namespace {

std::string randomHex(size_t bytes) {
    std::vector<unsigned char> data(bytes);
    if (BCryptGenRandom(nullptr, data.data(), static_cast<ULONG>(data.size()), BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0) {
        throw std::runtime_error("the system random number generator failed");
    }
    static const char* hex = "0123456789abcdef";
    std::string out;
    for (const unsigned char b : data) {
        out += hex[b >> 4];
        out += hex[b & 15];
    }
    return out;
}

std::string isoNow() {
    const std::time_t t = std::time(nullptr);
    std::tm utc{};
    gmtime_s(&utc, &t);
    char b[32];
    std::strftime(b, sizeof(b), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return b;
}

// Compares in time that does not depend on where the strings differ.
bool constantTimeEqual(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    unsigned char diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff |= static_cast<unsigned char>(a[i] ^ b[i]);
    return diff == 0;
}

}  // namespace

bool TokenInfo::allows(const std::string& permission) const {
    return std::find(permissions.begin(), permissions.end(), "admin") != permissions.end() ||
           std::find(permissions.begin(), permissions.end(), permission) != permissions.end();
}

std::vector<TokenStore::Entry> TokenStore::load() const {
    std::vector<Entry> entries;
    const auto text = core::readFile(file_);
    if (!text) return entries;
    const Json doc = Json::parse(*text, nullptr, false);
    if (!doc.is_object() || !doc.contains("tokens") || !doc["tokens"].is_array()) return entries;
    for (const auto& t : doc["tokens"]) {
        if (!t.is_object()) continue;
        Entry e;
        e.info.id = t.value("id", "");
        e.info.name = t.value("name", "");
        e.info.created = t.value("created", "");
        e.info.lastUsed = t.value("lastUsed", "");
        e.hash = t.value("hash", "");
        if (t.contains("permissions") && t["permissions"].is_array()) e.info.permissions = t["permissions"].get<std::vector<std::string>>();
        if (!e.info.id.empty() && !e.hash.empty()) entries.push_back(std::move(e));
    }
    return entries;
}

void TokenStore::save(const std::vector<Entry>& entries) const {
    Json list = Json::array();
    for (const auto& e : entries) {
        list.push_back({{"id", e.info.id}, {"name", e.info.name}, {"permissions", e.info.permissions},
                        {"created", e.info.created}, {"lastUsed", e.info.lastUsed}, {"hash", e.hash}});
    }
    core::writeFileAtomic(file_, Json{{"schemaVersion", 1}, {"tokens", list}}.dump(2));
}

TokenStore::Created TokenStore::create(const std::string& name, const std::vector<std::string>& permissions) {
    if (name.empty() || name.size() > 64) throw std::invalid_argument("a token needs a name of 1 to 64 characters");
    if (permissions.empty()) throw std::invalid_argument("give the token at least one permission: read, generate, manage or admin");
    for (const auto& p : permissions) {
        if (std::find(allPermissions().begin(), allPermissions().end(), p) == allPermissions().end()) {
            throw std::invalid_argument("unknown permission '" + p + "' (read, generate, manage, admin)");
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto entries = load();
    if (std::any_of(entries.begin(), entries.end(), [&](const Entry& e) { return e.info.name == name; })) {
        throw std::invalid_argument("a token named '" + name + "' already exists");
    }
    Created c;
    c.secret = "gh_" + randomHex(32);
    c.info.id = "tok_" + randomHex(4);
    c.info.name = name;
    c.info.permissions = permissions;
    std::sort(c.info.permissions.begin(), c.info.permissions.end());
    c.info.permissions.erase(std::unique(c.info.permissions.begin(), c.info.permissions.end()), c.info.permissions.end());
    c.info.created = isoNow();
    entries.push_back({c.info, core::sha256Hex(c.secret)});
    save(entries);
    return c;
}

std::vector<TokenInfo> TokenStore::list() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<TokenInfo> out;
    for (auto& e : load()) out.push_back(std::move(e.info));
    return out;
}

bool TokenStore::revoke(const std::string& idOrName) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto entries = load();
    const auto it = std::remove_if(entries.begin(), entries.end(), [&](const Entry& e) { return e.info.id == idOrName || e.info.name == idOrName; });
    if (it == entries.end()) return false;
    entries.erase(it, entries.end());
    save(entries);
    return true;
}

bool TokenStore::any() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !load().empty();
}

std::optional<TokenInfo> TokenStore::verify(const std::string& secret) {
    if (secret.size() < 8) return std::nullopt;
    const std::string hash = core::sha256Hex(secret);
    std::lock_guard<std::mutex> lock(mutex_);
    auto entries = load();
    for (auto& e : entries) {
        if (!constantTimeEqual(e.hash, hash)) continue;
        // lastUsed is informational: write it at most once a minute so a busy client does not rewrite the file per request
        const std::string now = isoNow();
        if (e.info.lastUsed.substr(0, 16) != now.substr(0, 16)) {
            e.info.lastUsed = now;
            try { save(entries); } catch (const std::exception&) {}
        }
        return e.info;
    }
    return std::nullopt;
}

}  // namespace gh::server
