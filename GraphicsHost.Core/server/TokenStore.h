#pragma once

#include <filesystem>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace gh::server {

// What a token may do.
//   read      list models, the gallery, jobs, settings (read only)
//   generate  start generations (also what OpenAI clients need)
//   manage    load/unload models, downloads, presets, delete images
//   admin     everything, including tokens and settings
inline const std::vector<std::string>& allPermissions() {
    static const std::vector<std::string> list = {"read", "generate", "manage", "admin"};
    return list;
}

struct TokenInfo {
    std::string id;                       // short, safe to show: "tok_1a2b3c4d"
    std::string name;
    std::vector<std::string> permissions;
    std::string created;                  // ISO 8601 UTC
    std::string lastUsed;                 // empty = never
    bool allows(const std::string& permission) const;   // admin allows everything
};

// API tokens. Only the SHA-256 hash of a secret is stored (tokens.json), so the secret can be shown once, when it is
// created, and never again. The secrets are 256 random bits, so a plain hash is enough.
class TokenStore {
public:
    explicit TokenStore(std::filesystem::path file) : file_(std::move(file)) {}

    struct Created {
        TokenInfo info;
        std::string secret;               // "gh_" + 64 hex characters; shown once
    };
    // Throws std::invalid_argument for an empty/duplicate name or an unknown permission.
    Created create(const std::string& name, const std::vector<std::string>& permissions);
    std::vector<TokenInfo> list() const;
    bool revoke(const std::string& idOrName);   // true if something was removed
    bool any() const;

    // The token a secret belongs to, or nullopt. Updates lastUsed (written at most once a minute per token).
    std::optional<TokenInfo> verify(const std::string& secret);

private:
    struct Entry {
        TokenInfo info;
        std::string hash;
    };
    std::vector<Entry> load() const;
    void save(const std::vector<Entry>& entries) const;

    std::filesystem::path file_;
    mutable std::mutex mutex_;
};

}  // namespace gh::server
