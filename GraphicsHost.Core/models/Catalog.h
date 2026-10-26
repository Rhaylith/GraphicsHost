#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gh::models {

struct CatalogCompanion {
    std::string role;      // "vae"
    std::string key;       // key of another catalog entry
    bool autoDownload = false;  // fetched together with the model unless the user opts out
    std::string note;
};

struct CatalogEntry {
    std::string key;
    std::string name;
    std::string kind = "checkpoint";   // checkpoint | vae | lora
    std::string family;                // "SD 1.x", "SDXL"
    std::string repo;                  // Hugging Face repository "publisher/name"
    std::string file;                  // file in the repository
    std::string revision;              // commit the hash was taken from; empty = main
    uint64_t sizeBytes = 0;
    std::string sha256;
    std::string license;
    bool gated = false;
    bool recommended = false;
    std::vector<std::string> tags;
    int minVramGb = 0;
    std::string notes;
    nlohmann::json defaults = nullptr;  // {"request": {...}, "load": {...}} applied as model defaults after the download
    std::vector<CatalogCompanion> companions;

    // https://huggingface.co/<repo>/resolve/<revision or main>/<file>
    std::string url() const;
    // <publisher>/<repo>/<file>: where it goes below the models folder
    std::filesystem::path relativePath() const;
};

struct CatalogFilter {
    std::string text;      // substring of key, name, family, tags or notes (case-insensitive)
    std::string family;    // exact, e.g. "SDXL"
    std::string kind;      // checkpoint | vae | lora
    bool onlyRecommended = false;
};

class Catalog {
public:
    // Reads the catalogs in order; entries of later files replace entries with the same key. Problems (a missing
    // file is not one) go into warnings() and the affected entries are skipped.
    static Catalog load(const std::vector<std::filesystem::path>& files);
    static Catalog fromJson(const nlohmann::json& document, std::vector<std::string>* warnings = nullptr);

    const std::vector<CatalogEntry>& entries() const { return entries_; }
    const std::vector<std::string>& warnings() const { return warnings_; }
    const CatalogEntry* find(const std::string& key) const;
    std::vector<const CatalogEntry*> search(const CatalogFilter& filter) const;

    // The entry and the companions it wants downloaded together: autoOnly keeps those marked auto.
    std::vector<const CatalogEntry*> withCompanions(const CatalogEntry& entry, bool autoOnly = true) const;

private:
    std::vector<CatalogEntry> entries_;
    std::vector<std::string> warnings_;
};

}  // namespace gh::models
