#pragma once

#include "models/Catalog.h"
#include "models/DownloadManager.h"
#include "models/HuggingFace.h"
#include "models/ModelLibrary.h"
#include "presets/PresetStore.h"

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace gh::models {

class ModelServiceError : public std::runtime_error {
public:
    enum class Code { UnknownEntry, NeedsToken, Gated, NotFound, Network, Other };
    ModelServiceError(Code code, const std::string& message) : std::runtime_error(message), code_(code) {}
    Code code() const { return code_; }

private:
    Code code_;
};

struct PlannedDownload {
    std::string role;                    // "checkpoint", "vae", "lora"
    std::string key;                     // catalog key, empty for a plain repository file
    std::string repo;
    std::string file;
    std::string revision;
    std::string url;
    std::filesystem::path destination;
    uint64_t sizeBytes = 0;
    std::string sha256;
    bool present = false;                // already on disk with the right size: nothing to download
    nlohmann::json source = nullptr;     // the record written next to the file
    nlohmann::json defaults = nullptr;   // model defaults to apply when it is there
};

struct DownloadPlan {
    std::vector<PlannedDownload> items;
    uint64_t bytesToDownload = 0;
    std::vector<std::string> notes;      // things worth telling the user (a companion is included, a token is used, ...)
};
nlohmann::json toJson(const DownloadPlan& plan);

struct ModelServiceConfig {
    std::filesystem::path modelsDir;
    std::string hubBaseUrl = "https://huggingface.co";
    std::function<std::string()> token;  // current Hugging Face token, empty if none
};

// Brings catalog, library, downloads, Hugging Face and per-model defaults together: "get this model" in one call.
class ModelService {
public:
    ModelService(Catalog catalog, ModelLibrary& library, DownloadManager& downloads, presets::ModelDefaultsStore& defaults,
                 ModelServiceConfig config);

    const Catalog& catalog() const { return catalog_; }
    ModelServiceConfig& config() { return config_; }

    // What getting a catalog entry would download: the entry and, with companions, the ones it wants automatically.
    // Gated entries are checked against Hugging Face first. Throws ModelServiceError.
    DownloadPlan planCatalog(const std::string& key, bool withCompanions = true) const;

    // A single file of a Hugging Face repository (from a search). Size and hash come from the repository listing.
    DownloadPlan planRepoFile(const std::string& repo, const std::string& file) const;

    // Enqueues what is not on disk yet; for what is, only the source record is (re)written. Returns the downloads.
    std::vector<std::shared_ptr<Download>> start(const DownloadPlan& plan);

    // Forwarded after the service has done its own bookkeeping (applying defaults when a download completes).
    void setObserver(DownloadManager::Observer observer);

private:
    PlannedDownload planFor(const CatalogEntry& entry, const std::string& role) const;
    void onDownload(const Download& d);
    void requireAccess(const CatalogEntry& entry) const;

    Catalog catalog_;
    ModelLibrary& library_;
    DownloadManager& downloads_;
    presets::ModelDefaultsStore& defaults_;
    ModelServiceConfig config_;

    mutable std::mutex mutex_;
    std::map<uint64_t, nlohmann::json> pendingDefaults_;  // download id -> {"file": ..., "defaults": ...}
    DownloadManager::Observer observer_;
};

}  // namespace gh::models
