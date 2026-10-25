#pragma once

#include "models/ModelCard.h"

#include <atomic>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace gh::models {

// File name of the record a download leaves next to a model: <model file>.source.json
std::filesystem::path sourceSidecarFor(const std::filesystem::path& modelFile);

// The model files below one or more folders. Layout is free (publisher/repo/file works as well as a flat folder).
// Analysing a file reads its header, so results are cached by path, size and modification time: scanning again is
// cheap, and only new or changed files are read.
class ModelLibrary {
public:
    explicit ModelLibrary(std::vector<std::filesystem::path> folders = {});
    ~ModelLibrary();
    ModelLibrary(const ModelLibrary&) = delete;
    ModelLibrary& operator=(const ModelLibrary&) = delete;

    void setFolders(std::vector<std::filesystem::path> folders);
    std::vector<std::filesystem::path> folders() const;

    // Every .safetensors / .sft / .gguf / .ckpt file, sorted by path, with VAE files linked to their checkpoints.
    std::vector<ModelCard> scan();
    std::vector<ModelCard> ofKind(Kind kind);

    struct Resolution {
        enum class Status { Found, NotFound, Ambiguous } status = Status::NotFound;
        std::optional<ModelCard> card;
        std::string problem;
    };
    // A path to a file, or a name: the file name, the name without extension, or a unique prefix of the file name.
    // Only checkpoints are considered for names.
    Resolution resolve(const std::string& arg);

    // Calls onChange (from a background thread) when a file appears, disappears or changes below the folders.
    void startWatching(std::function<void()> onChange);
    void stopWatching();

private:
    struct CacheEntry {
        uint64_t size = 0;
        std::filesystem::file_time_type time{};
        ModelCard card;
    };
    void watchLoop(std::function<void()> onChange);
    void linkCompanions(std::vector<ModelCard>& cards) const;

    mutable std::mutex mutex_;
    std::vector<std::filesystem::path> folders_;
    std::map<std::filesystem::path, CacheEntry> cache_;

    std::thread watcher_;
    void* stopEvent_ = nullptr;  // HANDLE
};

}  // namespace gh::models
