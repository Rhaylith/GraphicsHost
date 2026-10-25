#pragma once

#include "engine/GenerationRequest.h"
#include "engine/ImageCodec.h"
#include "engine/ImageEngine.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct sqlite3;

namespace gh::gallery {

struct GalleryEntry {
    int64_t id = 0;
    std::string path;       // relative to the outputs folder, forward slashes: 2026/10/23/213045-1234-1.png
    std::string created;    // ISO 8601 UTC
    std::string prompt;
    std::string model;      // checkpoint file name
    std::string family;
    std::string runtime;
    int64_t seed = 0;
    int width = 0;
    int height = 0;
    std::string format;     // png | jpeg | webp
    nlohmann::json parameters;  // the full record: request, model, runtime, seconds
};

struct GalleryQuery {
    std::string text;       // matches the prompt (case-insensitive substring); empty = everything
    int limit = 50;
    int offset = 0;
};

// Everything GraphicsHost generated: image files under outputs/YYYY/MM/DD/, a sidecar .json next to each, and an
// SQLite index (outputs/gallery.db) for listing and searching. The index can always be rebuilt from the sidecar files
// (rescan), so losing it loses nothing.
class GalleryStore {
public:
    explicit GalleryStore(std::filesystem::path outputsDir);
    ~GalleryStore();
    GalleryStore(const GalleryStore&) = delete;
    GalleryStore& operator=(const GalleryStore&) = delete;

    const std::filesystem::path& root() const { return root_; }

    // Writes the image (quality applies to JPEG and lossy WebP), its sidecar, and the index row.
    GalleryEntry add(const engine::Image& image, const engine::GenerationRequest& resolved, int64_t seed,
                     const engine::ModelInfo& model, double seconds, int quality = 92);
    // Stores every image of a result. The format and whether to embed the metadata come from result.request.
    std::vector<GalleryEntry> addResult(const engine::GenerationResult& result, const engine::ModelInfo& model,
                                        int quality = 92);

    std::vector<GalleryEntry> list(const GalleryQuery& query = {}) const;  // newest first
    size_t count(const std::string& text = {}) const;
    std::optional<GalleryEntry> get(int64_t id) const;
    std::filesystem::path absolutePath(const GalleryEntry& entry) const;

    // Deletes the image file, its sidecar and the index row. False if there is no such entry.
    bool remove(int64_t id);

    // Adds index rows for sidecar files that are not indexed yet and drops rows whose image is gone.
    // Returns the number of rows added.
    size_t rescan();

private:
    GalleryEntry insert(const GalleryEntry& entry);
    GalleryEntry fromRow(void* statement) const;
    void exec(const char* sql) const;

    std::filesystem::path root_;
    sqlite3* db_ = nullptr;
    mutable std::mutex mutex_;
};

}  // namespace gh::gallery
