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

// One finished graph job in the history (workflow plan, section 6.6). Every output file is also a normal gallery entry; `outputs`
// lists them per node. The job and the editor workflow are kept so a past run can be reopened or repeated.
struct HistoryEntry {
    int64_t id = 0;
    int64_t queueNumber = 0;     // the number the queue gave the job
    std::string clientId;
    std::string label;
    std::string status;          // done | failed | interrupted | cancelled
    std::string created;         // ISO 8601 UTC, when the job finished
    double seconds = 0;
    std::string error;
    std::string errorNode;
    nlohmann::json job;          // the job format
    nlohmann::json workflow;     // the editor format, or null
    nlohmann::json outputs = nlohmann::json::array();   // [{"node": "9", "gallery": 12, "path": "2026/10/25/..png"}]
    nlohmann::json timings = nlohmann::json::object();  // node id -> milliseconds
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

    // History of graph jobs. `history` lists newest first and leaves out the job and workflow (use historyItem for those).
    int64_t addHistory(const HistoryEntry& entry);
    std::vector<HistoryEntry> history(int limit = 50, int offset = 0) const;
    size_t historyCount() const;
    std::optional<HistoryEntry> historyItem(int64_t id) const;
    bool removeHistory(int64_t id);   // the row only; output files stay in the gallery
    // Keeps the newest `maxItems` rows (0 = keep everything); returns how many were removed. Never touches files.
    size_t pruneHistory(size_t maxItems);

    // Adds index rows for sidecar files that are not indexed yet and drops rows whose image is gone.
    // Returns the number of rows added.
    size_t rescan();

private:
    GalleryEntry insert(const GalleryEntry& entry);
    GalleryEntry fromRow(void* statement) const;
    void exec(const char* sql) const;
    void migrate();

    std::filesystem::path root_;
    sqlite3* db_ = nullptr;
    mutable std::mutex mutex_;
};

}  // namespace gh::gallery
