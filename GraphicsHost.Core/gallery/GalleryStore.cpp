#include "gallery/GalleryStore.h"

#include "core/AtomicFile.h"
#include "core/Log.h"
#include "core/Version.h"

#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <stdexcept>

namespace gh::gallery {

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

class Statement {
public:
    Statement(sqlite3* db, const char* sql) {
        if (sqlite3_prepare_v2(db, sql, -1, &stmt_, nullptr) != SQLITE_OK) {
            throw std::runtime_error(std::string("gallery index: ") + sqlite3_errmsg(db));
        }
    }
    ~Statement() { sqlite3_finalize(stmt_); }
    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;
    sqlite3_stmt* get() const { return stmt_; }

    void bind(int index, const std::string& value) const {
        sqlite3_bind_text(stmt_, index, value.c_str(), static_cast<int>(value.size()), SQLITE_TRANSIENT);
    }
    void bind(int index, int64_t value) const { sqlite3_bind_int64(stmt_, index, value); }

private:
    sqlite3_stmt* stmt_ = nullptr;
};

std::string columnText(sqlite3_stmt* s, int col) {
    const unsigned char* text = sqlite3_column_text(s, col);
    return text ? reinterpret_cast<const char*>(text) : std::string();
}

std::string isoUtcNow() {
    const auto now = std::chrono::system_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_s(&utc, &t);
    char buffer[40];
    std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", utc.tm_year + 1900, utc.tm_mon + 1,
                  utc.tm_mday, utc.tm_hour, utc.tm_min, utc.tm_sec, static_cast<int>(ms.count()));
    return buffer;
}

// The folder and file stem for "now" in local time: 2026/10/23 and 213045.
void localStamp(std::string& folder, std::string& stem) {
    const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local{};
    localtime_s(&local, &t);
    char f[16], s[16];
    std::snprintf(f, sizeof(f), "%04d/%02d/%02d", local.tm_year + 1900, local.tm_mon + 1, local.tm_mday);
    std::snprintf(s, sizeof(s), "%02d%02d%02d", local.tm_hour, local.tm_min, local.tm_sec);
    folder = f;
    stem = s;
}

fs::path pathFromRelative(const fs::path& root, const std::string& relative) {
    return root / fs::path(relative);  // forward slashes are fine for std::filesystem on Windows
}

GalleryEntry entryFromJson(const Json& sidecar, const std::string& relativePath) {
    GalleryEntry e;
    e.path = relativePath;
    e.created = sidecar.value("created", "");
    e.model = sidecar.value("model", "");
    e.family = sidecar.value("family", "");
    e.runtime = sidecar.value("runtime", "");
    e.seed = sidecar.value("seed", int64_t{0});
    e.width = sidecar.value("width", 0);
    e.height = sidecar.value("height", 0);
    e.format = sidecar.value("format", "png");
    if (sidecar.contains("request") && sidecar["request"].is_object()) {
        e.prompt = sidecar["request"].value("prompt", "");
    }
    e.parameters = sidecar;
    return e;
}

}  // namespace

GalleryStore::GalleryStore(fs::path outputsDir) : root_(std::move(outputsDir)) {
    fs::create_directories(root_);
    const std::string dbPath = (root_ / "gallery.db").string();
    if (sqlite3_open(dbPath.c_str(), &db_) != SQLITE_OK) {
        const std::string message = db_ ? sqlite3_errmsg(db_) : "out of memory";
        sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error("cannot open the gallery index " + dbPath + ": " + message);
    }
    exec("PRAGMA journal_mode=WAL");
    exec("PRAGMA synchronous=NORMAL");
    exec(
        "CREATE TABLE IF NOT EXISTS images ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " path TEXT NOT NULL UNIQUE,"
        " created TEXT NOT NULL,"
        " prompt TEXT NOT NULL,"
        " model TEXT NOT NULL,"
        " family TEXT NOT NULL,"
        " runtime TEXT NOT NULL,"
        " seed INTEGER NOT NULL,"
        " width INTEGER NOT NULL,"
        " height INTEGER NOT NULL,"
        " format TEXT NOT NULL,"
        " params TEXT NOT NULL)");
    exec("CREATE INDEX IF NOT EXISTS images_created ON images(created DESC, id DESC)");
    migrate();
}

GalleryStore::~GalleryStore() {
    if (db_) {
        sqlite3_close(db_);
    }
}

// Schema versions (PRAGMA user_version): 1 = the images table; 2 = adds graph_history.
void GalleryStore::migrate() {
    int version = 0;
    {
        Statement s(db_, "PRAGMA user_version");
        if (sqlite3_step(s.get()) == SQLITE_ROW) version = sqlite3_column_int(s.get(), 0);
    }
    if (version < 2) {
        exec(
            "CREATE TABLE IF NOT EXISTS graph_history ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " queue_number INTEGER NOT NULL,"
            " client_id TEXT NOT NULL,"
            " label TEXT NOT NULL,"
            " status TEXT NOT NULL,"
            " created TEXT NOT NULL,"
            " seconds REAL NOT NULL,"
            " error TEXT NOT NULL,"
            " error_node TEXT NOT NULL,"
            " job TEXT NOT NULL,"
            " workflow TEXT NOT NULL,"
            " outputs TEXT NOT NULL,"
            " timings TEXT NOT NULL)");
        exec("CREATE INDEX IF NOT EXISTS graph_history_created ON graph_history(created DESC, id DESC)");
        exec("PRAGMA user_version=2");
    }
}

void GalleryStore::exec(const char* sql) const {
    char* error = nullptr;
    if (sqlite3_exec(db_, sql, nullptr, nullptr, &error) != SQLITE_OK) {
        const std::string message = error ? error : "unknown error";
        sqlite3_free(error);
        throw std::runtime_error("gallery index: " + message + " (" + sql + ")");
    }
}

fs::path GalleryStore::absolutePath(const GalleryEntry& entry) const {
    return pathFromRelative(root_, entry.path).make_preferred();
}

GalleryEntry GalleryStore::fromRow(void* statement) const {
    auto* s = static_cast<sqlite3_stmt*>(statement);
    GalleryEntry e;
    e.id = sqlite3_column_int64(s, 0);
    e.path = columnText(s, 1);
    e.created = columnText(s, 2);
    e.prompt = columnText(s, 3);
    e.model = columnText(s, 4);
    e.family = columnText(s, 5);
    e.runtime = columnText(s, 6);
    e.seed = sqlite3_column_int64(s, 7);
    e.width = sqlite3_column_int(s, 8);
    e.height = sqlite3_column_int(s, 9);
    e.format = columnText(s, 10);
    e.parameters = Json::parse(columnText(s, 11), nullptr, false);
    return e;
}

GalleryEntry GalleryStore::insert(const GalleryEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    Statement s(db_,
                "INSERT INTO images(path,created,prompt,model,family,runtime,seed,width,height,format,params) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)");
    s.bind(1, entry.path);
    s.bind(2, entry.created);
    s.bind(3, entry.prompt);
    s.bind(4, entry.model);
    s.bind(5, entry.family);
    s.bind(6, entry.runtime);
    s.bind(7, entry.seed);
    s.bind(8, static_cast<int64_t>(entry.width));
    s.bind(9, static_cast<int64_t>(entry.height));
    s.bind(10, entry.format);
    s.bind(11, entry.parameters.dump());
    if (sqlite3_step(s.get()) != SQLITE_DONE) {
        throw std::runtime_error(std::string("gallery index: ") + sqlite3_errmsg(db_));
    }
    GalleryEntry stored = entry;
    stored.id = sqlite3_last_insert_rowid(db_);
    return stored;
}

GalleryEntry GalleryStore::add(const engine::Image& image, const engine::GenerationRequest& resolved, int64_t seed,
                               const engine::ModelInfo& model, double seconds, int quality) {
    const auto format = engine::parseImageFormat(resolved.format).value_or(engine::ImageFormat::Png);
    const std::string extension = engine::extensionFor(format);

    GalleryEntry entry;
    entry.created = isoUtcNow();
    entry.prompt = resolved.prompt;
    entry.model = fs::path(model.modelPath).filename().string();
    entry.family = model.family;
    entry.runtime = model.runtimeId;
    entry.seed = seed;
    entry.width = static_cast<int>(image.width);
    entry.height = static_cast<int>(image.height);
    entry.format = resolved.format.empty() ? "png" : resolved.format;

    engine::GenerationRequest recorded = resolved;
    recorded.seed = seed;  // the seed of this very image, so the record alone reproduces it
    entry.parameters = {{"app", kAppName},
                        {"version", kAppVersion},
                        {"created", entry.created},
                        {"model", entry.model},
                        {"family", entry.family},
                        {"runtime", entry.runtime},
                        {"seed", seed},
                        {"width", entry.width},
                        {"height", entry.height},
                        {"format", entry.format},
                        {"seconds", seconds},
                        {"request", engine::toJson(recorded)}};
    const std::string metadata = entry.parameters.dump();

    std::string folder, stem;
    localStamp(folder, stem);
    const fs::path dir = root_ / fs::path(folder);
    fs::create_directories(dir);
    fs::path file;
    for (int n = 1;; ++n) {
        file = dir / (stem + "-" + std::to_string(seed) + "-" + std::to_string(n) + extension);
        if (!fs::exists(file)) {
            break;
        }
    }
    entry.path = folder + "/" + file.filename().string();

    const auto bytes = engine::encodeImage(image, format,
                                           resolved.embedMetadata ? std::optional<std::string>(metadata) : std::nullopt,
                                           quality);
    core::writeFileAtomic(file, std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    fs::path sidecar = file;
    sidecar.replace_extension(".json");
    core::writeFileAtomic(sidecar, entry.parameters.dump(2));
    return insert(entry);
}

std::vector<GalleryEntry> GalleryStore::addResult(const engine::GenerationResult& result, const engine::ModelInfo& model,
                                                  int quality) {
    std::vector<GalleryEntry> stored;
    for (size_t i = 0; i < result.images.size(); ++i) {
        const int64_t seed = i < result.seeds.size() ? result.seeds[i] : result.request.seed;
        stored.push_back(add(result.images[i], result.request, seed, model, result.seconds / std::max<size_t>(1, result.images.size()), quality));
    }
    return stored;
}

std::vector<GalleryEntry> GalleryStore::list(const GalleryQuery& query) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool filter = !query.text.empty();
    Statement s(db_, filter ? "SELECT id,path,created,prompt,model,family,runtime,seed,width,height,format,params FROM images "
                             "WHERE prompt LIKE ? ESCAPE '\\' ORDER BY created DESC, id DESC LIMIT ? OFFSET ?"
                           : "SELECT id,path,created,prompt,model,family,runtime,seed,width,height,format,params FROM images "
                             "ORDER BY created DESC, id DESC LIMIT ? OFFSET ?");
    int index = 1;
    if (filter) {
        std::string pattern = "%";
        for (char c : query.text) {
            if (c == '%' || c == '_' || c == '\\') pattern += '\\';
            pattern += c;
        }
        pattern += "%";
        s.bind(index++, pattern);
    }
    s.bind(index++, static_cast<int64_t>(std::clamp(query.limit, 1, 1000)));
    s.bind(index, static_cast<int64_t>(std::max(query.offset, 0)));
    std::vector<GalleryEntry> out;
    while (sqlite3_step(s.get()) == SQLITE_ROW) {
        out.push_back(fromRow(s.get()));
    }
    return out;
}

size_t GalleryStore::count(const std::string& text) const {
    std::lock_guard<std::mutex> lock(mutex_);
    Statement s(db_, text.empty() ? "SELECT COUNT(*) FROM images" : "SELECT COUNT(*) FROM images WHERE prompt LIKE ? ESCAPE '\\'");
    if (!text.empty()) {
        std::string pattern = "%";
        for (char c : text) {
            if (c == '%' || c == '_' || c == '\\') pattern += '\\';
            pattern += c;
        }
        s.bind(1, pattern + "%");
    }
    return sqlite3_step(s.get()) == SQLITE_ROW ? static_cast<size_t>(sqlite3_column_int64(s.get(), 0)) : 0;
}

std::optional<GalleryEntry> GalleryStore::get(int64_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    Statement s(db_, "SELECT id,path,created,prompt,model,family,runtime,seed,width,height,format,params FROM images WHERE id=?");
    s.bind(1, id);
    if (sqlite3_step(s.get()) == SQLITE_ROW) {
        return fromRow(s.get());
    }
    return std::nullopt;
}

bool GalleryStore::remove(int64_t id) {
    const auto entry = get(id);
    if (!entry) {
        return false;
    }
    std::error_code ignored;
    const fs::path image = absolutePath(*entry);
    fs::path sidecar = image;
    sidecar.replace_extension(".json");
    fs::remove(image, ignored);
    fs::remove(sidecar, ignored);
    std::lock_guard<std::mutex> lock(mutex_);
    Statement s(db_, "DELETE FROM images WHERE id=?");
    s.bind(1, id);
    sqlite3_step(s.get());
    return true;
}

size_t GalleryStore::rescan() {
    size_t added = 0;
    std::vector<std::pair<std::string, Json>> found;  // relative image path, sidecar
    std::error_code ec;
    for (fs::recursive_directory_iterator it(root_, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec) || it->path().extension() != ".json") {
            continue;
        }
        const auto text = core::readFile(it->path());
        const Json sidecar = text ? Json::parse(*text, nullptr, false) : Json();
        if (!sidecar.is_object() || !sidecar.contains("request")) {
            continue;
        }
        const std::string ext = engine::extensionFor(engine::parseImageFormat(sidecar.value("format", "png")).value_or(engine::ImageFormat::Png));
        fs::path image = it->path();
        image.replace_extension(ext);
        if (!fs::exists(image, ec)) {
            continue;
        }
        found.emplace_back(fs::relative(image, root_, ec).generic_string(), sidecar);
    }
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [relative, sidecar] : found) {
        bool known;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            Statement s(db_, "SELECT 1 FROM images WHERE path=?");
            s.bind(1, relative);
            known = sqlite3_step(s.get()) == SQLITE_ROW;
        }
        if (!known) {
            insert(entryFromJson(sidecar, relative));
            ++added;
        }
    }
    // drop rows whose image file is gone
    std::vector<int64_t> missing;
    for (int offset = 0;; offset += 1000) {
        const auto page = list(GalleryQuery{"", 1000, offset});
        for (const auto& entry : page) {
            if (!fs::exists(absolutePath(entry), ec)) {
                missing.push_back(entry.id);
            }
        }
        if (page.size() < 1000) {
            break;
        }
    }
    for (const int64_t id : missing) {
        std::lock_guard<std::mutex> lock(mutex_);
        Statement s(db_, "DELETE FROM images WHERE id=?");
        s.bind(1, id);
        sqlite3_step(s.get());
    }
    if (added > 0 || !missing.empty()) {
        log::info("gallery rescan: " + std::to_string(added) + " added, " + std::to_string(missing.size()) + " dropped");
    }
    return added;
}

}  // namespace gh::gallery

// ---- history of graph jobs ----------------------------------------------------------------------------------------------

namespace gh::gallery {

namespace {
constexpr const char* kHistoryColumns = "id,queue_number,client_id,label,status,created,seconds,error,error_node";
constexpr const char* kHistoryAllColumns = "id,queue_number,client_id,label,status,created,seconds,error,error_node,job,workflow,outputs,timings";

HistoryEntry historyFromRow(sqlite3_stmt* s, bool full) {
    HistoryEntry e;
    e.id = sqlite3_column_int64(s, 0);
    e.queueNumber = sqlite3_column_int64(s, 1);
    e.clientId = columnText(s, 2);
    e.label = columnText(s, 3);
    e.status = columnText(s, 4);
    e.created = columnText(s, 5);
    e.seconds = sqlite3_column_double(s, 6);
    e.error = columnText(s, 7);
    e.errorNode = columnText(s, 8);
    if (full) {
        e.job = Json::parse(columnText(s, 9), nullptr, false);
        e.workflow = Json::parse(columnText(s, 10), nullptr, false);
        e.outputs = Json::parse(columnText(s, 11), nullptr, false);
        e.timings = Json::parse(columnText(s, 12), nullptr, false);
    }
    return e;
}
}  // namespace

int64_t GalleryStore::addHistory(const HistoryEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    Statement s(db_,
                "INSERT INTO graph_history(queue_number,client_id,label,status,created,seconds,error,error_node,job,workflow,outputs,timings) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)");
    s.bind(1, entry.queueNumber);
    s.bind(2, entry.clientId);
    s.bind(3, entry.label);
    s.bind(4, entry.status);
    s.bind(5, entry.created.empty() ? isoUtcNow() : entry.created);
    sqlite3_bind_double(s.get(), 6, entry.seconds);
    s.bind(7, entry.error);
    s.bind(8, entry.errorNode);
    s.bind(9, entry.job.dump());
    s.bind(10, entry.workflow.dump());
    s.bind(11, entry.outputs.dump());
    s.bind(12, entry.timings.dump());
    if (sqlite3_step(s.get()) != SQLITE_DONE) throw std::runtime_error(std::string("gallery index: ") + sqlite3_errmsg(db_));
    return sqlite3_last_insert_rowid(db_);
}

std::vector<HistoryEntry> GalleryStore::history(int limit, int offset) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string sql = std::string("SELECT ") + kHistoryColumns + " FROM graph_history ORDER BY created DESC, id DESC LIMIT ? OFFSET ?";
    Statement s(db_, sql.c_str());
    s.bind(1, static_cast<int64_t>(std::max(limit, 0)));
    s.bind(2, static_cast<int64_t>(std::max(offset, 0)));
    std::vector<HistoryEntry> out;
    while (sqlite3_step(s.get()) == SQLITE_ROW) out.push_back(historyFromRow(s.get(), false));
    return out;
}

size_t GalleryStore::historyCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Statement s(db_, "SELECT COUNT(*) FROM graph_history");
    return sqlite3_step(s.get()) == SQLITE_ROW ? static_cast<size_t>(sqlite3_column_int64(s.get(), 0)) : 0;
}

std::optional<HistoryEntry> GalleryStore::historyItem(int64_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string sql = std::string("SELECT ") + kHistoryAllColumns + " FROM graph_history WHERE id=?";
    Statement s(db_, sql.c_str());
    s.bind(1, id);
    if (sqlite3_step(s.get()) != SQLITE_ROW) return std::nullopt;
    return historyFromRow(s.get(), true);
}

bool GalleryStore::removeHistory(int64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    Statement s(db_, "DELETE FROM graph_history WHERE id=?");
    s.bind(1, id);
    sqlite3_step(s.get());
    return sqlite3_changes(db_) > 0;
}

size_t GalleryStore::pruneHistory(size_t maxItems) {
    if (maxItems == 0) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    Statement s(db_, "DELETE FROM graph_history WHERE id NOT IN (SELECT id FROM graph_history ORDER BY created DESC, id DESC LIMIT ?)");
    s.bind(1, static_cast<int64_t>(maxItems));
    sqlite3_step(s.get());
    return static_cast<size_t>(sqlite3_changes(db_));
}

}  // namespace gh::gallery
