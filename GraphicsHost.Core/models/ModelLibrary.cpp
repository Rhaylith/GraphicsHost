#include "models/ModelLibrary.h"

#include "core/AtomicFile.h"
#include "core/Sha256.h"

#include <windows.h>

#include <algorithm>

namespace gh::models {

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

std::string lowerExt(const fs::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return e;
}

bool isModelFile(const fs::path& p, bool kindKnown) {
    const std::string ext = lowerExt(p);
    if (ext == ".safetensors" || ext == ".sft" || ext == ".gguf" || ext == ".ckpt") return true;
    // pickled files cannot be inspected, so they only count where a folder says what they are
    return kindKnown && (ext == ".pth" || ext == ".pt" || ext == ".bin");
}

// The kind a kind folder in the path (loras/, vae/, text_encoder/ ...) asks for; the outermost one wins.
std::optional<Kind> kindFromRelative(const fs::path& relative) {
    for (const auto& part : relative.parent_path())
        if (const auto kind = kindFromFolder(part.string())) return kind;
    return std::nullopt;
}

fs::path previewFor(const fs::path& file) {
    for (const char* ext : {".png", ".jpg", ".jpeg", ".webp"}) {
        fs::path stemmed = file;
        stemmed.replace_extension(std::string(".preview") + ext);
        std::error_code ec;
        if (fs::exists(stemmed, ec)) return stemmed;
        fs::path full = file;
        full += std::string(".preview") + ext;
        if (fs::exists(full, ec)) return full;
    }
    return {};
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace

fs::path sourceSidecarFor(const fs::path& modelFile) {
    fs::path p = modelFile;
    p += ".source.json";
    return p;
}

ModelLibrary::ModelLibrary(std::vector<fs::path> folders) : folders_(std::move(folders)) {}

ModelLibrary::~ModelLibrary() {
    stopWatching();
}

void ModelLibrary::setFolders(std::vector<fs::path> folders) {
    std::lock_guard<std::mutex> lock(mutex_);
    folders_ = std::move(folders);
}

std::vector<fs::path> ModelLibrary::folders() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return folders_;
}

std::vector<ModelCard> ModelLibrary::scan() {
    std::lock_guard<std::mutex> lock(mutex_);

    // every folder to look in: the plain ones (the kind comes from the header, or from a kind subfolder) and the extra
    // ones from model-paths.json (everything below is of that kind)
    struct Source {
        fs::path root;
        std::optional<Kind> forced;
        bool fromRoot = false;
    };
    std::vector<Source> sources;
    for (const auto& folder : folders_) sources.push_back({folder, std::nullopt, false});
    for (const auto& [kind, list] : extraRoots_)
        for (const auto& folder : list) sources.push_back({folder, kind, true});

    struct Found {
        fs::path root;
        std::string relative;
        std::optional<Kind> forced;
        bool fromRoot = false;
    };
    std::map<fs::path, Found> found;
    for (const auto& source : sources) {
        std::error_code ec;
        for (fs::recursive_directory_iterator it(source.root, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file(ec)) continue;
            const fs::path& file = it->path();
            const fs::path relative = file.lexically_relative(source.root);
            const std::optional<Kind> forced = source.forced ? source.forced : kindFromRelative(relative);
            if (!isModelFile(file, forced.has_value())) continue;
            const auto existing = found.find(file);
            if (existing != found.end() && (existing->second.fromRoot || !source.fromRoot)) continue;   // an extra root's say wins
            found[file] = Found{source.root, relative.generic_string(), forced, source.fromRoot};
        }
    }

    std::vector<ModelCard> cards;
    std::map<fs::path, CacheEntry> fresh;
    for (const auto& [file, where] : found) {
        std::error_code ec;
        const uint64_t size = fs::file_size(file, ec);
        const auto time = fs::last_write_time(file, ec);
        ModelCard card;
        if (const auto hit = cache_.find(file); hit != cache_.end() && hit->second.size == size && hit->second.time == time && hit->second.forced == where.forced) {
            card = hit->second.card;
        } else {
            card = describeFile(file, where.forced);
        }
        card.root = where.root;
        card.relative = where.relative;
        card.kindSource = !where.forced ? "header" : where.fromRoot ? "root" : "folder";
        card.preview = previewFor(file);
        // the sidecar may change without the model changing: always read it
        card.source = nullptr;
        card.sha256.clear();
        if (const auto text = core::readFile(sourceSidecarFor(file))) {
            const Json doc = Json::parse(*text, nullptr, false);
            if (doc.is_object()) {
                card.source = doc;
                if (doc.contains("sha256") && doc["sha256"].is_string()) card.sha256 = doc["sha256"].get<std::string>();
            }
        }
        if (card.sha256.empty()) {
            if (const auto known = hashes_.find(file); known != hashes_.end() && known->second.size == size && known->second.time == time) card.sha256 = known->second.sha256;
        }
        fresh[file] = CacheEntry{size, time, where.forced, card};
        cards.push_back(std::move(card));
    }
    cache_ = std::move(fresh);
    linkCompanions(cards);
    return cards;
}

void ModelLibrary::linkCompanions(std::vector<ModelCard>& cards) const {
    for (auto& card : cards) {
        if (card.kind != Kind::Checkpoint) continue;
        card.vae.clear();
        card.companions.clear();
        // 1. recorded by a download
        if (card.source.is_object() && card.source.contains("companions") && card.source["companions"].is_array()) {
            for (const auto& c : card.source["companions"]) {
                const std::string file = c.value("file", "");
                if (file.empty() || !fs::exists(card.path.parent_path() / file)) continue;
                card.companions.push_back(file);
                if (c.value("role", "") == "vae") card.vae = file;
            }
        }
        if (!card.vae.empty()) continue;
        // 2. a VAE file next to it whose name starts with the checkpoint's name and mentions vae
        for (const auto& other : cards) {
            if (other.kind != Kind::Vae || other.path.parent_path() != card.path.parent_path()) continue;
            const std::string o = lower(other.name), c = lower(card.name);
            if (o.rfind(c, 0) == 0 && o.find("vae") != std::string::npos) {
                card.vae = other.file;
                card.companions.push_back(other.file);
                break;
            }
        }
    }
}

std::vector<ModelCard> ModelLibrary::ofKind(Kind kind) {
    auto all = scan();
    all.erase(std::remove_if(all.begin(), all.end(), [kind](const ModelCard& c) { return c.kind != kind; }), all.end());
    return all;
}

ModelLibrary::Resolution ModelLibrary::resolve(const std::string& arg) {
    Resolution r;
    if (arg.empty()) {
        r.problem = "no model given";
        return r;
    }
    std::error_code ec;
    if (fs::is_regular_file(arg, ec)) {
        const fs::path absolute = fs::absolute(arg, ec);
        // use the library's card when the file is one of ours (it carries the linked VAE and the source record)
        for (auto& card : scan()) {
            if (fs::equivalent(card.path, absolute, ec)) {
                r.status = Resolution::Status::Found;
                r.card = card;
                return r;
            }
        }
        r.status = Resolution::Status::Found;
        r.card = describeFile(absolute);
        return r;
    }
    const auto checkpoints = ofKind(Kind::Checkpoint);
    std::vector<const ModelCard*> prefix;
    for (const auto& c : checkpoints) {
        if (c.file == arg || c.name == arg || c.relative == arg) {
            r.status = Resolution::Status::Found;
            r.card = c;
            return r;
        }
        if (c.file.rfind(arg, 0) == 0) prefix.push_back(&c);
    }
    if (prefix.size() == 1) {
        r.status = Resolution::Status::Found;
        r.card = *prefix[0];
    } else if (prefix.empty()) {
        r.problem = "model not found: " + arg;
        if (!folders().empty()) r.problem += " (looked in " + folders().front().string() + ")";
    } else {
        r.status = Resolution::Status::Ambiguous;
        r.problem = "'" + arg + "' matches " + std::to_string(prefix.size()) + " models; be more specific";
    }
    return r;
}

void ModelLibrary::setExtraRoots(ExtraRoots roots) {
    std::lock_guard<std::mutex> lock(mutex_);
    extraRoots_ = std::move(roots);
}

ExtraRoots ModelLibrary::extraRoots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return extraRoots_;
}

std::vector<std::string> ModelLibrary::names(Kind kind) {
    std::vector<std::string> out;
    for (const auto& card : ofKind(kind)) out.push_back(card.relative);
    std::sort(out.begin(), out.end());
    return out;
}

std::optional<ModelCard> ModelLibrary::find(Kind kind, const std::string& name) {
    const auto cards = ofKind(kind);
    for (const auto& c : cards)
        if (c.relative == name) return c;
    // a file name or a name without extension, when that is unambiguous; then a unique prefix of the file name
    for (const int match : {0, 1, 2}) {
        std::vector<const ModelCard*> hits;
        for (const auto& c : cards) {
            const bool ok = match == 0 ? c.file == name : match == 1 ? c.name == name : c.file.rfind(name, 0) == 0;
            if (ok) hits.push_back(&c);
        }
        if (hits.size() == 1) return *hits[0];
        if (hits.size() > 1) return std::nullopt;
    }
    return std::nullopt;
}

void ModelLibrary::setHashCache(fs::path file) {
    std::lock_guard<std::mutex> lock(mutex_);
    hashFile_ = std::move(file);
    hashes_.clear();
    const auto text = core::readFile(hashFile_);
    if (!text) return;
    const Json doc = Json::parse(*text, nullptr, false);
    if (!doc.is_object()) return;
    for (const auto& [path, entry] : doc.items()) {
        if (!entry.is_object() || !entry.contains("sha256")) continue;
        HashEntry h;
        h.size = entry.value("size", uint64_t{0});
        h.time = fs::file_time_type(fs::file_time_type::duration(entry.value("time", int64_t{0})));
        h.sha256 = entry.value("sha256", std::string());
        hashes_[fs::path(path)] = h;
    }
}

std::string ModelLibrary::sha256(const ModelCard& card) {
    if (!card.sha256.empty()) return card.sha256;
    std::error_code ec;
    const uint64_t size = fs::file_size(card.path, ec);
    const auto time = fs::last_write_time(card.path, ec);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (const auto known = hashes_.find(card.path); known != hashes_.end() && known->second.size == size && known->second.time == time) return known->second.sha256;
    }
    const auto hash = core::sha256File(card.path);   // reads the whole file: not under the lock
    if (!hash) return {};
    std::lock_guard<std::mutex> lock(mutex_);
    hashes_[card.path] = HashEntry{size, time, *hash};
    if (!hashFile_.empty()) {
        Json doc = Json::object();
        for (const auto& [path, h] : hashes_)
            doc[path.generic_string()] = {{"size", h.size}, {"time", static_cast<int64_t>(h.time.time_since_epoch().count())}, {"sha256", h.sha256}};
        core::writeFileAtomic(hashFile_, doc.dump(1));
    }
    return *hash;
}

void ModelLibrary::startWatching(std::function<void()> onChange) {
    stopWatching();
    stopEvent_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    watcher_ = std::thread([this, onChange = std::move(onChange)] { watchLoop(onChange); });
}

void ModelLibrary::stopWatching() {
    if (stopEvent_) {
        SetEvent(static_cast<HANDLE>(stopEvent_));
    }
    if (watcher_.joinable()) {
        watcher_.join();
    }
    if (stopEvent_) {
        CloseHandle(static_cast<HANDLE>(stopEvent_));
        stopEvent_ = nullptr;
    }
}

void ModelLibrary::watchLoop(std::function<void()> onChange) {
    std::vector<HANDLE> handles;
    std::vector<fs::path> watched = folders();
    for (const auto& [kind, list] : extraRoots()) watched.insert(watched.end(), list.begin(), list.end());
    for (const auto& folder : watched) {
        std::error_code ec;
        if (!fs::is_directory(folder, ec)) continue;
        HANDLE h = FindFirstChangeNotificationW(folder.c_str(), TRUE,
                                                FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME |
                                                    FILE_NOTIFY_CHANGE_SIZE | FILE_NOTIFY_CHANGE_LAST_WRITE);
        if (h != INVALID_HANDLE_VALUE) handles.push_back(h);
    }
    std::vector<HANDLE> waitFor = handles;
    waitFor.push_back(static_cast<HANDLE>(stopEvent_));
    for (;;) {
        const DWORD r = WaitForMultipleObjects(static_cast<DWORD>(waitFor.size()), waitFor.data(), FALSE, INFINITE);
        if (r == WAIT_OBJECT_0 + waitFor.size() - 1 || r == WAIT_FAILED) break;
        if (r < WAIT_OBJECT_0 + handles.size()) {
            FindNextChangeNotification(handles[r - WAIT_OBJECT_0]);
            // a download writes many times: wait for it to settle before telling anyone
            if (WaitForSingleObject(static_cast<HANDLE>(stopEvent_), 400) == WAIT_OBJECT_0) break;
            if (onChange) onChange();
        }
    }
    for (HANDLE h : handles) FindCloseChangeNotification(h);
}

}  // namespace gh::models
