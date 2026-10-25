#include "models/ModelLibrary.h"

#include "core/AtomicFile.h"

#include <windows.h>

#include <algorithm>

namespace gh::models {

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

bool isModelFile(const fs::path& p) {
    const std::string ext = p.extension().string();
    return ext == ".safetensors" || ext == ".sft" || ext == ".gguf" || ext == ".ckpt";
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
    std::vector<fs::path> files;
    for (const auto& folder : folders_) {
        std::error_code ec;
        for (fs::recursive_directory_iterator it(folder, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec)) {
            if (it->is_regular_file(ec) && isModelFile(it->path())) {
                files.push_back(it->path());
            }
        }
    }
    std::sort(files.begin(), files.end());
    files.erase(std::unique(files.begin(), files.end()), files.end());

    std::vector<ModelCard> cards;
    std::map<fs::path, CacheEntry> fresh;
    for (const auto& file : files) {
        std::error_code ec;
        const uint64_t size = fs::file_size(file, ec);
        const auto time = fs::last_write_time(file, ec);
        ModelCard card;
        if (const auto hit = cache_.find(file); hit != cache_.end() && hit->second.size == size && hit->second.time == time) {
            card = hit->second.card;
        } else {
            card = describeFile(file);
        }
        // the sidecar may change without the model changing: always read it
        card.source = nullptr;
        if (const auto text = core::readFile(sourceSidecarFor(file))) {
            const Json doc = Json::parse(*text, nullptr, false);
            if (doc.is_object()) card.source = doc;
        }
        fresh[file] = CacheEntry{size, time, card};
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
        if (c.file == arg || c.name == arg) {
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
    for (const auto& folder : folders()) {
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
