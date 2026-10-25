#include "models/DownloadManager.h"

#include "core/AtomicFile.h"
#include "core/Log.h"
#include "core/Sha256.h"
#include "models/ModelLibrary.h"

#include <curl/curl.h>
#include <windows.h>

#include <algorithm>
#include <cstdio>

namespace gh::models {

namespace fs = std::filesystem;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

const bool g_curlReady = [] { return curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK; }();

uint64_t defaultFreeSpace(const fs::path& path) {
    fs::path probe = fs::absolute(path);
    std::error_code ec;
    while (!probe.empty() && !fs::exists(probe, ec)) {
        probe = probe.parent_path();
    }
    ULARGE_INTEGER freeBytes{};
    if (probe.empty() || !GetDiskFreeSpaceExW(probe.c_str(), &freeBytes, nullptr, nullptr)) {
        return UINT64_MAX;  // unknown: do not block the download
    }
    return freeBytes.QuadPart;
}

std::string gib(uint64_t bytes) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.2f GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return b;
}

std::string isoNow() {
    const std::time_t t = std::time(nullptr);
    std::tm utc{};
    gmtime_s(&utc, &t);
    char b[32];
    std::strftime(b, sizeof(b), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return b;
}

struct Transfer {
    FILE* file = nullptr;
    uint64_t resumeFrom = 0;
    uint64_t written = 0;
    bool failedToWrite = false;
};

size_t onBody(char* data, size_t size, size_t count, void* user) {
    auto* t = static_cast<Transfer*>(user);
    const size_t n = size * count;
    if (std::fwrite(data, 1, n, t->file) != n) {
        t->failedToWrite = true;
        return 0;
    }
    t->written += n;
    return n;
}

struct Progress {
    Download* download = nullptr;
    std::atomic<uint64_t>* done = nullptr;
    std::atomic<uint64_t>* total = nullptr;
    std::atomic<double>* speed = nullptr;
    uint64_t resumeFrom = 0;
    bool* stopRequested = nullptr;
    std::function<void()> tick;
    Clock::time_point lastNotify{};
    Clock::time_point lastSpeed{};
    uint64_t lastSpeedBytes = 0;
    std::function<bool()> shouldStop;
};

int onProgress(void* user, curl_off_t dltotal, curl_off_t dlnow, curl_off_t, curl_off_t) {
    auto* p = static_cast<Progress*>(user);
    p->done->store(p->resumeFrom + static_cast<uint64_t>(dlnow));
    if (dltotal > 0 && p->total->load() == 0) {
        p->total->store(p->resumeFrom + static_cast<uint64_t>(dltotal));
    }
    const auto now = Clock::now();
    if (p->lastSpeed == Clock::time_point{}) {
        p->lastSpeed = now;
        p->lastSpeedBytes = static_cast<uint64_t>(dlnow);
    } else if (now - p->lastSpeed >= std::chrono::milliseconds(500)) {
        const double seconds = std::chrono::duration<double>(now - p->lastSpeed).count();
        p->speed->store(static_cast<double>(static_cast<uint64_t>(dlnow) - p->lastSpeedBytes) / seconds);
        p->lastSpeed = now;
        p->lastSpeedBytes = static_cast<uint64_t>(dlnow);
    }
    if (now - p->lastNotify >= std::chrono::milliseconds(100)) {
        p->lastNotify = now;
        p->tick();
    }
    return p->shouldStop() ? 1 : 0;
}

struct CurlHandle {
    CURL* h = curl_easy_init();
    curl_slist* headers = nullptr;
    ~CurlHandle() {
        curl_slist_free_all(headers);
        curl_easy_cleanup(h);
    }
};

void configure(CurlHandle& c, const DownloadRequest& r) {
    curl_easy_setopt(c.h, CURLOPT_URL, r.url.c_str());
    curl_easy_setopt(c.h, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c.h, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(c.h, CURLOPT_USERAGENT, "GraphicsHost/0.1");
    curl_easy_setopt(c.h, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(c.h, CURLOPT_LOW_SPEED_LIMIT, 1L);   // a transfer that moves less than 1 byte/s for 60 s is dead
    curl_easy_setopt(c.h, CURLOPT_LOW_SPEED_TIME, 60L);
    curl_easy_setopt(c.h, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c.h, CURLOPT_ACCEPT_ENCODING, "identity");  // model files are already compressed; sizes must match
    if (!r.bearerToken.empty()) {
        // curl drops this header when a redirect leaves the original host (HF sends files from a CDN)
        c.headers = curl_slist_append(c.headers, ("Authorization: Bearer " + r.bearerToken).c_str());
        curl_easy_setopt(c.h, CURLOPT_HTTPHEADER, c.headers);
    }
}

}  // namespace

const char* toString(DownloadState s) {
    switch (s) {
        case DownloadState::Queued: return "queued";
        case DownloadState::Running: return "running";
        case DownloadState::Paused: return "paused";
        case DownloadState::Completed: return "completed";
        case DownloadState::Failed: return "failed";
        case DownloadState::Cancelled: return "cancelled";
    }
    return "unknown";
}

const char* toString(DownloadError e) {
    switch (e) {
        case DownloadError::None: return "none";
        case DownloadError::Network: return "network";
        case DownloadError::Http: return "http";
        case DownloadError::Unauthorized: return "unauthorized";
        case DownloadError::DiskSpace: return "disk-space";
        case DownloadError::Checksum: return "checksum";
        case DownloadError::Io: return "io";
    }
    return "unknown";
}

DownloadError Download::error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}
std::string Download::message() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return message_;
}
std::string Download::sha256() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sha256_;
}
bool Download::finished() const {
    const auto s = state_.load();
    return s == DownloadState::Completed || s == DownloadState::Failed || s == DownloadState::Cancelled;
}
bool Download::waitFor(std::chrono::milliseconds timeout) const {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock, timeout, [this] { return finished(); });
}

DownloadManager::DownloadManager(int maxConcurrent) : freeSpace_(defaultFreeSpace) {
    for (int i = 0; i < std::max(1, maxConcurrent); ++i) {
        workers_.emplace_back([this] { worker(); });
    }
}

DownloadManager::~DownloadManager() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        for (const auto& d : all_) d->pauseRequested_ = true;  // keep the partial files
    }
    wake_.notify_all();
    for (auto& w : workers_) w.join();
}

std::shared_ptr<Download> DownloadManager::enqueue(DownloadRequest request) {
    if (request.label.empty()) request.label = request.destination.filename().string();
    std::shared_ptr<Download> d;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        d.reset(new Download(nextId_++, std::move(request)));
        all_.push_back(d);
        queue_.push_back(d);
    }
    wake_.notify_one();
    notify(*d);
    return d;
}

std::shared_ptr<Download> DownloadManager::find(uint64_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& d : all_) if (d->id() == id) return d;
    return nullptr;
}

std::vector<std::shared_ptr<Download>> DownloadManager::list() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return all_;
}

void DownloadManager::setObserver(Observer observer) {
    std::lock_guard<std::mutex> lock(mutex_);
    observer_ = std::move(observer);
}

void DownloadManager::setFreeSpaceFunction(FreeSpaceFn fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    freeSpace_ = fn ? std::move(fn) : FreeSpaceFn(defaultFreeSpace);
}

void DownloadManager::notify(const Download& d) {
    Observer observer;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        observer = observer_;
    }
    if (observer) observer(d);
}

void DownloadManager::pause(uint64_t id) {
    const auto d = find(id);
    if (!d || d->finished()) return;
    d->pauseRequested_ = true;
    std::unique_lock<std::mutex> lock(mutex_);
    const auto it = std::find(queue_.begin(), queue_.end(), d);
    if (it != queue_.end()) {  // not started yet
        queue_.erase(it);
        lock.unlock();
        { std::lock_guard<std::mutex> l(d->mutex_); d->state_ = DownloadState::Paused; }
        d->changed_.notify_all();
        notify(*d);
    }
}

void DownloadManager::resume(uint64_t id) {
    const auto d = find(id);
    if (!d) return;
    const auto s = d->state();
    if (s != DownloadState::Paused && s != DownloadState::Failed) return;
    {
        std::lock_guard<std::mutex> l(d->mutex_);
        d->pauseRequested_ = false;
        d->cancelRequested_ = false;
        d->error_ = DownloadError::None;
        d->message_.clear();
        d->state_ = DownloadState::Queued;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(d);
    }
    wake_.notify_one();
    notify(*d);
}

void DownloadManager::cancel(uint64_t id) {
    const auto d = find(id);
    if (!d || d->finished()) return;
    d->cancelRequested_ = true;
    std::unique_lock<std::mutex> lock(mutex_);
    const auto it = std::find(queue_.begin(), queue_.end(), d);
    const bool idle = it != queue_.end() || d->state() == DownloadState::Paused || d->state() == DownloadState::Failed;
    if (it != queue_.end()) queue_.erase(it);
    lock.unlock();
    if (idle) {  // nothing is transferring: finish it here
        std::error_code ec;
        fs::path part = d->request().destination;
        part += ".part";
        fs::remove(part, ec);
        { std::lock_guard<std::mutex> l(d->mutex_); d->state_ = DownloadState::Cancelled; }
        d->changed_.notify_all();
        notify(*d);
    }
}

void DownloadManager::worker() {
    for (;;) {
        std::shared_ptr<Download> d;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_) return;
            d = queue_.front();
            queue_.pop_front();
        }
        if (d->state() != DownloadState::Queued) continue;
        transfer(d);
    }
}

void DownloadManager::transfer(const std::shared_ptr<Download>& dl) {
    Download& d = *dl;
    const DownloadRequest& req = d.request();

    auto setState = [&](DownloadState s) {
        { std::lock_guard<std::mutex> l(d.mutex_); d.state_ = s; }
        d.changed_.notify_all();
        notify(d);
    };
    auto fail = [&](DownloadError e, std::string message, int status = 0) {
        {
            std::lock_guard<std::mutex> l(d.mutex_);
            d.error_ = e;
            d.message_ = std::move(message);
            d.state_ = DownloadState::Failed;
        }
        d.http_ = status;
        d.changed_.notify_all();
        log::warn("download failed: " + req.label + ": " + d.message());
        notify(d);
    };
    FreeSpaceFn freeSpace;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        freeSpace = freeSpace_;
    }

    fs::path part = req.destination;
    part += ".part";
    std::error_code ec;
    fs::create_directories(req.destination.parent_path(), ec);
    setState(DownloadState::Running);

    auto stopRequested = [&] { return d.pauseRequested_.load() || d.cancelRequested_.load(); };
    auto finishStopped = [&] {
        if (d.cancelRequested_) {
            std::error_code e;
            fs::remove(part, e);
            setState(DownloadState::Cancelled);
        } else {
            setState(DownloadState::Paused);
        }
    };
    if (stopRequested()) { finishStopped(); return; }

    uint64_t resumeFrom = fs::exists(part, ec) ? fs::file_size(part, ec) : 0;

    // 1. how big is it? (HEAD, following redirects)
    uint64_t total = 0;
    {
        CurlHandle c;
        configure(c, req);
        curl_easy_setopt(c.h, CURLOPT_NOBODY, 1L);
        const CURLcode rc = curl_easy_perform(c.h);
        long status = 0;
        curl_easy_getinfo(c.h, CURLINFO_RESPONSE_CODE, &status);
        if (rc == CURLE_OK && (status == 401 || status == 403)) {
            fail(DownloadError::Unauthorized, "the server refused access (HTTP " + std::to_string(status) + "): this model may need an access token and its terms accepted", static_cast<int>(status));
            return;
        }
        if (rc == CURLE_OK && status == 404) {
            fail(DownloadError::Http, "not found (HTTP 404): " + req.url, 404);
            return;
        }
        if (rc == CURLE_OK && status >= 200 && status < 300) {
            curl_off_t length = -1;
            curl_easy_getinfo(c.h, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &length);
            if (length > 0) total = static_cast<uint64_t>(length);
        } else if (rc != CURLE_OK) {
            fail(DownloadError::Network, std::string("cannot reach the server: ") + curl_easy_strerror(rc));
            return;
        }  // other statuses (e.g. 405 for HEAD): size unknown, try the download anyway
    }
    d.total_ = total;
    if (stopRequested()) { finishStopped(); return; }

    // 2. room on the disk?
    if (total > 0) {
        if (resumeFrom > total) {  // not a part of this file
            fs::remove(part, ec);
            resumeFrom = 0;
        }
        const uint64_t needed = total - resumeFrom + (1u << 20);
        const uint64_t available = freeSpace(req.destination);
        if (available < needed) {
            fail(DownloadError::DiskSpace, "not enough disk space: " + gib(needed) + " needed, " + gib(available) + " free");
            return;
        }
    }

    // 3. transfer (a part that is already complete skips this)
    if (!(total > 0 && resumeFrom == total)) {
        for (int attempt = 0; attempt < 2; ++attempt) {
            Transfer t;
            t.resumeFrom = resumeFrom;
            t.file = nullptr;
            fopen_s(&t.file, part.string().c_str(), resumeFrom > 0 ? "ab" : "wb");
            if (!t.file) {
                fail(DownloadError::Io, "cannot write " + part.string());
                return;
            }
            Progress p;
            p.done = &d.done_;
            p.total = &d.total_;
            p.speed = &d.speed_;
            p.resumeFrom = resumeFrom;
            p.tick = [&] { notify(d); };
            p.shouldStop = stopRequested;
            d.done_ = resumeFrom;

            CurlHandle c;
            configure(c, req);
            curl_easy_setopt(c.h, CURLOPT_FAILONERROR, 1L);
            curl_easy_setopt(c.h, CURLOPT_WRITEFUNCTION, onBody);
            curl_easy_setopt(c.h, CURLOPT_WRITEDATA, &t);
            curl_easy_setopt(c.h, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(c.h, CURLOPT_XFERINFOFUNCTION, onProgress);
            curl_easy_setopt(c.h, CURLOPT_XFERINFODATA, &p);
            if (resumeFrom > 0) curl_easy_setopt(c.h, CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(resumeFrom));
            const CURLcode rc = curl_easy_perform(c.h);
            long status = 0;
            curl_easy_getinfo(c.h, CURLINFO_RESPONSE_CODE, &status);
            std::fclose(t.file);

            if (rc == CURLE_OK) break;
            if (rc == CURLE_ABORTED_BY_CALLBACK) { finishStopped(); return; }
            if (rc == CURLE_RANGE_ERROR && attempt == 0 && resumeFrom > 0) {  // the server cannot resume: start over
                fs::remove(part, ec);
                resumeFrom = 0;
                d.done_ = 0;
                continue;
            }
            if (rc == CURLE_HTTP_RETURNED_ERROR) {
                if (status == 401 || status == 403) {
                    fail(DownloadError::Unauthorized, "the server refused access (HTTP " + std::to_string(status) + "): this model may need an access token and its terms accepted", static_cast<int>(status));
                } else {
                    fail(DownloadError::Http, "the server answered HTTP " + std::to_string(status), static_cast<int>(status));
                }
                return;
            }
            if (t.failedToWrite) {
                fail(DownloadError::Io, "cannot write " + part.string() + " (disk full?)");
                return;
            }
            fail(DownloadError::Network, std::string("transfer failed: ") + curl_easy_strerror(rc) + " (the partial file is kept; resume to continue)");
            return;
        }
    }

    // 4. complete? verify
    const uint64_t size = fs::file_size(part, ec);
    if (total > 0 && size != total) {
        fail(DownloadError::Network, "the file is incomplete (" + std::to_string(size) + " of " + std::to_string(total) + " bytes); resume to continue");
        return;
    }
    d.done_ = size;
    d.total_ = size;
    const auto hash = core::sha256File(part, [&](uint64_t, uint64_t) { return !stopRequested(); });
    if (!hash) {
        if (stopRequested()) { finishStopped(); return; }
        fail(DownloadError::Io, "cannot read " + part.string() + " to verify it");
        return;
    }
    if (!req.sha256.empty()) {
        std::string expected = req.sha256;
        std::transform(expected.begin(), expected.end(), expected.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (*hash != expected) {
            fs::remove(part, ec);
            fail(DownloadError::Checksum, "checksum mismatch: expected " + expected + " but the download has " + *hash + " (the file was deleted)");
            return;
        }
    }

    // 5. publish: rename, then the source record
    if (!MoveFileExW(part.c_str(), req.destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        fail(DownloadError::Io, "cannot move the finished download to " + req.destination.string());
        return;
    }
    Json record = req.source.is_object() ? req.source : Json::object();
    record["url"] = req.url;
    if (!record.contains("file")) record["file"] = req.destination.filename().string();  // the name in the repository, when known
    record["localFile"] = req.destination.filename().string();
    record["sha256"] = *hash;
    record["sizeBytes"] = size;
    record["downloadedAt"] = isoNow();
    try {
        core::writeFileAtomic(sourceSidecarFor(req.destination), record.dump(2));
    } catch (const std::exception& e) {
        log::warn(std::string("download finished but its source record could not be written: ") + e.what());
    }
    {
        std::lock_guard<std::mutex> l(d.mutex_);
        d.sha256_ = *hash;
    }
    log::info("downloaded " + req.destination.string() + " (" + gib(size) + ")");
    setState(DownloadState::Completed);
}

}  // namespace gh::models
