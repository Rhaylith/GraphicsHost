#pragma once

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gh::models {

enum class DownloadState { Queued, Running, Paused, Completed, Failed, Cancelled };
const char* toString(DownloadState state);

enum class DownloadError {
    None,
    Network,       // connection problem, timeout, TLS
    Http,          // the server answered with an error status (httpStatus)
    Unauthorized,  // 401 or 403: a token is missing or the model is gated
    DiskSpace,
    Checksum,
    Io,            // cannot write the file
};
const char* toString(DownloadError error);

struct DownloadRequest {
    std::string url;
    std::filesystem::path destination;   // the final file; the download is written to <destination>.part first
    std::string label;                   // for display, defaults to the file name
    std::string sha256;                  // expected hash (hex); empty = do not verify, only record
    std::string bearerToken;             // sent as Authorization: Bearer ... (and dropped on redirects to other hosts)
    nlohmann::json source = nullptr;     // written, with size and hash, to <destination>.source.json when done
};

// One transfer. Safe to read from any thread.
class Download {
public:
    uint64_t id() const { return id_; }
    const DownloadRequest& request() const { return request_; }
    DownloadState state() const { return state_.load(); }
    uint64_t bytesDone() const { return done_.load(); }
    uint64_t bytesTotal() const { return total_.load(); }   // 0 while unknown
    double bytesPerSecond() const { return speed_.load(); }
    DownloadError error() const;
    int httpStatus() const { return http_.load(); }
    std::string message() const;                              // the failure reason, or empty
    std::string sha256() const;                               // of the finished file
    bool finished() const;

    // Blocks until the download is completed, failed or cancelled (paused downloads keep waiting).
    bool waitFor(std::chrono::milliseconds timeout) const;

private:
    friend class DownloadManager;
    Download(uint64_t id, DownloadRequest request) : id_(id), request_(std::move(request)) {}

    const uint64_t id_;
    const DownloadRequest request_;
    std::atomic<DownloadState> state_{DownloadState::Queued};
    std::atomic<uint64_t> done_{0}, total_{0};
    std::atomic<double> speed_{0};
    std::atomic<int> http_{0};
    std::atomic<bool> pauseRequested_{false}, cancelRequested_{false};

    mutable std::mutex mutex_;
    mutable std::condition_variable changed_;
    DownloadError error_ = DownloadError::None;
    std::string message_, sha256_;
};

// A queue of downloads with a few worker threads. Interrupted or paused downloads continue where they stopped (HTTP
// Range); a partial file left behind by a crash is picked up the same way.
class DownloadManager {
public:
    using Observer = std::function<void(const Download&)>;
    // Free space of the drive that holds a path; replaceable for tests.
    using FreeSpaceFn = std::function<uint64_t(const std::filesystem::path&)>;

    explicit DownloadManager(int maxConcurrent = 1);
    ~DownloadManager();  // stops running transfers (keeping their .part files) and joins the workers
    DownloadManager(const DownloadManager&) = delete;
    DownloadManager& operator=(const DownloadManager&) = delete;

    std::shared_ptr<Download> enqueue(DownloadRequest request);
    std::shared_ptr<Download> find(uint64_t id) const;
    std::vector<std::shared_ptr<Download>> list() const;

    void pause(uint64_t id);    // keeps the partial file
    void resume(uint64_t id);   // also retries a failed download
    void cancel(uint64_t id);   // removes the partial file

    void setObserver(Observer observer);    // called from worker threads, throttled to about 10 per second
    void setFreeSpaceFunction(FreeSpaceFn fn);

private:
    void worker();
    void transfer(const std::shared_ptr<Download>& d);
    void notify(const Download& d);

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::shared_ptr<Download>> queue_;
    std::vector<std::shared_ptr<Download>> all_;
    uint64_t nextId_ = 1;
    bool stopping_ = false;
    Observer observer_;
    FreeSpaceFn freeSpace_;
    std::vector<std::thread> workers_;
};

}  // namespace gh::models
