#pragma once

#include "engine/ImageEngine.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace gh::engine {

enum class JobState { Queued, Running, Succeeded, Failed, Cancelled };
const char* toString(JobState state);

// One generation request on its way through the queue. Created by JobQueue::submit; safe to read from any thread.
class Job {
public:
    uint64_t id() const { return id_; }
    const GenerationRequest& request() const { return request_; }
    JobState state() const { return state_.load(); }
    bool finished() const;
    StepProgress progress() const;
    // Valid once the job succeeded.
    GenerationResult result() const;
    // Valid once the job failed or was cancelled.
    std::string error() const;
    EngineErrorCode errorCode() const;

    // Cancels a queued job at once and a running one at its next sampling step. No effect once finished.
    void cancel();

    // Blocks until the job has finished; waitFor returns false on timeout.
    void wait() const;
    bool waitFor(std::chrono::milliseconds timeout) const;

private:
    friend class JobQueue;
    Job(uint64_t id, std::shared_ptr<ILoadedModel> model, GenerationRequest request)
        : id_(id), model_(std::move(model)), request_(std::move(request)) {}

    const uint64_t id_;
    std::shared_ptr<ILoadedModel> model_;  // released when the job finishes
    const GenerationRequest request_;
    std::stop_source stop_;
    std::atomic<JobState> state_{JobState::Queued};

    mutable std::mutex mutex_;
    mutable std::condition_variable done_;
    StepProgress progress_;
    std::optional<GenerationResult> result_;
    std::string error_;
    EngineErrorCode errorCode_ = EngineErrorCode::Failed;
};

// Runs jobs one after the other on a single worker thread: the runtime ABI allows one generation at a time per process,
// so the queue is what serialises everything (UI, CLI and HTTP all submit here). FIFO.
class JobQueue {
public:
    // Called from the worker thread on every state change and every sampling step; keep it quick.
    using Observer = std::function<void(const Job&)>;

    JobQueue();
    // Cancels everything queued and running, then joins the worker.
    ~JobQueue();
    JobQueue(const JobQueue&) = delete;
    JobQueue& operator=(const JobQueue&) = delete;

    std::shared_ptr<Job> submit(std::shared_ptr<ILoadedModel> model, GenerationRequest request);

    std::shared_ptr<Job> find(uint64_t id) const;
    // Newest last; finished jobs are kept (the last 100) so clients can poll for the outcome.
    std::vector<std::shared_ptr<Job>> list() const;
    bool cancel(uint64_t id);
    size_t pending() const;  // queued, not yet running
    size_t active() const;   // queued or running: zero means the queue is idle

    void setObserver(Observer observer);

private:
    void run();
    void notify(const Job& job);
    void trimFinished();  // callers hold mutex_

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::vector<std::shared_ptr<Job>> jobs_;
    uint64_t nextId_ = 1;
    bool stopping_ = false;
    Observer observer_;
    std::thread worker_;
};

}  // namespace gh::engine
