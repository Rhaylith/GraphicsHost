#pragma once

#include "engine/ImageEngine.h"

#include <nlohmann/json.hpp>

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

// What a job does: generate with a loaded model (the Generate tab, the CLI and the OpenAI-compatible API), or run a node
// graph (the executor wrapped in a function by the graph service). Both share one queue because they share the GPU.
enum class JobKind { Generation, Graph };
const char* toString(JobKind kind);

// What the queue hands to the work of a graph job.
struct JobControl {
    uint64_t jobId = 0;
    std::stop_token stop;                                  // set by Job::cancel() and when the queue shuts down
    std::function<void(const StepProgress&)> progress;     // may be called from any thread; StepProgress::node says which node
};
// Runs a graph job. Returns a JSON summary of the outcome (kept on the job); throws EngineError to fail, with the code
// Cancelled when it stopped because of JobControl::stop.
using GraphWork = std::function<nlohmann::json(JobControl&)>;

struct SubmitOptions {
    bool front = false;       // jump the queue (behind the running job)
    std::string clientId;     // who asked, so a client can recognise its own jobs
    std::string label;        // shown in lists
};

// One job on its way through the queue. Created by JobQueue::submit; safe to read from any thread.
class Job {
public:
    uint64_t id() const { return id_; }
    uint64_t number() const { return id_; }   // the queue number people see
    JobKind kind() const { return kind_; }
    const std::string& label() const { return options_.label; }
    const std::string& clientId() const { return options_.clientId; }
    // The request of a generation job; an empty request for a graph job.
    const GenerationRequest& request() const { return request_; }
    JobState state() const { return state_.load(); }
    bool finished() const;
    StepProgress progress() const;
    // Valid once a generation job succeeded.
    GenerationResult result() const;
    // Valid once a graph job succeeded: what its work returned.
    nlohmann::json graphResult() const;
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
    Job(uint64_t id, std::shared_ptr<ILoadedModel> model, GenerationRequest request, SubmitOptions options)
        : id_(id), kind_(JobKind::Generation), options_(std::move(options)), model_(std::move(model)), request_(std::move(request)) {}
    Job(uint64_t id, GraphWork work, SubmitOptions options)
        : id_(id), kind_(JobKind::Graph), options_(std::move(options)), graphWork_(std::move(work)) {}

    const uint64_t id_;
    const JobKind kind_;
    const SubmitOptions options_;
    std::shared_ptr<ILoadedModel> model_;  // released when the job finishes
    GraphWork graphWork_;                  // released when the job finishes
    const GenerationRequest request_;
    std::stop_source stop_;
    std::atomic<JobState> state_{JobState::Queued};

    mutable std::mutex mutex_;
    mutable std::condition_variable done_;
    StepProgress progress_;
    std::optional<GenerationResult> result_;
    nlohmann::json graphResult_;
    std::string error_;
    EngineErrorCode errorCode_ = EngineErrorCode::Failed;
};

// Runs jobs one after the other on a single worker thread: the runtime ABI allows one generation at a time per process,
// so the queue is what serialises everything (UI, CLI and HTTP all submit here). FIFO, except for jobs submitted with
// SubmitOptions::front.
class JobQueue {
public:
    // Called from the worker thread on every state change and every sampling step; keep it quick.
    using Observer = std::function<void(const Job&)>;

    JobQueue();
    // Cancels everything queued and running, then joins the worker.
    ~JobQueue();
    JobQueue(const JobQueue&) = delete;
    JobQueue& operator=(const JobQueue&) = delete;

    std::shared_ptr<Job> submit(std::shared_ptr<ILoadedModel> model, GenerationRequest request, SubmitOptions options = {});
    std::shared_ptr<Job> submitGraph(GraphWork work, SubmitOptions options = {});

    std::shared_ptr<Job> find(uint64_t id) const;
    // Newest last; finished jobs are kept (the last 100) so clients can poll for the outcome.
    std::vector<std::shared_ptr<Job>> list() const;
    bool cancel(uint64_t id);
    // Cancels every job that has not started; the running one is left alone. Returns how many.
    size_t clearPending();
    size_t pending() const;  // queued, not yet running
    size_t active() const;   // queued or running: zero means the queue is idle

    void setObserver(Observer observer);

private:
    std::shared_ptr<Job> enqueue(std::shared_ptr<Job> job, bool front);
    void run();
    void notify(const Job& job);
    void trimFinished();  // callers hold mutex_

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::vector<std::shared_ptr<Job>> jobs_;
    std::atomic<uint64_t> nextId_{1};
    bool stopping_ = false;
    Observer observer_;
    std::thread worker_;
};

}  // namespace gh::engine
