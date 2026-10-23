#include "engine/JobQueue.h"

#include "core/Log.h"

#include <algorithm>

namespace gh::engine {

namespace {
constexpr size_t kKeepFinished = 100;
}

const char* toString(JobState state) {
    switch (state) {
        case JobState::Queued: return "queued";
        case JobState::Running: return "running";
        case JobState::Succeeded: return "succeeded";
        case JobState::Failed: return "failed";
        case JobState::Cancelled: return "cancelled";
    }
    return "unknown";
}

bool Job::finished() const {
    const JobState s = state_.load();
    return s == JobState::Succeeded || s == JobState::Failed || s == JobState::Cancelled;
}

StepProgress Job::progress() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return progress_;
}

GenerationResult Job::result() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!result_) {
        throw std::logic_error("job " + std::to_string(id_) + " has no result (state " + toString(state_.load()) + ")");
    }
    return *result_;
}

std::string Job::error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

EngineErrorCode Job::errorCode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return errorCode_;
}

void Job::cancel() {
    stop_.request_stop();
}

void Job::wait() const {
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return finished(); });
}

bool Job::waitFor(std::chrono::milliseconds timeout) const {
    std::unique_lock<std::mutex> lock(mutex_);
    return done_.wait_for(lock, timeout, [this] { return finished(); });
}

JobQueue::JobQueue() : worker_([this] { run(); }) {}

JobQueue::~JobQueue() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        for (const auto& job : jobs_) {
            job->cancel();
        }
    }
    wake_.notify_all();
    worker_.join();
}

std::shared_ptr<Job> JobQueue::submit(std::shared_ptr<ILoadedModel> model, GenerationRequest request) {
    std::shared_ptr<Job> job;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job.reset(new Job(nextId_++, std::move(model), std::move(request)));
        queue_.push_back(job);
        jobs_.push_back(job);
        trimFinished();
    }
    wake_.notify_one();
    notify(*job);
    return job;
}

std::shared_ptr<Job> JobQueue::find(uint64_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = std::find_if(jobs_.begin(), jobs_.end(), [id](const auto& j) { return j->id() == id; });
    return it == jobs_.end() ? nullptr : *it;
}

std::vector<std::shared_ptr<Job>> JobQueue::list() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return jobs_;
}

bool JobQueue::cancel(uint64_t id) {
    const auto job = find(id);
    if (!job || job->finished()) {
        return false;
    }
    job->cancel();
    wake_.notify_all();
    return true;
}

size_t JobQueue::pending() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

void JobQueue::setObserver(Observer observer) {
    std::lock_guard<std::mutex> lock(mutex_);
    observer_ = std::move(observer);
}

void JobQueue::notify(const Job& job) {
    Observer observer;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        observer = observer_;
    }
    if (observer) {
        observer(job);
    }
}

void JobQueue::trimFinished() {
    size_t finished = 0;
    for (const auto& j : jobs_) {
        if (j->finished()) ++finished;
    }
    for (auto it = jobs_.begin(); finished > kKeepFinished && it != jobs_.end();) {
        if ((*it)->finished()) {
            it = jobs_.erase(it);
            --finished;
        } else {
            ++it;
        }
    }
}

void JobQueue::run() {
    for (;;) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) {
                return;  // stopping, nothing left
            }
            job = queue_.front();
            queue_.pop_front();
        }

        const auto finish = [&](JobState state) {
            {
                std::lock_guard<std::mutex> lock(job->mutex_);
                job->model_.reset();
                job->state_ = state;
            }
            job->done_.notify_all();
            notify(*job);
        };

        if (job->stop_.stop_requested()) {  // cancelled while it was waiting
            {
                std::lock_guard<std::mutex> lock(job->mutex_);
                job->error_ = "cancelled before it started";
                job->errorCode_ = EngineErrorCode::Cancelled;
            }
            finish(JobState::Cancelled);
            continue;
        }

        job->state_ = JobState::Running;
        notify(*job);
        try {
            GenerationResult result = job->model_->generate(
                job->request_,
                [&](const StepProgress& p) {
                    {
                        std::lock_guard<std::mutex> lock(job->mutex_);
                        job->progress_ = p;
                    }
                    notify(*job);
                },
                job->stop_.get_token());
            {
                std::lock_guard<std::mutex> lock(job->mutex_);
                job->result_ = std::move(result);
            }
            finish(JobState::Succeeded);
        } catch (const EngineError& e) {
            {
                std::lock_guard<std::mutex> lock(job->mutex_);
                job->error_ = e.what();
                job->errorCode_ = e.code();
            }
            if (e.code() != EngineErrorCode::Cancelled) {
                log::warn("job " + std::to_string(job->id()) + " failed: " + e.what());
            }
            finish(e.code() == EngineErrorCode::Cancelled ? JobState::Cancelled : JobState::Failed);
        } catch (const std::exception& e) {
            {
                std::lock_guard<std::mutex> lock(job->mutex_);
                job->error_ = e.what();
                job->errorCode_ = EngineErrorCode::Failed;
            }
            log::error("job " + std::to_string(job->id()) + " crashed: " + e.what());
            finish(JobState::Failed);
        }
    }
}

}  // namespace gh::engine
