#include "engine/JobQueue.h"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <thread>

using namespace gh::engine;
using namespace std::chrono_literals;

namespace {

// A model that needs no runtime: takes `stepMs` per step, can fail on demand, and records concurrency.
class FakeModel final : public ILoadedModel {
public:
    int stepMs = 5;
    EngineErrorCode failWith = EngineErrorCode::Failed;
    bool fail = false;
    std::atomic<int> running{0};
    std::atomic<int> maxRunning{0};
    std::vector<std::string> order;  // prompts in the order they ran
    std::mutex orderMutex;

    const ModelInfo& info() const override { return info_; }

    GenerationResult generate(const GenerationRequest& request, const ProgressSink& progress,
                              std::stop_token stop) override {
        const int now = ++running;
        int seen = maxRunning.load();
        while (now > seen && !maxRunning.compare_exchange_weak(seen, now)) {}
        struct Done {
            std::atomic<int>& r;
            ~Done() { --r; }
        } done{running};
        {
            std::lock_guard<std::mutex> lock(orderMutex);
            order.push_back(request.prompt);
        }
        const int steps = request.steps > 0 ? request.steps : 4;
        for (int i = 1; i <= steps; ++i) {
            if (stop.stop_requested()) {
                throw EngineError(EngineErrorCode::Cancelled, "cancelled");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(stepMs));
            if (progress) progress(StepProgress{i, steps, 0.0f});
        }
        if (fail) {
            throw EngineError(failWith, "fake failure");
        }
        GenerationResult r;
        r.request = request;
        r.seeds = {42};
        r.images.push_back(Image{1, 1, 3, {1, 2, 3}});
        return r;
    }

private:
    ModelInfo info_;
};

GenerationRequest req(const std::string& prompt, int steps = 4) {
    GenerationRequest r;
    r.prompt = prompt;
    r.steps = steps;
    return r;
}

}  // namespace

TEST_CASE("queue: a job runs and delivers its result", "[unit][queue]") {
    auto model = std::make_shared<FakeModel>();
    JobQueue queue;
    auto job = queue.submit(model, req("one"));
    REQUIRE(job->waitFor(5s));
    CHECK(job->state() == JobState::Succeeded);
    CHECK(job->result().seeds == std::vector<int64_t>{42});
    CHECK(job->progress().step == 4);
    CHECK(job->progress().steps == 4);
    CHECK(job->id() == 1);
}

TEST_CASE("queue: jobs run in submission order, one at a time", "[unit][queue]") {
    auto model = std::make_shared<FakeModel>();
    model->stepMs = 2;
    JobQueue queue;
    std::vector<std::shared_ptr<Job>> jobs;
    for (const char* p : {"a", "b", "c", "d", "e"}) {
        jobs.push_back(queue.submit(model, req(p, 3)));
    }
    for (auto& j : jobs) {
        REQUIRE(j->waitFor(10s));
        CHECK(j->state() == JobState::Succeeded);
    }
    CHECK(model->order == std::vector<std::string>{"a", "b", "c", "d", "e"});
    CHECK(model->maxRunning.load() == 1);
}

TEST_CASE("queue: cancelling a queued job means it never runs", "[unit][queue]") {
    auto model = std::make_shared<FakeModel>();
    model->stepMs = 20;
    JobQueue queue;
    auto first = queue.submit(model, req("first", 10));
    auto second = queue.submit(model, req("second", 10));
    REQUIRE(queue.cancel(second->id()));
    REQUIRE(second->waitFor(10s));
    CHECK(second->state() == JobState::Cancelled);
    CHECK(second->errorCode() == EngineErrorCode::Cancelled);
    REQUIRE(first->waitFor(10s));
    CHECK(first->state() == JobState::Succeeded);
    CHECK(model->order == std::vector<std::string>{"first"});
}

TEST_CASE("queue: cancelling the running job stops it at the next step", "[unit][queue]") {
    auto model = std::make_shared<FakeModel>();
    model->stepMs = 20;
    JobQueue queue;
    auto job = queue.submit(model, req("long", 500));  // 10 seconds if it ran to the end
    while (job->state() == JobState::Queued) std::this_thread::sleep_for(1ms);
    std::this_thread::sleep_for(60ms);
    const auto t0 = std::chrono::steady_clock::now();
    job->cancel();
    REQUIRE(job->waitFor(5s));
    CHECK(job->state() == JobState::Cancelled);
    CHECK(std::chrono::steady_clock::now() - t0 < 2s);
    CHECK(job->progress().step < 500);
    CHECK_THROWS(job->result());
}

TEST_CASE("queue: a failing job records code and message and does not stop the queue", "[unit][queue]") {
    auto bad = std::make_shared<FakeModel>();
    bad->fail = true;
    bad->failWith = EngineErrorCode::OutOfMemory;
    auto good = std::make_shared<FakeModel>();
    JobQueue queue;
    auto failed = queue.submit(bad, req("bad", 2));
    auto fine = queue.submit(good, req("good", 2));
    REQUIRE(failed->waitFor(5s));
    CHECK(failed->state() == JobState::Failed);
    CHECK(failed->errorCode() == EngineErrorCode::OutOfMemory);
    CHECK(failed->error() == "fake failure");
    REQUIRE(fine->waitFor(5s));
    CHECK(fine->state() == JobState::Succeeded);
}

TEST_CASE("queue: find, list and the pending count", "[unit][queue]") {
    auto model = std::make_shared<FakeModel>();
    model->stepMs = 30;
    JobQueue queue;
    auto a = queue.submit(model, req("a", 5));
    auto b = queue.submit(model, req("b", 5));
    auto c = queue.submit(model, req("c", 5));
    CHECK(queue.find(b->id()) == b);
    CHECK(queue.find(999) == nullptr);
    CHECK(queue.list().size() == 3);
    CHECK(queue.pending() >= 1);
    CHECK_FALSE(queue.cancel(999));
    for (auto& j : {a, b, c}) REQUIRE(j->waitFor(10s));
    CHECK(queue.pending() == 0);
    CHECK_FALSE(queue.cancel(a->id()));  // already finished
}

TEST_CASE("queue: the observer sees state changes and every step", "[unit][queue]") {
    auto model = std::make_shared<FakeModel>();
    JobQueue queue;
    std::mutex m;
    std::vector<std::string> seen;
    queue.setObserver([&](const Job& j) {
        std::lock_guard<std::mutex> lock(m);
        seen.push_back(std::string(toString(j.state())) + ":" + std::to_string(j.progress().step));
    });
    auto job = queue.submit(model, req("watched", 3));
    REQUIRE(job->waitFor(5s));
    std::this_thread::sleep_for(20ms);  // the final notification comes right after the job is marked finished
    std::lock_guard<std::mutex> lock(m);
    CHECK(seen.front() == "queued:0");
    CHECK(seen.back() == "succeeded:3");
    CHECK(std::count(seen.begin(), seen.end(), "running:1") + std::count(seen.begin(), seen.end(), "running:2") +
              std::count(seen.begin(), seen.end(), "running:3") >= 3);
}

TEST_CASE("queue: destroying it with work pending cancels and does not hang", "[unit][queue]") {
    auto model = std::make_shared<FakeModel>();
    model->stepMs = 20;
    std::shared_ptr<Job> running, waiting;
    const auto t0 = std::chrono::steady_clock::now();
    {
        JobQueue queue;
        running = queue.submit(model, req("r", 1000));
        waiting = queue.submit(model, req("w", 1000));
        std::this_thread::sleep_for(50ms);
    }
    CHECK(std::chrono::steady_clock::now() - t0 < 5s);
    CHECK(running->state() == JobState::Cancelled);
    CHECK(waiting->state() == JobState::Cancelled);
}

TEST_CASE("queue: only the last 100 finished jobs are kept", "[unit][queue]") {
    auto model = std::make_shared<FakeModel>();
    model->stepMs = 0;
    JobQueue queue;
    std::shared_ptr<Job> last;
    for (int i = 0; i < 130; ++i) {
        last = queue.submit(model, req("j", 1));
        last->wait();
    }
    queue.submit(model, req("trigger", 1))->wait();  // trimming happens on submit
    CHECK(queue.list().size() <= 102);
    CHECK(queue.find(1) == nullptr);
    CHECK(queue.find(last->id()) != nullptr);
}

// ---- graph jobs share the queue (workflow plan W2.5) --------------------------------------------------------------------

namespace {
GraphWork graphWork(std::vector<std::string>& log, std::mutex& logMutex, const std::string& name, int steps = 3, int stepMs = 5) {
    return [&log, &logMutex, name, steps, stepMs](JobControl& control) {
        {
            std::lock_guard<std::mutex> lock(logMutex);
            log.push_back(name);
        }
        for (int i = 1; i <= steps; ++i) {
            if (control.stop.stop_requested()) throw EngineError(EngineErrorCode::Cancelled, "interrupted");
            std::this_thread::sleep_for(std::chrono::milliseconds(stepMs));
            control.progress(StepProgress{i, steps, 0.0f, "node-" + name});
        }
        return nlohmann::json{{"name", name}};
    };
}
}  // namespace

TEST_CASE("queue: graph and generation jobs share one worker, in order", "[unit][queue]") {
    auto model = std::make_shared<FakeModel>();
    std::vector<std::string> log;
    std::mutex logMutex;
    JobQueue queue;
    auto a = queue.submit(model, req("gen-a", 3));
    SubmitOptions options;
    options.clientId = "editor-1";
    options.label = "my run";
    auto g = queue.submitGraph(graphWork(log, logMutex, "graph-1"), options);
    auto b = queue.submit(model, req("gen-b", 3));
    for (auto* j : {a.get(), g.get(), b.get()}) REQUIRE(j->waitFor(5s));

    CHECK(g->kind() == JobKind::Graph);
    CHECK(a->kind() == JobKind::Generation);
    CHECK(std::string(toString(g->kind())) == "graph");
    CHECK(g->state() == JobState::Succeeded);
    CHECK(g->graphResult()["name"] == "graph-1");
    CHECK(g->label() == "my run");
    CHECK(g->clientId() == "editor-1");
    CHECK(g->number() == g->id());
    CHECK(g->progress().node == "node-graph-1");
    CHECK(g->progress().step == 3);
    CHECK(model->maxRunning.load() == 1);
    CHECK(model->order == std::vector<std::string>{"gen-a", "gen-b"});
    CHECK(a->id() < g->id());
    CHECK(g->id() < b->id());
}

TEST_CASE("queue: front jobs jump the queue behind the running one, clear removes the pending ones", "[unit][queue]") {
    auto model = std::make_shared<FakeModel>();
    model->stepMs = 20;
    std::vector<std::string> log;
    std::mutex logMutex;
    JobQueue queue;
    auto running = queue.submit(model, req("running", 10));
    while (running->state() != JobState::Running) std::this_thread::sleep_for(1ms);
    auto late = queue.submitGraph(graphWork(log, logMutex, "late"));
    SubmitOptions front;
    front.front = true;
    auto urgent = queue.submitGraph(graphWork(log, logMutex, "urgent"), front);
    CHECK(queue.pending() == 2);

    SECTION("order") {
        for (auto* j : {running.get(), late.get(), urgent.get()}) REQUIRE(j->waitFor(5s));
        CHECK(log == std::vector<std::string>{"urgent", "late"});
    }
    SECTION("clear") {
        CHECK(queue.clearPending() == 2);
        CHECK(queue.pending() == 0);
        CHECK(late->state() == JobState::Cancelled);
        CHECK(urgent->state() == JobState::Cancelled);
        CHECK(late->errorCode() == EngineErrorCode::Cancelled);
        CHECK(running->state() != JobState::Cancelled);   // the running job is left alone
        REQUIRE(running->waitFor(5s));
        CHECK(running->state() == JobState::Succeeded);
        CHECK(log.empty());
        CHECK(queue.clearPending() == 0);
    }
}

TEST_CASE("queue: graph jobs fail and cancel like generation jobs", "[unit][queue]") {
    JobQueue queue;
    auto failing = queue.submitGraph([](JobControl&) -> nlohmann::json { throw EngineError(EngineErrorCode::OutOfMemory, "no memory"); });
    auto crashing = queue.submitGraph([](JobControl&) -> nlohmann::json { throw std::runtime_error("oops"); });
    REQUIRE(failing->waitFor(5s));
    REQUIRE(crashing->waitFor(5s));
    CHECK(failing->state() == JobState::Failed);
    CHECK(failing->errorCode() == EngineErrorCode::OutOfMemory);
    CHECK(failing->error() == "no memory");
    CHECK(crashing->state() == JobState::Failed);

    std::vector<std::string> log;
    std::mutex logMutex;
    auto slow = queue.submitGraph(graphWork(log, logMutex, "slow", 500, 10));
    while (slow->state() != JobState::Running) std::this_thread::sleep_for(1ms);
    REQUIRE(queue.cancel(slow->id()));
    REQUIRE(slow->waitFor(5s));
    CHECK(slow->state() == JobState::Cancelled);
    CHECK_THROWS(slow->result());   // a graph job has no generation result
}
