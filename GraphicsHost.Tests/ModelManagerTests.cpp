#include "engine/ModelManager.h"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <thread>

using namespace gh::engine;
using namespace std::chrono_literals;

namespace {

struct Counters {
    std::atomic<int> created{0};
    std::atomic<int> destroyed{0};
    std::atomic<int> engineAsked{0};
};

class FakeModel final : public ILoadedModel {
public:
    FakeModel(ModelInfo info, Counters& c, int stepMs) : info_(std::move(info)), c_(c), stepMs_(stepMs) { ++c_.created; }
    ~FakeModel() override { ++c_.destroyed; }
    const ModelInfo& info() const override { return info_; }
    GenerationResult generate(const GenerationRequest& request, const ProgressSink& progress,
                              std::stop_token stop) override {
        for (int i = 1; i <= 5; ++i) {
            if (stop.stop_requested()) throw EngineError(EngineErrorCode::Cancelled, "cancelled");
            std::this_thread::sleep_for(std::chrono::milliseconds(stepMs_));
            if (progress) progress(StepProgress{i, 5, 0});
        }
        GenerationResult r;
        r.request = request;
        r.images.push_back(Image{1, 1, 3, {0, 0, 0}});
        r.seeds = {1};
        return r;
    }

private:
    ModelInfo info_;
    Counters& c_;
    int stepMs_;
};

class FakeEngine final : public IImageEngine {
public:
    FakeEngine(Counters& c, int stepMs = 1) : c_(c), stepMs_(stepMs) {}
    std::string runtimeId() const override { return "fake"; }
    std::unique_ptr<ILoadedModel> load(const LoadOptions& options) override {
        if (options.modelPath == "broken") throw EngineError(EngineErrorCode::LoadFailed, "cannot load broken");
        ModelInfo info;
        info.modelPath = options.modelPath;
        info.family = "SD 1.x";
        return std::make_unique<FakeModel>(info, c_, stepMs_);
    }

private:
    Counters& c_;
    int stepMs_;
};

struct Fixture {
    Counters counters;
    std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();
    std::unique_ptr<ModelManager> manager;

    explicit Fixture(int stepMs = 1) {
        manager = std::make_unique<ModelManager>(
            [this, stepMs]() -> std::shared_ptr<IImageEngine> {
                ++counters.engineAsked;
                return std::make_shared<FakeEngine>(counters, stepMs);
            },
            [this] { return time; }, std::chrono::seconds(0));
    }
};

LoadOptions options(const std::string& path, int ttl = 10) {
    LoadOptions o;
    o.modelPath = path;
    o.ttlMinutes = ttl;
    return o;
}

GenerationRequest req() {
    GenerationRequest r;
    r.prompt = "x";
    return r;
}

}  // namespace

TEST_CASE("manager: load reports the loaded model", "[unit][manager]") {
    Fixture f;
    CHECK_FALSE(f.manager->loaded().has_value());
    auto model = f.manager->load(options("a.safetensors"));
    REQUIRE(model != nullptr);
    REQUIRE(f.manager->loaded().has_value());
    CHECK(f.manager->loaded()->info.modelPath == "a.safetensors");
    CHECK(f.manager->loaded()->options.ttlMinutes == 10);
}

TEST_CASE("manager: the same options reuse the loaded model", "[unit][manager]") {
    Fixture f;
    auto first = f.manager->load(options("a.safetensors"));
    auto second = f.manager->load(options("a.safetensors"));
    CHECK(first == second);
    CHECK(f.counters.created == 1);
    CHECK(f.counters.engineAsked == 1);
}

TEST_CASE("manager: another model evicts the first", "[unit][manager]") {
    Fixture f;
    f.manager->load(options("a.safetensors"));
    f.manager->load(options("b.safetensors"));
    CHECK(f.counters.created == 2);
    CHECK(f.counters.destroyed == 1);
    CHECK(f.manager->loaded()->info.modelPath == "b.safetensors");

    // different options for the same file also count as another model
    auto o = options("b.safetensors");
    o.device = "cpu";
    f.manager->load(o);
    CHECK(f.counters.created == 3);
    CHECK(f.counters.destroyed == 2);
}

TEST_CASE("manager: a failed load leaves nothing loaded and reports the reason", "[unit][manager]") {
    Fixture f;
    f.manager->load(options("a.safetensors"));
    try {
        f.manager->load(options("broken"));
        FAIL("expected an error");
    } catch (const EngineError& e) {
        CHECK(e.code() == EngineErrorCode::LoadFailed);
    }
    CHECK_FALSE(f.manager->loaded().has_value());  // the old one was evicted to make room
    CHECK(f.counters.destroyed == 1);
}

TEST_CASE("manager: no engine means a clear error", "[unit][manager]") {
    ModelManager manager([]() -> std::shared_ptr<IImageEngine> { return nullptr; }, {}, std::chrono::seconds(0));
    try {
        manager.load(options("a"));
        FAIL("expected an error");
    } catch (const EngineError& e) {
        CHECK(e.code() == EngineErrorCode::LoadFailed);
    }
}

TEST_CASE("manager: unload frees the model", "[unit][manager]") {
    Fixture f;
    f.manager->load(options("a.safetensors"));
    f.manager->unload();
    CHECK(f.counters.destroyed == 1);
    CHECK_FALSE(f.manager->loaded().has_value());
    CHECK(f.manager->current() == nullptr);
    f.manager->unload();  // harmless when nothing is loaded
}

TEST_CASE("manager: idle models are unloaded after their ttl", "[unit][manager][ttl]") {
    Fixture f;
    f.manager->load(options("a.safetensors", 10));
    f.time += 9min;
    CHECK_FALSE(f.manager->sweepIdle());
    CHECK(f.manager->loaded().has_value());
    f.time += 2min;
    CHECK(f.manager->sweepIdle());
    CHECK_FALSE(f.manager->loaded().has_value());
    CHECK(f.counters.destroyed == 1);
}

TEST_CASE("manager: using the model restarts the idle time, ttl 0 never unloads", "[unit][manager][ttl]") {
    Fixture f;
    f.manager->load(options("a.safetensors", 10));
    f.time += 8min;
    f.manager->current();  // used
    f.time += 8min;
    CHECK_FALSE(f.manager->sweepIdle());
    f.time += 3min;
    CHECK(f.manager->sweepIdle());

    f.manager->load(options("b.safetensors", 0));
    f.time += 24h;
    CHECK_FALSE(f.manager->sweepIdle());
    CHECK(f.manager->loaded().has_value());
}

TEST_CASE("manager: submit runs on the loaded model", "[unit][manager]") {
    Fixture f;
    CHECK_THROWS_AS(f.manager->submit(req()), EngineError);  // nothing loaded
    f.manager->load(options("a.safetensors"));
    auto job = f.manager->submit(req());
    REQUIRE(job->waitFor(5s));
    CHECK(job->state() == JobState::Succeeded);
}

TEST_CASE("manager: models cannot be switched or swept while jobs are active", "[unit][manager]") {
    Fixture f(30);  // slow steps: the job runs for about 150 ms
    f.manager->load(options("a.safetensors", 1));
    auto job = f.manager->submit(req());
    try {
        f.manager->load(options("b.safetensors"));
        FAIL("expected an error");
    } catch (const EngineError& e) {
        CHECK(std::string(e.what()).find("queued or running") != std::string::npos);
    }
    f.time += 2h;
    CHECK_FALSE(f.manager->sweepIdle());  // busy, so not unloaded despite the elapsed time
    REQUIRE(job->waitFor(5s));
    CHECK(f.manager->loaded()->info.modelPath == "a.safetensors");
    f.manager->load(options("b.safetensors"));  // idle now: allowed
    CHECK(f.manager->loaded()->info.modelPath == "b.safetensors");
}

TEST_CASE("manager: an unloaded model stays alive until its running job ends", "[unit][manager]") {
    Fixture f(20);
    f.manager->load(options("a.safetensors"));
    auto job = f.manager->submit(req());
    while (job->state() == JobState::Queued) std::this_thread::sleep_for(1ms);
    f.manager->unload();
    CHECK(f.counters.destroyed == 0);  // the job still holds it
    REQUIRE(job->waitFor(5s));
    CHECK(job->state() == JobState::Succeeded);
    std::this_thread::sleep_for(20ms);
    CHECK(f.counters.destroyed == 1);
}

TEST_CASE("manager: the background sweeper unloads idle models", "[unit][manager][ttl]") {
    Counters counters;
    std::atomic<long long> offsetMinutes{0};
    const auto base = std::chrono::steady_clock::now();
    ModelManager manager(
        [&]() -> std::shared_ptr<IImageEngine> { return std::make_shared<FakeEngine>(counters); },
        [&] { return base + std::chrono::minutes(offsetMinutes.load()); }, std::chrono::seconds(1));
    manager.load(options("a.safetensors", 5));
    offsetMinutes = 30;
    for (int i = 0; i < 40 && manager.loaded().has_value(); ++i) std::this_thread::sleep_for(100ms);
    CHECK_FALSE(manager.loaded().has_value());
}
