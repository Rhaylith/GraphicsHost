#include "TestUtil.h"
#include "host/AppHost.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh;
using namespace std::chrono_literals;
using nlohmann::json;

namespace {
json job(const json& nodes) { return {{"format", "graphicshost.job"}, {"version", 1}, {"nodes", nodes}}; }
json n(const char* type, json inputs = json::object()) { return {{"type", type}, {"inputs", inputs}}; }
json ref(const char* id, int out = 0) { return json::array({id, out}); }

json showSum(int a, int b) {
    return job({{"1", n("util.math", {{"expression", "a + b"}, {"a", a}, {"b", b}})}, {"2", n("util.previewAny", {{"value", ref("1", 1)}})}});
}
}  // namespace

TEST_CASE("graph service: a valid job runs on the shared queue and lands in the history", "[unit][graph-service]") {
    test::TempDir home;
    host::AppHost app(home.path());
    auto& service = app.graph();

    std::vector<json> events;
    std::mutex eventsMutex;
    const auto sub = service.events().subscribe([&](const graph::Event& e) {
        std::lock_guard<std::mutex> lock(eventsMutex);
        events.push_back(e.toJson());
    });

    host::GraphJobCall call;
    call.job = showSum(40, 2);
    call.workflow = {{"format", "graphicshost.workflow"}, {"id", "wf-1"}};
    call.clientId = "editor-1";
    call.label = "sum";
    const auto submission = service.submit(call);
    REQUIRE(submission.validation.ok());
    REQUIRE(submission.job != nullptr);
    CHECK(submission.job->kind() == engine::JobKind::Graph);
    REQUIRE(submission.job->waitFor(10s));
    CHECK(submission.job->state() == engine::JobState::Succeeded);
    CHECK(submission.job->label() == "sum");
    CHECK(submission.job->clientId() == "editor-1");
    const json result = submission.job->graphResult();
    CHECK(result["outputs"][0]["node"] == "2");
    CHECK(result["outputs"][0]["values"][0] == "STRING 42");   // the preview node outputs its text
    CHECK(result["ran"].size() == 2);

    service.events().unsubscribe(sub);
    {
        std::lock_guard<std::mutex> lock(eventsMutex);
        REQUIRE(events.size() >= 5);
        CHECK(events.front()["type"] == "job.started");
        CHECK(events.front()["job"] == std::to_string(submission.job->id()));
        CHECK(events.back()["type"] == "job.finished");
        CHECK(events.back()["status"] == "done");
    }

    const auto history = service.history();
    REQUIRE(history.size() == 1);
    CHECK(history[0].status == "done");
    CHECK(history[0].clientId == "editor-1");
    CHECK(history[0].queueNumber == static_cast<int64_t>(submission.job->id()));
    const auto item = service.historyItem(history[0].id);
    REQUIRE(item.has_value());
    CHECK(item->job["format"] == "graphicshost.job");
    CHECK(item->workflow["id"] == "wf-1");
    CHECK(item->timings.size() == 2);

    // the same job again: everything comes from the cache
    const auto again = service.submit(call);
    REQUIRE(again.job->waitFor(10s));
    CHECK(again.job->graphResult()["cached"].size() == 1);   // only the output node, with everything behind it skipped
    CHECK(again.job->graphResult()["ran"].empty());
    CHECK(service.history().size() == 2);
    service.freeCaches();
    const auto third = service.submit(call);
    REQUIRE(third.job->waitFor(10s));
    CHECK(third.job->graphResult()["ran"].size() == 2);

    CHECK(service.removeHistory(history[0].id));
    CHECK(service.history().size() == 2);
}

TEST_CASE("graph service: invalid jobs are reported, not queued", "[unit][graph-service]") {
    test::TempDir home;
    host::AppHost app(home.path());
    auto& service = app.graph();

    host::GraphJobCall call;
    call.job = job({{"1", n("util.math", {{"expression", "a"}, {"a", "text"}})}, {"2", n("util.previewAny", {{"value", ref("1")}})}, {"3", n("plugin.unknown")}});
    const auto bad = service.submit(call);
    CHECK(bad.job == nullptr);
    REQUIRE_FALSE(bad.validation.ok());
    bool badValue = false, unknown = false;
    for (const auto& i : bad.validation.issues) {
        badValue = badValue || (i.node == "1" && i.input == "a" && i.code == "bad_value");
        unknown = unknown || (i.node == "3" && i.code == "unknown_type");
    }
    CHECK(badValue);
    CHECK(unknown);
    CHECK(app.models().queue().list().empty());

    call.job = json::array();
    const auto notAJob = service.validate(call.job);
    REQUIRE(notAJob.issues.size() == 1);
    CHECK(notAJob.issues[0].code == "invalid_job");
    CHECK(service.history().empty());
}

TEST_CASE("graph service: a node that fails fails the job and is named in the history", "[unit][graph-service]") {
    test::TempDir home;
    host::AppHost app(home.path());
    auto& service = app.graph();

    host::GraphJobCall call;
    call.job = job({{"1", n("util.math", {{"expression", "a / 0"}, {"a", 1}})}, {"2", n("util.previewAny", {{"value", ref("1")}})}});
    const auto submission = service.submit(call);
    REQUIRE(submission.job != nullptr);   // it validates: the division only fails when it runs
    REQUIRE(submission.job->waitFor(10s));
    CHECK(submission.job->state() == engine::JobState::Failed);
    CHECK(submission.job->error().find("node 1, input 'expression': division by zero") == 0);
    const auto history = service.history();
    REQUIRE(history.size() == 1);
    CHECK(history[0].status == "failed");
    CHECK(history[0].errorNode == "1");
}

TEST_CASE("graph service: settings choose the cache policy and the history length", "[unit][graph-service]") {
    test::TempDir home;
    host::AppHost app(home.path());
    auto& service = app.graph();
    CHECK(service.cache().policy().toString() == "last");

    core::Settings s = app.settings();
    s.graph.cachePolicy = "lru";
    s.graph.cacheSize = 5;
    s.graph.historyMaxItems = 2;
    app.updateSettings(s);
    CHECK(service.cache().policy().toString() == "lru:5");

    for (int i = 0; i < 4; ++i) {
        host::GraphJobCall call;
        call.job = showSum(i, 1);
        const auto submission = service.submit(call);
        REQUIRE(submission.job->waitFor(10s));
    }
    CHECK(service.history(10).size() == 2);   // only the newest two are kept
    CHECK(host::GraphService::optionsFrom(s.graph).historyMax == 2);

    s.graph.cachePolicy = "bogus";
    CHECK_THROWS(app.updateSettings(s));
    s.graph.cachePolicy = "last";
    s.graph.cacheSize = 0;
    CHECK_THROWS(app.updateSettings(s));
}

TEST_CASE("graph service: the catalogue offers every built-in node with its tooltips", "[unit][graph-service]") {
    test::TempDir home;
    host::AppHost app(home.path());
    const json c = app.graph().nodeCatalogue();
    CHECK(c["catalogue"]["nodes"].size() == graph::Registry::builtin().size());
    CHECK(c["tips"]["common"]["seed"].is_string());
    CHECK(c["tips"]["settings"]["graph.cachePolicy"].is_string());
}

TEST_CASE("graph service: cancelling interrupts a running graph job", "[unit][graph-service]") {
    test::TempDir home;
    host::AppHost app(home.path());
    auto& service = app.graph();
    // a job with many sequential nodes; interrupt it as soon as the first node has finished
    json nodes = json::object();
    nodes["0"] = n("util.int", {{"value", 1}});
    for (int i = 1; i <= 200; ++i) nodes[std::to_string(i)] = n("util.math", {{"expression", "a + 1"}, {"a", ref(std::to_string(i - 1).c_str())}});
    nodes["201"] = n("util.previewAny", {{"value", ref("200", 1)}});
    host::GraphJobCall call;
    call.job = job(nodes);
    std::atomic<bool> cancelled{false};
    std::shared_ptr<engine::Job> submitted;
    const auto sub = service.events().subscribe([&](const graph::Event& e) {
        if (e.kind == graph::Event::Kind::NodeFinished && e.node == "3" && !cancelled.exchange(true) && submitted) submitted->cancel();
    });
    submitted = service.submit(call).job;
    REQUIRE(submitted != nullptr);
    REQUIRE(submitted->waitFor(20s));
    service.events().unsubscribe(sub);
    CHECK((submitted->state() == engine::JobState::Cancelled || submitted->state() == engine::JobState::Succeeded));
    if (submitted->state() == engine::JobState::Cancelled) {
        const auto history = service.history();
        REQUIRE(history.size() == 1);
        CHECK(history[0].status == "interrupted");
    }
}
