#include "ApiTestUtil.h"
#include "TestUtil.h"
#include "host/AppHost.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh;
using namespace std::chrono_literals;
using nlohmann::json;

namespace {
class Weights final : public models::ComponentObject {
public:
    explicit Weights(uint64_t bytes) : bytes_(bytes) {}
    uint64_t weightsBytes() const override { return bytes_; }
    void moveToRam() override {}
    void moveToVram() override {}

private:
    uint64_t bytes_;
};

models::ComponentCache::Lease open(host::AppHost& app, const char* file, uint64_t bytes) {
    models::ComponentCache::Use use;
    use.expectedBytes = bytes;
    return app.components().acquire({file, "diffusion", ""}, use, [bytes] { return std::make_shared<Weights>(bytes); });
}

host::GraphJobCall sum() {
    host::GraphJobCall call;
    call.job = {{"format", "graphicshost.job"},
                {"version", 1},
                {"nodes", {{"1", {{"type", "util.math"}, {"inputs", {{"expression", "a + b"}, {"a", 1}, {"b", 2}}}}}, {"2", {{"type", "util.previewAny"}, {"inputs", {{"value", json::array({"1", 0})}}}}}}}};
    return call;
}
}  // namespace

TEST_CASE("free memory: unused components close, components in use stay, node results are dropped", "[unit][free-memory]") {
    test::TempDir home;
    host::AppHost app(home.path());
    auto busy = open(app, "busy.safetensors", 3ull << 20);
    { auto idle = open(app, "idle.safetensors", 5ull << 20); }
    { auto idle = open(app, "idle2.safetensors", 7ull << 20); }

    auto job = app.graph().submit(sum());
    REQUIRE(job.job->waitFor(10s));
    REQUIRE(job.job->state() == engine::JobState::Succeeded);
    REQUIRE(app.graph().cache().size() > 0);

    const auto report = app.freeMemory();
    CHECK(report.componentsClosed == 2);
    CHECK(report.componentBytes == 12ull << 20);
    CHECK(report.resultsDropped > 0);
    CHECK(app.graph().cache().size() == 0);
    REQUIRE(report.kept.size() == 1);
    CHECK(report.kept[0] == "busy.safetensors is in use by a running node");
    CHECK_FALSE(report.generateModelUnloaded);   // nothing was loaded
    CHECK(app.components().stats().size() == 1);

    busy.release();
    const auto second = app.freeMemory();
    CHECK(second.componentsClosed == 1);
    CHECK(second.kept.empty());
    CHECK(app.components().stats().empty());

    // the next graph run starts from scratch
    auto again = app.graph().submit(sum());
    REQUIRE(again.job->waitFor(10s));
    CHECK(again.job->graphResult()["ran"].size() == 2);
}

TEST_CASE("free memory: each part can be asked for alone", "[unit][free-memory]") {
    test::TempDir home;
    host::AppHost app(home.path());
    { auto a = open(app, "a.safetensors", 1ull << 20); }
    auto job = app.graph().submit(sum());
    REQUIRE(job.job->waitFor(10s));

    auto onlyCaches = app.freeMemory({false, true});
    CHECK(onlyCaches.componentsClosed == 0);
    CHECK(onlyCaches.resultsDropped > 0);
    CHECK(app.components().stats().size() == 1);

    auto again = app.graph().submit(sum());
    REQUIRE(again.job->waitFor(10s));
    auto onlyModels = app.freeMemory({true, false});
    CHECK(onlyModels.componentsClosed == 1);
    CHECK(onlyModels.resultsDropped == 0);
    CHECK(app.graph().cache().size() > 0);
}

TEST_CASE("free memory: the API route reports what was released", "[integration][api][free-memory]") {
    test::TestApp app(test::TestApp::ephemeral());
    REQUIRE(app.error.empty());
    { auto a = open(*app.host, "a.safetensors", 2ull << 20); }
    auto c = app.client();

    auto res = c.Post("/api/v1/graph/free", "{}", "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    const json body = json::parse(res->body);
    CHECK(body["componentsClosed"] == 1);
    CHECK(body["componentBytes"] == 2 << 20);
    CHECK(body["generateModelUnloaded"] == false);
    CHECK(body["kept"].is_array());

    res = c.Post("/api/v1/graph/free", R"({"unloadModels": false, "freeCaches": false})", "application/json");
    REQUIRE(res);
    CHECK(json::parse(res->body)["componentsClosed"] == 0);

    res = c.Post("/api/v1/graph/free", "[1]", "application/json");
    REQUIRE(res);
    CHECK(res->status == 400);

    // the older route closes the components too
    { auto b = open(*app.host, "b.safetensors", 1ull << 20); }
    res = c.Post("/api/v1/models/unload", "{}", "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(json::parse(res->body)["componentsClosed"] == 1);
    CHECK(app.host->components().stats().empty());
}
