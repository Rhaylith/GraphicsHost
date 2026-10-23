#include "GenApiTestUtil.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh;
using gh::test::GenApp;
using nlohmann::json;

namespace {

json generateAndWait(GenApp& g, const json& body) {
    auto c = g.client();
    const auto r = c.Post("/api/v1/generate", body.dump(), "application/json");
    REQUIRE(r);
    INFO(r->body);
    REQUIRE(r->status == 202);
    const int64_t id = test::bodyJson(r)["job"]["id"];
    for (int i = 0; i < 100; ++i) {
        const auto state = test::bodyJson(c.Get("/api/v1/jobs/" + std::to_string(id)));
        const std::string s = state["state"];
        if (s == "succeeded" || s == "failed" || s == "cancelled") return state;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    FAIL("the job did not finish");
    return {};
}

}  // namespace

TEST_CASE("native: status, hardware, settings and logs", "[integration][api][native]") {
    GenApp g;
    auto c = g.client();

    const json status = test::bodyJson(c.Get("/api/v1/status"));
    CHECK(status["name"] == "GraphicsHost");
    CHECK(status["model"].is_null());
    CHECK(status["queue"]["active"] == 0);
    CHECK(status["counters"]["submitted"] == 0);

    const json hardware = test::bodyJson(c.Get("/api/v1/hardware"));
    CHECK(hardware.contains("cpu"));
    CHECK(hardware.contains("vulkan"));
    CHECK(hardware["memory"]["totalBytes"].get<uint64_t>() > 0);

    CHECK(test::bodyJson(c.Get("/api/v1/settings"))["backend"] == "auto");
    const auto changed = c.Put("/api/v1/settings", R"({"logging":{"level":"debug"},"server":{"port":4321}})", "application/json");
    REQUIRE(changed->status == 200);
    const json after = test::bodyJson(changed);
    CHECK(after["settings"]["logging"]["level"] == "debug");
    CHECK(after["settings"]["server"]["port"] == 4321);
    CHECK(after["settings"]["backend"] == "auto");   // untouched
    CHECK(after["restartRequired"] == true);
    CHECK(g.app.host->settings().logging.level == "debug");

    const auto bad = c.Put("/api/v1/settings", R"({"backend":"quantum"})", "application/json");
    CHECK(bad->status == 400);
    CHECK(test::bodyJson(bad)["error"]["code"] == "invalid_settings");
    CHECK(g.app.host->settings().backend == "auto");

    const json logs = test::bodyJson(c.Get("/api/v1/logs?limit=5"));
    CHECK(logs["entries"].is_array());
    CHECK(logs["entries"].size() <= 5);
    CHECK(c.Get("/api/v1/logs?limit=abc")->status == 400);
}

TEST_CASE("native: models can be listed, loaded and unloaded", "[integration][api][native]") {
    GenApp g;
    auto c = g.client();
    json list = test::bodyJson(c.Get("/api/v1/models"));
    REQUIRE(list["models"].size() == 1);
    CHECK(list["models"][0]["file"] == "tiny.safetensors");
    CHECK(list["models"][0]["loaded"] == false);
    CHECK(list["loaded"].is_null());

    const auto loaded = g.post("/api/v1/models/load", {{"model", "tiny"}, {"load", {{"ttlMinutes", 0}}}});
    INFO(loaded->body);
    REQUIRE(loaded->status == 200);
    list = test::bodyJson(c.Get("/api/v1/models"));
    CHECK(list["models"][0]["loaded"] == true);
    CHECK(list["loaded"]["options"]["ttlMinutes"] == 0);
    CHECK(test::bodyJson(c.Get("/api/v1/status"))["model"]["family"] == "SD 1.x");

    CHECK(g.post("/api/v1/models/load", {{"model", "ghost"}})->status == 404);
    CHECK(c.Post("/api/v1/models/unload", "", "application/json")->status == 200);
    CHECK(test::bodyJson(c.Get("/api/v1/models"))["loaded"].is_null());
}

TEST_CASE("native: generate returns a job that can be polled", "[integration][api][native]") {
    GenApp g;
    const json job = generateAndWait(g, {{"model", "tiny"}, {"prompt", "a lighthouse"}, {"n", 2}, {"seed", 9}});
    CHECK(job["state"] == "succeeded");
    REQUIRE(job["images"].size() == 2);
    CHECK(job["images"][0]["seed"] == 9);
    CHECK(job["images"][1]["seed"] == 10);
    CHECK(job["images"][0]["url"].get<std::string>().rfind("/outputs/", 0) == 0);
    CHECK(job["request"]["prompt"] == "a lighthouse");

    auto c = g.client();
    const json jobs = test::bodyJson(c.Get("/api/v1/jobs"));
    REQUIRE(jobs["jobs"].size() == 1);
    CHECK(jobs["jobs"][0]["id"] == job["id"]);
    CHECK(c.Get("/api/v1/jobs/9999")->status == 404);
    CHECK(c.Post("/api/v1/jobs/9999/cancel", "", "application/json")->status == 404);
    CHECK(g.post("/api/v1/generate", {{"model", "tiny"}})->status == 400);   // no prompt
}

TEST_CASE("native: a queued job can be cancelled", "[integration][api][native]") {
    GenApp g;
    g.state->stepMs = 300;
    auto c = g.client();
    const auto first = test::bodyJson(g.post("/api/v1/generate", {{"model", "tiny"}, {"prompt", "one"}}));
    const auto second = test::bodyJson(g.post("/api/v1/generate", {{"model", "tiny"}, {"prompt", "two"}}));
    const int64_t secondId = second["job"]["id"];
    const auto cancelled = c.Post("/api/v1/jobs/" + std::to_string(secondId) + "/cancel", "", "application/json");
    REQUIRE(cancelled->status == 200);
    for (int i = 0; i < 60; ++i) {
        if (test::bodyJson(c.Get("/api/v1/jobs/" + std::to_string(secondId)))["state"] == "cancelled") break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    CHECK(test::bodyJson(c.Get("/api/v1/jobs/" + std::to_string(secondId)))["state"] == "cancelled");
    const int64_t firstId = first["job"]["id"];
    for (int i = 0; i < 60 && test::bodyJson(c.Get("/api/v1/jobs/" + std::to_string(firstId)))["state"] != "succeeded"; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    CHECK(test::bodyJson(c.Get("/api/v1/jobs/" + std::to_string(firstId)))["state"] == "succeeded");
}

TEST_CASE("native: job events stream progress and the final document", "[integration][api][native]") {
    GenApp g;
    g.state->stepMs = 50;
    const auto submitted = test::bodyJson(g.post("/api/v1/generate", {{"model", "tiny"}, {"prompt", "stream"}}));
    const int64_t id = submitted["job"]["id"];
    auto c = g.client();
    std::string stream;
    const auto r = c.Get("/api/v1/jobs/" + std::to_string(id) + "/events", [&](const char* data, size_t n) {
        stream.append(data, n);
        return true;
    });
    REQUIRE(r);
    CHECK(r->status == 200);
    CHECK(stream.find("event: progress") != std::string::npos);
    const auto done = stream.find("event: done");
    REQUIRE(done != std::string::npos);
    const auto data = stream.find("data: ", done);
    const json final = json::parse(stream.substr(data + 6, stream.find('\n', data) - data - 6));
    CHECK(final["state"] == "succeeded");
    CHECK(final["images"].size() == 1);
}

TEST_CASE("native: the gallery lists, serves, thumbnails and deletes images", "[integration][api][native]") {
    GenApp g;
    generateAndWait(g, {{"model", "tiny"}, {"prompt", "sunset over water"}, {"width", 128}, {"height", 64}});
    generateAndWait(g, {{"model", "tiny"}, {"prompt", "a cat"}});
    auto c = g.client();

    json list = test::bodyJson(c.Get("/api/v1/gallery"));
    CHECK(list["total"] == 2);
    REQUIRE(list["items"].size() == 2);
    list = test::bodyJson(c.Get("/api/v1/gallery?text=sunset"));
    REQUIRE(list["items"].size() == 1);
    CHECK(list["total"] == 1);
    const json item = list["items"][0];
    const std::string id = std::to_string(item["id"].get<int64_t>());
    CHECK(item["width"] == 128);
    CHECK(item["parameters"]["request"]["prompt"] == "sunset over water");

    CHECK(test::bodyJson(c.Get("/api/v1/gallery/" + id))["prompt"] == "sunset over water");
    const auto image = c.Get("/api/v1/gallery/" + id + "/image");
    REQUIRE(image->status == 200);
    CHECK(image->get_header_value("Content-Type") == "image/png");
    CHECK(image->body.rfind("\x89PNG", 0) == 0);

    const auto thumb = c.Get("/api/v1/gallery/" + id + "/thumbnail?size=32");
    REQUIRE(thumb->status == 200);
    CHECK(thumb->get_header_value("Content-Type") == "image/jpeg");
    CHECK(thumb->body.size() < image->body.size());

    const auto file = c.Get(item["url"].get<std::string>());
    REQUIRE(file->status == 200);
    CHECK(file->body == image->body);

    CHECK(c.Get("/api/v1/gallery?limit=x")->status == 400);
    CHECK(c.Delete("/api/v1/gallery/" + id)->status == 204);
    CHECK(c.Delete("/api/v1/gallery/" + id)->status == 404);
    CHECK(c.Get("/api/v1/gallery/" + id)->status == 404);
    CHECK(test::bodyJson(c.Get("/api/v1/gallery"))["total"] == 1);
}

TEST_CASE("native: /outputs only serves image files inside the gallery folder", "[integration][api][native]") {
    GenApp g;
    generateAndWait(g, {{"model", "tiny"}, {"prompt", "x"}});
    auto c = g.client();
    CHECK(c.Get("/outputs/gallery.db")->status == 404);          // the index is not for download
    CHECK(c.Get("/outputs/missing.png")->status == 404);
    CHECK(c.Get("/outputs/../settings.json")->status == 404);
    CHECK(c.Get("/outputs/%2e%2e/settings.json")->status == 404);
    CHECK(c.Get("/outputs/..%5Csettings.json")->status == 404);
    CHECK(c.Get("/outputs/")->status == 404);
}

TEST_CASE("native: presets and model defaults can be managed", "[integration][api][native]") {
    GenApp g;
    auto c = g.client();
    CHECK(test::bodyJson(c.Get("/api/v1/presets"))["presets"].empty());
    const auto saved = c.Put("/api/v1/presets/fast", R"({"description":"quick drafts","request":{"steps":8,"cfgScale":2}})", "application/json");
    INFO(saved->body);
    REQUIRE(saved->status == 200);
    CHECK(test::bodyJson(c.Get("/api/v1/presets/fast"))["request"]["steps"] == 8);
    CHECK(test::bodyJson(c.Get("/api/v1/presets"))["presets"].size() == 1);
    CHECK(c.Put("/api/v1/presets/bad", R"({"request":{"sampler":"nonsense"}})", "application/json")->status == 400);
    CHECK(c.Put("/api/v1/presets/..%2Fescape", R"({})", "application/json")->status >= 400);   // never reaches the store
    CHECK(c.Put("/api/v1/presets/.hidden", R"({})", "application/json")->status == 400);

    // the preset applies through the API
    generateAndWait(g, {{"model", "tiny"}, {"prompt", "p"}, {"preset", "fast"}});
    {
        std::lock_guard<std::mutex> lock(g.state->mutex);
        CHECK(g.state->requests.back().steps == 8);
    }
    CHECK(c.Delete("/api/v1/presets/fast")->status == 204);
    CHECK(c.Delete("/api/v1/presets/fast")->status == 404);
    CHECK(c.Get("/api/v1/presets/fast")->status == 404);

    CHECK(c.Get("/api/v1/defaults/tiny.safetensors")->status == 404);
    REQUIRE(c.Put("/api/v1/defaults/tiny.safetensors", R"({"request":{"steps":11},"load":{"flashAttention":true}})", "application/json")->status == 200);
    CHECK(test::bodyJson(c.Get("/api/v1/defaults/tiny.safetensors"))["request"]["steps"] == 11);
    CHECK(test::bodyJson(c.Get("/api/v1/defaults"))["defaults"].size() == 1);
    CHECK(test::bodyJson(c.Get("/api/v1/models"))["models"][0]["hasDefaults"] == true);
    CHECK(c.Delete("/api/v1/defaults/tiny.safetensors")->status == 204);
    CHECK(c.Put("/api/v1/defaults/tiny.safetensors", R"({"request":{"steps":-5}})", "application/json")->status == 400);
}

TEST_CASE("native: catalog, getting models and downloads", "[integration][api][native]") {
    GenApp g;
    auto c = g.client();
    const json catalog = test::bodyJson(c.Get("/api/v1/catalog"));
    REQUIRE(catalog["entries"].size() >= 1);
    CHECK(catalog["entries"][0].contains("fit"));
    CHECK(catalog["entries"][0].contains("installed"));
    CHECK(test::bodyJson(c.Get("/api/v1/catalog?family=SDXL"))["entries"].size() < catalog["entries"].size());

    const std::string key = catalog["entries"][0]["key"];
    const auto plan = g.post("/api/v1/models/get", {{"target", key}, {"dryRun", true}});
    INFO(plan->body);
    REQUIRE(plan->status == 200);
    CHECK(test::bodyJson(plan)["plan"]["items"].size() >= 1);
    CHECK(test::bodyJson(plan)["started"].empty());

    CHECK(g.post("/api/v1/models/get", {{"target", "nothing-here"}})->status == 404);
    CHECK(g.post("/api/v1/models/get", json::object())->status == 400);
    CHECK(test::bodyJson(c.Get("/api/v1/downloads"))["downloads"].empty());
    CHECK(c.Post("/api/v1/downloads/77/pause", "", "application/json")->status == 404);
    CHECK(c.Get("/api/v1/hub/files?repo=noslash")->status == 400);
}

TEST_CASE("native: writing needs the matching permission", "[integration][api][native][auth]") {
    GenApp g;
    const auto reader = g.app.host->tokens().create("reader", {"read"});
    const auto maker = g.app.host->tokens().create("maker", {"read", "generate"});
    const auto manager = g.app.host->tokens().create("manager", {"read", "generate", "manage"});
    auto settings = g.app.host->settings();
    settings.server.requireToken = true;
    g.app.host->updateSettings(settings);
    auto c = g.client();
    auto as = [](const std::string& secret) { return httplib::Headers{{"Authorization", "Bearer " + secret}}; };

    CHECK(c.Get("/api/v1/status")->status == 401);
    CHECK(c.Get("/api/v1/status", as(reader.secret))->status == 200);
    CHECK(c.Post("/api/v1/models/load", as(reader.secret), R"({"model":"tiny"})", "application/json")->status == 403);
    CHECK(c.Post("/api/v1/models/load", as(maker.secret), R"({"model":"tiny"})", "application/json")->status == 403);
    CHECK(c.Post("/api/v1/models/load", as(manager.secret), R"({"model":"tiny"})", "application/json")->status == 200);
    CHECK(c.Post("/api/v1/generate", as(reader.secret), R"({"prompt":"x"})", "application/json")->status == 403);
    CHECK(c.Post("/api/v1/generate", as(maker.secret), R"({"prompt":"x"})", "application/json")->status == 202);
    CHECK(c.Put("/api/v1/settings", as(manager.secret), "{}", "application/json")->status == 403);   // admin only
    CHECK(c.Delete("/api/v1/presets/x", as(maker.secret))->status == 403);
}
