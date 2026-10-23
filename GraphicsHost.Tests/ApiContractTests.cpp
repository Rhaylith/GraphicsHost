#include "GenApiTestUtil.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh;
using gh::test::GenApp;
using nlohmann::json;

// Every route the documentation (docs/api.md) lists must exist. A request to a route that does not exist is answered
// with the generic 404 that has no error code; a route that exists answers with its own status, or a 404 that has one
// ("job_not_found" ...).
TEST_CASE("api contract: every documented route exists and speaks JSON errors", "[integration][api][contract]") {
    GenApp g;
    auto c = g.client();
    struct Route { const char* method; const char* path; };
    const Route routes[] = {
        {"GET", "/api/v1/health"},          {"GET", "/api/v1/whoami"},           {"GET", "/api/v1/status"},
        {"GET", "/api/v1/hardware"},        {"GET", "/api/v1/parameters"},       {"GET", "/api/v1/settings"},
        {"PUT", "/api/v1/settings"},        {"GET", "/api/v1/logs"},             {"GET", "/api/v1/requests"},
        {"GET", "/metrics"},                {"GET", "/api/v1/tokens"},           {"POST", "/api/v1/tokens"},
        {"DELETE", "/api/v1/tokens/nope"},  {"GET", "/v1/models"},               {"POST", "/v1/images/generations"},
        {"POST", "/v1/images/edits"},       {"GET", "/api/v1/models"},           {"POST", "/api/v1/models/load"},
        {"POST", "/api/v1/models/unload"},  {"POST", "/api/v1/models/get"},      {"GET", "/api/v1/defaults"},
        {"GET", "/api/v1/defaults/nope"},   {"PUT", "/api/v1/defaults/nope"},    {"DELETE", "/api/v1/defaults/nope"},
        {"GET", "/api/v1/presets"},         {"GET", "/api/v1/presets/nope"},     {"PUT", "/api/v1/presets/nope"},
        {"DELETE", "/api/v1/presets/nope"}, {"GET", "/api/v1/catalog"},          {"GET", "/api/v1/hub/search"},
        {"GET", "/api/v1/hub/files"},       {"GET", "/api/v1/downloads"},        {"POST", "/api/v1/downloads/1/pause"},
        {"POST", "/api/v1/downloads/1/resume"}, {"POST", "/api/v1/downloads/1/cancel"}, {"GET", "/api/v1/updates"},
        {"POST", "/api/v1/generate"},       {"GET", "/api/v1/jobs"},             {"GET", "/api/v1/jobs/1"},
        {"POST", "/api/v1/jobs/1/cancel"},  {"GET", "/api/v1/jobs/1/events"},    {"GET", "/api/v1/gallery"},
        {"GET", "/api/v1/gallery/1"},       {"DELETE", "/api/v1/gallery/1"},     {"GET", "/api/v1/gallery/1/image"},
        {"GET", "/api/v1/gallery/1/thumbnail"}, {"GET", "/outputs/x.png"},
    };
    for (const auto& r : routes) {
        INFO(r.method << " " << r.path);
        httplib::Result res;
        const std::string m = r.method;
        // an offline Hugging Face must not make the test slow or flaky
        const std::string path = std::string(r.path) == "/api/v1/hub/search" ? "/api/v1/hub/search?q=" : r.path == std::string("/api/v1/hub/files") ? "/api/v1/hub/files?repo=bad" : r.path;
        if (m == "GET") res = c.Get(path);
        else if (m == "POST") res = c.Post(path, "{}", "application/json");
        else if (m == "PUT") res = c.Put(path, "{}", "application/json");
        else res = c.Delete(path);
        REQUIRE(res);
        const json body = json::parse(res->body, nullptr, false);
        if (res->status == 404 || res->status == 405) {
            // only a route's own "not found" is acceptable, and it names itself with a code
            REQUIRE(body.is_object());
            CHECK(body["error"]["code"].is_string());
        }
        if (res->status >= 400) {   // every failure is the one JSON shape
            REQUIRE(body.is_object());
            CHECK(body["error"].contains("message"));
            CHECK(body["error"].contains("type"));
        }
    }

    // routes that do not exist
    for (const char* path : {"/api/v1/nope", "/v2/images", "/api/v1/gallery/abc"}) {
        const auto res = c.Get(path);
        REQUIRE(res);
        CHECK(res->status == 404);
        CHECK(json::parse(res->body)["error"]["code"].is_null());
    }
}
