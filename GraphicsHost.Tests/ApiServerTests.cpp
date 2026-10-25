#include "ApiTestUtil.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::test;
using nlohmann::json;

TEST_CASE("server: starts on a free port, answers health, stops", "[unit][api][server]") {
    TestApp app(TestApp::ephemeral());
    REQUIRE(app.error.empty());
    CHECK(app.server->running());
    CHECK(app.server->port() > 0);
    CHECK(app.server->url() == "http://127.0.0.1:" + std::to_string(app.server->port()));

    auto c = app.client();
    const auto r = c.Get("/api/v1/health");
    REQUIRE(r);
    CHECK(r->status == 200);
    CHECK(r->get_header_value("Content-Type").find("application/json") != std::string::npos);
    CHECK(bodyJson(r)["status"] == "ok");
    CHECK(bodyJson(r)["name"] == "GraphicsHost");

    app.server->stop();
    CHECK_FALSE(app.server->running());
    CHECK_FALSE(app.client().Get("/api/v1/health"));   // nobody is listening any more
    app.server->stop();                                // harmless twice
}

TEST_CASE("server: it can be started again after it was stopped", "[unit][api][server]") {
    TestApp app(TestApp::ephemeral());
    REQUIRE(app.error.empty());
    app.server->stop();
    // a stopped httplib server cannot be reused, so the ApiServer must build a fresh one
    CHECK(app.server->start(TestApp::ephemeral()).empty());
    CHECK(app.client().Get("/api/v1/health"));
}

TEST_CASE("server: a port that is taken is reported with advice, not a crash", "[unit][api][server]") {
    TestApp first(TestApp::ephemeral());
    REQUIRE(first.error.empty());
    TestApp second(TestApp::ephemeral(), false);
    auto options = TestApp::ephemeral();
    options.port = first.server->port();
    const std::string error = second.server->start(options);
    CHECK(error.find(std::to_string(first.server->port())) != std::string::npos);
    CHECK(error.find("another program") != std::string::npos);
    CHECK_FALSE(second.server->running());
    CHECK(first.client().Get("/api/v1/health"));   // the first one is unaffected
    CHECK(second.server->start(TestApp::ephemeral()).empty());   // and the second can still use another port
}

TEST_CASE("server: every failure is JSON in one shape", "[unit][api][server]") {
    TestApp app(TestApp::ephemeral());
    auto c = app.client();
    const auto missing = c.Get("/api/v1/nope");
    REQUIRE(missing);
    CHECK(missing->status == 404);
    const json e = bodyJson(missing);
    REQUIRE(e.contains("error"));
    CHECK(e["error"]["type"] == "not_found");
    CHECK_FALSE(e["error"]["message"].get<std::string>().empty());
    CHECK(missing->get_header_value("Content-Type").find("application/json") != std::string::npos);
}

TEST_CASE("server: the browser UI folder is served at /", "[unit][api][server]") {
    TempDir web;
    writeText(web.path() / "index.html", "<html><body>hello ui</body></html>");
    writeText(web.path() / "app.js", "console.log(1)");
    auto options = TestApp::ephemeral();
    options.webRoot = web.path();
    TestApp app(options);
    auto c = app.client();
    const auto index = c.Get("/");
    REQUIRE(index);
    CHECK(index->status == 200);
    CHECK(index->body.find("hello ui") != std::string::npos);
    CHECK(index->get_header_value("Content-Type").find("text/html") != std::string::npos);
    CHECK(c.Get("/app.js")->status == 200);
    CHECK(c.Get("/../secret")->status == 404);   // no way out of the folder
    CHECK(c.Get("/api/v1/health")->status == 200);   // the API still wins its routes
}

TEST_CASE("server: without a web folder / is a JSON 404", "[unit][api][server]") {
    auto options = TestApp::ephemeral();
    options.webRoot = TempDir().path() / "does-not-exist";
    TestApp app(options);
    const auto r = app.client().Get("/");
    REQUIRE(r);
    CHECK(r->status == 404);
    CHECK(bodyJson(r)["error"]["type"] == "not_found");
}
