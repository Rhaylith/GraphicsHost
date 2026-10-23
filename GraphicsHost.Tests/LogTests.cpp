#include "TestUtil.h"

#include "core/Log.h"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <sstream>

TEST_CASE("log: entries reach the ring buffer and the JSON-lines file", "[unit][log]") {
    gh::test::TempDir dir;
    gh::log::init(dir.path(), "info");
    gh::log::debug("hidden: below the level");
    gh::log::info("hello \"quoted\"\nsecond line");
    gh::log::warn("careful");

    const auto tail = gh::log::tail(10);
    REQUIRE(tail.size() == 2);
    CHECK(tail[0].level == "info");
    CHECK(tail[0].message == "hello \"quoted\"\nsecond line");
    CHECK(tail[1].level == "warn");
    CHECK(tail[1].time.size() == 24);  // 2026-10-22T21:38:45.123Z

    gh::log::shutdown();  // flushes the file

    std::istringstream lines(gh::test::readText(dir.path() / "graphicshost.log"));
    std::string line;
    int count = 0;
    while (std::getline(lines, line)) {
        const auto json = nlohmann::json::parse(line);  // every line is one valid JSON object
        CHECK(json.contains("ts"));
        CHECK(json.contains("level"));
        CHECK(json.contains("msg"));
        if (count == 0) {
            CHECK(json["msg"] == "hello \"quoted\"\nsecond line");
        }
        ++count;
    }
    CHECK(count == 2);
}

TEST_CASE("log: the level can be changed at run time", "[unit][log]") {
    gh::test::TempDir dir;
    gh::log::init(dir.path(), "error");
    gh::log::info("dropped");
    gh::log::error("kept");
    gh::log::setLevel("debug");
    gh::log::debug("now kept");
    const auto tail = gh::log::tail(10);
    REQUIRE(tail.size() == 2);
    CHECK(tail[0].message == "kept");
    CHECK(tail[1].message == "now kept");
    gh::log::shutdown();
}

TEST_CASE("log: native engine messages are tagged and trimmed", "[unit][log]") {
    gh::test::TempDir dir;
    gh::log::init(dir.path(), "debug");
    gh::log::nativeLog(3, "something odd\n");
    gh::log::nativeLog(2, "\n");  // empty after trimming: ignored
    const auto tail = gh::log::tail(10);
    REQUIRE(tail.size() == 1);
    CHECK(tail[0].level == "warn");
    CHECK(tail[0].message == "[engine] something odd");
    gh::log::shutdown();
}

TEST_CASE("log: tail returns at most the requested number, newest last", "[unit][log]") {
    gh::test::TempDir dir;
    gh::log::init(dir.path(), "info");
    for (int i = 0; i < 5; ++i) {
        gh::log::info("m" + std::to_string(i));
    }
    const auto tail = gh::log::tail(2);
    REQUIRE(tail.size() == 2);
    CHECK(tail[0].message == "m3");
    CHECK(tail[1].message == "m4");
    gh::log::shutdown();
}
