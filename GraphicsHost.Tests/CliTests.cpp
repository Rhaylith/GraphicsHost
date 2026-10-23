#include "TestUtil.h"

#include "cli/Cli.h"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <sstream>
#include <string>
#include <vector>

namespace {

struct Result {
    int code = 0;
    std::string out;
    std::string err;
};

Result runCli(const std::filesystem::path& home, std::vector<std::string> args) {
    args.insert(args.begin(), "graphicshost");
    std::vector<char*> argv;
    for (auto& a : args) {
        argv.push_back(a.data());
    }
    std::ostringstream out, err;
    Result r;
    r.code = gh::cli::run(static_cast<int>(argv.size()), argv.data(), out, err, home);
    r.out = out.str();
    r.err = err.str();
    return r;
}

}  // namespace

TEST_CASE("cli: GUI or CLI is decided by the arguments", "[unit][cli]") {
    auto decide = [](std::vector<std::string> args) {
        args.insert(args.begin(), "graphicshost");
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        return gh::cli::wantsCli(static_cast<int>(argv.size()), argv.data());
    };
    CHECK_FALSE(decide({}));
    CHECK_FALSE(decide({"--smoke-test"}));
    CHECK_FALSE(decide({"gui"}));
    CHECK(decide({"hardware"}));
    CHECK(decide({"--json", "version"}));
    CHECK(decide({"--help"}));
}

TEST_CASE("cli: version", "[unit][cli]") {
    gh::test::TempDir home;
    const Result text = runCli(home.path(), {"version"});
    CHECK(text.code == gh::cli::kOk);
    CHECK(text.out.find("GraphicsHost 0.1.0") != std::string::npos);

    const Result json = runCli(home.path(), {"--json", "version"});
    CHECK(json.code == gh::cli::kOk);
    CHECK(nlohmann::json::parse(json.out)["name"] == "GraphicsHost");
}

TEST_CASE("cli: hardware --json is valid and complete", "[unit][cli]") {
    gh::test::TempDir home;
    const Result r = runCli(home.path(), {"--json", "hardware"});
    REQUIRE(r.code == gh::cli::kOk);
    const auto json = nlohmann::json::parse(r.out);
    CHECK(json["cpu"]["logicalCores"].get<int>() > 0);
    CHECK(json["vulkan"].contains("available"));
    CHECK(json["runtimes"].is_array());
    CHECK(json["backendSetting"] == "auto");
    CHECK(json.contains("selectedRuntime"));
}

TEST_CASE("cli: hardware text mentions the CPU and the backend", "[unit][cli]") {
    gh::test::TempDir home;
    const Result r = runCli(home.path(), {"hardware"});
    CHECK(r.code == gh::cli::kOk);
    CHECK(r.out.find("CPU") != std::string::npos);
    CHECK(r.out.find("Backend") != std::string::npos);
}

TEST_CASE("cli: config set changes and persists a setting", "[unit][cli][settings]") {
    gh::test::TempDir home;
    CHECK(runCli(home.path(), {"config", "set", "backend", "cpu"}).code == gh::cli::kOk);
    CHECK(runCli(home.path(), {"config", "set", "server.port", "4321"}).code == gh::cli::kOk);
    CHECK(runCli(home.path(), {"config", "set", "safety.enabled", "true"}).code == gh::cli::kOk);

    const Result shown = runCli(home.path(), {"--json", "config", "show"});
    REQUIRE(shown.code == gh::cli::kOk);
    const auto json = nlohmann::json::parse(shown.out);
    CHECK(json["backend"] == "cpu");
    CHECK(json["server"]["port"] == 4321);
    CHECK(json["safety"]["enabled"] == true);
}

TEST_CASE("cli: config set rejects bad input with the right exit code", "[unit][cli][settings]") {
    gh::test::TempDir home;
    const Result badValue = runCli(home.path(), {"config", "set", "backend", "cuda"});
    CHECK(badValue.code == gh::cli::kUsage);
    CHECK(badValue.err.find("backend") != std::string::npos);

    CHECK(runCli(home.path(), {"config", "set", "server.port", "banana"}).code == gh::cli::kUsage);
    CHECK(runCli(home.path(), {"config", "set", "server.port", "0"}).code == gh::cli::kUsage);
    CHECK(runCli(home.path(), {"config", "set", "no.such.key", "1"}).code == gh::cli::kNotFound);
    CHECK(runCli(home.path(), {"config", "set", "schemaVersion", "9"}).code == gh::cli::kNotFound);
    CHECK(runCli(home.path(), {"config", "set", "backend"}).code == gh::cli::kUsage);

    // nothing of that was saved
    const auto json = nlohmann::json::parse(runCli(home.path(), {"--json", "config", "show"}).out);
    CHECK(json["backend"] == "auto");
    CHECK(json["server"]["port"] == 1234);
}

TEST_CASE("cli: config path points into the data folder", "[unit][cli]") {
    gh::test::TempDir home;
    const Result r = runCli(home.path(), {"config", "path"});
    CHECK(r.code == gh::cli::kOk);
    CHECK(r.out.find("settings.json") != std::string::npos);
}

TEST_CASE("cli: unknown commands are usage errors", "[unit][cli]") {
    gh::test::TempDir home;
    CHECK(runCli(home.path(), {"frobnicate"}).code == gh::cli::kUsage);
    CHECK(runCli(home.path(), {}).code == gh::cli::kUsage);
}

TEST_CASE("cli: runtime info loads the selected runtime", "[integration][cli]") {
    gh::test::TempDir home;
    const Result r = runCli(home.path(), {"--json", "runtime", "info"});
    if (r.code != gh::cli::kOk) {
        // no runtime next to the test exe, or the DLL was refused by Application Control
        WARN("runtime info failed with code " << r.code << ": " << r.err);
        SKIP("no loadable runtime on this machine");
    }
    const auto json = nlohmann::json::parse(r.out);
    CHECK_FALSE(json["id"].get<std::string>().empty());
    CHECK_FALSE(json["engineVersion"].get<std::string>().empty());
}
