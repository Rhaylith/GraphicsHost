#include "TestUtil.h"

#include "cli/Cli.h"
#include "cli/Commands.h"
#include "server/ApiServer.h"

#include <catch2/catch_test_macros.hpp>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <sstream>
#include <thread>

using namespace gh;

TEST_CASE("cli serve: runs the API until it is told to stop", "[integration][cli][serve]") {
    test::TempDir home;

    // a port that is free right now
    int port = 0;
    {
        host::AppHost probe(home.path());
        server::ApiServer api(probe);
        server::ServerOptions o;
        o.port = 0;
        REQUIRE(api.start(o).empty());
        port = api.port();
    }

    std::ostringstream out, err;
    int code = -1;
    std::thread server([&] {
        std::vector<std::string> args = {"graphicshost", "serve", "--port", std::to_string(port)};
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        code = cli::run(static_cast<int>(argv.size()), argv.data(), out, err, home.path());
    });

    httplib::Client c("127.0.0.1", port);
    c.set_connection_timeout(1);
    bool up = false;
    for (int i = 0; i < 100 && !up; ++i) {
        const auto r = c.Get("/api/v1/health");
        up = r && r->status == 200;
        if (!up) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    CHECK(up);
    if (up) CHECK(nlohmann::json::parse(c.Get("/api/v1/health")->body)["status"] == "ok");

    cli::requestServeStop();
    server.join();
    CHECK(code == 0);
    CHECK(out.str().find("listening on http://127.0.0.1:" + std::to_string(port)) != std::string::npos);
    CHECK(out.str().find("stopped") != std::string::npos);
    CHECK(err.str().empty());
    CHECK_FALSE(c.Get("/api/v1/health"));   // gone
}

TEST_CASE("cli serve: a port that is taken is reported", "[integration][cli][serve]") {
    test::TempDir home;
    host::AppHost first(home.path());
    server::ApiServer api(first);
    server::ServerOptions o;
    o.port = 0;
    REQUIRE(api.start(o).empty());

    std::ostringstream out, err;
    std::vector<std::string> args = {"graphicshost", "serve", "--port", std::to_string(api.port())};
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    CHECK(cli::run(static_cast<int>(argv.size()), argv.data(), out, err, home.path()) == 1);
    CHECK(err.str().find("is another program using the port") != std::string::npos);
}

namespace {
int runWith(const test::TempDir& home, std::vector<std::string> args, std::string& out, std::string& err) {
    args.insert(args.begin(), "graphicshost");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::ostringstream o, e;
    const int code = cli::run(static_cast<int>(argv.size()), argv.data(), o, e, home.path());
    out = o.str();
    err = e.str();
    return code;
}
}  // namespace

TEST_CASE("cli service: status can be asked by anyone, and --service needs Windows to start it", "[unit][cli][serve]") {
    test::TempDir home;
    std::string out, err;
    REQUIRE(runWith(home, {"--json", "service", "status"}, out, err) == 0);   // installed or not, it answers
    const auto status = nlohmann::json::parse(out);
    REQUIRE(status.contains("installed"));
    if (status["installed"].get<bool>()) CHECK(status.contains("state"));

    // started from a console, not by the service manager: refused at once, nothing is served
    CHECK(runWith(home, {"serve", "--service"}, out, err) == 1);
    CHECK(err.find("service") != std::string::npos);

    CHECK(runWith(home, {"service", "bogus"}, out, err) == 2);
    CHECK(err.find("usage") != std::string::npos);
}
