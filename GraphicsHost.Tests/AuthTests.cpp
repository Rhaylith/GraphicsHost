#include "ApiTestUtil.h"

#include "core/Sha256.h"
#include "server/Auth.h"
#include "server/TokenStore.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::server;
using gh::test::TempDir;
using nlohmann::json;

TEST_CASE("tokens: create shows the secret once, only a hash is stored", "[unit][auth]") {
    TempDir dir;
    TokenStore store(dir.path() / "tokens.json");
    CHECK_FALSE(store.any());
    const auto created = store.create("laptop", {"generate", "read"});
    CHECK(created.secret.rfind("gh_", 0) == 0);
    CHECK(created.secret.size() == 3 + 64);
    CHECK(created.info.id.rfind("tok_", 0) == 0);
    CHECK(created.info.permissions == std::vector<std::string>{"generate", "read"});
    CHECK(store.any());

    const std::string file = gh::test::readText(dir.path() / "tokens.json");
    CHECK(file.find(created.secret) == std::string::npos);                 // the secret is nowhere in the file
    CHECK(file.find(gh::core::sha256Hex(created.secret)) != std::string::npos);   // its hash is

    const auto listed = store.list();
    REQUIRE(listed.size() == 1);
    CHECK(listed[0].name == "laptop");
    CHECK(listed[0].lastUsed.empty());
}

TEST_CASE("tokens: verify, last-used, revoke", "[unit][auth]") {
    TempDir dir;
    TokenStore store(dir.path() / "tokens.json");
    const auto a = store.create("a", {"read"});
    const auto b = store.create("b", {"admin"});
    CHECK(a.secret != b.secret);
    CHECK(a.info.id != b.info.id);

    const auto found = store.verify(a.secret);
    REQUIRE(found.has_value());
    CHECK(found->name == "a");
    CHECK_FALSE(store.verify("gh_" + std::string(64, '0')).has_value());
    CHECK_FALSE(store.verify("").has_value());
    CHECK_FALSE(store.verify("short").has_value());
    CHECK_FALSE(store.verify(a.secret + "x").has_value());
    CHECK_FALSE(store.list()[0].lastUsed.empty());     // used now

    // a second store object sees the same file (the server and the CLI are different processes)
    TokenStore other(dir.path() / "tokens.json");
    CHECK(other.verify(b.secret).has_value());

    CHECK(store.revoke("a"));                           // by name
    CHECK_FALSE(store.verify(a.secret).has_value());
    CHECK(store.revoke(b.info.id));                     // by id
    CHECK_FALSE(store.revoke("a"));
    CHECK_FALSE(store.any());
}

TEST_CASE("tokens: bad requests are refused", "[unit][auth]") {
    TempDir dir;
    TokenStore store(dir.path() / "tokens.json");
    CHECK_THROWS_AS(store.create("", {"read"}), std::invalid_argument);
    CHECK_THROWS_AS(store.create(std::string(65, 'x'), {"read"}), std::invalid_argument);
    CHECK_THROWS_AS(store.create("x", {}), std::invalid_argument);
    CHECK_THROWS_AS(store.create("x", {"root"}), std::invalid_argument);
    store.create("x", {"read"});
    CHECK_THROWS_AS(store.create("x", {"read"}), std::invalid_argument);   // names are unique
    CHECK(store.list().size() == 1);
}

TEST_CASE("tokens: permissions, admin allows everything; duplicates collapse", "[unit][auth]") {
    TempDir dir;
    TokenStore store(dir.path() / "tokens.json");
    const auto reader = store.create("r", {"read", "read"}).info;
    CHECK(reader.permissions == std::vector<std::string>{"read"});
    CHECK(reader.allows("read"));
    CHECK_FALSE(reader.allows("generate"));
    const auto admin = store.create("a", {"admin"}).info;
    for (const char* p : {"read", "generate", "manage", "admin"}) CHECK(admin.allows(p));
}

TEST_CASE("auth: the decision table", "[unit][auth]") {
    TempDir dir;
    TokenStore store(dir.path() / "tokens.json");
    bool requireToken = false;
    Auth auth(store, [&] { return requireToken; });
    const auto reader = store.create("reader", {"read"});
    const auto gen = store.create("gen", {"generate"});

    // this computer, no token: allowed (all permissions) unless tokens are required
    CHECK(auth.decide("", "127.0.0.1", "admin").allowed);
    CHECK(auth.decide("", "::1", "generate").allowed);
    CHECK(auth.decide("", "127.0.0.1", "admin").principal == "(local)");
    requireToken = true;
    const auto needed = auth.decide("", "127.0.0.1", "read");
    CHECK_FALSE(needed.allowed);
    CHECK(needed.status == 401);
    CHECK(needed.code == "missing_api_key");
    CHECK(needed.message.find("token create") != std::string::npos);
    requireToken = false;

    // another computer, no token: refused
    const auto remote = auth.decide("", "192.168.1.20", "read");
    CHECK_FALSE(remote.allowed);
    CHECK(remote.status == 401);

    // with a token: judged by the token, wherever it comes from
    CHECK(auth.decide(reader.secret, "192.168.1.20", "read").allowed);
    CHECK(auth.decide(reader.secret, "192.168.1.20", "read").principal == "reader");
    const auto tooLittle = auth.decide(reader.secret, "192.168.1.20", "generate");
    CHECK_FALSE(tooLittle.allowed);
    CHECK(tooLittle.status == 403);
    CHECK(tooLittle.code == "insufficient_permissions");
    CHECK(auth.decide(gen.secret, "127.0.0.1", "generate").allowed);

    // a wrong token is an error even from this computer
    const auto wrong = auth.decide("gh_" + std::string(64, 'f'), "127.0.0.1", "read");
    CHECK_FALSE(wrong.allowed);
    CHECK(wrong.status == 401);
    CHECK(wrong.code == "invalid_api_key");
}

TEST_CASE("auth: loopback addresses and header forms", "[unit][auth]") {
    for (const char* a : {"127.0.0.1", "127.1.2.3", "::1", "::ffff:127.0.0.1", "localhost"}) CHECK(Auth::isLoopback(a));
    for (const char* a : {"192.168.0.2", "10.0.0.1", "8.8.8.8", "::ffff:10.0.0.1", ""}) CHECK_FALSE(Auth::isLoopback(a));

    CHECK(Auth::secretFrom("Bearer abc", "") == "abc");
    CHECK(Auth::secretFrom("bearer   abc  ", "") == "abc");
    CHECK(Auth::secretFrom("", "key123") == "key123");            // x-api-key
    CHECK(Auth::secretFrom("Bearer abc", "key123") == "abc");     // the Authorization header wins
    CHECK(Auth::secretFrom("Basic dXNlcjpwYXNz", "").empty());    // other schemes are not tokens
    CHECK(Auth::secretFrom("", "").empty());
}

TEST_CASE("api: the token routes and the policy over HTTP", "[unit][api][auth]") {
    gh::test::TestApp app(gh::test::TestApp::ephemeral());
    auto c = app.client();

    // no tokens yet: this computer may do anything, including creating the first token
    const auto me = c.Get("/api/v1/whoami");
    REQUIRE(me);
    CHECK(me->status == 200);
    CHECK(gh::test::bodyJson(me)["principal"] == "(local)");
    CHECK(gh::test::bodyJson(me)["local"] == true);
    CHECK(gh::test::bodyJson(me)["permissions"].size() == 4);

    const auto made = c.Post("/api/v1/tokens", R"({"name":"ci","permissions":["read","generate"]})", "application/json");
    REQUIRE(made);
    CHECK(made->status == 201);
    const std::string secret = gh::test::bodyJson(made)["secret"];
    CHECK(secret.rfind("gh_", 0) == 0);
    const std::string id = gh::test::bodyJson(made)["token"]["id"];

    CHECK(c.Post("/api/v1/tokens", R"({"name":"ci","permissions":["read"]})", "application/json")->status == 400);   // duplicate
    CHECK(c.Post("/api/v1/tokens", R"({"name":"x","permissions":["root"]})", "application/json")->status == 400);
    CHECK(c.Post("/api/v1/tokens", "not json", "application/json")->status == 400);
    CHECK(c.Post("/api/v1/tokens", R"([1])", "application/json")->status == 400);

    // the token works, with what it is allowed to do
    httplib::Headers bearer = {{"Authorization", "Bearer " + secret}};
    const auto mine = c.Get("/api/v1/whoami", bearer);
    CHECK(mine->status == 200);
    CHECK(gh::test::bodyJson(mine)["principal"] == "ci");
    CHECK(gh::test::bodyJson(mine)["permissions"] == json::array({"generate", "read"}));
    CHECK(c.Get("/api/v1/whoami", httplib::Headers{{"x-api-key", secret}})->status == 200);   // the other header form
    const auto forbidden = c.Get("/api/v1/tokens", bearer);                                   // needs admin
    CHECK(forbidden->status == 403);
    CHECK(gh::test::bodyJson(forbidden)["error"]["code"] == "insufficient_permissions");
    CHECK(c.Get("/api/v1/whoami", httplib::Headers{{"Authorization", "Bearer gh_wrong"}})->status == 401);

    // the secret is never listed again
    const auto list = c.Get("/api/v1/tokens");
    REQUIRE(list->status == 200);
    CHECK(list->body.find(secret) == std::string::npos);
    CHECK(gh::test::bodyJson(list)["tokens"].size() == 1);

    // requiring a token for this computer too
    REQUIRE(gh::test::bodyJson(c.Get("/api/v1/whoami"))["tokenRequired"] == false);
    auto settings = app.host->settings();
    settings.server.requireToken = true;
    app.host->updateSettings(settings);
    const auto denied = c.Get("/api/v1/whoami");
    CHECK(denied->status == 401);
    CHECK(denied->get_header_value("WWW-Authenticate") == "Bearer");
    CHECK(c.Get("/api/v1/whoami", bearer)->status == 200);
    CHECK(c.Get("/api/v1/health")->status == 200);   // health stays public

    // revoking
    CHECK(c.Delete("/api/v1/tokens/" + id, bearer)->status == 403);
    settings.server.requireToken = false;
    app.host->updateSettings(settings);
    CHECK(c.Delete("/api/v1/tokens/" + id)->status == 204);
    CHECK(c.Delete("/api/v1/tokens/" + id)->status == 404);
    CHECK(c.Get("/api/v1/whoami", bearer)->status == 401);   // revoked
}

TEST_CASE("api: listening for the LAN is refused until a token exists", "[unit][api][auth]") {
    gh::test::TestApp app(gh::test::TestApp::ephemeral(), false);
    auto options = gh::test::TestApp::ephemeral();
    options.bindAddress = "0.0.0.0";
    const std::string refused = app.server->start(options);
    CHECK(refused.find("access token") != std::string::npos);
    CHECK_FALSE(app.server->running());

    app.host->tokens().create("lan", {"read"});
    CHECK(app.server->start(options).empty());
    CHECK(app.server->running());
    // this computer is still trusted on the LAN port; a request from elsewhere would need the token (see the decision table)
    CHECK(app.client().Get("/api/v1/health")->status == 200);
}

#include "cli/Cli.h"

#include <sstream>

TEST_CASE("cli token: create, list, revoke", "[unit][cli][auth]") {
    TempDir home;
    auto run = [&](std::vector<std::string> args) {
        args.insert(args.begin(), "graphicshost");
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        std::ostringstream out, err;
        const int code = gh::cli::run(static_cast<int>(argv.size()), argv.data(), out, err, home.path());
        return std::make_tuple(code, out.str(), err.str());
    };
    auto [c1, o1, e1] = run({"--json", "token", "list"});
    CHECK(c1 == gh::cli::kOk);
    CHECK(json::parse(o1)["tokens"].empty());

    auto [c2, o2, e2] = run({"--json", "token", "create", "phone", "--perm", "read,generate"});
    REQUIRE(c2 == gh::cli::kOk);
    const json made = json::parse(o2);
    CHECK(made["secret"].get<std::string>().rfind("gh_", 0) == 0);
    CHECK(made["permissions"] == json::array({"generate", "read"}));

    auto [c3, o3, e3] = run({"token", "list"});
    CHECK(o3.find("phone") != std::string::npos);
    CHECK(o3.find(made["secret"].get<std::string>()) == std::string::npos);   // never shown again

    auto [c4, o4, e4] = run({"token", "create", "phone"});
    CHECK(c4 == gh::cli::kUsage);                        // duplicate name
    CHECK(e4.find("already exists") != std::string::npos);
    CHECK(std::get<0>(run({"token", "create", "x", "--perm", "root"})) == gh::cli::kUsage);
    CHECK(std::get<0>(run({"token", "create"})) == gh::cli::kUsage);

    CHECK(std::get<0>(run({"token", "revoke", "phone"})) == gh::cli::kOk);
    CHECK(std::get<0>(run({"token", "revoke", "phone"})) == gh::cli::kNotFound);
    // the default permissions are read and generate
    auto [c5, o5, e5] = run({"--json", "token", "create", "default"});
    CHECK(json::parse(o5)["permissions"] == json::array({"generate", "read"}));
}

#include "server/ClientLimits.h"

TEST_CASE("limits: the allow-list understands ranges, single addresses and loopback", "[unit][api][limits]") {
    using gh::server::IpAllowList;
    CHECK(IpAllowList().allows("203.0.113.9"));                                   // no list: no restriction
    const IpAllowList list({"192.168.1.0/24", "10.1.2.3", "fe80::1"});
    CHECK(list.restricted());
    CHECK(list.allows("192.168.1.77"));
    CHECK_FALSE(list.allows("192.168.2.1"));
    CHECK(list.allows("10.1.2.3"));
    CHECK_FALSE(list.allows("10.1.2.4"));
    CHECK(list.allows("fe80::1"));
    CHECK_FALSE(list.allows("fe80::2"));
    CHECK(list.allows("::ffff:192.168.1.5"));                                       // IPv4 on a dual-stack socket
    CHECK(list.allows("127.0.0.1"));                                                // this computer is always allowed
    CHECK(list.allows("::1"));
    CHECK(IpAllowList({"0.0.0.0/0"}).allows("8.8.8.8"));
    CHECK_THROWS_AS(IpAllowList({"192.168.1.0/33"}), std::invalid_argument);
    CHECK_THROWS_AS(IpAllowList({"example.com"}), std::invalid_argument);
    CHECK_THROWS_AS(IpAllowList({"300.1.1.1"}), std::invalid_argument);
    CHECK(IpAllowList::valid("10.0.0.0/8"));
    CHECK_FALSE(IpAllowList::valid("10.0.0.0/"));
}

TEST_CASE("limits: the rate limiter counts a sliding minute per client", "[unit][api][limits]") {
    using gh::server::RateLimiter;
    RateLimiter off(0);
    for (int i = 0; i < 1000; ++i) REQUIRE(off.allow("a"));

    RateLimiter limiter(3);
    const auto t0 = RateLimiter::Clock::now();
    int retry = 0;
    CHECK(limiter.allow("a", &retry, t0));
    CHECK(limiter.allow("a", &retry, t0 + std::chrono::seconds(1)));
    CHECK(limiter.allow("a", &retry, t0 + std::chrono::seconds(2)));
    CHECK_FALSE(limiter.allow("a", &retry, t0 + std::chrono::seconds(3)));
    CHECK(retry >= 57);
    CHECK(retry <= 60);
    CHECK(limiter.allow("b", &retry, t0 + std::chrono::seconds(3)));                 // another client has its own count
    CHECK(limiter.allow("a", &retry, t0 + std::chrono::seconds(61)));                // the first two are a minute old, one request is left over
    CHECK(limiter.allow("a", &retry, t0 + std::chrono::seconds(61)));
    CHECK_FALSE(limiter.allow("a", &retry, t0 + std::chrono::seconds(61)));
}
