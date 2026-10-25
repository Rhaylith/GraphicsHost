#include "ApiTestUtil.h"

#include "server/Tls.h"

#include <catch2/catch_test_macros.hpp>

#include <fstream>

using namespace gh;
using nlohmann::json;
namespace fs = std::filesystem;

TEST_CASE("tls: a self-signed certificate is made once and kept", "[integration][api][tls]") {
    test::TempDir dir;
    fs::path cert, key;
    REQUIRE(server::ensureSelfSignedCertificate(dir.path() / "tls", cert, key).empty());
    REQUIRE(fs::is_regular_file(cert));
    REQUIRE(fs::is_regular_file(key));
    const auto stamp = fs::last_write_time(cert);
    const auto size = fs::file_size(cert);
    fs::path cert2, key2;
    REQUIRE(server::ensureSelfSignedCertificate(dir.path() / "tls", cert2, key2).empty());
    CHECK(cert2 == cert);
    CHECK(fs::last_write_time(cert) == stamp);   // still good: not made again
    CHECK(fs::file_size(cert) == size);

    // a damaged file is replaced
    { std::ofstream(cert) << "junk"; }
    REQUIRE(server::ensureSelfSignedCertificate(dir.path() / "tls", cert, key).empty());
    CHECK(fs::file_size(cert) > 200);
}

TEST_CASE("tls: the server speaks https with a generated certificate", "[integration][api][tls]") {
    test::TempDir tlsDir;
    auto options = test::TestApp::ephemeral();
    options.https = true;
    options.tlsDir = tlsDir.path();
    test::TestApp app(options);
    REQUIRE(app.error.empty());
    CHECK(app.server->url().rfind("https://", 0) == 0);

    httplib::SSLClient secure("127.0.0.1", app.server->port());
    secure.enable_server_certificate_verification(false);   // self-signed
    secure.set_connection_timeout(10);
    const auto r = secure.Get("/api/v1/health");
    REQUIRE(r);
    CHECK(r->status == 200);
    CHECK(json::parse(r->body)["status"] == "ok");

    httplib::Client plain("127.0.0.1", app.server->port());   // http to an https port does not work
    plain.set_connection_timeout(3);
    plain.set_read_timeout(3);
    const auto p = plain.Get("/api/v1/health");
    CHECK((!p || p->status != 200));
}

TEST_CASE("tls: a certificate and key of the user's are used, bad ones are reported", "[integration][api][tls]") {
    test::TempDir dir;
    fs::path cert, key;
    REQUIRE(server::ensureSelfSignedCertificate(dir.path(), cert, key).empty());

    auto options = test::TestApp::ephemeral();
    options.https = true;
    options.tlsCertFile = cert;
    options.tlsKeyFile = key;
    test::TestApp good(options);
    REQUIRE(good.error.empty());
    httplib::SSLClient c("127.0.0.1", good.server->port());
    c.enable_server_certificate_verification(false);
    REQUIRE(c.Get("/api/v1/health"));

    { std::ofstream(dir.path() / "bad.pem") << "not a certificate"; }
    options.tlsCertFile = dir.path() / "bad.pem";
    test::TestApp bad(options);
    CHECK_FALSE(bad.error.empty());
    CHECK(bad.error.find("certificate") != std::string::npos);
    CHECK_FALSE(bad.server->running());
}

TEST_CASE("limits: the address list lets this computer in and refuses nonsense", "[integration][api][limits]") {
    auto options = test::TestApp::ephemeral();
    options.allowedAddresses = {"10.0.0.0/8"};
    test::TestApp app(options);
    REQUIRE(app.error.empty());
    auto c = app.client();
    CHECK(c.Get("/api/v1/health")->status == 200);   // loopback is always allowed

    auto broken = test::TestApp::ephemeral();
    broken.allowedAddresses = {"not an address"};
    test::TestApp bad(broken);
    CHECK(bad.error.find("address") != std::string::npos);
    CHECK_FALSE(bad.server->running());
}

TEST_CASE("limits: too many requests get a 429 with Retry-After, the health check is exempt", "[integration][api][limits]") {
    auto options = test::TestApp::ephemeral();
    options.rateLimitPerMinute = 5;
    test::TestApp app(options);
    REQUIRE(app.error.empty());
    auto c = app.client();
    for (int i = 0; i < 5; ++i) REQUIRE(c.Get("/api/v1/status")->status == 200);
    const auto r = c.Get("/api/v1/status");
    REQUIRE(r);
    CHECK(r->status == 429);
    CHECK(std::stoi(r->get_header_value("Retry-After")) >= 1);
    const json body = json::parse(r->body);
    CHECK(body["error"]["code"] == "rate_limited");
    CHECK(body["error"]["type"] == "rate_limit_error");
    for (int i = 0; i < 20; ++i) CHECK(c.Get("/api/v1/health")->status == 200);   // monitors can keep polling

    // another token is another client
    const auto token = app.host->tokens().create("other", {"read"});
    const httplib::Headers as = {{"Authorization", "Bearer " + token.secret}};
    CHECK(c.Get("/api/v1/status", as)->status == 200);
}
