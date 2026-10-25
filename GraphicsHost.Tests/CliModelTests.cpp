#include "TestHttpServer.h"
#include "TestUtil.h"

#include "cli/Cli.h"
#include "core/SecretStore.h"
#include "core/Sha256.h"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <sstream>

using nlohmann::json;
namespace fs = std::filesystem;

namespace {

constexpr uint64_t kSize = 120000, kVaeSize = 70000;
constexpr uint64_t kBig = 21ull * 1024 * 1024;   // above the 20 MB the application needs to call a file a model
std::string hashOf(uint64_t size) { return gh::core::sha256Hex(gh::test::blobBytes(0, size)); }

struct Result {
    int code = 0;
    std::string out, err;
    json asJson() const { return json::parse(out); }
};

Result runCli(const fs::path& home, std::vector<std::string> args) {
    args.insert(args.begin(), "graphicshost");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::ostringstream out, err;
    Result r;
    r.code = gh::cli::run(static_cast<int>(argv.size()), argv.data(), out, err, home);
    r.out = out.str();
    r.err = err.str();
    return r;
}

// A Hugging Face look-alike plus the environment switch that points the application at it.
struct Hub {
    gh::test::TestHttpServer server;
    std::map<std::string, uint64_t> files{{"pub/xl-model/xl.safetensors", kSize}, {"vaes/xl-vae/vae.safetensors", kVaeSize},
                                          {"gated/closed/c.safetensors", kSize}, {"two/files/a.safetensors", kBig},
                                          {"two/files/b.safetensors", kBig}, {"one/file/only.safetensors", kBig}};
    Hub() {
        server.server().Get(R"(/api/models)", [](const httplib::Request& req, httplib::Response& res) {
            if (req.get_param_value("search") == "nothing") { res.set_content("[]", "application/json"); return; }
            res.set_content(json::array({{{"id", "pub/xl-model"}, {"downloads", 1000}, {"likes", 7}, {"gated", false}, {"tags", json::array({"license:openrail"})}},
                                         {{"id", "gated/closed"}, {"downloads", 5}, {"likes", 1}, {"gated", "manual"}}}).dump(), "application/json");
        });
        server.server().Get(R"(/api/models/([^/]+)/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
            const std::string repo = req.matches[1].str() + "/" + req.matches[2].str();
            json doc = {{"id", repo}, {"sha", "c0ffee"}, {"gated", repo.rfind("gated/", 0) == 0 ? json("manual") : json(false)}, {"siblings", json::array()}};
            for (const auto& [path, size] : files) {
                if (path.rfind(repo + "/", 0) == 0) doc["siblings"].push_back({{"rfilename", path.substr(repo.size() + 1)}, {"size", size}, {"lfs", {{"sha256", hashOf(size)}}}});
            }
            if (doc["siblings"].empty()) { res.status = 404; return; }
            res.set_content(doc.dump(), "application/json");
        });
        server.server().Get("/api/whoami-v2", [](const httplib::Request& req, httplib::Response& res) {
            if (req.get_header_value("Authorization") != "Bearer good") { res.status = 401; return; }
            res.set_content(R"({"name":"alice"})", "application/json");
        });
        server.server().Get(R"(/([^/]+)/([^/]+)/resolve/([^/]+)/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
            const std::string key = req.matches[1].str() + "/" + req.matches[2].str() + "/" + req.matches[4].str();
            const auto it = files.find(key);
            if (it == files.end()) { res.status = 404; return; }
            if (key.rfind("gated/", 0) == 0 && req.get_header_value("Authorization") != "Bearer good") { res.status = 401; return; }
            const uint64_t size = it->second;
            res.set_content_provider(size, "application/octet-stream", [size](size_t offset, size_t length, httplib::DataSink& sink) {
                const std::string data = gh::test::blobBytes(offset, std::min<uint64_t>(offset + std::min<size_t>(length, 16384), size));
                return sink.write(data.data(), data.size());
            });
        });
        _putenv_s("GRAPHICSHOST_HUB_URL", server.url("").c_str());
    }
    ~Hub() { _putenv_s("GRAPHICSHOST_HUB_URL", ""); }
};

void writeUserCatalog(const fs::path& home) {
    auto entry = [](const std::string& key, const std::string& repo, const std::string& file, uint64_t size, json extra = json::object()) {
        json e = {{"key", key}, {"name", "Model " + key}, {"kind", "checkpoint"}, {"family", "SDXL"}, {"repo", repo}, {"file", file},
                  {"revision", "rev1"}, {"sizeBytes", size}, {"sha256", hashOf(size)}, {"license", "test"}};
        for (const auto& [k, v] : extra.items()) e[k] = v;
        return e;
    };
    json doc = {{"entries", json::array({entry("xl", "pub/xl-model", "xl.safetensors", kSize,
                                                {{"companions", json::array({{{"role", "vae"}, {"key", "xlvae"}, {"auto", true}}})},
                                                 {"defaults", {{"request", {{"steps", 6}}}}}}),
                                         entry("xlvae", "vaes/xl-vae", "vae.safetensors", kVaeSize, {{"kind", "vae"}}),
                                         entry("closed", "gated/closed", "c.safetensors", kSize, {{"gated", true}})})}};
    gh::test::writeText(home / "catalog.json", doc.dump());
}

}  // namespace

TEST_CASE("cli catalog: lists the shipped and the user entries with fit hints", "[unit][cli][models]") {
    gh::test::TempDir home;
    writeUserCatalog(home.path());
    const auto r = runCli(home.path(), {"--json", "catalog"});
    REQUIRE(r.code == gh::cli::kOk);
    const auto entries = r.asJson()["entries"];
    CHECK(entries.size() >= 9);   // the shipped six and the three of this test
    bool sawSdxl = false, sawUser = false;
    for (const auto& e : entries) {
        CHECK_FALSE(e["fit"]["label"].get<std::string>().empty());
        CHECK(e["installed"] == false);
        sawSdxl |= e["key"] == "sdxl";
        sawUser |= e["key"] == "xl";
    }
    CHECK(sawSdxl);
    CHECK(sawUser);

    CHECK(runCli(home.path(), {"--json", "catalog", "--kind", "vae"}).asJson()["entries"].size() >= 3);
    const auto sdxlOnly = runCli(home.path(), {"--json", "catalog", "--family", "SDXL", "turbo"}).asJson()["entries"];
    REQUIRE(sdxlOnly.size() == 1);
    CHECK(sdxlOnly[0]["key"] == "sdxl-turbo");
    CHECK(runCli(home.path(), {"catalog", "zzzznothing"}).out.find("nothing matches") != std::string::npos);
    const auto text = runCli(home.path(), {"catalog"});
    CHECK(text.out.find("KEY") != std::string::npos);
    CHECK(text.out.find("get one with") != std::string::npos);
}

TEST_CASE("cli search and info talk to the hub", "[unit][cli][models]") {
    Hub hub;
    gh::test::TempDir home;
    const auto s = runCli(home.path(), {"--json", "search", "xl", "--limit", "5"});
    REQUIRE(s.code == gh::cli::kOk);
    CHECK(s.asJson()["models"].size() == 2);
    CHECK(s.asJson()["models"][0]["id"] == "pub/xl-model");
    CHECK(s.asJson()["models"][1]["gated"] == "manual");
    CHECK(runCli(home.path(), {"search", "nothing"}).out.find("nothing found") != std::string::npos);

    const auto i = runCli(home.path(), {"--json", "info", "two/files"});
    REQUIRE(i.code == gh::cli::kOk);
    CHECK(i.asJson()["files"].size() == 2);
    CHECK(i.asJson()["files"][0]["sha256"] == hashOf(kBig));
    CHECK(runCli(home.path(), {"info", "nobody/nothing"}).code == gh::cli::kNotFound);
}

TEST_CASE("cli: an unreachable hub is exit code 5", "[unit][cli][models]") {
    gh::test::TempDir home;
    _putenv_s("GRAPHICSHOST_HUB_URL", "http://127.0.0.1:1");
    CHECK(runCli(home.path(), {"search", "x"}).code == gh::cli::kServerUnreachable);
    CHECK(runCli(home.path(), {"info", "a/b"}).code == gh::cli::kServerUnreachable);
    _putenv_s("GRAPHICSHOST_HUB_URL", "");
}

TEST_CASE("cli get: a catalog model arrives with its VAE, defaults and source record", "[unit][cli][models]") {
    Hub hub;
    gh::test::TempDir home;
    writeUserCatalog(home.path());

    const auto dry = runCli(home.path(), {"--json", "get", "xl", "--dry-run"});
    REQUIRE(dry.code == gh::cli::kOk);
    CHECK(dry.asJson()["items"].size() == 2);
    CHECK(dry.asJson()["bytesToDownload"] == kSize + kVaeSize);
    CHECK_FALSE(fs::exists(home.path() / "models" / "pub"));   // a dry run downloads nothing

    const auto r = runCli(home.path(), {"--json", "get", "xl"});
    INFO(r.err);
    REQUIRE(r.code == gh::cli::kOk);
    const fs::path model = home.path() / "models" / "pub" / "xl-model" / "xl.safetensors";
    CHECK(gh::test::readText(model) == gh::test::blobBytes(0, kSize));
    CHECK(gh::test::readText(home.path() / "models" / "vaes" / "xl-vae" / "vae.safetensors") == gh::test::blobBytes(0, kVaeSize));

    const auto record = json::parse(gh::test::readText(fs::path(model.string() + ".source.json")));
    CHECK(record["sha256"] == hashOf(kSize));
    CHECK(record["companions"][0]["file"] == "../../vaes/xl-vae/vae.safetensors");
    CHECK(runCli(home.path(), {"defaults", "show", "xl.safetensors"}).asJson()["request"]["steps"] == 6);   // catalog defaults

    // the second time there is nothing to download
    const auto again = runCli(home.path(), {"--json", "get", "xl"});
    REQUIRE(again.code == gh::cli::kOk);
    CHECK(again.asJson()["plan"]["bytesToDownload"] == 0);
    bool installedXl = false;
    const json catalogNow = runCli(home.path(), {"--json", "catalog"}).asJson();   // (a named value: a range-for over a temporary would dangle)
    for (const auto& e : catalogNow["entries"]) if (e["key"] == "xl") installedXl = e["installed"].get<bool>();
    CHECK(installedXl);

    // without companions only the checkpoint is wanted
    CHECK(runCli(home.path(), {"--json", "get", "xl", "--no-companions", "--dry-run"}).asJson()["items"].size() == 1);
}

TEST_CASE("cli get: problems have their own exit codes", "[unit][cli][models]") {
    Hub hub;
    gh::test::TempDir home;
    writeUserCatalog(home.path());
    CHECK(runCli(home.path(), {"get", "no-such-key"}).code == gh::cli::kNotFound);
    CHECK(runCli(home.path(), {"get", "not a key or repo"}).code == gh::cli::kNotFound);

    const auto gated = runCli(home.path(), {"get", "closed"});
    CHECK(gated.code == gh::cli::kNotAuthorised);
    CHECK(gated.err.find("login") != std::string::npos);

    CHECK(runCli(home.path(), {"get", "nobody/nothing", "x.safetensors"}).code == gh::cli::kNotFound);
    CHECK(runCli(home.path(), {"get", "one/file", "missing.safetensors"}).code == gh::cli::kNotFound);
}

TEST_CASE("cli get: a plain repository, with the file named or implied", "[unit][cli][models]") {
    Hub hub;
    gh::test::TempDir home;

    const auto ambiguous = runCli(home.path(), {"get", "two/files"});   // two usable files: ask which
    CHECK(ambiguous.code == gh::cli::kUsage);
    CHECK(ambiguous.err.find("a.safetensors") != std::string::npos);
    CHECK(ambiguous.err.find("b.safetensors") != std::string::npos);

    const auto named = runCli(home.path(), {"get", "two/files", "b.safetensors"});
    INFO(named.err);
    REQUIRE(named.code == gh::cli::kOk);
    CHECK(fs::exists(home.path() / "models" / "two" / "files" / "b.safetensors"));

    const auto implied = runCli(home.path(), {"get", "one/file"});     // only one usable file: no need to name it
    INFO(implied.err);
    REQUIRE(implied.code == gh::cli::kOk);
    CHECK(fs::exists(home.path() / "models" / "one" / "file" / "only.safetensors"));
}

TEST_CASE("cli downloads: leftovers of an interrupted download are listed and cleared", "[unit][cli][models]") {
    gh::test::TempDir home;
    CHECK(runCli(home.path(), {"downloads"}).out.find("no interrupted downloads") != std::string::npos);
    gh::test::writeText(home.path() / "models" / "pub" / "m.safetensors.part", std::string(2000, 'x'));
    const auto list = runCli(home.path(), {"--json", "downloads"});
    REQUIRE(list.asJson()["partial"].size() == 1);
    CHECK(list.asJson()["partial"][0]["bytes"] == 2000);
    CHECK(runCli(home.path(), {"downloads", "clear"}).code == gh::cli::kOk);
    CHECK(runCli(home.path(), {"--json", "downloads"}).asJson()["partial"].empty());
    CHECK(runCli(home.path(), {"downloads", "frobnicate"}).code == gh::cli::kUsage);
}

TEST_CASE("cli updates: reports changed, current and unreachable models", "[unit][cli][models]") {
    Hub hub;
    gh::test::TempDir home;
    writeUserCatalog(home.path());
    REQUIRE(runCli(home.path(), {"get", "xl", "--no-companions"}).code == gh::cli::kOk);
    auto status = [&] { return runCli(home.path(), {"--json", "updates"}).asJson()["updates"]; };
    REQUIRE(status().size() == 1);
    CHECK(status()[0]["status"] == "up-to-date");

    hub.files["pub/xl-model/xl.safetensors"] = kSize;     // same name, but the hub now serves other content:
    // (the mock computes the hash from the size, so change the recorded hash in the record instead)
    const fs::path record = home.path() / "models" / "pub" / "xl-model" / "xl.safetensors.source.json";
    auto doc = json::parse(gh::test::readText(record));
    doc["sha256"] = std::string(64, '0');
    gh::test::writeText(record, doc.dump());
    CHECK(status()[0]["status"] == "update-available");

    _putenv_s("GRAPHICSHOST_HUB_URL", "http://127.0.0.1:1");
    CHECK(status()[0]["status"] == "unknown");
}

TEST_CASE("cli login: bad tokens are refused and never stored; whoami uses the environment token", "[unit][cli][models]") {
    Hub hub;
    gh::test::TempDir home;
    CHECK(runCli(home.path(), {"login", "bad-token"}).code == gh::cli::kNotAuthorised);
    CHECK_FALSE(gh::core::readSecret(gh::core::kHuggingFaceSecret).value_or("") == "bad-token");

    _putenv_s("HF_TOKEN", "good");
    const auto me = runCli(home.path(), {"--json", "whoami"});
    CHECK(me.code == gh::cli::kOk);
    CHECK(me.asJson()["user"] == "alice");
    _putenv_s("HF_TOKEN", "bad");
    CHECK(runCli(home.path(), {"whoami"}).code == gh::cli::kNotAuthorised);
    _putenv_s("HF_TOKEN", "");

    // a gated model downloads with the token from the environment
    writeUserCatalog(home.path());
    _putenv_s("HF_TOKEN", "good");
    const auto gated = runCli(home.path(), {"get", "closed"});
    INFO(gated.err);
    CHECK(gated.code == gh::cli::kOk);
    CHECK(fs::exists(home.path() / "models" / "gated" / "closed" / "c.safetensors"));
    _putenv_s("HF_TOKEN", "");
}
