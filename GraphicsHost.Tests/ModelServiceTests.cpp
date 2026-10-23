#include "SafetensorsFixture.h"
#include "TestHttpServer.h"
#include "TestUtil.h"

#include "core/Sha256.h"
#include "models/ModelService.h"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

using namespace gh::models;
using nlohmann::json;
namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {

constexpr uint64_t kCkptSize = 150000, kVaeSize = 90000;

std::string hashOf(uint64_t size) { return gh::core::sha256Hex(gh::test::blobBytes(0, size)); }

json entry(const std::string& key, const std::string& repo, const std::string& file, uint64_t size, json extra = json::object()) {
    json e = {{"key", key}, {"name", "Model " + key}, {"kind", "checkpoint"}, {"family", "SDXL"}, {"repo", repo}, {"file", file},
              {"revision", "rev1"}, {"sizeBytes", size}, {"sha256", hashOf(size)}, {"license", "test-license"}};
    for (const auto& [k, v] : extra.items()) e[k] = v;
    return e;
}

// A mock hub that serves the catalog's files, a model listing and a gated file.
struct Hub {
    gh::test::TestHttpServer server;
    Hub() {
        auto serve = [this](const httplib::Request& req, httplib::Response& res) {
            const std::string file = req.matches[4];
            const std::string repo = req.matches[1].str() + "/" + req.matches[2].str();
            const uint64_t size = sizes.count(repo + "/" + file) ? sizes[repo + "/" + file] : 0;
            if (size == 0) { res.status = 404; return; }
            if (repo.rfind("gated/", 0) == 0 && req.get_header_value("Authorization") != "Bearer good") { res.status = 401; return; }
            res.set_content_provider(size, "application/octet-stream", [size](size_t offset, size_t length, httplib::DataSink& sink) {
                const std::string data = gh::test::blobBytes(offset, std::min<uint64_t>(offset + std::min<size_t>(length, 16384), size));
                return sink.write(data.data(), data.size());
            });
        };
        server.server().Get(R"(/([^/]+)/([^/]+)/resolve/([^/]+)/(.+))", serve);
        server.server().Get(R"(/api/models/([^/]+)/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
            const std::string repo = req.matches[1].str() + "/" + req.matches[2].str();
            json doc = {{"id", repo}, {"sha", "commit42"}, {"gated", repo.rfind("gated/", 0) == 0 ? json("manual") : json(false)}, {"tags", json::array({"license:openrail"})}};
            doc["siblings"] = json::array();
            for (const auto& [path, size] : sizes) {
                if (path.rfind(repo + "/", 0) == 0) {
                    doc["siblings"].push_back({{"rfilename", path.substr(repo.size() + 1)}, {"size", size}, {"lfs", {{"sha256", hashOf(size)}, {"size", size}}}});
                }
            }
            if (doc["siblings"].empty()) { res.status = 404; return; }
            res.set_content(doc.dump(), "application/json");
        });
    }
    std::map<std::string, uint64_t> sizes;
    std::string url() const { return server.url(""); }
};

struct Fixture {
    gh::test::TempDir dir;
    Hub hub;
    ModelLibrary library{{dir.path() / "models"}};
    DownloadManager downloads{2};
    gh::presets::ModelDefaultsStore defaults{dir.path() / "model-defaults.json"};
    std::string token;
    std::unique_ptr<ModelService> service;

    Fixture() {
        hub.sizes = {{"pub/xl-model/xl.safetensors", kCkptSize}, {"pub/xl-model-two/xl2.safetensors", kCkptSize + 2},
                     {"vaes/xl-vae/vae.safetensors", kVaeSize}, {"gated/closed-model/c.safetensors", kCkptSize},
                     {"other/plain/plain.safetensors", kCkptSize}};
        json doc = {{"entries", json::array({
            entry("xl", "pub/xl-model", "xl.safetensors", kCkptSize,
                  {{"companions", json::array({{{"role", "vae"}, {"key", "xlvae"}, {"auto", true}, {"note", "stable in fp16"}}})},
                   {"defaults", {{"request", {{"steps", 4}, {"cfgScale", 1.0}}}, {"load", {{"flashAttention", true}}}}}}),
            entry("xl2", "pub/xl-model-two", "xl2.safetensors", kCkptSize + 2,
                  {{"companions", json::array({{{"role", "vae"}, {"key", "xlvae"}, {"auto", true}}})}}),
            entry("xlvae", "vaes/xl-vae", "vae.safetensors", kVaeSize, {{"kind", "vae"}}),
            entry("closed", "gated/closed-model", "c.safetensors", kCkptSize, {{"gated", true}}),
            entry("plain", "other/plain", "plain.safetensors", kCkptSize)})}};
        // the catalog has to carry the second file's own size for xl2
        service = std::make_unique<ModelService>(Catalog::fromJson(doc), library, downloads, defaults,
                                                 ModelServiceConfig{dir.path() / "models", hub.url(), [this] { return token; }});
    }
    fs::path models() const { return dir.path() / "models"; }
    void waitAll(const std::vector<std::shared_ptr<Download>>& all) {
        for (const auto& d : all) { REQUIRE(d->waitFor(30s)); INFO(d->message()); CHECK(d->state() == DownloadState::Completed); }
    }
};

}  // namespace

TEST_CASE("service: a catalog plan contains the model and its automatic companion", "[unit][service]") {
    Fixture f;
    const auto plan = f.service->planCatalog("xl");
    REQUIRE(plan.items.size() == 2);
    CHECK(plan.items[0].role == "checkpoint");
    CHECK(plan.items[0].destination == f.models() / "pub" / "xl-model" / "xl.safetensors");
    CHECK(plan.items[0].url == f.hub.url() + "/pub/xl-model/resolve/rev1/xl.safetensors");
    CHECK(plan.items[1].role == "vae");
    CHECK(plan.items[1].destination == f.models() / "vaes" / "xl-vae" / "vae.safetensors");
    CHECK(plan.bytesToDownload == kCkptSize + kVaeSize);
    CHECK_FALSE(plan.items[0].present);
    // the checkpoint's record points at the VAE relative to itself
    CHECK(plan.items[0].source["companions"][0]["file"] == "../../vaes/xl-vae/vae.safetensors");
    CHECK(plan.items[0].source["companions"][0]["role"] == "vae");
    bool mentionsVae = false, mentionsLicense = false;
    for (const auto& n : plan.notes) { mentionsVae |= n.find("includes Model xlvae") != std::string::npos && n.find("stable in fp16") != std::string::npos; mentionsLicense |= n == "license: test-license"; }
    CHECK(mentionsVae);
    CHECK(mentionsLicense);

    const auto bare = f.service->planCatalog("xl", false);
    CHECK(bare.items.size() == 1);
    CHECK(bare.bytesToDownload == kCkptSize);
    CHECK(toJson(plan)["items"].size() == 2);
}

TEST_CASE("service: unknown keys and gated models are refused with advice", "[unit][service]") {
    Fixture f;
    try { f.service->planCatalog("nope"); FAIL("expected an error"); }
    catch (const ModelServiceError& e) { CHECK(e.code() == ModelServiceError::Code::UnknownEntry); CHECK(std::string(e.what()).find("catalog") != std::string::npos); }

    try { f.service->planCatalog("closed"); FAIL("expected an error"); }
    catch (const ModelServiceError& e) { CHECK(e.code() == ModelServiceError::Code::NeedsToken); CHECK(std::string(e.what()).find("login") != std::string::npos); }

    f.token = "wrong";
    try { f.service->planCatalog("closed"); FAIL("expected an error"); }
    catch (const ModelServiceError& e) { CHECK(e.code() == ModelServiceError::Code::Gated); }

    f.token = "good";
    CHECK_NOTHROW(f.service->planCatalog("closed"));
}

TEST_CASE("service: start downloads everything, links the VAE and applies the catalog defaults", "[unit][service]") {
    Fixture f;
    std::atomic<int> completed{0};
    f.service->setObserver([&](const Download& d) { if (d.state() == DownloadState::Completed) ++completed; });
    const auto started = f.service->start(f.service->planCatalog("xl"));
    REQUIRE(started.size() == 2);
    f.waitAll(started);
    for (int i = 0; i < 100 && completed < 2; ++i) std::this_thread::sleep_for(10ms);
    CHECK(completed == 2);   // the user's observer still hears about every download

    CHECK(gh::test::readText(f.models() / "pub" / "xl-model" / "xl.safetensors") == gh::test::blobBytes(0, kCkptSize));
    CHECK(gh::test::readText(f.models() / "vaes" / "xl-vae" / "vae.safetensors") == gh::test::blobBytes(0, kVaeSize));

    // the catalog defaults became the model's defaults
    const auto d = f.defaults.get("xl.safetensors");
    REQUIRE(d.has_value());
    CHECK(d->request["steps"] == 4);
    CHECK(d->request["cfgScale"] == 1.0);
    CHECK(d->load["flashAttention"] == true);

    // the library knows where it came from and which VAE belongs to it
    const auto cards = f.library.scan();
    const ModelCard* xl = nullptr;
    for (const auto& c : cards) if (c.file == "xl.safetensors") xl = &c;
    REQUIRE(xl != nullptr);
    CHECK(xl->source["key"] == "xl");
    CHECK(xl->source["revision"] == "rev1");
    CHECK(xl->source["sha256"] == hashOf(kCkptSize));
    CHECK(xl->vae == "../../vaes/xl-vae/vae.safetensors");
}

TEST_CASE("service: what is already on disk is not downloaded again, a shared VAE is fetched once", "[unit][service]") {
    Fixture f;
    f.waitAll(f.service->start(f.service->planCatalog("xl")));

    const auto second = f.service->planCatalog("xl2");
    REQUIRE(second.items.size() == 2);
    CHECK_FALSE(second.items[0].present);
    CHECK(second.items[1].present);                       // the VAE of the first model
    CHECK(second.bytesToDownload == kCkptSize + 2);
    const auto started = f.service->start(second);
    CHECK(started.size() == 1);                           // only the new checkpoint
    f.waitAll(started);
    // both checkpoints point at the same VAE file
    int linked = 0;
    for (const auto& c : f.library.scan()) if (c.kind == Kind::Checkpoint && c.vae.find("vae.safetensors") != std::string::npos) ++linked;
    CHECK(linked == 2);

    const auto again = f.service->planCatalog("xl");
    CHECK(again.bytesToDownload == 0);
    CHECK(f.service->start(again).empty());

    // a lost record is written again for a model that is already there
    const auto record = sourceSidecarFor(f.models() / "pub" / "xl-model" / "xl.safetensors");
    fs::remove(record);
    f.service->start(again);
    CHECK(fs::exists(record));
    CHECK(json::parse(gh::test::readText(record))["companions"][0]["role"] == "vae");
}

TEST_CASE("service: a gated model downloads once the token has access", "[unit][service]") {
    Fixture f;
    f.token = "good";
    f.waitAll(f.service->start(f.service->planCatalog("closed")));
    CHECK(fs::exists(f.models() / "gated" / "closed-model" / "c.safetensors"));
    CHECK(f.hub.server.authorization() == "Bearer good");
}

TEST_CASE("service: a plain repository file, with size and hash from the repository listing", "[unit][service]") {
    Fixture f;
    const auto plan = f.service->planRepoFile("other/plain", "plain.safetensors");
    REQUIRE(plan.items.size() == 1);
    CHECK(plan.items[0].sizeBytes == kCkptSize);
    CHECK(plan.items[0].sha256 == hashOf(kCkptSize));
    CHECK(plan.items[0].revision == "commit42");
    CHECK(plan.items[0].destination == f.models() / "other" / "plain" / "plain.safetensors");
    CHECK(plan.items[0].url == f.hub.url() + "/other/plain/resolve/commit42/plain.safetensors");
    bool license = false;
    for (const auto& n : plan.notes) license |= n == "license: openrail";
    CHECK(license);
    f.waitAll(f.service->start(plan));
    CHECK(gh::test::readText(plan.items[0].destination) == gh::test::blobBytes(0, kCkptSize));

    try { f.service->planRepoFile("other/plain", "missing.safetensors"); FAIL("expected an error"); }
    catch (const ModelServiceError& e) { CHECK(e.code() == ModelServiceError::Code::NotFound); }
    try { f.service->planRepoFile("nobody/nothing", "x.safetensors"); FAIL("expected an error"); }
    catch (const ModelServiceError& e) { CHECK(e.code() == ModelServiceError::Code::NotFound); }
    const auto gated = f.service->planRepoFile("gated/closed-model", "c.safetensors");
    bool note = false;
    for (const auto& n : gated.notes) note |= n.find("gated") != std::string::npos;
    CHECK(note);
}

TEST_CASE("service: the record keeps the repository path of a file in a sub-folder (needed by the update check)", "[unit][service]") {
    Fixture f;
    f.hub.sizes["other/plain/sub/dir/nested.safetensors"] = kCkptSize;
    const auto plan = f.service->planRepoFile("other/plain", "sub/dir/nested.safetensors");
    f.waitAll(f.service->start(plan));
    CHECK(plan.items[0].destination.filename() == "nested.safetensors");   // flat in the library
    const auto record = json::parse(gh::test::readText(sourceSidecarFor(plan.items[0].destination)));
    CHECK(record["file"] == "sub/dir/nested.safetensors");                 // the name to look for in the repository
    CHECK(record["localFile"] == "nested.safetensors");
    CHECK(record["repo"] == "other/plain");
}
