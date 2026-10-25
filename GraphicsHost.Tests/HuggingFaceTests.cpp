#include "TestHttpServer.h"

#include "core/SecretStore.h"
#include <nlohmann/json.hpp>
#include "models/HuggingFace.h"

#include <catch2/catch_test_macros.hpp>

#include <random>

using namespace gh::models;
using nlohmann::json;

namespace {

// A tiny stand-in for the Hugging Face API.
struct MockHub {
    gh::test::TestHttpServer server;
    std::mutex m;
    std::string lastQuery, lastAuth;

    MockHub() {
        server.server().Get("/api/models", [this](const httplib::Request& req, httplib::Response& res) {
            {
                std::lock_guard<std::mutex> lock(m);
                lastQuery = req.get_param_value("search") + "|" + req.get_param_value("filter") + "|" + req.get_param_value("limit") + "|" + req.get_param_value("sort");
                lastAuth = req.get_header_value("Authorization");
            }
            if (req.get_param_value("search") == "limited") {
                res.status = 429;
                return;
            }
            res.set_content(json::array({
                json{{"id", "stabilityai/sdxl-turbo"}, {"downloads", 12345}, {"likes", 99}, {"gated", false},
                     {"pipeline_tag", "text-to-image"}, {"library_name", "diffusers"},
                     {"tags", {"diffusers", "license:other", "text-to-image"}}, {"lastModified", "2025-01-01T00:00:00.000Z"}},
                json{{"id", "somebody/gated-model"}, {"downloads", 10}, {"likes", 1}, {"gated", "manual"}, {"tags", {"license:openrail++"}}},
                json{{"modelId", "old/style-id"}, {"downloads", 5}},
                json{{"noid", true}},
                json("junk")}).dump(), "application/json");
        });
        server.server().Get(R"(/api/models/([^/]+)/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
            {
                std::lock_guard<std::mutex> lock(m);
                lastAuth = req.get_header_value("Authorization");
            }
            const std::string repo = req.matches[1].str() + "/" + req.matches[2].str();
            if (repo == "private/secret") { res.status = 401; return; }
            if (repo == "meta/closed") { res.status = 403; return; }
            if (repo == "broken/json") { res.set_content("<html>oops</html>", "text/html"); return; }
            if (repo != "stabilityai/sdxl-turbo") { res.status = 404; return; }
            auto file = [](const std::string& path, uint64_t size, const std::string& sha = {}) {
                json f = {{"rfilename", path}, {"size", size}};
                if (!sha.empty()) f["lfs"] = {{"sha256", sha}, {"size", size}};
                return f;
            };
            json doc = {{"id", repo}, {"sha", "71153311d3dbb46851df1931d3ca6e939de83304"}, {"gated", false},
                        {"cardData", {{"license", "other"}}}, {"tags", json::array({"text-to-image"})}};
            doc["siblings"] = json::array({file("sd_xl_turbo_1.0_fp16.safetensors", 6938081905ull, std::string(64, 'e')),
                                           file("README.md", 1200),
                                           file("unet/diffusion_pytorch_model.fp16.safetensors", 5135149760ull, std::string(64, 'f')),
                                           file("vae/diffusion_pytorch_model.safetensors", 334643276),
                                           file("sdxl_vae.safetensors", 334641162, std::string(64, 'a')),
                                           file("tiny.safetensors", 1000)});
            res.set_content(doc.dump(), "application/json");
        });
        server.server().Get("/api/whoami-v2", [this](const httplib::Request& req, httplib::Response& res) {
            if (req.get_header_value("Authorization") != "Bearer good") { res.status = 401; return; }
            res.set_content(R"({"name":"alice","type":"user"})", "application/json");
        });
        server.server().Get(R"(/open/model/resolve/main/file.safetensors)", [](const httplib::Request&, httplib::Response& res) { res.status = 200; });
        server.server().Get(R"(/closed/model/resolve/main/file.safetensors)", [](const httplib::Request& req, httplib::Response& res) {
            res.status = req.get_header_value("Authorization") == "Bearer good" ? 200 : 401;
        });
    }
    HuggingFace hub(const std::string& token = {}) { return HuggingFace(token, server.url("")); }
};

}  // namespace

TEST_CASE("hf: search sends the right query and parses the answer", "[unit][hf]") {
    MockHub mock;
    const auto models = mock.hub("sometoken").search("sdxl turbo & more", 5);
    {
        std::lock_guard<std::mutex> lock(mock.m);
        CHECK(mock.lastQuery == "sdxl turbo & more|text-to-image|5|downloads");   // encoded on the wire, decoded by the server
        CHECK(mock.lastAuth == "Bearer sometoken");
    }
    REQUIRE(models.size() == 3);   // the entries without an id and the junk are skipped
    CHECK(models[0].id == "stabilityai/sdxl-turbo");
    CHECK(models[0].downloads == 12345);
    CHECK(models[0].likes == 99);
    CHECK(models[0].license == "other");
    CHECK(models[0].library == "diffusers");
    CHECK_FALSE(models[0].isGated());
    CHECK(models[1].isGated());
    CHECK(models[1].gated == "manual");
    CHECK(models[1].license == "openrail++");
    CHECK(models[2].id == "old/style-id");   // the older "modelId" key
}

TEST_CASE("hf: model info lists files with sizes and hashes", "[unit][hf]") {
    MockHub mock;
    const auto info = mock.hub().info("stabilityai/sdxl-turbo");
    CHECK(info.model.commit == "71153311d3dbb46851df1931d3ca6e939de83304");
    CHECK(info.model.license == "other");   // from cardData when there is no license tag
    REQUIRE(info.files.size() == 6);
    CHECK(info.files[0].path == "sd_xl_turbo_1.0_fp16.safetensors");
    CHECK(info.files[0].sizeBytes == 6938081905ull);
    CHECK(info.files[0].sha256 == std::string(64, 'e'));
    CHECK(info.files[1].sha256.empty());   // README: no LFS hash

    const auto usable = usableFiles(info);
    REQUIRE(usable.size() == 2);           // not the README, the diffusers component folders or the 1 KB file
    CHECK(usable[0].path == "sd_xl_turbo_1.0_fp16.safetensors");
    CHECK(usable[1].path == "sdxl_vae.safetensors");
    CHECK(guessFileKind(usable[1].path) == "vae");
    CHECK(guessFileKind(usable[0].path) == "checkpoint");
    CHECK(guessFileKind("MyStyle_LoRA_v2.safetensors") == "lora");
    CHECK(guessFileKind("README.md") == "other");
}

TEST_CASE("hf: HTTP problems become typed errors with advice", "[unit][hf]") {
    MockHub mock;
    auto code = [&](const std::function<void()>& f) {
        try { f(); } catch (const HfError& e) { return std::make_pair(e.code(), std::string(e.what())); }
        return std::make_pair(HfErrorCode::Other, std::string("no error"));
    };
    auto hub = mock.hub();

    auto r = code([&] { hub.info("nobody/nothing"); });
    CHECK(r.first == HfErrorCode::NotFound);
    r = code([&] { hub.info("private/secret"); });
    CHECK(r.first == HfErrorCode::Unauthorized);
    CHECK(r.second.find("log in") != std::string::npos);
    r = code([&] { hub.info("meta/closed"); });
    CHECK(r.first == HfErrorCode::Gated);
    CHECK(r.second.find("accept its terms") != std::string::npos);
    r = code([&] { hub.search("limited"); });
    CHECK(r.first == HfErrorCode::RateLimited);
    r = code([&] { hub.info("broken/json"); });
    CHECK(r.first == HfErrorCode::BadAnswer);
    r = code([&] { hub.info("not-a-repo"); });
    CHECK(r.first == HfErrorCode::NotFound);
    r = code([&] { hub.info("../../etc/passwd"); });
    CHECK(r.first == HfErrorCode::NotFound);

    HuggingFace dead("", "http://127.0.0.1:1");
    r = code([&] { dead.search("x"); });
    CHECK(r.first == HfErrorCode::Network);
}

TEST_CASE("hf: the gated-model flow: no token, wrong token, good token", "[unit][hf]") {
    MockHub mock;
    CHECK_NOTHROW(mock.hub().checkAccess("open/model", "file.safetensors"));
    CHECK_THROWS_AS(mock.hub().checkAccess("closed/model", "file.safetensors"), HfError);
    CHECK_THROWS_AS(mock.hub("bad").checkAccess("closed/model", "file.safetensors"), HfError);
    CHECK_NOTHROW(mock.hub("good").checkAccess("closed/model", "file.safetensors"));

    CHECK(mock.hub("good").whoami() == "alice");
    CHECK_THROWS_AS(mock.hub("bad").whoami(), HfError);
    CHECK_THROWS_AS(mock.hub().whoami(), HfError);
}

TEST_CASE("hf: file URLs", "[unit][hf]") {
    HuggingFace hub("", "https://huggingface.co/");
    CHECK(hub.fileUrl("a/b", "x/y.safetensors") == "https://huggingface.co/a/b/resolve/main/x/y.safetensors");
    CHECK(hub.fileUrl("a/b", "y.safetensors", "abc") == "https://huggingface.co/a/b/resolve/abc/y.safetensors");
}

TEST_CASE("secrets: stored in the Windows Credential Manager and removed again", "[unit][secrets]") {
    std::random_device rd;
    const std::string name = "test-secret-" + std::to_string(rd());
    CHECK_FALSE(gh::core::readSecret(name).has_value());
    REQUIRE(gh::core::storeSecret(name, "hf_abc123 with spaces \xC3\xA4"));
    CHECK(gh::core::readSecret(name) == "hf_abc123 with spaces \xC3\xA4");
    REQUIRE(gh::core::storeSecret(name, "second"));   // overwrite
    CHECK(gh::core::readSecret(name) == "second");
    CHECK(gh::core::deleteSecret(name));
    CHECK_FALSE(gh::core::readSecret(name).has_value());
    CHECK_FALSE(gh::core::deleteSecret(name));
}

TEST_CASE("secrets: the Hugging Face token comes from the environment first", "[unit][secrets]") {
    const auto before = gh::core::huggingFaceToken();   // whatever this machine has
    _putenv_s("HF_TOKEN", "from-env");
    CHECK(gh::core::huggingFaceToken() == "from-env");
    _putenv_s("HF_TOKEN", "");   // removes the variable
    _putenv_s("HUGGING_FACE_HUB_TOKEN", "from-hub-env");
    CHECK(gh::core::huggingFaceToken() == "from-hub-env");
    _putenv_s("HUGGING_FACE_HUB_TOKEN", "");
    CHECK(gh::core::huggingFaceToken() == before);
}

TEST_CASE("hf: the real Hugging Face API still answers the way the parser expects", "[network][hf]") {
    char* enabled = nullptr;
    size_t size = 0;
    if (_dupenv_s(&enabled, &size, "GRAPHICSHOST_NETWORK_TESTS") != 0 || !enabled) SKIP("set GRAPHICSHOST_NETWORK_TESTS=1 to run");
    std::free(enabled);
    HuggingFace hub;
    const auto models = hub.search("stable diffusion xl", 5);
    REQUIRE_FALSE(models.empty());
    for (const auto& m : models) CHECK(m.id.find('/') != std::string::npos);

    const auto info = hub.info("stabilityai/stable-diffusion-xl-base-1.0");
    REQUIRE_FALSE(info.files.empty());
    const auto usable = usableFiles(info);
    const auto base = std::find_if(usable.begin(), usable.end(), [](const HfFile& f) { return f.path == "sd_xl_base_1.0.safetensors"; });
    REQUIRE(base != usable.end());
    CHECK(base->sizeBytes == 6938078334ull);
    CHECK(base->sha256 == "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b");  // the value in catalog.json
    CHECK_NOTHROW(hub.checkAccess("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors"));
}
