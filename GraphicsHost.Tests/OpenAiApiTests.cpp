#include "ApiTestUtil.h"
#include "SafetensorsFixture.h"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <mutex>
#include <thread>

using namespace gh;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

// An engine that "generates" gradient images without a runtime, and remembers what it was asked.
struct FakeState {
    std::mutex mutex;
    std::vector<engine::GenerationRequest> requests;
    std::vector<engine::LoadOptions> loads;
    std::atomic<int> stepMs{1};
};

class FakeModel final : public engine::ILoadedModel {
public:
    FakeModel(engine::ModelInfo info, std::shared_ptr<FakeState> state) : info_(std::move(info)), state_(std::move(state)) {}
    const engine::ModelInfo& info() const override { return info_; }
    engine::GenerationResult generate(const engine::GenerationRequest& request, const engine::ProgressSink& progress,
                                      std::stop_token stop) override {
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->requests.push_back(request);
        }
        if (request.prompt == "boom") throw engine::EngineError(engine::EngineErrorCode::OutOfMemory, "out of memory while sampling");
        engine::GenerationResult r;
        r.request = engine::resolve(request, info_.defaults);
        r.request.width = request.width > 0 ? request.width : 64;
        r.request.height = request.height > 0 ? request.height : 64;
        for (int step = 1; step <= 3; ++step) {
            if (stop.stop_requested()) throw engine::EngineError(engine::EngineErrorCode::Cancelled, "cancelled");
            std::this_thread::sleep_for(std::chrono::milliseconds(state_->stepMs.load()));
            if (progress) progress(engine::StepProgress{step, 3, 0});
        }
        for (int i = 0; i < std::max(1, request.batchCount); ++i) {
            engine::Image image;
            image.width = static_cast<uint32_t>(r.request.width);
            image.height = static_cast<uint32_t>(r.request.height);
            image.data.resize(static_cast<size_t>(image.width) * image.height * 3);
            for (size_t p = 0; p < image.data.size(); ++p) image.data[p] = static_cast<uint8_t>(p * 7 + i * 40);
            r.images.push_back(std::move(image));
            r.seeds.push_back(request.seed >= 0 ? request.seed + i : 1000 + i);
        }
        r.request.seed = r.seeds.front();
        r.seconds = 0.01;
        return r;
    }

private:
    engine::ModelInfo info_;
    std::shared_ptr<FakeState> state_;
};

class FakeEngine final : public engine::IImageEngine {
public:
    explicit FakeEngine(std::shared_ptr<FakeState> state) : state_(std::move(state)) {}
    std::string runtimeId() const override { return "fake"; }
    std::unique_ptr<engine::ILoadedModel> load(const engine::LoadOptions& options) override {
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->loads.push_back(options);
        }
        engine::ModelInfo info;
        info.modelPath = options.modelPath;
        info.family = "SD 1.x";
        info.runtimeId = "fake";
        info.device = options.device;
        return std::make_unique<FakeModel>(info, state_);
    }

private:
    std::shared_ptr<FakeState> state_;
};

struct GenApp {
    test::TestApp app;
    std::shared_ptr<FakeState> state = std::make_shared<FakeState>();

    explicit GenApp(server::ServerOptions options = test::TestApp::ephemeral()) : app(options, false) {
        test::st::writeSafetensors(app.home.path() / "models" / "tiny.safetensors", test::st::sd1Tensors());
        app.host->overrideEngine(std::make_shared<FakeEngine>(state));
        app.error = app.server->start(options);
        REQUIRE(app.error.empty());
    }
    httplib::Client client() const { return app.client(); }
    httplib::Result post(const std::string& path, const json& body) {
        auto c = client();
        return c.Post(path, body.dump(), "application/json");
    }
};

}  // namespace

TEST_CASE("openai: /v1/models lists the checkpoints that can run", "[integration][api][openai]") {
    GenApp g;
    auto c = g.client();
    const auto r = c.Get("/v1/models");
    REQUIRE(r);
    REQUIRE(r->status == 200);
    const json body = test::bodyJson(r);
    CHECK(body["object"] == "list");
    REQUIRE(body["data"].size() == 1);
    CHECK(body["data"][0]["id"] == "tiny.safetensors");
    CHECK(body["data"][0]["object"] == "model");
    CHECK(body["data"][0]["loaded"] == false);
}

TEST_CASE("openai: generations returns a PNG as base64 and files it in the gallery", "[integration][api][openai]") {
    GenApp g;
    const auto r = g.post("/v1/images/generations", {{"model", "tiny"}, {"prompt", "a red cube"}, {"size", "128x64"}, {"seed", 5}});
    REQUIRE(r);
    INFO(r->body);
    REQUIRE(r->status == 200);
    const json body = test::bodyJson(r);
    REQUIRE(body["data"].size() == 1);
    const std::string b64 = body["data"][0]["b64_json"];
    CHECK(b64.rfind("iVBORw0KGgo", 0) == 0);   // the PNG signature, base64 encoded
    CHECK(body["data"][0]["seed"] == 5);
    CHECK(body["model"] == "tiny.safetensors");
    CHECK(body["created"].get<int64_t>() > 0);

    CHECK(g.app.host->gallery().count() == 1);
    {
        std::lock_guard<std::mutex> lock(g.state->mutex);
        REQUIRE(g.state->requests.size() == 1);
        CHECK(g.state->requests[0].width == 128);
        CHECK(g.state->requests[0].height == 64);
        CHECK(g.state->loads.size() == 1);
    }
    CHECK(g.app.host->generation().counters().succeeded == 1);
    CHECK(g.app.host->generation().counters().images == 1);
}

TEST_CASE("openai: n gives several images, the native names work as well, and store:false skips the gallery", "[integration][api][openai]") {
    GenApp g;
    const auto r = g.post("/v1/images/generations", {{"model", "tiny.safetensors"}, {"prompt", "x"}, {"n", 3}, {"width", 64}, {"height", 64},
                                                     {"cfgScale", 4.5}, {"negative_prompt", "blurry"}, {"store", false}});
    REQUIRE(r);
    INFO(r->body);
    REQUIRE(r->status == 200);
    CHECK(test::bodyJson(r)["data"].size() == 3);
    CHECK(g.app.host->gallery().count() == 0);
    std::lock_guard<std::mutex> lock(g.state->mutex);
    CHECK(g.state->requests[0].batchCount == 3);
    CHECK(g.state->requests[0].cfgScale == 4.5f);
    CHECK(g.state->requests[0].negativePrompt == "blurry");
}

TEST_CASE("openai: response_format url points at /outputs", "[integration][api][openai]") {
    GenApp g;
    const auto r = g.post("/v1/images/generations", {{"model", "tiny"}, {"prompt", "x"}, {"response_format", "url"}});
    REQUIRE(r);
    REQUIRE(r->status == 200);
    const std::string url = test::bodyJson(r)["data"][0]["url"];
    CHECK(url.rfind("http://127.0.0.1:" + std::to_string(g.app.server->port()) + "/outputs/", 0) == 0);
    CHECK(url.find(".png") != std::string::npos);

    const auto bad = g.post("/v1/images/generations", {{"model", "tiny"}, {"prompt", "x"}, {"response_format", "url"}, {"store", false}});
    CHECK(bad->status == 400);
}

TEST_CASE("openai: model defaults sit below the request", "[integration][api][openai]") {
    GenApp g;
    g.app.host->modelDefaults().set("tiny.safetensors", {{{"steps", 7}, {"sampler", "euler_a"}}, json::object()});
    REQUIRE(g.post("/v1/images/generations", {{"model", "tiny"}, {"prompt", "a"}})->status == 200);
    REQUIRE(g.post("/v1/images/generations", {{"model", "tiny"}, {"prompt", "b"}, {"steps", 3}})->status == 200);
    std::lock_guard<std::mutex> lock(g.state->mutex);
    REQUIRE(g.state->requests.size() == 2);
    CHECK(g.state->requests[0].steps == 7);
    CHECK(g.state->requests[0].sampler == "euler_a");
    CHECK(g.state->requests[1].steps == 3);
    CHECK(g.state->requests[1].sampler == "euler_a");
    CHECK(g.state->loads.size() == 1);   // the same model and options: loaded once
}

TEST_CASE("openai: problems come back as OpenAI style errors", "[integration][api][openai]") {
    GenApp g;
    auto expect = [&](const json& body, int status, const std::string& code) {
        const auto r = g.post("/v1/images/generations", body);
        REQUIRE(r);
        INFO(r->body);
        CHECK(r->status == status);
        const json e = test::bodyJson(r);
        REQUIRE(e.contains("error"));
        CHECK(e["error"]["code"] == code);
        CHECK(e["error"].contains("message"));
        CHECK(e["error"].contains("type"));
    };
    expect({{"model", "tiny"}}, 400, "missing_prompt");
    expect({{"model", "tiny"}, {"prompt", "x"}, {"size", "big"}}, 400, "invalid_parameters");
    expect({{"model", "tiny"}, {"prompt", "x"}, {"size", "100x100"}}, 400, "invalid_parameters");   // not a multiple of 8
    expect({{"model", "tiny"}, {"prompt", "x"}, {"steps", "many"}}, 400, "invalid_parameters");
    expect({{"model", "tiny"}, {"prompt", "x"}, {"response_format", "html"}}, 400, "invalid_parameters");
    expect({{"model", "no-such-model"}, {"prompt", "x"}}, 404, "model_not_found");
    expect({{"prompt", "x"}}, 404, "model_not_found");   // nothing loaded and no model named
    expect({{"model", "tiny"}, {"prompt", "x"}, {"preset", "ghost"}}, 404, "model_not_found");
    expect({{"model", "tiny"}, {"prompt", "boom"}}, 507, "out_of_memory");

    auto c = g.client();
    const auto notJson = c.Post("/v1/images/generations", "not json", "application/json");
    CHECK(notJson->status == 400);
    CHECK(g.app.host->generation().counters().failed == 1);
}

TEST_CASE("openai: without a model named, the loaded one is used", "[integration][api][openai]") {
    GenApp g;
    REQUIRE(g.post("/v1/images/generations", {{"model", "tiny"}, {"prompt", "a"}})->status == 200);
    const auto r = g.post("/v1/images/generations", {{"prompt", "b"}});
    REQUIRE(r);
    CHECK(r->status == 200);
    CHECK(test::bodyJson(r)["model"] == "tiny.safetensors");
    auto c = g.client();
    CHECK(test::bodyJson(c.Get("/v1/models"))["data"][0]["loaded"] == true);
}

TEST_CASE("openai: edits are not implemented yet", "[integration][api][openai]") {
    GenApp g;
    auto c = g.client();
    const auto r = c.Post("/v1/images/edits", "{}", "application/json");
    REQUIRE(r);
    CHECK(r->status == 501);
    CHECK(test::bodyJson(r)["error"]["code"] == "not_implemented");
}

TEST_CASE("openai: needs the generate permission when tokens are required", "[integration][api][openai][auth]") {
    GenApp g;
    const auto reader = g.app.host->tokens().create("reader", {"read"});
    const auto maker = g.app.host->tokens().create("maker", {"read", "generate"});
    auto settings = g.app.host->settings();
    settings.server.requireToken = true;
    g.app.host->updateSettings(settings);

    const std::string body = json{{"model", "tiny"}, {"prompt", "x"}}.dump();
    auto c = g.client();
    CHECK(c.Post("/v1/images/generations", body, "application/json")->status == 401);
    CHECK(c.Get("/v1/models")->status == 401);
    const httplib::Headers asReader = {{"Authorization", "Bearer " + reader.secret}};
    const httplib::Headers asMaker = {{"Authorization", "Bearer " + maker.secret}};
    CHECK(c.Get("/v1/models", asReader)->status == 200);
    CHECK(c.Post("/v1/images/generations", asReader, body, "application/json")->status == 403);
    CHECK(c.Post("/v1/images/generations", asMaker, body, "application/json")->status == 200);
}

TEST_CASE("openai: a request larger than the limit is answered with a JSON 413", "[integration][api][openai]") {
    server::ServerOptions options = test::TestApp::ephemeral();
    options.maxBodyBytes = 1024;
    GenApp g(options);
    auto c = g.client();
    const std::string big = json{{"model", "tiny"}, {"prompt", std::string(4096, 'a')}}.dump();
    const auto r = c.Post("/v1/images/generations", big, "application/json");
    REQUIRE(r);
    CHECK(r->status == 413);
    CHECK(test::bodyJson(r)["error"]["type"] == "payload_too_large");
}

TEST_CASE("openai: hanging up cancels the generation", "[integration][api][openai]") {
    GenApp g;
    g.state->stepMs = 400;   // three steps, about 1.2 s
    {
        httplib::Client c("127.0.0.1", g.app.server->port());
        c.set_read_timeout(0, 300000);   // give up after 0.3 s
        const auto r = c.Post("/v1/images/generations", json{{"model", "tiny"}, {"prompt", "slow"}}.dump(), "application/json");
        CHECK_FALSE(r);
    }
    for (int i = 0; i < 40 && g.app.host->generation().counters().cancelled == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK(g.app.host->generation().counters().cancelled == 1);
    CHECK(g.app.host->gallery().count() == 0);
}
