#pragma once

#include "ApiTestUtil.h"
#include "SafetensorsFixture.h"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <mutex>
#include <thread>

namespace gh::test {

using nlohmann::json;
namespace fs = std::filesystem;

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
    TestApp app;
    std::shared_ptr<FakeState> state = std::make_shared<FakeState>();

    explicit GenApp(server::ServerOptions options = TestApp::ephemeral()) : app(options, false) {
        st::writeSafetensors(app.home.path() / "models" / "tiny.safetensors", st::sd1Tensors());
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


}  // namespace gh::test
