#include "ModelFixture.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <set>

using namespace gh::engine;

namespace {

// Loads the SD1.5 checkpoint on the CPU runtime or skips the test.
struct Loaded {
    std::shared_ptr<Runtime> runtime;
    std::unique_ptr<IImageEngine> engine;
    std::unique_ptr<ILoadedModel> model;
};

Loaded loadSd15OrSkip() {
    const auto modelPath = gh::test::findTestModel();
    if (!modelPath) {
        SKIP("no SD1.5 checkpoint (put models/v1-5-pruned-emaonly.safetensors in the repository or set GRAPHICSHOST_TEST_MODEL)");
    }
    std::string problem;
    Loaded l;
    l.runtime = gh::test::loadCpuRuntime(problem);
    if (!l.runtime) {
        WARN(problem);
        SKIP("CPU runtime not available");
    }
    l.engine = makeRuntimeEngine(l.runtime);
    LoadOptions options;
    options.modelPath = modelPath->string();
    options.device = "cpu";
    l.model = l.engine->load(options);
    return l;
}

GenerationRequest tiny() {
    GenerationRequest r;
    r.prompt = "a red apple on a table";
    r.width = 256;
    r.height = 256;
    r.steps = 3;
    r.seed = 7;
    return r;
}

}  // namespace

TEST_CASE("engine: loading problems are typed errors", "[unit][engine]") {
    std::string problem;
    auto runtime = gh::test::loadCpuRuntime(problem);
    if (!runtime) {
        WARN(problem);
        SKIP("CPU runtime not available");
    }
    auto engine = makeRuntimeEngine(runtime);

    LoadOptions missing;
    missing.modelPath = "C:/definitely/not/here.safetensors";
    try {
        engine->load(missing);
        FAIL("expected an error");
    } catch (const EngineError& e) {
        CHECK(e.code() == EngineErrorCode::NotFound);
    }

    LoadOptions bad;  // no model path at all
    try {
        engine->load(bad);
        FAIL("expected an error");
    } catch (const EngineError& e) {
        CHECK(e.code() == EngineErrorCode::InvalidRequest);
    }

    LoadOptions wrongDevice;
    wrongDevice.modelPath = "x";
    wrongDevice.device = "vulkan0";  // the CPU runtime cannot do that
    std::filesystem::path real = gh::core::AppPaths::executableDir() / "GraphicsHost.Tests.exe";
    wrongDevice.modelPath = real.string();
    try {
        engine->load(wrongDevice);
        FAIL("expected an error");
    } catch (const EngineError& e) {
        CHECK(e.code() == EngineErrorCode::LoadFailed);
        CHECK(std::string(e.what()).find("Vulkan runtime") != std::string::npos);
    }

    LoadOptions notAModel;
    notAModel.modelPath = real.string();  // exists, but is not a checkpoint
    notAModel.device = "cpu";
    CHECK_THROWS_AS(engine->load(notAModel), EngineError);
}

TEST_CASE("engine: txt2img on the CPU with SD1.5", "[model][engine]") {
    auto l = loadSd15OrSkip();
    CHECK_FALSE(l.model->info().family.empty());
    CHECK(l.model->info().runtimeId == "cpu-avx2");

    std::vector<int> steps;
    const auto result = l.model->generate(tiny(), [&](const StepProgress& p) { steps.push_back(p.step); }, {});
    REQUIRE(result.images.size() == 1);
    const Image& image = result.images[0];
    CHECK(image.width == 256);
    CHECK(image.height == 256);
    CHECK(image.channels == 3);
    CHECK(image.data.size() == 256u * 256u * 3u);
    // not a flat picture
    const std::set<uint8_t> distinct(image.data.begin(), image.data.end());
    CHECK(distinct.size() > 40);

    REQUIRE(result.seeds.size() == 1);
    CHECK(result.seeds[0] == 7);
    CHECK(result.request.seed == 7);
    CHECK(result.request.steps == 3);
    CHECK(result.seconds > 0);
    CHECK(steps == std::vector<int>{1, 2, 3});
}

TEST_CASE("engine: the same seed gives the same image, another seed does not", "[model][engine]") {
    auto l = loadSd15OrSkip();
    const auto a = l.model->generate(tiny(), {}, {});
    const auto b = l.model->generate(tiny(), {}, {});
    auto other = tiny();
    other.seed = 8;
    const auto c = l.model->generate(other, {}, {});
    CHECK(a.images[0].data == b.images[0].data);
    CHECK(a.images[0].data != c.images[0].data);
}

TEST_CASE("engine: a random seed is replaced by the real one and batches get consecutive seeds", "[model][engine]") {
    auto l = loadSd15OrSkip();
    auto r = tiny();
    r.seed = -1;
    r.batchCount = 2;
    r.steps = 2;
    const auto result = l.model->generate(r, {}, {});
    REQUIRE(result.images.size() == 2);
    REQUIRE(result.seeds.size() == 2);
    CHECK(result.seeds[0] >= 0);
    CHECK(result.seeds[1] == result.seeds[0] + 1);
    CHECK(result.request.seed == result.seeds[0]);

    // reproduce the second image from its reported seed
    auto again = tiny();
    again.steps = 2;
    again.seed = result.seeds[1];
    CHECK(l.model->generate(again, {}, {}).images[0].data == result.images[1].data);
}

TEST_CASE("engine: bad parameters are rejected before anything runs", "[model][engine]") {
    auto l = loadSd15OrSkip();
    auto r = tiny();
    r.width = 250;
    try {
        l.model->generate(r, {}, {});
        FAIL("expected an error");
    } catch (const EngineError& e) {
        CHECK(e.code() == EngineErrorCode::InvalidRequest);
        CHECK(std::string(e.what()).find("width") != std::string::npos);
    }
}

TEST_CASE("engine: a stop request cancels at the next step", "[model][engine]") {
    auto l = loadSd15OrSkip();
    auto r = tiny();
    r.width = 512;
    r.height = 512;
    r.steps = 30;  // would take far longer than the test allows if it ran to the end

    std::stop_source stop;
    int lastStep = 0;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        l.model->generate(r,
                          [&](const StepProgress& p) {
                              lastStep = p.step;
                              if (p.step == 2) stop.request_stop();
                          },
                          stop.get_token());
        FAIL("expected the generation to be cancelled");
    } catch (const EngineError& e) {
        CHECK(e.code() == EngineErrorCode::Cancelled);
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(lastStep < 30);
    CHECK(seconds < 25.0);

    // the model is still usable afterwards
    CHECK(l.model->generate(tiny(), {}, {}).images.size() == 1);

    // a token that is already stopped never starts
    std::stop_source already;
    already.request_stop();
    CHECK_THROWS_AS(l.model->generate(tiny(), {}, already.get_token()), EngineError);
}

TEST_CASE("engine: a negative prompt with cfgScale 1 produces a warning", "[model][engine]") {
    auto l = loadSd15OrSkip();
    auto r = tiny();
    r.steps = 2;
    r.cfgScale = 1.0f;
    r.negativePrompt = "blurry";
    const auto result = l.model->generate(r, {}, {});
    REQUIRE(result.warnings.size() == 1);
    CHECK(result.warnings[0].find("negative prompt") != std::string::npos);
}
