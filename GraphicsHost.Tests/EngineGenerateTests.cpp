#include "ModelFixture.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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

TEST_CASE("engine: every known sampler works on SD1.5", "[model][engine][samplers]") {
    auto l = loadSd15OrSkip();
    for (const auto& sampler : knownSamplers()) {
        DYNAMIC_SECTION(sampler) {
            auto r = tiny();
            r.width = r.height = 128;
            r.steps = 2;
            r.sampler = sampler;
            const auto result = l.model->generate(r, {}, {});
            REQUIRE(result.images.size() == 1);
            CHECK(result.images[0].data.size() == 128u * 128u * 3u);
        }
    }
}

TEST_CASE("engine: every known scheduler works on SD1.5", "[model][engine][samplers]") {
    auto l = loadSd15OrSkip();
    for (const auto& scheduler : knownSchedulers()) {
        DYNAMIC_SECTION(scheduler) {
            auto r = tiny();
            r.width = r.height = 128;
            r.steps = 2;
            r.scheduler = scheduler;
            const auto result = l.model->generate(r, {}, {});
            REQUIRE(result.images.size() == 1);
        }
    }
}

TEST_CASE("engine: sampler and scheduler change the picture", "[model][engine][samplers]") {
    auto l = loadSd15OrSkip();
    auto a = tiny();
    a.steps = 4;
    a.sampler = "euler";
    auto b = a;
    b.sampler = "dpm++2m";
    auto c = a;
    c.scheduler = "karras";
    const auto ra = l.model->generate(a, {}, {}).images[0].data;
    CHECK(ra != l.model->generate(b, {}, {}).images[0].data);
    CHECK(ra != l.model->generate(c, {}, {}).images[0].data);
}

TEST_CASE("engine: odd sizes and the vae tiling switch", "[model][engine]") {
    auto l = loadSd15OrSkip();
    auto r = tiny();
    r.width = 320;
    r.height = 192;
    r.steps = 2;
    auto result = l.model->generate(r, {}, {});
    CHECK(result.images[0].width == 320);
    CHECK(result.images[0].height == 192);
    r.vaeTiling = true;
    result = l.model->generate(r, {}, {});
    CHECK(result.images[0].width == 320);
}

TEST_CASE("engine: SDXL generates at its own default family settings", "[model][engine][sdxl]") {
    const auto sdxl = gh::test::findTestModel("sd_xl_base_1.0.safetensors");
    if (!sdxl) {
        SKIP("no SDXL checkpoint (models/sd_xl_base_1.0.safetensors)");
    }
    std::string problem;
    auto runtime = gh::test::loadCpuRuntime(problem);
    if (!runtime) {
        WARN(problem);
        SKIP("CPU runtime not available");
    }
    auto engine = makeRuntimeEngine(runtime);
    LoadOptions options;
    options.modelPath = sdxl->string();
    options.device = "cpu";
    auto model = engine->load(options);
    CHECK(model->info().family.find("XL") != std::string::npos);
    CHECK(model->info().defaults.width == 1024);

    GenerationRequest r;
    r.prompt = "a red apple on a table";
    r.width = 512;  // smaller than the default to keep the test short
    r.height = 512;
    r.steps = 2;
    r.seed = 3;
    const auto result = model->generate(r, {}, {});
    REQUIRE(result.images.size() == 1);
    CHECK(result.images[0].width == 512);
    CHECK(result.request.cfgScale == 7.0f);  // SDXL family default filled in
}

namespace {

// How far apart two same-sized pictures are: the mean absolute difference per byte.
double meanDifference(const Image& a, const Image& b) {
    REQUIRE(a.data.size() == b.data.size());
    double sum = 0;
    for (size_t i = 0; i < a.data.size(); ++i) sum += std::abs(static_cast<int>(a.data[i]) - static_cast<int>(b.data[i]));
    return sum / static_cast<double>(a.data.size());
}

}  // namespace

TEST_CASE("engine: image to image keeps the source the more the lower the strength", "[model][engine][img2img]") {
    auto l = loadSd15OrSkip();

    // the source: an ordinary text to image result
    GenerationRequest source = tiny();
    source.steps = 6;
    const auto base = l.model->generate(source, {}, {});
    REQUIRE(base.images.size() == 1);
    const auto init = std::make_shared<const Image>(base.images[0]);

    auto edit = [&](float strength, int* progressSteps = nullptr, int* lastStep = nullptr) {
        GenerationRequest r = tiny();
        r.prompt = "a green apple on a table";
        r.steps = 8;
        r.seed = 99;
        r.initImage = init;
        r.strength = strength;
        std::atomic<int> total{0}, last{0};
        const auto result = l.model->generate(r, [&](const StepProgress& p) { total = p.steps; last = p.step; }, {});
        if (progressSteps) *progressSteps = total;
        if (lastStep) *lastStep = last;
        REQUIRE(result.images.size() == 1);
        CHECK(result.images[0].width == 256);
        CHECK(result.images[0].height == 256);
        return result.images[0];
    };

    int steps = 0, last = 0;
    const Image gentle = edit(0.3f, &steps, &last);
    CHECK(steps == 3);    // 8 steps * 0.3 = 2, plus the starting point: only the last part of the schedule runs
    CHECK(last == 3);     // and its progress is reported
    const Image strong = edit(0.95f);
    CHECK(meanDifference(*init, gentle) < meanDifference(*init, strong));
    // the same inputs give the same picture
    const Image again = edit(0.3f);
    CHECK(meanDifference(gentle, again) == 0.0);
    // a source that does not have the result's size is refused before anything runs
    GenerationRequest wrong = tiny();
    wrong.initImage = std::make_shared<const Image>(Image{64, 64, 3, std::vector<uint8_t>(64 * 64 * 3, 128)});
    try {
        l.model->generate(wrong, {}, {});
        FAIL("expected an error");
    } catch (const EngineError& e) {
        CHECK(e.code() == EngineErrorCode::InvalidRequest);
    }
}
