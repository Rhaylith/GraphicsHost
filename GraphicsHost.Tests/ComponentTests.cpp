#include "ModelFixture.h"

#include "core/Paths.h"
#include "engine/RuntimeLoader.h"
#include "sampling/Euler.h"

#include <chrono>
#include <cmath>
#include <filesystem>

using namespace gh::engine;

namespace {

// A loaded SD 1.5 on the Vulkan runtime, or a skipped test. Component calls need gh_session_begin (not while gh_generate runs).
struct VulkanModel {
    std::unique_ptr<Runtime> runtime;
    gh_model* model = nullptr;
    ~VulkanModel() {
        if (model && runtime) {
            runtime->api().sessionEnd(model);
            runtime->api().free(model);
        }
    }
};

std::unique_ptr<VulkanModel> loadOrSkip(const char* checkpoint = "v1-5-pruned-emaonly.safetensors") {
    const auto path = gh::test::findTestModel(checkpoint);
    if (!path) SKIP("no checkpoint " << checkpoint << " (see ModelFixture.h)");
    const auto dll = gh::core::AppPaths::executableDir() / "runtimes" / "sd-vulkan.dll";
    if (!std::filesystem::exists(dll)) SKIP("sd-vulkan.dll is not next to the test executable");
    auto m = std::make_unique<VulkanModel>();
    std::string error;
    m->runtime = Runtime::load(dll, error);
    if (!m->runtime) SKIP("cannot load the Vulkan runtime: " << error);
    REQUIRE(m->runtime->hasComponents());
    gh_load_params lp{};
    lp.struct_size = sizeof(lp);
    const std::string file = path->string();
    lp.model_path = file.c_str();
    lp.flash_attention = 1;
    m->model = m->runtime->api().load(&lp);
    if (!m->model) SKIP("the model could not be loaded: " << m->runtime->api().lastError());
    return m;
}

struct Tensor {
    gh_tensor t{};
    const Runtime::Api* api;
    explicit Tensor(const Runtime::Api& a) : api(&a) { t.struct_size = sizeof(t); }
    ~Tensor() { api->tensorFree(&t); }
    int64_t count() const {
        int64_t n = 1;
        for (int i = 0; i < t.ndim; ++i) n *= t.shape[i];
        return n;
    }
};

}  // namespace

TEST_CASE("components: the pieces of SD 1.5 produce plausible tensors on Vulkan", "[model][vulkan][components]") {
    auto m = loadOrSkip();
    const auto& api = m->runtime->api();

    REQUIRE(api.sessionBegin(m->model) == GH_OK);
    Tensor hidden(api), pooled(api);
    REQUIRE(api.textEncode(m->model, "a lighthouse on a cliff", -1, 512, 512, 0, &hidden.t, &pooled.t) == GH_OK);
    CHECK(hidden.t.ndim == 3);
    CHECK(hidden.t.shape[0] == 768);   // the CLIP width of SD 1.x
    CHECK(hidden.t.shape[1] >= 77);

    float* sigmas = nullptr;
    int32_t count = 0;
    gh_generate_params gp{};
    gp.struct_size = sizeof(gp);
    gp.width = gp.height = 512;
    gp.steps = 4;
    gp.sampler = "euler";
    gp.scheduler = "discrete";
    REQUIRE(api.sigmaSchedule(m->model, &gp, &sigmas, &count) == GH_OK);
    REQUIRE(count == 5);   // steps + 1
    CHECK(sigmas[count - 1] == 0.0f);
    for (int i = 1; i < count; ++i) CHECK(sigmas[i] < sigmas[i - 1]);

    Tensor x(api);
    REQUIRE(api.initialLatent(m->model, 42, 512, 512, sigmas[0], &x.t) == GH_OK);
    REQUIRE(x.t.ndim == 4);
    CHECK(x.t.shape[0] == 64);
    CHECK(x.t.shape[1] == 64);
    CHECK(x.t.shape[2] == 4);

    Tensor uncondHidden(api);
    REQUIRE(api.textEncode(m->model, "", -1, 512, 512, 0, &uncondHidden.t, nullptr) == GH_OK);
    gh_denoise_args args{};
    args.struct_size = sizeof(args);
    args.x = &x.t;
    args.sigma = sigmas[0];
    args.cond_hidden = &hidden.t;
    args.uncond_hidden = &uncondHidden.t;
    args.cfg_scale = 7.5f;
    Tensor denoised(api);
    REQUIRE(api.denoise(m->model, &args, &denoised.t) == GH_OK);
    REQUIRE(denoised.count() == x.count());
    bool finite = true;
    for (int64_t i = 0; i < denoised.count(); ++i) finite = finite && std::isfinite(denoised.t.data[i]);
    CHECK(finite);

    Tensor image(api);
    REQUIRE(api.vaeDecode(m->model, &denoised.t, &image.t) == GH_OK);
    REQUIRE(image.t.ndim >= 3);
    CHECK(image.t.shape[0] == 512);
    CHECK(image.t.shape[1] == 512);
    CHECK(image.t.shape[2] == 3);
    float lo = 1e9f, hi = -1e9f;
    for (int64_t i = 0; i < image.count(); ++i) {
        lo = std::min(lo, image.t.data[i]);
        hi = std::max(hi, image.t.data[i]);
    }
    CHECK(hi > lo);   // not a flat picture
    api.floatArrayFree(sigmas);
}

namespace {

struct PipelineSettings {
    std::string prompt = "a lighthouse on a cliff at sunset";
    std::string negative = "";
    int64_t seed = 42;
    int size = 512;
    int steps = 6;
    float cfg = 7.5f;
    const char* sampler = "euler";
    const char* scheduler = "discrete";
};

// The whole pipeline from component calls and the host's Euler loop: an RGB float image in 0..1, row by row.
std::vector<float> runFromComponents(const Runtime::Api& api, gh_model* model, const PipelineSettings& s, double* secondsInLoop = nullptr) {
    REQUIRE(api.sessionBegin(model) == GH_OK);
    Tensor hidden(api), uncond(api), pooled(api), unpooled(api);
    REQUIRE(api.textEncode(model, s.prompt.c_str(), -1, s.size, s.size, 0, &hidden.t, &pooled.t) == GH_OK);
    // SDXL (a pooled vector exists) with an empty negative prompt: all zeros, as the engine does
    const bool zeroOut = s.negative.empty() && pooled.t.ndim > 0;
    REQUIRE(api.textEncode(model, s.negative.c_str(), -1, s.size, s.size, zeroOut ? 1 : 0, &uncond.t, &unpooled.t) == GH_OK);

    gh_generate_params gp{};
    gp.struct_size = sizeof(gp);
    gp.width = gp.height = s.size;
    gp.steps = s.steps;
    gp.sampler = s.sampler;
    gp.scheduler = s.scheduler;
    float* raw = nullptr;
    int32_t count = 0;
    REQUIRE(api.sigmaSchedule(model, &gp, &raw, &count) == GH_OK);
    const std::vector<float> sigmas(raw, raw + count);
    api.floatArrayFree(raw);

    Tensor start(api);
    REQUIRE(api.initialLatent(model, s.seed, s.size, s.size, sigmas[0], &start.t) == GH_OK);
    gh_tensor shape = start.t;   // the shape of every latent in the loop
    gh::sampling::Latent x(start.t.data, start.t.data + start.count());

    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = gh::sampling::euler(x, sigmas, [&](const gh::sampling::Latent& cur, float sigma, int, gh::sampling::Latent& out) {
        gh_tensor in = shape;
        in.data = const_cast<float*>(cur.data());
        gh_denoise_args args{};
        args.struct_size = sizeof(args);
        args.x = &in;
        args.sigma = sigma;
        args.cond_hidden = &hidden.t;
        args.cond_pooled = pooled.t.ndim > 0 ? &pooled.t : nullptr;   // SDXL: pooled text + size conditioning
        args.uncond_pooled = unpooled.t.ndim > 0 ? &unpooled.t : nullptr;
        args.uncond_hidden = &uncond.t;
        args.cfg_scale = s.cfg;
        Tensor result(api);
        if (api.denoise(model, &args, &result.t) != GH_OK) return false;
        out.assign(result.t.data, result.t.data + result.count());
        return true;
    });
    if (secondsInLoop) *secondsInLoop = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    REQUIRE(ok);

    gh_tensor latent = shape;
    latent.data = x.data();
    Tensor image(api);
    REQUIRE(api.vaeDecode(model, &latent, &image.t) == GH_OK);
    // ggml order {W, H, 3, 1} -> rows of RGB triples
    const int w = static_cast<int>(image.t.shape[0]), h = static_cast<int>(image.t.shape[1]);
    std::vector<float> rgb(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y)
        for (int xx = 0; xx < w; ++xx)
            for (int c = 0; c < 3; ++c) rgb[(static_cast<size_t>(y) * w + xx) * 3 + c] = image.t.data[(static_cast<size_t>(c) * h + y) * w + xx];
    api.sessionEnd(model);
    return rgb;
}

// What gh_generate makes for the same settings, as floats in 0..1.
std::vector<float> runWithGenerate(const Runtime::Api& api, gh_model* model, const PipelineSettings& s) {
    gh_generate_params gp{};
    gp.struct_size = sizeof(gp);
    gp.prompt = s.prompt.c_str();
    gp.negative_prompt = s.negative.c_str();
    gp.width = gp.height = s.size;
    gp.steps = s.steps;
    gp.cfg_scale = s.cfg;
    gp.seed = s.seed;
    gp.batch_count = 1;
    gp.sampler = s.sampler;
    gp.scheduler = s.scheduler;
    gp.clip_skip = -1;
    gh_result result{};
    REQUIRE(api.generate(model, &gp, nullptr, nullptr, &result) == GH_OK);
    REQUIRE(result.image_count == 1);
    const gh_image& img = result.images[0];
    std::vector<float> rgb(static_cast<size_t>(img.width) * img.height * 3);
    for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = img.data[i] / 255.0f;
    api.freeResult(&result);
    return rgb;
}

double meanAbsDifference255(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) sum += std::fabs(a[i] - b[i]) * 255.0;
    return sum / static_cast<double>(a.size());
}

}  // namespace

TEST_CASE("components: the host's Euler loop over gh_denoise makes the image gh_generate makes", "[model][vulkan][components]") {
    auto m = loadOrSkip();
    const auto& api = m->runtime->api();
    PipelineSettings s;
    const auto reference = runWithGenerate(api, m->model, s);
    const auto fromParts = runFromComponents(api, m->model, s);
    const double difference = meanAbsDifference255(reference, fromParts);
    INFO("mean absolute difference per colour value (0..255): " << difference);
    CHECK(difference < 0.5);   // the same picture (measured 0.23): GPU maths is not bit-exact, a different path would differ by tens

    // a different seed is a different picture (the comparison can tell)
    PipelineSettings other = s;
    other.seed = 43;
    CHECK(meanAbsDifference255(reference, runFromComponents(api, m->model, other)) > 10.0);
}

TEST_CASE("components: gh_generate repeats itself, so differences to it mean something", "[model][vulkan][components]") {
    auto m = loadOrSkip();
    PipelineSettings s;
    const auto a = runWithGenerate(m->runtime->api(), m->model, s);
    const auto b = runWithGenerate(m->runtime->api(), m->model, s);
    const double self = meanAbsDifference255(a, b);
    INFO("gh_generate against itself: " << self);
    CHECK(self < 0.05);
}

TEST_CASE("components: the host loop costs less than 3 % per step over the engine's own loop", "[model][vulkan][components][perf]") {
    auto m = loadOrSkip();
    const auto& api = m->runtime->api();
    PipelineSettings s;

    // Seconds per sampling step from two runs of different length: the encoders, the decode and the first-call set-up
    // are the same in both and drop out of the difference.
    auto seconds = [&](bool host, int steps) {
        PipelineSettings p = s;
        p.steps = steps;
        if (host) {
            double loop = 0;
            runFromComponents(api, m->model, p, &loop);
            return loop;
        }
        const auto t0 = std::chrono::steady_clock::now();
        runWithGenerate(api, m->model, p);
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    runWithGenerate(api, m->model, s);   // warm-up: shader compilation, first allocations
    runFromComponents(api, m->model, s);

    const int shortRun = 6, longRun = 26;
    auto perStep = [&](bool host) {
        double best = 1e9;
        for (int repeat = 0; repeat < 2; ++repeat) {
            const double a = seconds(host, shortRun), b = seconds(host, longRun);
            best = std::min(best, (b - a) / (longRun - shortRun));
        }
        return best;
    };
    const double engine = perStep(false);
    const double hostLoop = perStep(true);
    const double overhead = (hostLoop - engine) / engine * 100.0;
    WARN("engine loop " << engine << " s/step, host loop " << hostLoop << " s/step, overhead " << overhead << " %");
    CHECK(overhead < 3.0);
}

TEST_CASE("components: SDXL through the same path (two encoders, pooled and size conditioning)", "[model][vulkan][components][sdxl]") {
    auto m = loadOrSkip("sd_xl_base_1.0.safetensors");
    const auto& api = m->runtime->api();

    // the second encoder and the size conditioning show in the tensors
    REQUIRE(api.sessionBegin(m->model) == GH_OK);
    {
        Tensor hidden(api), pooled(api);
        REQUIRE(api.textEncode(m->model, "a lighthouse", -1, 1024, 1024, 0, &hidden.t, &pooled.t) == GH_OK);
        CHECK(hidden.t.shape[0] == 2048);   // CLIP-L (768) and CLIP-G (1280) side by side
        REQUIRE(pooled.t.ndim >= 1);
        CHECK(pooled.t.shape[0] == 2816);   // pooled CLIP-G (1280) plus six size and crop values embedded (6 x 256)
    }
    api.sessionEnd(m->model);

    PipelineSettings s;
    s.size = 1024;
    s.steps = 6;
    const auto reference = runWithGenerate(api, m->model, s);
    const auto fromParts = runFromComponents(api, m->model, s);
    const double difference = meanAbsDifference255(reference, fromParts);
    INFO("mean absolute difference per colour value (0..255): " << difference);
    CHECK(difference < 0.5);
}
