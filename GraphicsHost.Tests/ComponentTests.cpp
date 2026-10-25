#include "ModelFixture.h"

#include "core/Paths.h"
#include "engine/RuntimeLoader.h"

#include <cmath>
#include <filesystem>

using namespace gh::engine;

namespace {

// A loaded SD 1.5 on the Vulkan runtime, with a component session open, or a skipped test.
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
    REQUIRE(m->runtime->api().sessionBegin(m->model) == GH_OK);
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

    Tensor hidden(api), pooled(api);
    REQUIRE(api.textEncode(m->model, "a lighthouse on a cliff", -1, 512, 512, &hidden.t, &pooled.t) == GH_OK);
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
    REQUIRE(api.textEncode(m->model, "", -1, 512, 512, &uncondHidden.t, nullptr) == GH_OK);
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
