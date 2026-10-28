#include "core/Settings.h"
#include "graph/Inference.h"
#include "graph/SampleRun.h"
#include "sampling/Preview.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace gh::sampling;
using gh::tensor::Tensor;
using Catch::Approx;

TEST_CASE("preview: latents become small colour pictures", "[unit][sampling][preview]") {
    Tensor zero({1, 4, 6, 8}, 0.0f);
    const Tensor grey = latentToRgb(zero, "SD 1.x");
    REQUIRE(grey.shape() == std::vector<int64_t>{1, 6, 8, 3});
    for (float v : grey.values()) CHECK(v == Approx(0.5f));   // (0 + 1) / 2

    // one channel switched on moves the colour along that channel's row of the map
    Tensor one({1, 4, 2, 2}, 0.0f);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) one.at({0, 0, y, x}) = 1.0f;
    const auto f = latentRgbFactors("SD 1.x");
    const Tensor img = latentToRgb(one, "SD 1.x");
    for (int c = 0; c < 3; ++c) CHECK(img.at({0, 0, 0, c}) == Approx(std::clamp((f[static_cast<size_t>(c)] + 1.0f) * 0.5f, 0.0f, 1.0f)));
    CHECK(latentRgbFactors("SDXL") != latentRgbFactors("SD 1.x"));
    CHECK(latentRgbFactors("FLUX").empty());
    CHECK(latentToRgb(one, "something else").shape() == img.shape());   // 4 channels: the SD 1.x map

    // values stay in 0..1 even for wild latents
    Tensor wild({2, 4, 3, 3}, 50.0f);
    wild.at({1, 2, 1, 1}) = -80.0f;
    for (float v : latentToRgb(wild, "SDXL").values()) CHECK((v >= 0.0f && v <= 1.0f));

    // other channel counts show the first channels stretched over -3..3
    Tensor sixteen({1, 16, 2, 2}, 0.0f);
    sixteen.at({0, 1, 0, 0}) = 3.0f;
    const Tensor wide = latentToRgb(sixteen, "FLUX");
    CHECK(wide.at({0, 0, 0, 0}) == Approx(0.5f));
    CHECK(wide.at({0, 0, 0, 1}) == Approx(1.0f));

    CHECK(latentToRgb(Tensor({4, 4}, 0.0f), "SD 1.x").empty());   // not a latent
}

TEST_CASE("preview: pictures are shrunk to the longest side, never enlarged", "[unit][sampling][preview]") {
    Tensor big({1, 4, 128, 64}, 0.1f);   // a 1024 x 512 picture
    const Tensor small = previewImage(big, "SD 1.x", 256);
    CHECK(small.dim(1) == 256);
    CHECK(small.dim(2) == 128);
    CHECK(small.dim(3) == 3);
    CHECK(previewImage(big, "SD 1.x", 4096).dim(1) == 128);   // no enlarging: the latent size stays
    CHECK(previewImage(big, "SD 1.x", 0).dim(1) == 128);
}

TEST_CASE("preview: the sampler offers a picture every few steps", "[unit][sampling][preview]") {
    using namespace gh::graph;
    class Flat final : public InferenceService {
    public:
        Value encodeText(const TextEncoderHandle&, const gh::text::ParsedPrompt&, int, int, int, bool) override { return {}; }
        void denoise(const ModelHandle&, const Value&, const LatentView& x, float, std::vector<float>& out) override { out.assign(x.count(), 0.25f); }
        Tensor decode(const VaeHandle&, const ModelHandle&, const LatentView&) override { return {}; }
    } inference;
    SampleRequest req;
    req.inference = &inference;
    req.guider.kind = GuiderHandle::Kind::Basic;
    ModelHandle model;
    model.file.family = "SDXL";
    req.guider.model = handleValue(ValueType::Model, model);
    req.guider.conditioning = {Value::makeInt(0)};
    req.sampler.name = "euler";
    req.sigmas = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.5f, 0.0f};
    req.latent = Tensor({1, 4, 64, 32}, 0.0f);
    int previews = 0;
    req.emitPreview = [&](const Tensor& image) {
        ++previews;
        CHECK(image.dim(3) == 3);
        CHECK(image.dim(1) == 32);   // the longest side is 64, shrunk to the limit
    };
    req.previewMethod = "fast";
    req.previewEvery = 2;
    req.previewMaxSize = 32;
    runSampling(req);
    CHECK(previews == 3);   // steps 2, 4 and 6
    previews = 0;
    req.previewMethod = "none";
    runSampling(req);
    CHECK(previews == 0);
}

TEST_CASE("settings: preview settings round trip and are validated", "[unit][sampling][preview][settings]") {
    gh::core::Settings s;
    CHECK(s.graph.previewMethod == "fast");
    CHECK(s.graph.previewEveryNSteps == 1);
    CHECK(s.graph.previewMaxSize == 512);
    s.graph.previewMethod = "none";
    s.graph.previewEveryNSteps = 5;
    s.graph.previewMaxSize = 256;
    const auto back = gh::core::settingsFromJson(gh::core::toJson(s));
    CHECK(back.graph.previewMethod == "none");
    CHECK(back.graph.previewEveryNSteps == 5);
    CHECK(back.graph.previewMaxSize == 256);
    CHECK(gh::core::validate(back).empty());
    s.graph.previewMethod = "magic";
    s.graph.previewEveryNSteps = 0;
    s.graph.previewMaxSize = 8;
    CHECK(gh::core::validate(s).size() == 3);
}
