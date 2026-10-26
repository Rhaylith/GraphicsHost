#include "SafetensorsFixture.h"
#include "TestUtil.h"
#include "core/Settings.h"
#include "graph/Executor.h"
#include "graph/Handles.h"
#include "host/GraphService.h"
#include "models/Precision.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace gh::models;
using gh::tensor::Tensor;
using Catch::Approx;
using nlohmann::json;

TEST_CASE("precision: names and widths", "[unit][precision]") {
    for (const Precision p : {Precision::Auto, Precision::Fp32, Precision::Fp16, Precision::Bf16, Precision::Fp8E4m3, Precision::Fp8E5m2}) {
        CHECK(parsePrecision(toString(p)) == p);
    }
    CHECK(parsePrecision("fp8_e4m3") == Precision::Fp8E4m3);
    CHECK(parsePrecision("default") == Precision::Auto);
    CHECK(parsePrecision("fp8 e4m3 fast") == Precision::Fp8E4m3);
    CHECK_FALSE(parsePrecision("fp4").has_value());
    CHECK(bytesPerElement(Precision::Fp32) == 4);
    CHECK(bytesPerElement(Precision::Fp16) == 2);
    CHECK(bytesPerElement(Precision::Bf16) == 2);
    CHECK(bytesPerElement(Precision::Fp8E5m2) == 1);
    CHECK(isFp8(Precision::Fp8E4m3));
    CHECK_FALSE(isFp8(Precision::Bf16));
}

TEST_CASE("precision: a node's dtype wins over the setting, default follows it", "[unit][precision]") {
    auto c = resolveDtype("default", "fp8 e4m3");
    CHECK(c.precision == Precision::Fp8E4m3);
    CHECK_FALSE(c.fast);
    c = resolveDtype("fp8 e4m3 fast", "fp16");
    CHECK(c.precision == Precision::Fp8E4m3);
    CHECK(c.fast);
    CHECK(resolveDtype("fp8 e5m2", "fp16").precision == Precision::Fp8E5m2);
    CHECK(resolveDtype("default", "auto").precision == Precision::Auto);
    CHECK(resolveDtype("", "bf16").precision == Precision::Bf16);
    CHECK(resolveDtype("nonsense", "fp16").precision == Precision::Auto);
}

TEST_CASE("fp8 e4m3: known bit patterns", "[unit][precision][fp8]") {
    CHECK(encodeFp8E4m3(1.0f) == 0x38);
    CHECK(encodeFp8E4m3(-1.0f) == 0xB8);
    CHECK(encodeFp8E4m3(0.0f) == 0x00);
    CHECK(encodeFp8E4m3(-0.0f) == 0x80);
    CHECK(encodeFp8E4m3(448.0f) == 0x7E);
    CHECK(encodeFp8E4m3(1000.0f) == 0x7E);          // saturates
    CHECK(encodeFp8E4m3(-1e9f) == 0xFE);
    CHECK(encodeFp8E4m3(std::numeric_limits<float>::infinity()) == 0x7E);
    CHECK(encodeFp8E4m3(std::nanf("")) == 0x7F);
    CHECK(encodeFp8E4m3(0.001953125f) == 0x01);     // 2^-9: the smallest subnormal
    CHECK(encodeFp8E4m3(0.015625f) == 0x08);        // 2^-6: the smallest normal number
    CHECK(encodeFp8E4m3(1.0625f) == 0x38);          // halfway between 1.0 and 1.125: ties to even (1.0)
    CHECK(encodeFp8E4m3(1.1875f) == 0x3A);          // halfway between 1.125 and 1.25: ties to even (1.25)
    CHECK(encodeFp8E4m3(1e-6f) == 0x00);            // below half the smallest step
    CHECK(decodeFp8E4m3(0x38) == 1.0f);
    CHECK(decodeFp8E4m3(0x7E) == 448.0f);
    CHECK(decodeFp8E4m3(0xB8) == -1.0f);
    CHECK(decodeFp8E4m3(0x01) == 0.001953125f);
    CHECK(decodeFp8E4m3(0x08) == 0.015625f);
    CHECK(std::isnan(decodeFp8E4m3(0x7F)));
    CHECK(std::isnan(decodeFp8E4m3(0xFF)));
}

TEST_CASE("fp8 e5m2: known bit patterns", "[unit][precision][fp8]") {
    CHECK(encodeFp8E5m2(1.0f) == 0x3C);
    CHECK(encodeFp8E5m2(-2.0f) == 0xC0);
    CHECK(encodeFp8E5m2(57344.0f) == 0x7B);
    CHECK(encodeFp8E5m2(1e9f) == 0x7B);             // saturates
    CHECK(encodeFp8E5m2(std::nanf("")) == 0x7E);
    CHECK(encodeFp8E5m2(1.1f) == 0x3C);             // mantissa steps of 0.25: 1.1 rounds to 1.0
    CHECK(encodeFp8E5m2(1.2f) == 0x3D);             // 1.25
    CHECK(encodeFp8E5m2(1.0f / 65536.0f) == 0x01);  // 2^-16: the smallest subnormal
    CHECK(decodeFp8E5m2(0x3C) == 1.0f);
    CHECK(decodeFp8E5m2(0x7B) == 57344.0f);
    CHECK(decodeFp8E5m2(0x01) == 1.0f / 65536.0f);
    CHECK(std::isinf(decodeFp8E5m2(0x7C)));
    CHECK(std::isnan(decodeFp8E5m2(0x7D)));
    CHECK(decodeFp8E5m2(0xFC) < 0);
}

TEST_CASE("fp8: every code survives a round trip and the error stays small", "[unit][precision][fp8]") {
    for (int bits = 0; bits < 256; ++bits) {
        const float a = decodeFp8E4m3(static_cast<uint8_t>(bits));
        if (std::isnan(a)) continue;
        CHECK(decodeFp8E4m3(encodeFp8E4m3(a)) == a);
        const float b = decodeFp8E5m2(static_cast<uint8_t>(bits));
        if (std::isnan(b) || std::isinf(b)) continue;
        CHECK(decodeFp8E5m2(encodeFp8E5m2(b)) == b);
    }
    // relative error of normal numbers is at most half a mantissa step: 2^-4 for e4m3, 2^-3 for e5m2
    for (float v = 0.02f; v < 400.0f; v *= 1.037f) {
        CHECK(std::fabs(decodeFp8E4m3(encodeFp8E4m3(v)) - v) <= v * 0.0625f * 1.001f);
        CHECK(std::fabs(decodeFp8E5m2(encodeFp8E5m2(v)) - v) <= v * 0.125f * 1.001f);
    }
}

TEST_CASE("fp16 and bf16: encoding matches the standard formats", "[unit][precision]") {
    CHECK(encodeFp16(1.0f) == 0x3C00);
    CHECK(encodeFp16(-2.0f) == 0xC000);
    CHECK(encodeFp16(0.5f) == 0x3800);
    CHECK(encodeFp16(65504.0f) == 0x7BFF);
    CHECK(encodeFp16(1e6f) == 0x7C00);              // overflows to infinity, as fp16 does
    CHECK(encodeFp16(5.9604645e-8f) == 0x0001);     // the smallest subnormal
    CHECK(encodeFp16(1.0f + 1.0f / 2048.0f) == 0x3C00);   // halfway: ties to even
    CHECK(decodeFp16(0x3C00) == 1.0f);
    CHECK(decodeFp16(0x7BFF) == 65504.0f);
    CHECK(decodeFp16(0x0001) == Approx(5.9604645e-8f));
    CHECK(std::isinf(decodeFp16(0x7C00)));
    CHECK(encodeBf16(1.0f) == 0x3F80);
    CHECK(encodeBf16(3.0f) == 0x4040);
    CHECK(decodeBf16(0x4040) == 3.0f);
    CHECK(encodeBf16(1.00390625f) == 0x3F80);       // halfway between two bf16 numbers: ties to even
    for (float v : {0.1f, 3.14159f, -123.456f, 1e-3f, 6.5e4f}) {
        CHECK(decodeFp16(encodeFp16(v)) == Approx(v).epsilon(1.0 / 1024));
        CHECK(decodeBf16(encodeBf16(v)) == Approx(v).epsilon(1.0 / 128));
    }
}

TEST_CASE("packing: tensors keep their shape and shrink to the storage width", "[unit][precision]") {
    std::vector<float> values;
    for (int i = 0; i < 24; ++i) values.push_back(0.05f * static_cast<float>(i) - 0.5f);
    const Tensor t({2, 3, 4}, values);
    for (const auto& [precision, tolerance, size] : {std::tuple{Precision::Fp32, 0.0, size_t{96}}, std::tuple{Precision::Fp16, 0.001, size_t{48}}, std::tuple{Precision::Bf16, 0.004, size_t{48}},
                                                      std::tuple{Precision::Fp8E4m3, 0.07, size_t{24}}, std::tuple{Precision::Fp8E5m2, 0.13, size_t{24}}}) {
        INFO(toString(precision));
        const PackedTensor packed = pack(t, precision);
        CHECK(packed.sizeBytes() == size);
        const Tensor back = unpack(packed);
        REQUIRE(back.shape() == t.shape());
        for (int64_t i = 0; i < t.numel(); ++i) CHECK(std::fabs(back.data()[i] - t.data()[i]) <= std::fabs(t.data()[i]) * tolerance + 0.002);
    }
    CHECK(pack(t, Precision::Auto).precision == Precision::Fp16);   // auto packs as the usual file type

    Tensor odd({3}, std::vector<float>{std::nanf(""), 1e9f, -1e9f});
    const Tensor saturated = unpack(pack(odd, Precision::Fp8E4m3));
    CHECK(std::isnan(saturated.data()[0]));
    CHECK(saturated.data()[1] == 448.0f);
    CHECK(saturated.data()[2] == -448.0f);
    CHECK(hasInvalidValues(odd));
    CHECK_FALSE(hasInvalidValues(t));
    CHECK(hasInvalidValues(Tensor({1}, std::vector<float>{std::numeric_limits<float>::infinity()})));
}

TEST_CASE("precision: weights take less memory at a smaller width", "[unit][precision]") {
    gh::test::TempDir dir;
    gh::test::st::writeSafetensors(dir.path() / "a.safetensors", {{"w1", {1000, 1000}, "F32"}, {"w2", {500, 500}, "F16"}, {"ids", {10}, "I64"}});
    const auto header = readCheckpointHeader(dir.path() / "a.safetensors");
    REQUIRE(header.has_value());
    // the fixture writes one byte per element for types it does not know (I64 above): that tensor keeps its size
    const uint64_t ids = header->find("ids")->bytes;
    CHECK(weightsBytesAt(*header, Precision::Auto) == header->fileSize);
    CHECK(weightsBytesAt(*header, Precision::Fp32) == 1000000ull * 4 + 250000 * 4 + ids);
    CHECK(weightsBytesAt(*header, Precision::Fp16) == 1000000ull * 2 + 250000 * 2 + ids);
    CHECK(weightsBytesAt(*header, Precision::Fp8E4m3) == 1000000ull + 250000 + ids);
    CHECK(weightsBytesAt(*header, Precision::Fp8E4m3) * 2 < weightsBytesAt(*header, Precision::Fp16) + 2 * ids);
}

TEST_CASE("settings: the precision section round trips and is validated", "[unit][precision][settings]") {
    gh::core::Settings s;
    CHECK(s.precision.diffusion == "auto");
    CHECK(s.precision.vaeFallbackFp32);
    s.precision.diffusion = "fp8 e4m3";
    s.precision.textEncoder = "bf16";
    s.precision.vae = "fp32";
    s.precision.vaeFallbackFp32 = false;
    const auto back = gh::core::settingsFromJson(gh::core::toJson(s));
    CHECK(back.precision.diffusion == "fp8 e4m3");
    CHECK(back.precision.textEncoder == "bf16");
    CHECK(back.precision.vae == "fp32");
    CHECK_FALSE(back.precision.vaeFallbackFp32);
    CHECK(gh::core::validate(back).empty());

    s.precision.textEncoder = "fp8 e5m2";   // not offered for text encoders
    s.precision.vae = "fp8 e4m3";           // nor for VAEs
    const auto errors = gh::core::validate(s);
    REQUIRE(errors.size() == 2);
    CHECK(errors[0].field == "precision.textEncoder");
    CHECK(errors[1].field == "precision.vae");
}

TEST_CASE("loaders: the precision settings reach the handles", "[unit][precision][loaders]") {
    using namespace gh::graph;
    gh::test::TempDir dir;
    const auto root = dir.path() / "models";
    gh::test::st::writeSafetensors(root / "base.safetensors", gh::test::st::sd1Tensors());
    gh::test::st::writeSafetensors(root / "diffusion_models" / "dit.safetensors", gh::test::st::sd1Tensors());
    gh::test::st::writeSafetensors(root / "vae" / "v.safetensors", {{"decoder.conv_in.weight", {512, 4, 3, 3}}, {"encoder.conv_in.weight", {128, 3, 3, 3}}});
    gh::test::st::writeSafetensors(root / "text_encoders" / "clip.safetensors", {{"text_model.embeddings.token_embedding.weight", {128, 768}}});
    gh::models::ModelLibrary library({root});
    gh::host::LibraryNodeServices services(library, [] { return PrecisionPolicy{"fp8 e4m3", "bf16", "fp32", false}; });
    OutputCache cache(*CachePolicy::parse("none"));
    const auto run = [&](const json& nodes) {
        const JobGraph job = jobFromJson({{"format", "graphicshost.job"}, {"version", 1}, {"nodes", nodes}, {"outputs", json::array({"1"})}});
        RunOptions options;
        options.services = &services;
        return Executor(Registry::builtin(), cache).run(job, options);
    };
    auto r = run({{"1", {{"type", "loader.checkpoint"}, {"inputs", {{"checkpoint", "base"}}}}}});
    REQUIRE(r.ok());
    CHECK(r.outputs.at("1")[0].asObject<ModelHandle>()->load.weightDtype == "fp8 e4m3");
    CHECK(r.outputs.at("1")[1].asObject<TextEncoderHandle>()->load.weightDtype == "bf16");
    CHECK(r.outputs.at("1")[2].asObject<VaeHandle>()->load.weightDtype == "fp32");
    CHECK_FALSE(r.outputs.at("1")[2].asObject<VaeHandle>()->fallbackFp32);

    // the diffusion model node: "default" follows the setting, an explicit value wins ("fast" is kept)
    r = run({{"1", {{"type", "loader.diffusionModel"}, {"inputs", {{"model", "dit"}}}}}});
    CHECK(r.outputs.at("1")[0].asObject<ModelHandle>()->load.weightDtype == "fp8 e4m3");
    r = run({{"1", {{"type", "loader.diffusionModel"}, {"inputs", {{"model", "dit"}, {"weightDtype", "fp8 e5m2"}}}}}});
    CHECK(r.outputs.at("1")[0].asObject<ModelHandle>()->load.weightDtype == "fp8 e5m2");
    r = run({{"1", {{"type", "loader.diffusionModel"}, {"inputs", {{"model", "dit"}, {"weightDtype", "fp8 e4m3 fast"}}}}}});
    CHECK(r.outputs.at("1")[0].asObject<ModelHandle>()->load.weightDtype == "fp8 e4m3 fast");

    r = run({{"1", {{"type", "loader.textEncoders"}, {"inputs", {{"encoder1", "clip"}}}}}});
    CHECK(r.outputs.at("1")[0].asObject<TextEncoderHandle>()->load.weightDtype == "bf16");
    r = run({{"1", {{"type", "loader.vae"}, {"inputs", {{"vae", "v"}}}}}});
    CHECK(r.outputs.at("1")[0].asObject<VaeHandle>()->load.weightDtype == "fp32");

    // without the settings the files keep their own format
    gh::host::LibraryNodeServices plain(library);
    RunOptions options;
    options.services = &plain;
    const JobGraph job = jobFromJson({{"format", "graphicshost.job"}, {"version", 1}, {"nodes", {{"1", {{"type", "loader.checkpoint"}, {"inputs", {{"checkpoint", "base"}}}}}}}, {"outputs", json::array({"1"})}});
    const auto plainRun = Executor(Registry::builtin(), cache).run(job, options);
    CHECK(plainRun.outputs.at("1")[0].asObject<ModelHandle>()->load.weightDtype == "default");
}
