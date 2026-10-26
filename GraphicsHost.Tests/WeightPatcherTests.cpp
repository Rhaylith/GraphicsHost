#include "ModelFixture.h"
#include "SafetensorsFixture.h"
#include "TestUtil.h"
#include "models/CheckpointHeader.h"
#include "models/SafetensorsReader.h"
#include "models/WeightPatcher.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstring>
#include <fstream>

using namespace gh::models;
using gh::tensor::Tensor;
using Catch::Approx;

namespace {

// Numbers that are easy to follow: 0.1, 0.2 ... repeating, shifted per tensor.
Tensor filled(std::vector<int64_t> shape, int seed) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = 0.1f * static_cast<float>((i * 7 + seed * 3) % 11) - 0.5f;
    return Tensor(std::move(shape), std::move(v));
}

Tensor scalar(float v) { return Tensor({1}, std::vector<float>{v}); }

AdapterModule module(const std::string& name, std::map<std::string, Tensor> tensors, std::optional<double> alpha = std::nullopt) {
    AdapterModule m;
    m.name = name;
    m.tensors = std::move(tensors);
    m.alpha = alpha;
    return m;
}

// The same sums written the slow way, so a mistake in the fast code cannot hide in a shared helper.
Tensor naiveLinearLora(const Tensor& down, const Tensor& up, double scale) {
    const int64_t out = up.dim(0), r = up.dim(1), in = down.dim(1);
    Tensor w({out, in});
    for (int64_t o = 0; o < out; ++o)
        for (int64_t i = 0; i < in; ++i) {
            double sum = 0;
            for (int64_t k = 0; k < r; ++k) sum += static_cast<double>(up.at({o, k})) * down.at({k, i});
            w.at({o, i}) = static_cast<float>(sum * scale);
        }
    return w;
}

void sameValues(const Tensor& a, const Tensor& b, float eps = 1e-5f) {
    REQUIRE(a.numel() == b.numel());
    for (int64_t i = 0; i < a.numel(); ++i) {
        if (std::fabs(a.data()[i] - b.data()[i]) > eps) {
            FAIL("values differ at " << i << ": " << a.data()[i] << " vs " << b.data()[i]);
        }
    }
}

// a weight store backed by a map, counting writes
class MapTarget final : public WeightTarget {
public:
    std::map<std::string, Tensor> weights;
    int writes = 0;
    std::vector<std::string> keys() const override {
        std::vector<std::string> k;
        for (const auto& [name, t] : weights) k.push_back(name);
        return k;
    }
    bool read(const std::string& key, Tensor& out) const override {
        const auto it = weights.find(key);
        if (it == weights.end()) return false;
        out = it->second;
        return true;
    }
    void write(const std::string& key, const Tensor& value) override {
        weights[key] = value;
        ++writes;
    }
};
}  // namespace

TEST_CASE("patcher: LoRA on a linear layer is up @ down times alpha over rank", "[unit][patcher]") {
    const Tensor down = filled({3, 5}, 1), up = filled({4, 3}, 2);
    const AdapterModule m = module("lora_unet_x", {{"down", down}, {"up", up}}, 6.0);
    sameValues(adapterDelta(m, {4, 5}), naiveLinearLora(down, up, 6.0 / 3.0));
    const AdapterModule noAlpha = module("lora_unet_x", {{"down", down}, {"up", up}});
    sameValues(adapterDelta(noAlpha, {4, 5}), naiveLinearLora(down, up, 1.0));   // no alpha: scale 1
    CHECK_THROWS_WITH(adapterDelta(m, {4, 6}), Catch::Matchers::ContainsSubstring("changes 20 values"));
    CHECK_THROWS_WITH(adapterDelta(module("m", {{"junk", down}}), {4, 5}), Catch::Matchers::ContainsSubstring("no tensors that change a weight"));
}

TEST_CASE("patcher: LoCon on a convolution, plain and with a Tucker core", "[unit][patcher]") {
    // plain: down [r, in, k, k], up [out, r, 1, 1]
    const Tensor down = filled({2, 3, 3, 3}, 1), up = filled({4, 2, 1, 1}, 2);
    const Tensor plain = adapterDelta(module("m", {{"down", down}, {"up", up}}, 2.0), {4, 3, 3, 3});
    for (int64_t o = 0; o < 4; ++o)
        for (int64_t i = 0; i < 3; ++i)
            for (int64_t kk = 0; kk < 9; ++kk) {
                double sum = 0;
                for (int64_t r = 0; r < 2; ++r) sum += static_cast<double>(up.at({o, r, 0, 0})) * down.data()[(r * 3 + i) * 9 + kk];
                CHECK(plain.data()[(o * 3 + i) * 9 + kk] == Approx(sum).margin(1e-6));   // alpha 2 / rank 2 = 1
            }

    // Tucker: down [r, in, 1, 1], mid [r, r, k, k], up [out, r, 1, 1]: W[o,i,k] = sum_ab up[o,b] mid[b,a,k] down[a,i]
    const Tensor d2 = filled({2, 3, 1, 1}, 3), mid = filled({2, 2, 3, 3}, 4);
    const Tensor tuck = adapterDelta(module("m", {{"down", d2}, {"mid", mid}, {"up", up}}), {4, 3, 3, 3});
    for (int64_t o = 0; o < 4; ++o)
        for (int64_t i = 0; i < 3; ++i)
            for (int64_t kk = 0; kk < 9; ++kk) {
                double sum = 0;
                for (int64_t a = 0; a < 2; ++a)
                    for (int64_t b = 0; b < 2; ++b) sum += static_cast<double>(up.at({o, b, 0, 0})) * mid.data()[(b * 2 + a) * 9 + kk] * d2.at({a, i, 0, 0});
                CHECK(tuck.data()[(o * 3 + i) * 9 + kk] == Approx(sum).margin(1e-5));
            }
}

TEST_CASE("patcher: LoHa is the element-wise product of two low-rank matrices", "[unit][patcher]") {
    const Tensor w1a = filled({4, 2}, 1), w1b = filled({2, 5}, 2), w2a = filled({4, 3}, 3), w2b = filled({3, 5}, 4);
    const Tensor delta = adapterDelta(module("m", {{"hada_w1_a", w1a}, {"hada_w1_b", w1b}, {"hada_w2_a", w2a}, {"hada_w2_b", w2b}}, 4.0), {4, 5});
    const Tensor first = naiveLinearLora(w1b, w1a, 1.0), second = naiveLinearLora(w2b, w2a, 1.0);   // (w1a @ w1b), (w2a @ w2b)
    for (int64_t i = 0; i < delta.numel(); ++i) CHECK(delta.data()[i] == Approx(first.data()[i] * second.data()[i] * (4.0 / 2.0)).margin(1e-6));   // alpha / rank of w1_b

    // Tucker for convolutions: t [r, r, k, k], a [r, out], b [r, in]
    const Tensor t1 = filled({2, 2, 3, 3}, 5), t2 = filled({2, 2, 3, 3}, 6), a1 = filled({2, 4}, 7), b1 = filled({2, 3}, 8), a2 = filled({2, 4}, 9), b2 = filled({2, 3}, 10);
    const Tensor conv = adapterDelta(module("m", {{"hada_t1", t1}, {"hada_w1_a", a1}, {"hada_w1_b", b1}, {"hada_t2", t2}, {"hada_w2_a", a2}, {"hada_w2_b", b2}}), {4, 3, 3, 3});
    const auto cp = [](const Tensor& t, const Tensor& a, const Tensor& b, int64_t p, int64_t r, int64_t kk) {
        double sum = 0;
        for (int64_t i = 0; i < 2; ++i)
            for (int64_t j = 0; j < 2; ++j) sum += static_cast<double>(t.data()[(i * 2 + j) * 9 + kk]) * a.at({i, p}) * b.at({j, r});
        return sum;
    };
    for (int64_t p = 0; p < 4; ++p)
        for (int64_t r = 0; r < 3; ++r)
            for (int64_t kk = 0; kk < 9; ++kk) CHECK(conv.data()[(p * 3 + r) * 9 + kk] == Approx(cp(t1, a1, b1, p, r, kk) * cp(t2, a2, b2, p, r, kk)).margin(1e-4));
}

TEST_CASE("patcher: LoKr is a Kronecker product, with full or decomposed factors", "[unit][patcher]") {
    const Tensor w1 = filled({2, 3}, 1), w2 = filled({4, 2}, 2);
    const Tensor full = adapterDelta(module("m", {{"lokr_w1", w1}, {"lokr_w2", w2}}, 5.0), {8, 6});   // both full: alpha is not used
    for (int64_t i = 0; i < 2; ++i)
        for (int64_t j = 0; j < 3; ++j)
            for (int64_t p = 0; p < 4; ++p)
                for (int64_t q = 0; q < 2; ++q) CHECK(full.at({i * 4 + p, j * 2 + q}) == Approx(w1.at({i, j}) * w2.at({p, q})));

    const Tensor a = filled({4, 2}, 3), b = filled({2, 2}, 4);   // w2 = a @ b, rank 2
    const Tensor dec = adapterDelta(module("m", {{"lokr_w1", w1}, {"lokr_w2_a", a}, {"lokr_w2_b", b}}, 4.0), {8, 6});
    const Tensor w2d = naiveLinearLora(b, a, 1.0);
    for (int64_t i = 0; i < 2; ++i)
        for (int64_t j = 0; j < 3; ++j)
            for (int64_t p = 0; p < 4; ++p)
                for (int64_t q = 0; q < 2; ++q) CHECK(dec.at({i * 4 + p, j * 2 + q}) == Approx(w1.at({i, j}) * w2d.at({p, q}) * (4.0 / 2.0)));

    // a convolution keeps the kernel dimensions of the second factor
    const Tensor w2c = filled({4, 2, 3, 3}, 5);
    const Tensor conv = adapterDelta(module("m", {{"lokr_w1", w1}, {"lokr_w2", w2c}}), {8, 6, 3, 3});
    CHECK(conv.at({1 * 4 + 2, 2 * 2 + 1, 1, 2}) == Approx(w1.at({1, 2}) * w2c.at({2, 1, 1, 2})));
}

TEST_CASE("patcher: a weight difference adds itself", "[unit][patcher]") {
    const Tensor diff = filled({3, 4}, 1);
    sameValues(adapterDelta(module("m", {{"diff", diff}}), {3, 4}), diff);
}

TEST_CASE("unet paths: the original layout maps to the diffusers names", "[unit][patcher][keys]") {
    CHECK(diffusersUnetPath("input_blocks.1.1.transformer_blocks.0.attn1.to_q") == "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q");
    CHECK(diffusersUnetPath("input_blocks.5.1.proj_in") == "down_blocks.1.attentions.1.proj_in");
    CHECK(diffusersUnetPath("input_blocks.2.0.in_layers.2") == "down_blocks.0.resnets.1.conv1");
    CHECK(diffusersUnetPath("input_blocks.4.0.emb_layers.1") == "down_blocks.1.resnets.0.time_emb_proj");
    CHECK(diffusersUnetPath("input_blocks.3.0.op") == "down_blocks.0.downsamplers.0.conv");
    CHECK(diffusersUnetPath("input_blocks.9.0.op") == "down_blocks.2.downsamplers.0.conv");
    CHECK(diffusersUnetPath("input_blocks.11.0.out_layers.3") == "down_blocks.3.resnets.1.conv2");   // the last level has two resnets
    CHECK(diffusersUnetPath("input_blocks.4.0.skip_connection") == "down_blocks.1.resnets.0.conv_shortcut");
    CHECK(diffusersUnetPath("middle_block.0.in_layers.0") == "mid_block.resnets.0.norm1");
    CHECK(diffusersUnetPath("middle_block.1.transformer_blocks.0.attn2.to_k") == "mid_block.attentions.0.transformer_blocks.0.attn2.to_k");
    CHECK(diffusersUnetPath("middle_block.2.out_layers.0") == "mid_block.resnets.1.norm2");
    CHECK(diffusersUnetPath("output_blocks.0.0.in_layers.2") == "up_blocks.0.resnets.0.conv1");
    CHECK(diffusersUnetPath("output_blocks.2.1.conv") == "up_blocks.0.upsamplers.0.conv");   // no attention in the first up block
    CHECK(diffusersUnetPath("output_blocks.4.1.transformer_blocks.0.ff.net.2") == "up_blocks.1.attentions.1.transformer_blocks.0.ff.net.2");
    CHECK(diffusersUnetPath("output_blocks.5.2.conv") == "up_blocks.1.upsamplers.0.conv");
    CHECK(diffusersUnetPath("output_blocks.11.1.proj_out") == "up_blocks.3.attentions.2.proj_out");
    CHECK(diffusersUnetPath("time_embed.0") == "time_embedding.linear_1");
    CHECK(diffusersUnetPath("label_emb.0.2") == "add_embedding.linear_2");
    CHECK(diffusersUnetPath("input_blocks.0.0") == "conv_in");
    CHECK(diffusersUnetPath("out.2") == "conv_out");
    CHECK(diffusersUnetPath("something.else") == "something.else");
}

TEST_CASE("key resolver: kohya, diffusers and PEFT names find the same weights", "[unit][patcher][keys]") {
    const std::vector<std::string> keys = {
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
        "model.diffusion_model.input_blocks.1.0.in_layers.2.weight",
        "model.diffusion_model.output_blocks.2.1.conv.weight",
        "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.q_proj.weight",
        "conditioner.embedders.0.transformer.text_model.encoder.layers.1.mlp.fc1.weight",
        "conditioner.embedders.1.model.transformer.resblocks.4.attn.in_proj_weight",
        "conditioner.embedders.1.model.transformer.resblocks.4.attn.out_proj.weight",
        "conditioner.embedders.1.model.transformer.resblocks.4.mlp.c_fc.weight",
        "conditioner.embedders.1.model.transformer.resblocks.4.mlp.c_proj.weight",
    };
    const KeyResolver r(keys);
    const std::string q = "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight";
    for (const char* name : {"lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q", "lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q",
                             "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q", "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q",
                             "base_model.model.unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"}) {
        INFO(name);
        const auto hit = r.resolve(name);
        REQUIRE(hit.has_value());
        CHECK(hit->key == q);
        CHECK(hit->parts == 1);
    }
    CHECK(r.resolve("lora_unet_down_blocks_0_resnets_0_conv1")->key == "model.diffusion_model.input_blocks.1.0.in_layers.2.weight");
    CHECK(r.resolve("lora_unet_up_blocks_0_upsamplers_0_conv")->key == "model.diffusion_model.output_blocks.2.1.conv.weight");
    CHECK(r.resolve("lora_te_text_model_encoder_layers_3_self_attn_q_proj")->key == keys[3]);
    CHECK(r.resolve("text_encoder.text_model.encoder.layers.3.self_attn.q_proj")->key == keys[3]);
    CHECK(r.resolve("lora_te1_text_model_encoder_layers_1_mlp_fc1")->key == keys[4]);
    CHECK(r.resolve("lora_te2_text_model_encoder_layers_4_self_attn_out_proj")->key == keys[6]);
    CHECK(r.resolve("lora_te2_text_model_encoder_layers_4_mlp_fc1")->key == keys[7]);
    CHECK(r.resolve("lora_te2_text_model_encoder_layers_4_mlp_fc2")->key == keys[8]);
    for (int part = 0; part < 3; ++part) {
        const char* names[] = {"q_proj", "k_proj", "v_proj"};
        const auto hit = r.resolve(std::string("lora_te2_text_model_encoder_layers_4_self_attn_") + names[part]);
        REQUIRE(hit.has_value());
        CHECK(hit->key == keys[5]);
        CHECK(hit->part == part);
        CHECK(hit->parts == 3);
    }
    // the weight itself, for "diff" files
    CHECK(r.resolve("diffusion_model.input_blocks.1.0.in_layers.2.weight")->key == "model.diffusion_model.input_blocks.1.0.in_layers.2.weight");
    CHECK_FALSE(r.resolve("lora_unet_down_blocks_9_attentions_0_proj_in").has_value());
    CHECK_FALSE(r.resolve("nonsense").has_value());
}

TEST_CASE("patcher: apply adds the change, undo gives the original back, a new list starts from the original", "[unit][patcher]") {
    MapTarget target;
    target.weights["model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"] = filled({4, 5}, 9);
    target.weights["model.diffusion_model.input_blocks.1.1.proj_in.weight"] = filled({4, 4}, 1);
    target.weights["cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"] = filled({4, 5}, 4);
    const auto original = target.weights;

    const Tensor down = filled({2, 5}, 1), up = filled({4, 2}, 2);
    const Tensor down2 = filled({2, 5}, 5), up2 = filled({4, 2}, 6);
    const auto first = AdapterFile::fromTensors({{"lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight", down},
                                                 {"lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight", up},
                                                 {"lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.alpha", scalar(2)},
                                                 {"lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight", down2},
                                                 {"lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight", up2},
                                                 {"lora_unet_down_blocks_7_resnets_0_conv1.lora_down.weight", down},
                                                 {"lora_unet_down_blocks_7_resnets_0_conv1.lora_up.weight", up}},
                                                "first");
    REQUIRE(first->modules.size() == 3);

    WeightPatcher patcher;
    PatchSpec spec{first, 0.5, 2.0};
    const auto diffusion = patcher.apply(target, TargetKind::Diffusion, {spec});
    CHECK(diffusion.modules == 2);   // the text module is meant for the other component
    CHECK(diffusion.patched == 1);
    CHECK(diffusion.unmatched == std::vector<std::string>{"lora_unet_down_blocks_7_resnets_0_conv1"});
    const std::string qKey = "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight";
    const Tensor expected = [&] {
        Tensor w = original.at(qKey);
        const Tensor delta = naiveLinearLora(down, up, 2.0 / 2.0);
        for (int64_t i = 0; i < w.numel(); ++i) w.data()[i] += 0.5f * delta.data()[i];   // strengthModel 0.5
        return w;
    }();
    sameValues(target.weights.at(qKey), expected);
    sameValues(target.weights.at("model.diffusion_model.input_blocks.1.1.proj_in.weight"), original.at("model.diffusion_model.input_blocks.1.1.proj_in.weight"));
    sameValues(target.weights.at("cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"), original.at("cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"));
    CHECK(patcher.applied());

    // lazy: the same list again touches nothing
    const int writes = target.writes;
    patcher.ensure(target, TargetKind::Diffusion, {spec});
    CHECK(target.writes == writes);
    // a different strength starts again from the originals, it does not pile up
    PatchSpec stronger{first, 1.0, 2.0};
    patcher.ensure(target, TargetKind::Diffusion, {stronger});
    Tensor twice = original.at(qKey);
    const Tensor delta = naiveLinearLora(down, up, 1.0);
    for (int64_t i = 0; i < twice.numel(); ++i) twice.data()[i] += delta.data()[i];
    sameValues(target.weights.at(qKey), twice);

    patcher.undo(target);
    CHECK_FALSE(patcher.applied());
    for (const auto& [key, w] : original) sameValues(target.weights.at(key), w, 0.0f);

    // the text encoder uses strengthText
    const auto text = patcher.apply(target, TargetKind::TextEncoder, {spec});
    CHECK(text.patched == 1);
    const std::string tKey = "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight";
    Tensor te = original.at(tKey);
    const Tensor tdelta = naiveLinearLora(down2, up2, 1.0);   // no alpha
    for (int64_t i = 0; i < te.numel(); ++i) te.data()[i] += 2.0f * tdelta.data()[i];
    sameValues(target.weights.at(tKey), te);
    patcher.undo(target);

    // two adapters add up; strength 0 skips a module; an empty list is a plain undo
    const auto second = AdapterFile::fromTensors({{"lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q.lora_down.weight", down2},
                                                  {"lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q.lora_up.weight", up2}},
                                                 "second");
    patcher.apply(target, TargetKind::Diffusion, {PatchSpec{first, 1.0, 1.0}, PatchSpec{second, 1.0, 1.0}, PatchSpec{second, 0.0, 0.0}});
    Tensor both = original.at(qKey);
    const Tensor d1 = naiveLinearLora(down, up, 1.0), d2 = naiveLinearLora(down2, up2, 1.0);
    for (int64_t i = 0; i < both.numel(); ++i) both.data()[i] += d1.data()[i] + d2.data()[i];
    sameValues(target.weights.at(qKey), both);
    patcher.apply(target, TargetKind::Diffusion, {});
    sameValues(target.weights.at(qKey), original.at(qKey), 0.0f);
}

TEST_CASE("patcher: packed attention weights are changed one third at a time, and DoRA rescales rows", "[unit][patcher]") {
    MapTarget target;
    const std::string packed = "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight";
    target.weights[packed] = filled({6, 4}, 3);   // q, k, v of 2 rows each
    const Tensor original = target.weights[packed];
    const Tensor down = filled({1, 4}, 1), up = filled({2, 1}, 2);
    const auto file = AdapterFile::fromTensors({{"lora_te2_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight", down},
                                                {"lora_te2_text_model_encoder_layers_0_self_attn_k_proj.lora_up.weight", up}});
    WeightPatcher patcher;
    CHECK(patcher.apply(target, TargetKind::TextEncoder, {PatchSpec{file, 1.0, 1.0}}).patched == 1);
    const Tensor delta = naiveLinearLora(down, up, 1.0);
    for (int64_t row = 0; row < 6; ++row)
        for (int64_t c = 0; c < 4; ++c) {
            const float extra = row >= 2 && row < 4 ? delta.at({row - 2, c}) : 0.0f;   // only the k rows move
            CHECK(target.weights[packed].at({row, c}) == Approx(original.at({row, c}) + extra));
        }

    // DoRA: the row norm becomes the stored magnitude
    MapTarget dora;
    dora.weights["model.diffusion_model.input_blocks.1.1.proj_in.weight"] = filled({3, 4}, 5);
    const Tensor w = dora.weights.begin()->second;
    const Tensor d = filled({1, 4}, 2), u = filled({3, 1}, 3), magnitude = filled({3, 1}, 6);
    const auto doraFile = AdapterFile::fromTensors({{"lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight", d},
                                                    {"lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight", u},
                                                    {"lora_unet_down_blocks_0_attentions_0_proj_in.dora_scale", magnitude}});
    WeightPatcher p2;
    p2.apply(dora, TargetKind::Diffusion, {PatchSpec{doraFile, 1.0, 1.0}});
    const Tensor delta2 = naiveLinearLora(d, u, 1.0);
    for (int64_t r = 0; r < 3; ++r) {
        double norm = 0;
        for (int64_t c = 0; c < 4; ++c) norm += std::pow(static_cast<double>(w.at({r, c})) + delta2.at({r, c}), 2);
        norm = std::sqrt(norm);
        for (int64_t c = 0; c < 4; ++c) CHECK(dora.weights.begin()->second.at({r, c}) == Approx((w.at({r, c}) + delta2.at({r, c})) * magnitude.data()[r] / norm).margin(1e-5));
    }
    // half strength sits halfway between the original and the DoRA weight
    p2.apply(dora, TargetKind::Diffusion, {PatchSpec{doraFile, 0.5, 0.5}});
    double norm0 = 0;
    for (int64_t c = 0; c < 4; ++c) norm0 += std::pow(static_cast<double>(w.at({0, c})) + delta2.at({0, c}), 2);
    const double full = (w.at({0, 0}) + delta2.at({0, 0})) * magnitude.data()[0] / std::sqrt(norm0);
    CHECK(dora.weights.begin()->second.at({0, 0}) == Approx(0.5 * w.at({0, 0}) + 0.5 * full).margin(1e-5));
}

TEST_CASE("patcher: weight differences, bias differences and unreadable weights", "[unit][patcher]") {
    MapTarget target;
    target.weights["model.diffusion_model.out.2.weight"] = filled({3, 3}, 1);
    target.weights["model.diffusion_model.out.2.bias"] = filled({3}, 2);
    const Tensor w = target.weights["model.diffusion_model.out.2.weight"], b = target.weights["model.diffusion_model.out.2.bias"];
    const Tensor diff = filled({3, 3}, 7), diffB = filled({3}, 8);
    const auto file = AdapterFile::fromTensors({{"diffusion_model.out.2.weight.diff", diff}, {"diffusion_model.out.2.diff_b", diffB}});
    WeightPatcher patcher;
    const auto report = patcher.apply(target, TargetKind::Diffusion, {PatchSpec{file, 0.5, 0.5}});
    CHECK(report.patched >= 1);
    for (int64_t i = 0; i < 9; ++i) CHECK(target.weights["model.diffusion_model.out.2.weight"].data()[i] == Approx(w.data()[i] + 0.5f * diff.data()[i]));
    patcher.undo(target);
    for (int64_t i = 0; i < 9; ++i) CHECK(target.weights["model.diffusion_model.out.2.weight"].data()[i] == w.data()[i]);
    for (int64_t i = 0; i < 3; ++i) CHECK(target.weights["model.diffusion_model.out.2.bias"].data()[i] == b.data()[i]);
}

TEST_CASE("safetensors reader: float types are converted and files become adapters", "[unit][patcher][reader]") {
    gh::test::TempDir dir;
    const auto file = dir.path() / "lora.safetensors";
    // F32 [2,2], F16 [3] (1.0, -2.0, 0.5), BF16 [2] (1.0, 3.0), F32 scalar alpha
    std::string data;
    const auto put = [&](const void* p, size_t n) { data.append(static_cast<const char*>(p), n); };
    const float f32[] = {1, 2, 3, 4};
    const uint16_t f16[] = {0x3C00, 0xC000, 0x3800};
    const uint16_t bf16[] = {0x3F80, 0x4040};
    const float alpha = 8;
    put(f32, 16);
    put(f16, 6);
    put(bf16, 4);
    put(&alpha, 4);
    nlohmann::json header = {{"__metadata__", {{"format", "pt"}}},
                             {"lora_unet_x.lora_down.weight", {{"dtype", "F32"}, {"shape", {2, 2}}, {"data_offsets", {0, 16}}}},
                             {"half", {{"dtype", "F16"}, {"shape", {3}}, {"data_offsets", {16, 22}}}},
                             {"brain", {{"dtype", "BF16"}, {"shape", {2}}, {"data_offsets", {22, 26}}}},
                             {"lora_unet_x.alpha", {{"dtype", "F32"}, {"shape", nlohmann::json::array()}, {"data_offsets", {26, 30}}}}};
    const std::string text = header.dump();
    {
        std::ofstream out(file, std::ios::binary);
        const uint64_t length = text.size();
        out.write(reinterpret_cast<const char*>(&length), 8);
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
    }
    const auto tensors = readAllSafetensors(file);
    REQUIRE(tensors.size() == 4);
    CHECK(tensors.at("lora_unet_x.lora_down.weight").at({1, 0}) == 3.0f);
    CHECK(tensors.at("half").data()[0] == 1.0f);
    CHECK(tensors.at("half").data()[1] == -2.0f);
    CHECK(tensors.at("half").data()[2] == 0.5f);
    CHECK(tensors.at("brain").data()[1] == 3.0f);
    CHECK(tensors.at("lora_unet_x.alpha").shape() == std::vector<int64_t>{1});   // a scalar gets the shape {1}

    const auto adapter = AdapterFile::load(file);
    REQUIRE(adapter != nullptr);
    REQUIRE(adapter->modules.size() == 1);
    CHECK(adapter->modules[0].name == "lora_unet_x");
    CHECK(adapter->modules[0].alpha == 8.0);
    CHECK(adapter->modules[0].tensors.count("down") == 1);
    CHECK(adapter->identity.find("lora.safetensors") != std::string::npos);

    std::string error;
    CHECK(AdapterFile::load(dir.path() / "missing.safetensors", &error) == nullptr);
    CHECK_FALSE(error.empty());
    gh::test::TempDir other;
    std::vector<gh::test::st::FakeTensor> junk = {{"model.diffusion_model.x.weight", {2, 2}}};
    gh::test::st::writeSafetensors(other.path() / "junk.safetensors", junk);
    CHECK(AdapterFile::load(other.path() / "junk.safetensors", &error) == nullptr);
    CHECK(error.find("no LoRA") != std::string::npos);

    CHECK(AdapterFile::targetOf("lora_unet_down_blocks_0") == TargetKind::Diffusion);
    CHECK(AdapterFile::targetOf("lora_te2_text_model") == TargetKind::TextEncoder);
    CHECK(AdapterFile::targetOf("text_encoder.text_model.x") == TargetKind::TextEncoder);
    CHECK_FALSE(AdapterFile::targetOf("something").has_value());
}

TEST_CASE("key resolver: the layer names of a real SD 1.5 checkpoint are all found", "[model][patcher][keys]") {
    const auto path = gh::test::findTestModel();
    if (!path) SKIP("no SD 1.5 checkpoint available");
    const auto header = readCheckpointHeader(*path);
    REQUIRE(header.has_value());
    std::vector<std::string> keys;
    for (const auto& t : header->tensors) keys.push_back(t.name);
    const KeyResolver resolver(keys);

    // every attention and feed-forward weight of the UNet and of the text encoder has a diffusers-style kohya name
    int unet = 0, text = 0;
    for (const auto& t : header->tensors) {
        const std::string& k = t.name;
        const bool linear = t.shape.size() == 2 || (t.shape.size() == 4 && t.shape[2] == 1);
        if (k.rfind("model.diffusion_model.", 0) == 0 && k.size() > 7 && k.substr(k.size() - 7) == ".weight" && k.find("transformer_blocks") != std::string::npos && linear &&
            k.find("norm") == std::string::npos) {
            const std::string rel = k.substr(22, k.size() - 22 - 7);
            std::string alias = "lora_unet_" + diffusersUnetPath(rel);
            std::replace(alias.begin(), alias.end(), '.', '_');
            const auto hit = resolver.resolve(alias);
            INFO(k << " -> " << alias);
            REQUIRE(hit.has_value());
            CHECK(hit->key == k);
            ++unet;
        }
        if (k.rfind("cond_stage_model.transformer.text_model.encoder.layers.", 0) == 0 && k.size() > 7 && k.substr(k.size() - 7) == ".weight" && t.shape.size() == 2) {
            std::string alias = "lora_te_" + k.substr(29, k.size() - 29 - 7);
            std::replace(alias.begin(), alias.end(), '.', '_');
            REQUIRE(resolver.resolve(alias).has_value());
            ++text;
        }
    }
    CHECK(unet > 100);   // SD 1.5: 16 transformer blocks with 10 weights each
    CHECK(text == 12 * 6);
}
