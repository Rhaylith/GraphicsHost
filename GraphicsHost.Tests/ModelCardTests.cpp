#include "ModelFixture.h"
#include "TestUtil.h"

#include "models/ModelCard.h"

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <fstream>

using namespace gh::models;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

struct FakeTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype = "F16";
};

uint64_t dtypeSize(const std::string& d) { return d == "F32" ? 4 : d == "F16" || d == "BF16" ? 2 : 1; }

// A real, valid safetensors file with zero-filled tensors.
void writeSafetensors(const fs::path& file, const std::vector<FakeTensor>& tensors) {
    json header = json::object();
    header["__metadata__"] = {{"format", "pt"}};
    uint64_t offset = 0;
    for (const auto& t : tensors) {
        uint64_t n = 1;
        for (auto d : t.shape) n *= static_cast<uint64_t>(d);
        const uint64_t bytes = n * dtypeSize(t.dtype);
        header[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {offset, offset + bytes}}};
        offset += bytes;
    }
    const std::string text = header.dump();
    fs::create_directories(file.parent_path());
    std::ofstream out(file, std::ios::binary);
    const uint64_t length = text.size();
    out.write(reinterpret_cast<const char*>(&length), 8);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    const std::string zeros(4096, '\0');
    for (uint64_t written = 0; written < offset; written += zeros.size()) {
        out.write(zeros.data(), static_cast<std::streamsize>(std::min<uint64_t>(zeros.size(), offset - written)));
    }
}

std::vector<FakeTensor> sd1Tensors() {
    return {{"model.diffusion_model.input_blocks.0.0.weight", {320, 4, 3, 3}},
            {"model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight", {320, 768}},
            {"cond_stage_model.transformer.text_model.embeddings.token_embedding.weight", {128, 768}},
            {"first_stage_model.decoder.conv_in.weight", {512, 4, 3, 3}}};
}

CheckpointHeader load(const fs::path& file) {
    std::string error;
    const auto h = readSafetensorsHeader(file, &error);
    INFO(error);
    REQUIRE(h.has_value());
    return *h;
}

}  // namespace

TEST_CASE("header: safetensors tensors, shapes and metadata are read", "[unit][models]") {
    gh::test::TempDir dir;
    writeSafetensors(dir.path() / "a.safetensors", sd1Tensors());
    const auto h = load(dir.path() / "a.safetensors");
    CHECK(h.format == CheckpointHeader::Format::Safetensors);
    CHECK(h.tensors.size() == 4);
    CHECK(h.metadata["format"] == "pt");
    const auto* conv = h.find("model.diffusion_model.input_blocks.0.0.weight");
    REQUIRE(conv != nullptr);
    CHECK(conv->shape == std::vector<int64_t>{320, 4, 3, 3});
    CHECK(conv->bytes == 320u * 4 * 3 * 3 * 2);
    CHECK(conv->elements() == 320 * 4 * 3 * 3);
    CHECK(h.hasPrefix("cond_stage_model."));
    CHECK_FALSE(h.hasPrefix("conditioner."));
    CHECK(h.hasSubstring("transformer_blocks"));
}

TEST_CASE("header: damaged safetensors files are refused with a reason", "[unit][models]") {
    gh::test::TempDir dir;
    std::string error;

    gh::test::writeText(dir.path() / "short.safetensors", "abc");
    CHECK_FALSE(readSafetensorsHeader(dir.path() / "short.safetensors", &error).has_value());
    CHECK(error.find("short") != std::string::npos);

    { // a header length far beyond the file
        std::ofstream out(dir.path() / "huge.safetensors", std::ios::binary);
        const uint64_t length = 1ull << 40;
        out.write(reinterpret_cast<const char*>(&length), 8);
        out << "{}   ";
    }
    CHECK_FALSE(readSafetensorsHeader(dir.path() / "huge.safetensors", &error).has_value());
    CHECK(error.find("header length") != std::string::npos);

    { // not JSON
        const std::string text = "this is not json";
        std::ofstream out(dir.path() / "bad.safetensors", std::ios::binary);
        const uint64_t length = text.size();
        out.write(reinterpret_cast<const char*>(&length), 8);
        out << text;
    }
    CHECK_FALSE(readSafetensorsHeader(dir.path() / "bad.safetensors", &error).has_value());
    CHECK(error.find("JSON") != std::string::npos);

    { // a tensor that claims more data than the file has
        const std::string text = R"({"t":{"dtype":"F16","shape":[4],"data_offsets":[0,800]}})";
        std::ofstream out(dir.path() / "lies.safetensors", std::ios::binary);
        const uint64_t length = text.size();
        out.write(reinterpret_cast<const char*>(&length), 8);
        out << text << "12345678";
    }
    CHECK_FALSE(readSafetensorsHeader(dir.path() / "lies.safetensors", &error).has_value());
    CHECK(error.find("outside") != std::string::npos);

    CHECK_FALSE(readSafetensorsHeader(dir.path() / "missing.safetensors", &error).has_value());
    CHECK_FALSE(readCheckpointHeader(dir.path() / "x.ckpt", &error).has_value());
}

TEST_CASE("header: a minimal GGUF file", "[unit][models]") {
    gh::test::TempDir dir;
    const auto file = dir.path() / "m.gguf";
    {
        std::ofstream out(file, std::ios::binary);
        auto u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
        auto u64 = [&](uint64_t v) { out.write(reinterpret_cast<const char*>(&v), 8); };
        auto str = [&](const std::string& s) { u64(s.size()); out.write(s.data(), static_cast<std::streamsize>(s.size())); };
        out.write("GGUF", 4);
        u32(3);
        u64(1);   // tensors
        u64(2);   // key/values
        str("general.architecture"); u32(8); str("sdxl");
        str("general.file_type"); u32(4); u32(1);
        str("model.diffusion_model.input_blocks.0.0.weight"); u32(4); u64(3); u64(3); u64(4); u64(320); u32(1); u64(0);
        out << std::string(100, 'x');
    }
    std::string error;
    const auto h = readGgufHeader(file, &error);
    INFO(error);
    REQUIRE(h.has_value());
    CHECK(h->format == CheckpointHeader::Format::Gguf);
    CHECK(h->metadata["general.architecture"] == "sdxl");
    CHECK(h->metadata["general.file_type"] == 1);
    REQUIRE(h->tensors.size() == 1);
    CHECK(h->tensors[0].shape == std::vector<int64_t>{3, 3, 4, 320});
    CHECK(readCheckpointHeader(file).has_value());

    gh::test::writeText(dir.path() / "no.gguf", "NOPE");
    CHECK_FALSE(readGgufHeader(dir.path() / "no.gguf", &error).has_value());
}

TEST_CASE("family: SD 1.x, SD 2.x and SDXL are told apart", "[unit][models]") {
    gh::test::TempDir dir;
    auto classify = [&](const std::string& name, std::vector<FakeTensor> tensors) {
        writeSafetensors(dir.path() / (name + ".safetensors"), tensors);
        return detectFamily(load(dir.path() / (name + ".safetensors")));
    };

    auto sd1 = classify("sd1", sd1Tensors());
    CHECK(sd1.kind == Kind::Checkpoint);
    CHECK(sd1.family == Family::Sd1);
    CHECK(sd1.supported);
    CHECK_FALSE(sd1.inpainting);

    auto xl = sd1Tensors();
    xl.push_back({"conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight", {64, 64}});
    xl.push_back({"model.diffusion_model.label_emb.0.0.weight", {1280, 2816}});
    const auto sdxl = classify("sdxl", xl);
    CHECK(sdxl.family == Family::SdXl);
    CHECK(sdxl.supported);

    auto v2 = sd1Tensors();
    v2.push_back({"cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight", {64, 64}});
    const auto sd2 = classify("sd2", v2);
    CHECK(sd2.family == Family::Sd2);
    CHECK(sd2.supported);
    CHECK_FALSE(sd2.note.empty());  // flagged as untested

    auto inpaint = sd1Tensors();
    inpaint[0] = {"model.diffusion_model.input_blocks.0.0.weight", {320, 9, 3, 3}};
    CHECK(classify("inpaint", inpaint).inpainting);
}

TEST_CASE("family: newer families are recognised and reported as unsupported", "[unit][models]") {
    gh::test::TempDir dir;
    writeSafetensors(dir.path() / "flux.safetensors", {{"double_blocks.0.img_attn.qkv.weight", {16, 16}}, {"single_blocks.0.linear1.weight", {16, 16}}});
    const auto flux = detectFamily(load(dir.path() / "flux.safetensors"));
    CHECK(flux.family == Family::Flux);
    CHECK_FALSE(flux.supported);
    CHECK(flux.note.find("not supported") != std::string::npos);

    writeSafetensors(dir.path() / "sd3.safetensors", {{"model.diffusion_model.joint_blocks.0.context_block.attn.qkv.weight", {16, 16}}});
    const auto sd3 = detectFamily(load(dir.path() / "sd3.safetensors"));
    CHECK(sd3.family == Family::Sd3);
    CHECK_FALSE(sd3.supported);
}

TEST_CASE("family: VAE, LoRA and unknown files", "[unit][models]") {
    gh::test::TempDir dir;
    writeSafetensors(dir.path() / "vae.safetensors", {{"encoder.conv_in.weight", {128, 3, 3, 3}}, {"decoder.conv_in.weight", {512, 4, 3, 3}}, {"quant_conv.weight", {8, 8, 1, 1}}});
    const auto vae = detectFamily(load(dir.path() / "vae.safetensors"));
    CHECK(vae.kind == Kind::Vae);
    CHECK(vae.supported);

    writeSafetensors(dir.path() / "lora.safetensors", {{"lora_unet_down_blocks_0_attentions_0.lora_down.weight", {8, 320}}, {"lora_unet_down_blocks_0_attentions_0.lora_up.weight", {320, 8}}, {"lora_te_text_model.lora_down.weight", {8, 768}}});
    const auto lora = detectFamily(load(dir.path() / "lora.safetensors"));
    CHECK(lora.kind == Kind::Lora);
    CHECK(lora.family == Family::Sd1);

    writeSafetensors(dir.path() / "lora-xl.safetensors", {{"lora_unet_x.lora_down.weight", {8, 320}}, {"lora_te2_text_model.lora_down.weight", {8, 1280}}});
    CHECK(detectFamily(load(dir.path() / "lora-xl.safetensors")).family == Family::SdXl);

    writeSafetensors(dir.path() / "other.safetensors", {{"something.weight", {4, 4}}});
    const auto other = detectFamily(load(dir.path() / "other.safetensors"));
    CHECK(other.kind == Kind::Other);
    CHECK_FALSE(other.supported);
}

TEST_CASE("weights: measured load ratios for fp32 and fp16, other types keep their size", "[unit][models]") {
    gh::test::TempDir dir;
    writeSafetensors(dir.path() / "w.safetensors", {{"a", {1000}, "F32"}, {"b", {1000}, "F16"}, {"c", {1000}, "BF16"}, {"d", {1000}, "I8"}});
    const auto h = load(dir.path() / "w.safetensors");
    CHECK(estimateWeightsBytes(h) == static_cast<uint64_t>(4000 * 0.653 + 2000 * 0.955 + 2000 * 0.955 + 1000));
    CHECK(h.fileSize > 9000);  // the file holds 4000 + 2000 + 2000 + 1000 bytes of tensors plus the header
}

TEST_CASE("card: describeFile fills in what the application needs to know", "[unit][models]") {
    gh::test::TempDir dir;
    writeSafetensors(dir.path() / "my model.safetensors", sd1Tensors());
    const auto card = describeFile(dir.path() / "my model.safetensors");
    CHECK(card.name == "my model");
    CHECK(card.file == "my model.safetensors");
    CHECK(card.kind == Kind::Checkpoint);
    CHECK(card.familyName == "SD 1.x");
    CHECK(card.supported);
    CHECK(card.headerRead);
    CHECK(card.defaults.width == 512);
    CHECK(card.weightsBytes > 0);
    CHECK(toJson(card)["family"] == "SD 1.x");

    // .ckpt files are not inspected: a guess, flagged as such
    gh::test::writeText(dir.path() / "old.ckpt", "pickle");
    const auto ckpt = describeFile(dir.path() / "old.ckpt");
    CHECK_FALSE(ckpt.headerRead);
    CHECK(ckpt.supported);
    CHECK(ckpt.note.find("guessed") != std::string::npos);

    // damaged safetensors: unsupported, with the reason
    gh::test::writeText(dir.path() / "broken.safetensors", "xx");
    const auto broken = describeFile(dir.path() / "broken.safetensors");
    CHECK_FALSE(broken.supported);
    CHECK_FALSE(broken.problem.empty());
}

TEST_CASE("card: the real SD 1.5 and SDXL checkpoints", "[model][models]") {
    const auto sd15 = gh::test::findTestModel();
    const auto sdxl = gh::test::findTestModel("sd_xl_base_1.0.safetensors");
    if (!sd15 || !sdxl) SKIP("needs both checkpoints in models/");
    const auto a = describeFile(*sd15);
    CHECK(a.familyName == "SD 1.x");
    CHECK(a.kind == Kind::Checkpoint);
    CHECK(a.supported);
    // the engine reports 2784.45 MB for this file
    CHECK(std::abs(static_cast<double>(a.weightsBytes) / 1e6 - 2784.45) < 2784.45 * 0.02);
    const auto b = describeFile(*sdxl);
    CHECK(b.familyName == "SDXL");
    CHECK(std::abs(static_cast<double>(b.weightsBytes) / 1e6 - 6624.11) < 6624.11 * 0.02);
}
