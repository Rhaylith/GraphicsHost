#include "SafetensorsFixture.h"
#include "TestUtil.h"
#include "core/AtomicFile.h"
#include "core/Sha256.h"
#include "models/ModelLibrary.h"
#include "models/ModelPaths.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::models;
using gh::test::st::FakeTensor;
using gh::test::st::writeSafetensors;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {
Kind kindOf(const fs::path& dir, const std::string& name, const std::vector<FakeTensor>& tensors) {
    writeSafetensors(dir / name, tensors);
    return describeFile(dir / name).kind;
}
void touch(const fs::path& p, const std::string& content = "x") {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << content;
}
}  // namespace

TEST_CASE("kinds: folder names and kind names map both ways", "[unit][models][kinds]") {
    CHECK(std::string(folderName(Kind::Lora)) == "loras");
    CHECK(std::string(folderName(Kind::PreviewDecoder)) == "vae_approx");
    CHECK(std::string(folderName(Kind::Other)).empty());
    CHECK(kindFromFolder("loras") == Kind::Lora);
    CHECK(kindFromFolder("LoRAs") == Kind::Lora);
    CHECK(kindFromFolder("lora") == Kind::Lora);
    CHECK(kindFromFolder("text_encoders") == Kind::TextEncoder);
    CHECK(kindFromFolder("diffusion_model") == Kind::DiffusionModel);
    CHECK_FALSE(kindFromFolder("other").has_value());
    CHECK_FALSE(kindFromFolder("publisher").has_value());
    for (const Kind k : libraryKinds()) {
        CHECK(kindFromFolder(folderName(k)) == k);
        CHECK(kindFromFolder(toString(k)) == k);
    }
    CHECK(libraryKinds().size() == 14);
}

TEST_CASE("kinds: headers tell the kinds apart", "[unit][models][kinds]") {
    gh::test::TempDir dir;
    CHECK(kindOf(dir.path(), "sd.safetensors", gh::test::st::sd1Tensors()) == Kind::Checkpoint);
    CHECK(kindOf(dir.path(), "cn.safetensors", {{"control_model.input_blocks.0.0.weight", {320, 4, 3, 3}}, {"control_model.zero_convs.0.0.weight", {320, 320, 1, 1}}}) == Kind::ControlNet);
    CHECK(kindOf(dir.path(), "t2i.safetensors", {{"adapter.body.0.block1.weight", {320, 320, 3, 3}}}) == Kind::ControlNet);
    CHECK(kindOf(dir.path(), "clip-l.safetensors", {{"text_model.embeddings.token_embedding.weight", {128, 768}}, {"text_model.encoder.layers.0.mlp.fc1.weight", {3072, 768}}}) == Kind::TextEncoder);
    CHECK(kindOf(dir.path(), "t5.safetensors", {{"encoder.block.0.layer.0.SelfAttention.q.weight", {64, 64}}, {"shared.weight", {32128, 64}}}) == Kind::TextEncoder);
    CHECK(kindOf(dir.path(), "vision.safetensors", {{"vision_model.encoder.layers.0.mlp.fc1.weight", {4096, 1024}}, {"vision_model.embeddings.class_embedding", {1024}}}) == Kind::ClipVision);
    CHECK(kindOf(dir.path(), "redux.safetensors", {{"redux_down.weight", {4096, 1152}}, {"redux_up.weight", {1152, 4096}}, {"redux_up.bias", {1152}}, {"redux_down.bias", {4096}}}) == Kind::StyleModel);
    CHECK(kindOf(dir.path(), "taesd.safetensors", {{"decoder.layers.0.weight", {64, 4, 3, 3}}, {"encoder.layers.0.weight", {64, 3, 3, 3}}}) == Kind::PreviewDecoder);
    CHECK(kindOf(dir.path(), "esrgan.safetensors", {{"model.1.sub.0.RDB1.conv1.0.weight", {32, 64, 3, 3}}, {"model.0.weight", {64, 3, 3, 3}}}) == Kind::UpscaleModel);
    CHECK(kindOf(dir.path(), "gligen.safetensors", {{"model.diffusion_model.position_net.linears.0.weight", {512, 1024}}, {"model.diffusion_model.fuser.gated.weight", {64, 64}}}) == Kind::Gligen);
    CHECK(kindOf(dir.path(), "pm.safetensors", {{"id_encoder.vision_model.embeddings.weight", {64, 64}}, {"id_encoder.fuse_module.weight", {64, 64}}}) == Kind::PhotoMaker);
    CHECK(kindOf(dir.path(), "vae.safetensors", {{"decoder.conv_in.weight", {512, 4, 3, 3}}, {"encoder.conv_in.weight", {128, 3, 3, 3}}}) == Kind::Vae);
    CHECK(kindOf(dir.path(), "lora.safetensors", {{"lora_unet_down_blocks_0.lora_down.weight", {4, 320}}, {"lora_unet_down_blocks_0.lora_up.weight", {320, 4}}}) == Kind::Lora);
    // new families: only the weights of the network is a diffusion model; with the text encoder and VAE inside it is a checkpoint
    CHECK(kindOf(dir.path(), "flux-dit.safetensors", {{"double_blocks.0.img_attn.qkv.weight", {64, 64}}, {"single_blocks.0.linear1.weight", {64, 64}}}) == Kind::DiffusionModel);
    CHECK(kindOf(dir.path(), "flux-all.safetensors", {{"double_blocks.0.img_attn.qkv.weight", {64, 64}}, {"single_blocks.0.linear1.weight", {64, 64}}, {"vae.decoder.conv_in.weight", {512, 16, 3, 3}}}) == Kind::Checkpoint);
    CHECK(kindOf(dir.path(), "sd35.safetensors", {{"joint_blocks.0.x_block.attn.qkv.weight", {64, 64}}}) == Kind::DiffusionModel);
    CHECK(kindOf(dir.path(), "junk.safetensors", {{"something.else", {2, 2}}}) == Kind::Other);
}

TEST_CASE("library: kind folders decide, names are relative, previews and pickled files are found", "[unit][library][kinds]") {
    gh::test::TempDir dir;
    const fs::path root = dir.path() / "models";
    writeSafetensors(root / "loras" / "style" / "watercolor.safetensors", {{"lora_unet_x.lora_down.weight", {4, 8}}, {"lora_unet_x.lora_up.weight", {8, 4}}});
    writeSafetensors(root / "pub" / "repo" / "base.safetensors", gh::test::st::sd1Tensors());
    writeSafetensors(root / "diffusion_models" / "dit.safetensors", gh::test::st::sd1Tensors());   // the folder says it is a diffusion model
    writeSafetensors(root / "text_encoder" / "enc.safetensors", {{"something.else", {2, 2}}});      // so does a diffusers-style folder name
    touch(root / "upscale_models" / "4x.pth");                                                      // pickled: counts only inside a kind folder
    touch(root / "pub" / "repo" / "ignored.pth");
    touch(root / "loras" / "style" / "watercolor.preview.png", "png");

    ModelLibrary library({root});
    const auto cards = library.scan();
    std::map<std::string, ModelCard> by;
    for (const auto& c : cards) by[c.relative] = c;
    REQUIRE(by.size() == 5);
    CHECK(by.at("loras/style/watercolor.safetensors").kind == Kind::Lora);
    CHECK(by.at("loras/style/watercolor.safetensors").kindSource == "folder");
    CHECK_FALSE(by.at("loras/style/watercolor.safetensors").preview.empty());
    CHECK(by.at("pub/repo/base.safetensors").kind == Kind::Checkpoint);
    CHECK(by.at("pub/repo/base.safetensors").kindSource == "header");
    CHECK(by.at("pub/repo/base.safetensors").preview.empty());
    CHECK(by.at("diffusion_models/dit.safetensors").kind == Kind::DiffusionModel);
    CHECK(by.at("diffusion_models/dit.safetensors").kindSource == "folder");
    CHECK(by.at("text_encoder/enc.safetensors").kind == Kind::TextEncoder);
    CHECK(by.at("upscale_models/4x.pth").kind == Kind::UpscaleModel);
    CHECK_FALSE(by.at("upscale_models/4x.pth").headerRead);
    CHECK(by.at("upscale_models/4x.pth").supported);
    CHECK(by.count("pub/repo/ignored.pth") == 0);
    CHECK(by.at("loras/style/watercolor.safetensors").root == root);

    CHECK(library.names(Kind::Lora) == std::vector<std::string>{"loras/style/watercolor.safetensors"});
    CHECK(library.names(Kind::Checkpoint) == std::vector<std::string>{"pub/repo/base.safetensors"});
    CHECK(library.names(Kind::Gligen).empty());

    // name lookup: exact relative path, unique file name, unique name without extension, unique prefix
    CHECK(library.find(Kind::Lora, "loras/style/watercolor.safetensors").has_value());
    CHECK(library.find(Kind::Lora, "watercolor.safetensors").has_value());
    CHECK(library.find(Kind::Lora, "watercolor").has_value());
    CHECK(library.find(Kind::Lora, "water").has_value());
    CHECK_FALSE(library.find(Kind::Lora, "nothing").has_value());
    CHECK_FALSE(library.find(Kind::Checkpoint, "watercolor").has_value());   // the right name, the wrong kind

    // the JSON the API serves carries the new fields
    const json j = toJson(by.at("loras/style/watercolor.safetensors"));
    CHECK(j["kind"] == "lora");
    CHECK(j["relative"] == "loras/style/watercolor.safetensors");
    CHECK(j["kindSource"] == "folder");
    CHECK(!j["preview"].get<std::string>().empty());
}

TEST_CASE("library: extra roots from model-paths.json hold files of one kind", "[unit][library][kinds]") {
    gh::test::TempDir dir;
    const fs::path main = dir.path() / "main";
    const fs::path nas = dir.path() / "nas-loras";
    writeSafetensors(main / "pub" / "base.safetensors", gh::test::st::sd1Tensors());
    writeSafetensors(nas / "sub" / "anime.safetensors", gh::test::st::sd1Tensors());   // not a LoRA by its header; the root decides
    touch(nas / "old.pt");                                                              // pickled files count in an extra root

    ModelLibrary library({main});
    library.setExtraRoots({{Kind::Lora, {nas}}});
    CHECK(library.names(Kind::Lora) == std::vector<std::string>{"old.pt", "sub/anime.safetensors"});
    CHECK(library.names(Kind::Checkpoint) == std::vector<std::string>{"pub/base.safetensors"});   // names do not change when roots are added
    const auto anime = library.find(Kind::Lora, "anime");
    REQUIRE(anime.has_value());
    CHECK(anime->kindSource == "root");
    CHECK(anime->root == nas);

    // an extra root inside the main folder wins for the files below it
    library.setExtraRoots({{Kind::Vae, {main / "pub"}}});
    CHECK(library.names(Kind::Vae) == std::vector<std::string>{"base.safetensors"});
    CHECK(library.names(Kind::Checkpoint).empty());

    // a changed root invalidates the cached card
    library.setExtraRoots({});
    CHECK(library.names(Kind::Checkpoint) == std::vector<std::string>{"pub/base.safetensors"});
}

TEST_CASE("model-paths.json: kinds by folder or kind name, problems become warnings", "[unit][library][kinds]") {
    gh::test::TempDir dir;
    const fs::path file = dir.path() / "model-paths.json";
    std::vector<std::string> warnings;
    CHECK(loadModelPaths(file, &warnings).empty());   // no file: nothing, no warning
    CHECK(warnings.empty());

    touch(file, R"({"loras": ["D:/a", "D:/b"], "checkpoint": ["E:/c"], "nonsense": ["F:/x"], "vae": "G:/not-a-list", "controlnet": [3, ""]})");
    const auto roots = loadModelPaths(file, &warnings);
    REQUIRE(roots.count(Kind::Lora) == 1);
    CHECK(roots.at(Kind::Lora).size() == 2);
    CHECK(roots.at(Kind::Lora)[1] == fs::path("D:/b"));
    CHECK(roots.at(Kind::Checkpoint).size() == 1);
    CHECK(roots.count(Kind::Vae) == 0);
    CHECK(roots.count(Kind::ControlNet) == 0);
    CHECK(warnings.size() == 4);

    touch(file, "[1,2");
    warnings.clear();
    CHECK(loadModelPaths(file, &warnings).empty());
    REQUIRE(warnings.size() == 1);
    CHECK(warnings[0].find("not a JSON object") != std::string::npos);
}

TEST_CASE("library: hashes come from the download record or are computed once and remembered", "[unit][library][kinds]") {
    gh::test::TempDir dir;
    const fs::path root = dir.path() / "models";
    writeSafetensors(root / "a.safetensors", gh::test::st::sd1Tensors());
    writeSafetensors(root / "b.safetensors", gh::test::st::sd1Tensors());
    touch(root / "b.safetensors.source.json", R"({"sha256": "feedface", "repo": "x/y"})");
    const fs::path cache = dir.path() / "hashes.json";

    std::string computed;
    {
        ModelLibrary library({root});
        library.setHashCache(cache);
        auto cards = library.scan();
        REQUIRE(cards.size() == 2);
        const ModelCard& a = cards[0];
        const ModelCard& b = cards[1];
        CHECK(a.sha256.empty());
        CHECK(b.sha256 == "feedface");                     // from the download record, no reading
        CHECK(library.sha256(b) == "feedface");
        computed = library.sha256(a);
        CHECK(computed.size() == 64);
        CHECK(computed == *gh::core::sha256File(a.path));
        CHECK(fs::exists(cache));
    }
    {   // a new library with the same cache file: the hash is known after a scan, without reading the file
        ModelLibrary library({root});
        library.setHashCache(cache);
        const auto cards = library.scan();
        CHECK(cards[0].sha256 == computed);
    }
    {   // a changed file is hashed again
        writeSafetensors(root / "a.safetensors", {{"model.diffusion_model.input_blocks.0.0.weight", {320, 4, 3, 3}}, {"cond_stage_model.transformer.text_model.x", {8, 8}}, {"first_stage_model.decoder.conv_in.weight", {512, 4, 3, 3}}});
        ModelLibrary library({root});
        library.setHashCache(cache);
        const auto cards = library.scan();
        CHECK(cards[0].sha256.empty());
        CHECK(library.sha256(cards[0]) != computed);
    }
}
