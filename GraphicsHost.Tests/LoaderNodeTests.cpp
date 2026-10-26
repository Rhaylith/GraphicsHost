#include "SafetensorsFixture.h"
#include "TestUtil.h"
#include "graph/Executor.h"
#include "graph/Handles.h"
#include "host/AppHost.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh;
using namespace gh::graph;
using gh::test::st::writeSafetensors;
using nlohmann::json;
namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {
json n(const char* type, json inputs = json::object()) { return {{"type", type}, {"inputs", inputs}}; }
json ref(const char* id, int out = 0) { return json::array({id, out}); }

std::vector<gh::test::st::FakeTensor> loraTensors() {
    return {{"lora_unet_x.lora_down.weight", {4, 8}}, {"lora_unet_x.lora_up.weight", {8, 4}}};
}

// A models folder with a little of everything.
struct Models {
    test::TempDir dir;
    fs::path root = dir.path() / "models";
    Models() {
        writeSafetensors(root / "pub" / "repo" / "base-sd15.safetensors", test::st::sd1Tensors());
        writeSafetensors(root / "diffusion_models" / "dit.safetensors", test::st::sd1Tensors());
        writeSafetensors(root / "text_encoders" / "clip_l.safetensors", {{"text_model.embeddings.token_embedding.weight", {128, 768}}});
        writeSafetensors(root / "text_encoders" / "t5.safetensors", {{"encoder.block.0.layer.0.weight", {64, 64}}, {"shared.weight", {128, 64}}});
        writeSafetensors(root / "vae" / "good-vae.safetensors", {{"decoder.conv_in.weight", {512, 4, 3, 3}}, {"encoder.conv_in.weight", {128, 3, 3, 3}}});
        writeSafetensors(root / "vae_approx" / "taesd.safetensors", {{"decoder.layers.0.weight", {64, 4, 3, 3}}, {"encoder.layers.0.weight", {64, 3, 3, 3}}});
        writeSafetensors(root / "loras" / "watercolor_v2.safetensors", loraTensors());
        writeSafetensors(root / "loras" / "detail-a.safetensors", loraTensors());
        writeSafetensors(root / "loras" / "detail-b.safetensors", loraTensors());
        writeSafetensors(root / "embeddings" / "bad-hands.safetensors", {{"emb_params", {4, 768}}});
    }
};

struct Rig {
    Models models;
    models::ModelLibrary library{{models.root}};
    host::LibraryNodeServices services{library};
    OutputCache cache{*CachePolicy::parse("none")};

    ExecResult run(const json& nodes, const json& outputs, bool withServices = true) {
        const JobGraph job = jobFromJson({{"format", "graphicshost.job"}, {"version", 1}, {"nodes", nodes}, {"outputs", outputs}});
        RunOptions options;
        if (withServices) options.services = &services;
        return Executor(Registry::builtin(), cache).run(job, options);
    }
};
}  // namespace

TEST_CASE("loaders: a checkpoint gives model, text encoder and VAE handles", "[unit][loaders]") {
    Rig rig;
    const auto r = rig.run({{"1", n("loader.checkpoint", {{"checkpoint", "pub/repo/base-sd15.safetensors"}, {"flashAttention", "on"}, {"device", "vulkan1"}, {"idleUnloadMinutes", 0}})}}, json::array({"1"}));
    REQUIRE(r.ok());
    const auto& out = r.outputs.at("1");
    REQUIRE(out.size() == 3);
    const auto model = out[0].asObject<ModelHandle>();
    CHECK(model->file.kind == "checkpoint");
    CHECK(model->file.name == "pub/repo/base-sd15.safetensors");
    CHECK(model->file.family == "SD 1.x");
    CHECK(fs::path(model->file.path).filename() == "base-sd15.safetensors");
    CHECK(model->load.flashAttention == "on");
    CHECK(model->load.device == "vulkan1");
    CHECK(model->load.idleUnloadMinutes == 0);
    CHECK(model->load.offloadToRam == "auto");
    const auto encoders = out[1].asObject<TextEncoderHandle>();
    CHECK(encoders->fromCheckpoint);
    CHECK(encoders->files.size() == 1);
    CHECK(encoders->load == model->load);
    const auto vae = out[2].asObject<VaeHandle>();
    CHECK(vae->fromCheckpoint);
    CHECK_FALSE(vae->pixelSpace);
    CHECK(out[0].type == ValueType::Model);
    CHECK(out[1].type == ValueType::TextEncoder);
    CHECK(out[2].type == ValueType::Vae);

    // names are also found loosely, the way prompt tags are
    CHECK(rig.run({{"1", n("loader.checkpoint", {{"checkpoint", "base-sd15"}})}}, json::array({"1"})).ok());

    const auto missing = rig.run({{"1", n("loader.checkpoint", {{"checkpoint", "nothing.safetensors"}})}}, json::array({"1"}));
    CHECK_FALSE(missing.ok());
    CHECK(missing.errorInput == "checkpoint");
    CHECK(missing.errorMessage.find("nothing.safetensors") != std::string::npos);

    const auto bare = rig.run({{"1", n("loader.checkpoint", {{"checkpoint", "base-sd15"}})}}, json::array({"1"}), false);
    CHECK_FALSE(bare.ok());
    CHECK(bare.errorMessage.find("without a model library") != std::string::npos);
}

TEST_CASE("loaders: diffusion model, text encoders and VAEs", "[unit][loaders]") {
    Rig rig;
    auto r = rig.run({{"1", n("loader.diffusionModel", {{"model", "dit"}, {"weightDtype", "fp8 e4m3"}})}}, json::array({"1"}));
    REQUIRE(r.ok());
    CHECK(r.outputs.at("1")[0].asObject<ModelHandle>()->file.kind == "diffusion_model");
    CHECK(r.outputs.at("1")[0].asObject<ModelHandle>()->load.weightDtype == "fp8 e4m3");

    r = rig.run({{"1", n("loader.textEncoders", {{"encoder1", "clip_l"}, {"encoder2", "t5"}, {"family", "flux"}, {"device", "GPU weights in RAM"}})}}, json::array({"1"}));
    REQUIRE(r.ok());
    const auto enc = r.outputs.at("1")[0].asObject<TextEncoderHandle>();
    CHECK(enc->files.size() == 2);
    CHECK(enc->files[0].kind == "text_encoder");
    CHECK(enc->files[1].name == "text_encoders/t5.safetensors");
    CHECK(enc->family == "flux");
    CHECK(enc->load.device == "GPU weights in RAM");
    CHECK_FALSE(enc->fromCheckpoint);

    r = rig.run({{"1", n("loader.textEncoders", {{"encoder1", "clip_l"}, {"encoder3", ""}})}}, json::array({"1"}));   // empty optional entries are skipped
    REQUIRE(r.ok());
    CHECK(r.outputs.at("1")[0].asObject<TextEncoderHandle>()->files.size() == 1);

    r = rig.run({{"1", n("loader.vae", {{"vae", "good-vae"}})}}, json::array({"1"}));
    REQUIRE(r.ok());
    CHECK(r.outputs.at("1")[0].asObject<VaeHandle>()->file.kind == "vae");
    r = rig.run({{"1", n("loader.vae", {{"vae", "taesd"}})}}, json::array({"1"}));   // a preview decoder in the VAE list
    REQUIRE(r.ok());
    CHECK(r.outputs.at("1")[0].asObject<VaeHandle>()->file.kind == "preview_decoder");
    r = rig.run({{"1", n("loader.vae", {{"vae", "pixel space"}})}}, json::array({"1"}));
    REQUIRE(r.ok());
    CHECK(r.outputs.at("1")[0].asObject<VaeHandle>()->pixelSpace);
    CHECK_FALSE(rig.run({{"1", n("loader.vae", {{"vae", "no-such-vae"}})}}, json::array({"1"})).ok());
}

TEST_CASE("loaders: LoRAs are added to the handles without touching the originals", "[unit][loaders]") {
    Rig rig;
    const json base = {{"1", n("loader.checkpoint", {{"checkpoint", "base-sd15"}})},
                       {"2", n("loader.lora", {{"model", ref("1", 0)}, {"textEncoder", ref("1", 1)}, {"lora", "watercolor"}, {"strengthModel", 0.7}, {"strengthText", 0.5}})},
                       {"3", n("loader.loraModel", {{"model", ref("2", 0)}, {"lora", "loras/detail-a.safetensors"}, {"strengthModel", -0.5}})}};
    const auto r = rig.run(base, json::array({"3", "2"}));
    REQUIRE(r.ok());
    const auto chained = r.outputs.at("3")[0].asObject<ModelHandle>();
    REQUIRE(chained->patches.size() == 2);
    CHECK(chained->patches[0].lora.name == "loras/watercolor_v2.safetensors");   // "watercolor" was found like <lora:watercolor:w>
    CHECK(chained->patches[0].strengthModel == 0.7);
    CHECK(chained->patches[0].strengthText == 0.5);
    CHECK(chained->patches[1].lora.name == "loras/detail-a.safetensors");
    CHECK(chained->patches[1].strengthModel == -0.5);
    CHECK(chained->patches[1].strengthText == 0.0);
    const auto first = r.outputs.at("2");
    CHECK(first[0].asObject<ModelHandle>()->patches.size() == 1);   // the earlier handle is unchanged
    REQUIRE(first[1].asObject<TextEncoderHandle>()->patches.size() == 1);
    CHECK(first[1].asObject<TextEncoderHandle>()->patches[0].strengthText == 0.5);

    // a prefix that matches several files is refused, naming the input
    auto ambiguous = base;
    ambiguous["2"]["inputs"]["lora"] = "detail";
    const auto bad = rig.run(ambiguous, json::array({"2"}));
    CHECK_FALSE(bad.ok());
    CHECK(bad.errorNode == "2");
    CHECK(bad.errorInput == "lora");
    CHECK(bad.errorMessage.find("matches 2") != std::string::npos);
}

TEST_CASE("loaders: the host validates model names and lists them for combos", "[unit][loaders][graph-service]") {
    Models models;
    test::TempDir home;
    host::AppHost app(home.path());
    app.library().setFolders({models.root});
    auto& service = app.graph();

    CHECK(service.modelNames("checkpoints") == std::vector<std::string>{"pub/repo/base-sd15.safetensors"});
    CHECK(service.modelNames("loras") == std::vector<std::string>{"loras/detail-a.safetensors", "loras/detail-b.safetensors", "loras/watercolor_v2.safetensors"});
    const auto vaes = service.modelNames("vae");
    CHECK(vaes == std::vector<std::string>{"vae/good-vae.safetensors", "vae_approx/taesd.safetensors", "pixel space"});
    CHECK(service.modelNames("nonsense").empty());
    CHECK(service.embeddingNames() == std::vector<std::string>{"bad-hands"});

    const auto job = [](const char* checkpoint, const char* vae) {
        return json{{"format", "graphicshost.job"},
                    {"version", 1},
                    {"nodes", {{"1", n("loader.checkpoint", {{"checkpoint", checkpoint}})}, {"2", n("loader.vae", {{"vae", vae}})},
                               {"3", n("util.previewAny", {{"value", ref("1", 0)}})}, {"4", n("util.previewAny", {{"value", ref("2", 0)}})}}}};
    };
    host::GraphJobCall call;
    call.job = job("pub/repo/base-sd15.safetensors", "pixel space");
    const auto good = service.submit(call);
    REQUIRE(good.validation.ok());
    REQUIRE(good.job->waitFor(10s));
    CHECK(good.job->state() == engine::JobState::Succeeded);
    CHECK(good.job->graphResult()["outputs"][0]["values"][0] == "STRING MODEL");

    call.job = job("gone.safetensors", "no-vae");
    const auto bad = service.submit(call);
    CHECK(bad.job == nullptr);
    int missing = 0;
    for (const auto& i : bad.validation.issues) missing += i.code == "missing_model";
    CHECK(missing == 2);
}

TEST_CASE("loaders: an optional model left empty validates", "[unit][loaders][graph-service]") {
    Models models;
    test::TempDir home;
    host::AppHost app(home.path());
    app.library().setFolders({models.root});
    host::GraphJobCall call;
    call.job = {{"format", "graphicshost.job"}, {"version", 1}, {"nodes", {{"1", n("loader.textEncoders", {{"encoder1", "clip_l"}, {"encoder2", ""}})}, {"2", n("util.previewAny", {{"value", ref("1")}})}}}};
    CHECK(app.graph().validate(call.job).ok());
    call.job["nodes"]["1"]["inputs"]["encoder1"] = "";   // the first one is required
    CHECK_FALSE(app.graph().validate(call.job).ok());
}
