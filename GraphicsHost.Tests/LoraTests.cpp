#include "GenApiTestUtil.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace gh;
using gh::test::GenApp;
using nlohmann::json;

namespace {

// A LoRA file the library recognises (SD 1.x style, or SDXL when it also carries a second text encoder).
void addLora(GenApp& g, const std::string& name, bool sdxl = false) {
    using gh::test::st::FakeTensor;
    const std::string base = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q";
    std::vector<FakeTensor> tensors = {{base + ".lora_down.weight", {4, 320}}, {base + ".lora_up.weight", {320, 4}}};
    if (sdxl) tensors.push_back({"lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight", {4, 1280}});
    gh::test::st::writeSafetensors(g.app.home.path() / "models" / (name + ".safetensors"), tensors);
}

engine::GenerationRequest lastRequest(GenApp& g) {
    std::lock_guard<std::mutex> lock(g.state->mutex);
    REQUIRE_FALSE(g.state->requests.empty());
    return g.state->requests.back();
}

}  // namespace

TEST_CASE("lora: tags in the prompt and the loras list reach the engine as files", "[integration][api][lora]") {
    GenApp g;
    addLora(g, "detail");
    addLora(g, "watercolor");

    auto r = g.post("/v1/images/generations", {{"model", "tiny"}, {"prompt", "a cat <lora:detail:0.7> sitting"}});
    REQUIRE(r->status == 200);
    auto request = lastRequest(g);
    REQUIRE(request.loras.size() == 1);
    CHECK(request.loras[0].name == "detail");
    CHECK(request.loras[0].weight == Catch::Approx(0.7f));
    CHECK(request.loras[0].path.find("detail.safetensors") != std::string::npos);
    CHECK(request.prompt == "a cat sitting");   // the tag is not left in the text

    // a list, with the three spellings; a name may be shortened to a unique prefix
    r = g.post("/v1/images/generations", {{"model", "tiny"}, {"prompt", "p"}, {"loras", json::array({"water:0.5", {{"name", "detail"}, {"weight", 1.2}}, "detail"})}});
    REQUIRE(r->status == 200);
    request = lastRequest(g);
    REQUIRE(request.loras.size() == 3);
    CHECK(request.loras[0].name == "watercolor");
    CHECK(request.loras[0].weight == Catch::Approx(0.5f));
    CHECK(request.loras[1].weight == Catch::Approx(1.2f));
    CHECK(request.loras[2].weight == Catch::Approx(1.0f));

    // a tag wins over a list entry of the same name; a bare tag has weight 1
    r = g.post("/v1/images/generations", {{"model", "tiny"}, {"prompt", "<lora:detail> p <LORA:watercolor:0.3>"}, {"loras", json::array({"detail:0.2"})}});
    REQUIRE(r->status == 200);
    request = lastRequest(g);
    REQUIRE(request.loras.size() == 2);
    CHECK(request.loras[0].weight == Catch::Approx(1.0f));
    CHECK(request.loras[1].name == "watercolor");
    CHECK(request.prompt == "p");

    // no LoRA: nothing is passed
    REQUIRE(g.post("/v1/images/generations", {{"model", "tiny"}, {"prompt", "plain"}})->status == 200);
    CHECK(lastRequest(g).loras.empty());

    // the gallery records them, so a picture can be reproduced
    const auto entries = g.app.host->gallery().list();
    CHECK(entries[1].parameters["request"]["loras"].size() == 2);
}

TEST_CASE("lora: bad references are refused before anything runs", "[integration][api][lora]") {
    GenApp g;
    addLora(g, "detail");
    addLora(g, "detail-xl", true);
    addLora(g, "style");
    addLora(g, "style2");
    auto expect = [&](const json& body, int status, const std::string& needle) {
        const auto r = g.post("/v1/images/generations", body);
        REQUIRE(r);
        INFO(r->body);
        CHECK(r->status == status);
        CHECK(test::bodyJson(r)["error"]["message"].get<std::string>().find(needle) != std::string::npos);
    };
    expect({{"model", "tiny"}, {"prompt", "<lora:nothere:1>"}}, 404, "no LoRA named");
    expect({{"model", "tiny"}, {"prompt", "<lora:detail-xl:1> p"}}, 400, "SDXL");   // the wrong family
    expect({{"model", "tiny"}, {"prompt", "<lora:sty:1> p"}}, 400, "several files");   // ambiguous prefix
    expect({{"model", "tiny"}, {"prompt", "p"}, {"loras", json::array({{{"name", "detail"}, {"weight", 9}}})}}, 400, "weight");
    expect({{"model", "tiny"}, {"prompt", "p"}, {"loras", "detail"}}, 400, "list");
    expect({{"model", "tiny"}, {"prompt", "p"}, {"loras", json::array({"detail:much"})}}, 400, "not a number");
    expect({{"model", "tiny"}, {"prompt", "p"}, {"loras", json::array({5})}}, 400, "name");
    CHECK(g.state->requests.empty());
}

TEST_CASE("lora: they are listed with the models", "[integration][api][lora]") {
    GenApp g;
    addLora(g, "detail");
    const auto models = test::bodyJson(g.app.client().Get("/api/v1/models"))["models"];
    bool found = false;
    for (const auto& m : models) {
        if (m["file"] == "detail.safetensors") {
            found = true;
            CHECK(m["kind"] == "lora");
            CHECK(m["family"] == "SD 1.x");
        }
    }
    CHECK(found);
    // a LoRA is not a checkpoint: it is not found as a model to generate with
    CHECK(g.post("/v1/images/generations", {{"model", "detail"}, {"prompt", "p"}})->status == 404);
}
