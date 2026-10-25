#include "engine/GenerationRequest.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::engine;
using nlohmann::json;

namespace {
GenerationRequest valid() {
    GenerationRequest r;
    r.prompt = "a lighthouse";
    return r;
}
}  // namespace

TEST_CASE("request: defaults are all 'use the model default'", "[unit][request]") {
    const GenerationRequest r;
    CHECK(r.width == 0);
    CHECK(r.height == 0);
    CHECK(r.steps == 0);
    CHECK(r.cfgScale < 0);
    CHECK(r.seed == -1);
    CHECK(r.sampler.empty());
    CHECK(r.safetyFilter == false);  // the safety filter is off unless asked for
    CHECK(validate(valid()).empty());
}

TEST_CASE("request: family defaults", "[unit][request]") {
    const auto sd15 = familyDefaults("SD 1.x");
    CHECK(sd15.width == 512);
    CHECK(sd15.steps == 20);
    const auto sdxl = familyDefaults("SDXL");
    CHECK(sdxl.width == 1024);
    CHECK(sdxl.height == 1024);
    CHECK(sdxl.steps == 25);
    CHECK(familyDefaults("something new").width == 512);
}

TEST_CASE("request: resolve fills only the unset fields", "[unit][request]") {
    GenerationRequest r = valid();
    r.width = 768;
    r.cfgScale = 3.0f;
    const auto resolved = resolve(r, familyDefaults("SDXL"));
    CHECK(resolved.width == 768);       // kept
    CHECK(resolved.height == 1024);     // filled
    CHECK(resolved.steps == 25);        // filled
    CHECK(resolved.cfgScale == 3.0f);   // kept
}

TEST_CASE("request: validation reports every bad field with its name", "[unit][request]") {
    GenerationRequest r;  // empty prompt
    r.width = 500;        // not a multiple of 8
    r.height = 8192;
    r.steps = 500;
    r.cfgScale = 99;
    r.seed = -5;
    r.batchCount = 0;
    r.sampler = "magic";
    r.scheduler = "chaos";
    r.clipSkip = 40;
    r.format = "bmp";
    const auto errors = validate(r);
    std::vector<std::string> fields;
    for (const auto& e : errors) fields.push_back(e.field);
    CHECK(fields == std::vector<std::string>{"prompt", "width", "height", "steps", "cfgScale", "seed", "batchCount",
                                             "sampler", "scheduler", "clipSkip", "format"});
}

TEST_CASE("request: every known sampler and scheduler is accepted", "[unit][request]") {
    for (const auto& s : knownSamplers()) {
        auto r = valid();
        r.sampler = s;
        CHECK(validate(r).empty());
    }
    for (const auto& s : knownSchedulers()) {
        auto r = valid();
        r.scheduler = s;
        CHECK(validate(r).empty());
    }
}

TEST_CASE("request: JSON round trip", "[unit][request]") {
    GenerationRequest r = valid();
    r.negativePrompt = "blurry";
    r.width = 640;
    r.height = 384;
    r.steps = 30;
    r.cfgScale = 6.5f;
    r.seed = 123456789012345LL;
    r.batchCount = 4;
    r.sampler = "dpm++2m";
    r.scheduler = "karras";
    r.clipSkip = 2;
    r.vaeTiling = true;
    r.format = "webp";
    r.embedMetadata = false;
    r.safetyFilter = true;

    const GenerationRequest back = requestFromJson(toJson(r));
    CHECK(toJson(back) == toJson(r));
    CHECK(back.seed == 123456789012345LL);
}

TEST_CASE("request: a partial document only changes the keys it contains", "[unit][request]") {
    GenerationRequest base = valid();
    base.steps = 30;
    base.sampler = "euler_a";
    const auto merged = applyRequestJson(base, json::parse(R"({"steps": 12, "scheduler": "karras", "unknownKey": 1})"));
    CHECK(merged.steps == 12);
    CHECK(merged.scheduler == "karras");
    CHECK(merged.sampler == "euler_a");
    CHECK(merged.prompt == "a lighthouse");
}

TEST_CASE("request: wrong JSON types are errors, nulls are ignored", "[unit][request]") {
    CHECK_THROWS(requestFromJson(json::parse(R"({"steps": "many"})")));
    CHECK_THROWS(requestFromJson(json::parse("[1,2]")));
    CHECK(requestFromJson(json::parse(R"({"steps": null})")).steps == 0);
}

TEST_CASE("load options: validation", "[unit][request]") {
    LoadOptions o;
    CHECK_FALSE(validate(o).empty());  // no model path
    o.modelPath = "m.safetensors";
    CHECK(validate(o).empty());
    for (const char* ok : {"auto", "cpu", "vulkan0", "vulkan12"}) {
        o.device = ok;
        CHECK(validate(o).empty());
    }
    for (const char* bad : {"gpu", "vulkan", "vulkanX", "cuda0", ""}) {
        o.device = bad;
        const auto errors = validate(o);
        REQUIRE(errors.size() == 1);
        CHECK(errors[0].field == "device");
    }
    o.device = "cpu";
    o.threads = -1;
    o.ttlMinutes = 5000;
    CHECK(validate(o).size() == 2);
}

TEST_CASE("load options: JSON round trip", "[unit][request]") {
    LoadOptions o;
    o.modelPath = "D:/models/sdxl.safetensors";
    o.vaePath = "vae.safetensors";
    o.device = "vulkan0";
    o.threads = 8;
    o.flashAttention = true;
    o.offloadParamsToCpu = true;
    o.clipOnCpu = true;
    o.vaeOnCpu = true;
    o.ttlMinutes = 0;
    CHECK(loadOptionsFromJson(toJson(o)) == o);
}
