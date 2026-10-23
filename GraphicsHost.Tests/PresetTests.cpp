#include "TestUtil.h"

#include "presets/PresetStore.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::presets;
using nlohmann::json;

namespace {
Preset portrait() {
    Preset p;
    p.name = "portrait";
    p.description = "tall pictures, slow sampler";
    p.request = {{"width", 512}, {"height", 768}, {"steps", 30}, {"sampler", "dpm++2m"}, {"scheduler", "karras"}};
    p.load = {{"flashAttention", true}};
    return p;
}
}  // namespace

TEST_CASE("presets: names must be safe file names", "[unit][presets]") {
    for (const char* ok : {"portrait", "my preset", "a-b_c.d", "SDXL 1024", "x"}) CHECK(validPresetName(ok));
    for (const char* bad : {"", ".hidden", "../evil", "a/b", "a\\b", "c:x", "tab\tname", "emoji\xF0\x9F\x98\x80"}) {
        CHECK_FALSE(validPresetName(bad));
    }
    CHECK_FALSE(validPresetName(std::string(65, 'a')));
    CHECK(validPresetName(std::string(64, 'a')));
}

TEST_CASE("presets: save, list, get, remove", "[unit][presets]") {
    gh::test::TempDir dir;
    PresetStore store(dir.path() / "presets");
    CHECK(store.list().empty());  // the folder does not even exist yet
    store.save(portrait());
    auto second = portrait();
    second.name = "landscape";
    second.request = {{"width", 768}, {"height", 512}};
    store.save(second);

    CHECK(store.list() == std::vector<std::string>{"landscape", "portrait"});
    const auto got = store.get("portrait");
    REQUIRE(got.has_value());
    CHECK(got->description == "tall pictures, slow sampler");
    CHECK(got->request["steps"] == 30);
    CHECK(got->load["flashAttention"] == true);

    CHECK_FALSE(store.get("missing").has_value());
    CHECK_FALSE(store.get("../portrait").has_value());
    CHECK(store.remove("portrait"));
    CHECK_FALSE(store.remove("portrait"));
    CHECK(store.list() == std::vector<std::string>{"landscape"});
}

TEST_CASE("presets: invalid content is refused with field names", "[unit][presets]") {
    gh::test::TempDir dir;
    PresetStore store(dir.path());
    auto p = portrait();
    p.request["sampler"] = "magic";
    p.request["width"] = 500;
    p.load["device"] = "cuda0";
    p.load["modelPath"] = "x.safetensors";
    try {
        store.save(p);
        FAIL("expected an error");
    } catch (const std::invalid_argument& e) {
        const std::string message = e.what();
        CHECK(message.find("request.sampler") != std::string::npos);
        CHECK(message.find("request.width") != std::string::npos);
        CHECK(message.find("load.device") != std::string::npos);
        CHECK(message.find("load.modelPath") != std::string::npos);
    }
    CHECK(store.list().empty());

    auto badName = portrait();
    badName.name = "../x";
    CHECK_THROWS_AS(store.save(badName), std::invalid_argument);

    auto badType = portrait();
    badType.request["steps"] = "many";
    CHECK_THROWS_AS(store.save(badType), std::invalid_argument);
}

TEST_CASE("presets: a preset without a prompt is fine", "[unit][presets]") {
    gh::test::TempDir dir;
    PresetStore store(dir.path());
    Preset p;
    p.name = "just steps";
    p.request = {{"steps", 12}};
    CHECK_NOTHROW(store.save(p));
}

TEST_CASE("presets: files edited or added by hand show up immediately", "[unit][presets][reload]") {
    gh::test::TempDir dir;
    PresetStore store(dir.path());
    store.save(portrait());
    CHECK(store.get("portrait")->request["steps"] == 30);

    // edit in place
    gh::test::writeText(dir.path() / "portrait.json", R"({"name":"portrait","request":{"steps":99}})");
    CHECK(store.get("portrait")->request["steps"] == 99);

    // drop a new file in
    gh::test::writeText(dir.path() / "dropped.json", R"({"request":{"width":1024,"height":1024}})");
    CHECK(store.list() == std::vector<std::string>{"dropped", "portrait"});
    CHECK(store.get("dropped")->request["width"] == 1024);

    // a damaged file is not a preset, and does not break the others
    gh::test::writeText(dir.path() / "broken.json", "{ nope");
    CHECK_FALSE(store.get("broken").has_value());
    CHECK(store.get("portrait").has_value());

    // delete by hand
    std::filesystem::remove(dir.path() / "dropped.json");
    CHECK_FALSE(store.get("dropped").has_value());
}

TEST_CASE("presets: import and export", "[unit][presets]") {
    gh::test::TempDir dir;
    PresetStore store(dir.path() / "store");
    store.save(portrait());
    const auto exported = dir.path() / "shared" / "portrait-copy.json";
    store.exportFile("portrait", exported);
    CHECK(std::filesystem::exists(exported));

    PresetStore other(dir.path() / "other");
    CHECK(other.importFile(exported) == "portrait-copy");
    CHECK(other.importFile(exported, "renamed") == "renamed");
    CHECK(other.get("renamed")->request["sampler"] == "dpm++2m");

    gh::test::writeText(dir.path() / "evil.json", R"({"request":{"steps":-5}})");
    CHECK_THROWS_AS(other.importFile(dir.path() / "evil.json"), std::invalid_argument);
    CHECK_THROWS_AS(other.importFile(dir.path() / "absent.json"), std::runtime_error);
    CHECK_THROWS_AS(store.exportFile("nope", dir.path() / "x.json"), std::runtime_error);
}

TEST_CASE("model defaults: stored per checkpoint file name", "[unit][presets][defaults]") {
    gh::test::TempDir dir;
    ModelDefaultsStore store(dir.path() / "model-defaults.json");
    CHECK_FALSE(store.get("sdxl.safetensors").has_value());

    ModelDefaultsStore::Defaults d;
    d.request = {{"steps", 25}, {"cfgScale", 6.0}};
    d.load = {{"offloadParamsToCpu", true}};
    store.set("sdxl.safetensors", d);
    d.request = {{"sampler", "euler_a"}};
    d.load = json::object();
    store.set("sd15.safetensors", d);

    CHECK(store.models() == std::vector<std::string>{"sd15.safetensors", "sdxl.safetensors"});
    CHECK(store.get("sdxl.safetensors")->request["steps"] == 25);
    CHECK(store.get("sdxl.safetensors")->load["offloadParamsToCpu"] == true);
    CHECK(store.get("sd15.safetensors")->request["sampler"] == "euler_a");

    CHECK(store.remove("sd15.safetensors"));
    CHECK_FALSE(store.remove("sd15.safetensors"));
    CHECK(store.models() == std::vector<std::string>{"sdxl.safetensors"});

    ModelDefaultsStore::Defaults bad;
    bad.request = {{"steps", 9999}};
    CHECK_THROWS_AS(store.set("x.safetensors", bad), std::invalid_argument);
    CHECK_THROWS_AS(store.set("", d), std::invalid_argument);
}

TEST_CASE("layering: later layers win, absent keys fall through", "[unit][presets]") {
    const auto modelDefaults = json::parse(R"({"steps":25,"cfgScale":6.0,"sampler":"euler_a"})");
    const auto preset = json::parse(R"({"steps":30,"scheduler":"karras"})");
    const auto explicitValues = json::parse(R"({"prompt":"a cat","steps":8})");

    const auto r = layerRequest({modelDefaults, preset, explicitValues});
    CHECK(r.prompt == "a cat");
    CHECK(r.steps == 8);               // explicit beats preset beats model default
    CHECK(r.scheduler == "karras");    // preset only
    CHECK(r.sampler == "euler_a");     // model default only
    CHECK(r.cfgScale == 6.0f);
    CHECK(r.width == 0);               // nobody set it: stays "model default"

    CHECK(layerRequest({}).steps == 0);
    CHECK(layerRequest({json::object(), json(nullptr)}).steps == 0);

    const auto load = layerLoadOptions({json::parse(R"({"device":"cpu"})"), json::parse(R"({"device":"vulkan0","flashAttention":true})")});
    CHECK(load.device == "vulkan0");
    CHECK(load.flashAttention);
}
