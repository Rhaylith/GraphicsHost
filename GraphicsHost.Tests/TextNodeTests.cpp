#include "graph/Executor.h"
#include "graph/Handles.h"
#include "graph/Inference.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::graph;
using nlohmann::json;

namespace {

class RecordingInference final : public InferenceService {
public:
    std::vector<std::string> plain;
    std::vector<int> skips;
    std::vector<bool> zeroOuts;
    Value encodeText(const TextEncoderHandle&, const gh::text::ParsedPrompt& p, int clipSkip, int, int, bool zeroOut) override {
        plain.push_back(gh::text::plainText(p));
        skips.push_back(clipSkip);
        zeroOuts.push_back(zeroOut);
        return Value::makeInt(static_cast<int64_t>(plain.size()));
    }
    void denoise(const ModelHandle&, const Value&, const LatentView&, float, std::vector<float>&) override {}
    gh::tensor::Tensor decode(const VaeHandle&, const ModelHandle&, const LatentView&) override { return {}; }
};

class Services final : public NodeServices {
public:
    RecordingInference inf;
    std::string refuse;   // words that make the filter refuse
    std::optional<ModelFileRef> findModel(const std::string&, const std::string&, std::string*) const override { return std::nullopt; }
    std::vector<std::string> modelNames(const std::string&) const override { return {}; }
    std::vector<std::string> embeddingNames() const override { return {}; }
    InferenceService* inference() const override { return const_cast<RecordingInference*>(&inf); }
    std::string refusePrompt(const std::string& p) const override { return !refuse.empty() && p.find(refuse) != std::string::npos ? "it mentions " + refuse : std::string(); }
};

struct Rig {
    Services services;
    OutputCache cache{*CachePolicy::parse("none")};
    Registry reg;
    Rig(const char* family = "SD 1.x") {
        for (const NodeDef* d : Registry::builtin().all()) reg.add(*d);
        reg.add(NodeDef("t.encoder").output(Out("e", ValueType::TextEncoder)).run([family](NodeContext&) {
            TextEncoderHandle h;
            ModelFileRef f;
            f.family = family;
            h.files = {f};
            return NodeResult{{handleValue(ValueType::TextEncoder, std::move(h))}};
        }));
    }
    ExecResult run(json inputs, json extra = json::object()) {
        json nodes = {{"0", {{"type", "t.encoder"}, {"inputs", json::object()}}}};
        for (const auto& [k, v] : extra.items()) nodes[k] = v;
        inputs["textEncoder"] = extra.contains("1") ? json::array({"1", 0}) : json::array({"0", 0});
        nodes["9"] = {{"type", "text.encode"}, {"inputs", inputs}};
        const JobGraph job = jobFromJson({{"format", "graphicshost.job"}, {"version", 1}, {"nodes", nodes}, {"outputs", json::array({"9"})}});
        RunOptions options;
        options.services = &services;
        return Executor(reg, cache).run(job, options);
    }
};
}  // namespace

TEST_CASE("text nodes: the prompt is parsed with the seed and handed to the model side", "[unit][text][nodes]") {
    Rig rig;
    auto r = rig.run({{"text", "a (red:1.2) fox // not this\n, {winter|winter} forest"}, {"seed", 3}});
    REQUIRE(r.ok());
    REQUIRE(rig.services.inf.plain.size() == 1);
    CHECK(rig.services.inf.plain[0].find("red") != std::string::npos);
    CHECK(rig.services.inf.plain[0].find("not this") == std::string::npos);
    CHECK(rig.services.inf.plain[0].find("{") == std::string::npos);
    CHECK(rig.services.inf.skips[0] == 1);
    CHECK_FALSE(rig.services.inf.zeroOuts[0]);
    CHECK(r.outputs.at("9")[0].type == ValueType::Conditioning);
}

TEST_CASE("text nodes: clip skip is carried by the encoder", "[unit][text][nodes]") {
    Rig rig;
    json extra = {{"1", {{"type", "text.clipSkip"}, {"inputs", {{"stopAtLayer", -2}, {"textEncoder", {"0", 0}}}}}}};
    auto r = rig.run({{"text", "x"}}, extra);
    REQUIRE(r.ok());
    CHECK(rig.services.inf.skips[0] == 2);
}

TEST_CASE("text nodes: an empty prompt of an SDXL checkpoint is zeroed out, a refused prompt fails the node", "[unit][text][nodes]") {
    Rig xl("SDXL");
    REQUIRE(xl.run({{"text", "  "}}).ok());
    CHECK(xl.services.inf.zeroOuts[0]);
    REQUIRE(xl.run({{"text", "a cat"}}).ok());
    CHECK_FALSE(xl.services.inf.zeroOuts[1]);

    Rig strict;
    strict.services.refuse = "forbidden";
    auto r = strict.run({{"text", "a forbidden thing"}});
    CHECK_FALSE(r.ok());
    CHECK(r.errorInput == "text");
    CHECK(r.errorMessage.find("safety filter") != std::string::npos);
    CHECK(strict.services.inf.plain.empty());

    // no runtime: a message, no crash
    OutputCache cache(*CachePolicy::parse("none"));
    const JobGraph job = jobFromJson({{"format", "graphicshost.job"}, {"version", 1}, {"nodes", {{"0", {{"type", "t.encoder"}, {"inputs", json::object()}}}, {"9", {{"type", "text.encode"}, {"inputs", {{"text", "x"}, {"textEncoder", {"0", 0}}}}}}}}, {"outputs", json::array({"9"})}});
    CHECK_FALSE(Executor(strict.reg, cache).run(job).ok());
}
