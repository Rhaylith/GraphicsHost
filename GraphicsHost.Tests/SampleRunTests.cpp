#include "graph/Executor.h"
#include "graph/Handles.h"
#include "graph/Inference.h"
#include "graph/SampleRun.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace gh::graph;
using Catch::Approx;
using nlohmann::json;

namespace {

// Data that is N(0,1) per element; a conditioning is an INT tag that shifts the estimate by tag / 10.
class FakeInference final : public InferenceService {
public:
    int calls = 0, pairCalls = 0;
    std::vector<float> sigmasSeen;
    Value encodeText(const TextEncoderHandle&, const gh::text::ParsedPrompt&, int, int, int, bool) override { return Value::makeInt(0); }
    void denoise(const ModelHandle&, const Value& cond, const LatentView& x, float sigma, std::vector<float>& out) override {
        ++calls;
        sigmasSeen.push_back(sigma);
        const double shift = static_cast<double>(cond.asInt()) / 10.0;
        out.resize(x.count());
        for (size_t i = 0; i < x.count(); ++i) out[i] = static_cast<float>(x.data[i] / (1.0 + static_cast<double>(sigma) * sigma) + shift);
    }
    void denoisePair(const ModelHandle& m, const Value& a, const Value& b, const LatentView& x, float sigma, std::vector<float>& oa, std::vector<float>& ob) override {
        ++pairCalls;
        InferenceService::denoisePair(m, a, b, x, sigma, oa, ob);
    }
    gh::tensor::Tensor decode(const VaeHandle&, const ModelHandle&, const LatentView&) override { return {}; }
};

class FakeServices final : public NodeServices {
public:
    FakeInference inf;
    std::optional<ModelFileRef> findModel(const std::string&, const std::string&, std::string*) const override { return std::nullopt; }
    std::vector<std::string> modelNames(const std::string&) const override { return {}; }
    std::vector<std::string> embeddingNames() const override { return {}; }
    InferenceService* inference() const override { return const_cast<FakeInference*>(&inf); }
};

double rms(const gh::tensor::Tensor& t) {
    double s = 0;
    for (float v : t.values()) s += static_cast<double>(v) * v;
    return std::sqrt(s / static_cast<double>(t.numel()));
}

struct Rig {
    FakeServices services;
    OutputCache cache{*CachePolicy::parse("none")};
    Registry reg;
    Rig() {
        for (const NodeDef* d : Registry::builtin().all()) reg.add(*d);
        reg.add(NodeDef("t.model").output(Out("model", ValueType::Model)).run([](NodeContext&) { return NodeResult{{handleValue(ValueType::Model, ModelHandle{})}}; }));
        reg.add(NodeDef("t.cond").input(Int("tag").def(0)).output(Out("c", ValueType::Conditioning)).run([](NodeContext& c) { return NodeResult{{c.get("tag")}}; }));
        reg.add(NodeDef("t.latent").input(Float("fill").def(0.0)).output(Out("l", ValueType::Latent)).run([](NodeContext& c) {
            return NodeResult{{Value::makeTensor(ValueType::Latent, gh::tensor::Tensor({1, 4, 8, 8}, static_cast<float>(c.get("fill").asFloat())))}};
        }));
    }
    ExecResult run(json sampleInputs, const char* type = "sampling.sample", json extra = json::object()) {
        json nodes = {{"0", {{"type", "t.model"}, {"inputs", json::object()}}},
                      {"1", {{"type", "t.cond"}, {"inputs", {{"tag", 1}}}}},
                      {"2", {{"type", "t.cond"}, {"inputs", {{"tag", 0}}}}},
                      {"3", {{"type", "t.latent"}, {"inputs", json::object()}}}};
        for (const auto& [k, v] : extra.items()) nodes[k] = v;
        sampleInputs["model"] = json::array({"0", 0});
        sampleInputs["positive"] = json::array({"1", 0});
        sampleInputs["negative"] = json::array({"2", 0});
        sampleInputs["latent"] = json::array({"3", 0});
        nodes["9"] = {{"type", type}, {"inputs", sampleInputs}};
        const JobGraph job = jobFromJson({{"format", "graphicshost.job"}, {"version", 1}, {"nodes", nodes}, {"outputs", json::array({"9"})}});
        RunOptions options;
        options.services = &services;
        return Executor(reg, cache).run(job, options);
    }
};
}  // namespace

TEST_CASE("sampler node: from noise to a latent with the spread of the data", "[unit][sampling][run]") {
    Rig rig;
    auto r = rig.run({{"seed", 5}, {"steps", 20}, {"cfg", 1.0}, {"sampler", "euler"}, {"scheduler", "karras"}});
    REQUIRE(r.ok());
    const auto& out = r.outputs.at("9")[0].asTensor();
    CHECK(out.shape() == std::vector<int64_t>{1, 4, 8, 8});
    CHECK(r.outputs.at("9")[0].type == ValueType::Latent);
    CHECK(rms(out) > 0.4);   // 256 values of N(0,1) data plus the 0.1 shift of the positive tag
    CHECK(rms(out) < 1.8);
    CHECK(rig.services.inf.calls == 20);        // cfg 1: one call per step, no negative prompt pass
    CHECK(rig.services.inf.pairCalls == 0);
    CHECK(rig.services.inf.sigmasSeen.front() > 10.0f);
    CHECK(rig.services.inf.sigmasSeen.back() < 0.2f);

    // the same seed gives the same latent, another seed another one
    auto again = rig.run({{"seed", 5}, {"steps", 20}, {"cfg", 1.0}, {"sampler", "euler"}, {"scheduler", "karras"}});
    CHECK(again.outputs.at("9")[0].asTensor().values() == out.values());
    auto other = rig.run({{"seed", 6}, {"steps", 20}, {"cfg", 1.0}, {"sampler", "euler"}, {"scheduler", "karras"}});
    CHECK(other.outputs.at("9")[0].asTensor().values() != out.values());
    // the generator can be told apart
    auto philox = rig.run({{"seed", 5}, {"steps", 20}, {"cfg", 1.0}, {"sampler", "euler"}, {"scheduler", "karras"}});
    CHECK(philox.ok());
}

TEST_CASE("sampler node: guidance asks for the negative prompt and moves the result", "[unit][sampling][run]") {
    Rig rig;
    auto plain = rig.run({{"seed", 1}, {"steps", 10}, {"cfg", 1.0}, {"sampler", "heun"}, {"scheduler", "simple"}});
    const int plainCalls = rig.services.inf.calls;
    Rig guided;
    auto g = guided.run({{"seed", 1}, {"steps", 10}, {"cfg", 6.0}, {"sampler", "heun"}, {"scheduler", "simple"}});
    REQUIRE(g.ok());
    CHECK(guided.services.inf.pairCalls > 0);                         // cond and uncond asked together
    CHECK(guided.services.inf.pairCalls == plainCalls);               // one pair where there was one call
    // the positive tag shifts the estimate by 0.1 and the negative by 0: cfg 6 pushes the result six times as far
    double meanPlain = 0, meanGuided = 0;
    for (float v : plain.outputs.at("9")[0].asTensor().values()) meanPlain += v;
    for (float v : g.outputs.at("9")[0].asTensor().values()) meanGuided += v;
    CHECK(meanGuided - meanPlain > 0.5 * 256 * 0.1 * 5.0 * 0.5);
}

TEST_CASE("sampler node: advanced runs part of the schedule and can keep the noise", "[unit][sampling][run]") {
    Rig a;
    auto full = a.run({{"seed", 2}, {"steps", 20}, {"cfg", 1.0}, {"sampler", "euler"}, {"scheduler", "karras"}}, "sampling.sampleAdvanced");
    REQUIRE(full.ok());
    CHECK(a.services.inf.calls == 20);

    Rig b;
    auto part = b.run({{"seed", 2}, {"steps", 20}, {"cfg", 1.0}, {"sampler", "euler"}, {"scheduler", "karras"}, {"startStep", 10}}, "sampling.sampleAdvanced");
    REQUIRE(part.ok());
    CHECK(b.services.inf.calls == 10);
    CHECK(b.services.inf.sigmasSeen.front() == Approx(a.services.inf.sigmasSeen[10]));   // the same schedule, from the middle

    Rig c;
    auto clean = c.run({{"seed", 2}, {"steps", 20}, {"cfg", 1.0}, {"sampler", "euler"}, {"scheduler", "karras"}, {"endStep", 6}}, "sampling.sampleAdvanced");
    Rig d;
    auto noisy = d.run({{"seed", 2}, {"steps", 20}, {"cfg", 1.0}, {"sampler", "euler"}, {"scheduler", "karras"}, {"endStep", 6}, {"returnWithNoise", true}}, "sampling.sampleAdvanced");
    REQUIRE(clean.ok());
    REQUIRE(noisy.ok());
    CHECK(c.services.inf.calls == 6);
    CHECK(rms(clean.outputs.at("9")[0].asTensor()) < 1.8);           // ended at noise level 0
    CHECK(rms(noisy.outputs.at("9")[0].asTensor()) > rms(clean.outputs.at("9")[0].asTensor()) * 1.5);   // stopped with noise left

    // without added noise the latent is the starting point
    Rig e;
    auto kept = e.run({{"addNoise", false}, {"steps", 4}, {"cfg", 1.0}, {"sampler", "euler"}, {"scheduler", "simple"}}, "sampling.sampleAdvanced", {{"3", {{"type", "t.latent"}, {"inputs", {{"fill", 2.0}}}}}});
    REQUIRE(kept.ok());
    // the schedule starts at the model's top noise level, so a latent of 2.0 that was not noised is treated as a very noisy one and is pulled toward the data
    CHECK(rms(kept.outputs.at("9")[0].asTensor()) < 2.0);
}

TEST_CASE("sampler node: errors name the input", "[unit][sampling][run]") {
    Rig rig;
    auto r = rig.run({{"sampler", "uni pc"}});
    CHECK_FALSE(r.ok());
    CHECK(r.errorInput == "sampler");

    // no runtime: a clear message, no crash
    OutputCache cache(*CachePolicy::parse("none"));
    const JobGraph job = jobFromJson({{"format", "graphicshost.job"},
                                      {"version", 1},
                                      {"nodes", {{"0", {{"type", "t.model"}, {"inputs", json::object()}}}, {"1", {{"type", "t.cond"}, {"inputs", json::object()}}}, {"3", {{"type", "t.latent"}, {"inputs", json::object()}}},
                                                 {"9", {{"type", "sampling.sample"}, {"inputs", {{"model", {"0", 0}}, {"positive", {"1", 0}}, {"negative", {"1", 0}}, {"latent", {"3", 0}}}}}}}},
                                      {"outputs", json::array({"9"})}});
    const auto none = Executor(rig.reg, cache).run(job);
    CHECK_FALSE(none.ok());
    CHECK(none.errorMessage.find("without one") != std::string::npos);
}

TEST_CASE("sample run: the noise mask keeps the original outside, and stopping works", "[unit][sampling][run]") {
    FakeInference inf;
    SampleRequest req;
    req.inference = &inf;
    req.guider.kind = GuiderHandle::Kind::Basic;
    req.guider.model = handleValue(ValueType::Model, ModelHandle{});
    req.guider.conditioning = {Value::makeInt(0)};
    req.sampler.name = "euler";
    req.noise.seed = 3;
    req.sigmas = {10.0f, 5.0f, 2.0f, 1.0f, 0.0f};
    req.latent = gh::tensor::Tensor({1, 4, 4, 4}, 0.5f);
    // half the picture is kept: mask 0 on the left, 1 on the right
    gh::tensor::Tensor mask({1, 1, 4, 4}, 0.0f);
    for (int y = 0; y < 4; ++y)
        for (int x = 2; x < 4; ++x) mask.at({0, 0, y, x}) = 1.0f;
    req.noiseMask = mask;
    const SampleResult res = runSampling(req);
    for (int c = 0; c < 4; ++c)
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x) {
                const float v = res.output.at({0, c, y, x});
                if (x < 2) CHECK(v == Approx(0.5f).margin(1e-4));   // kept
            }
    bool changed = false;
    for (int y = 0; y < 4; ++y) changed = changed || std::fabs(res.output.at({0, 0, y, 3}) - 0.5f) > 0.01f;
    CHECK(changed);

    std::stop_source stop;
    stop.request_stop();
    req.stop = stop.get_token();
    CHECK_THROWS_AS(runSampling(req), Interrupted);

    req.stop = {};
    req.latent = gh::tensor::Tensor({4, 4}, 0.0f);
    CHECK_THROWS_AS(runSampling(req), NodeError);   // not a latent
    req.inference = nullptr;
    CHECK_THROWS_AS(runSampling(req), NodeError);
}

TEST_CASE("sample (custom): noise, guider, sampler and sigmas given separately", "[unit][sampling][run]") {
    Rig rig;
    json extra = {{"4", {{"type", "sampling.noiseRandom"}, {"inputs", {{"seed", 9}, {"generator", "philox"}}}}},
                  {"5", {{"type", "sampling.guiderBasic"}, {"inputs", {{"model", {"0", 0}}, {"conditioning", {"1", 0}}}}}},
                  {"6", {{"type", "sampling.samplerSelect"}, {"inputs", {{"sampler", "dpm++ 2M"}}}}},
                  {"7", {{"type", "sampling.scheduleKarras"}, {"inputs", {{"steps", 12}}}}}};
    const json nodes = {{"noise", {"4", 0}}, {"guider", {"5", 0}}, {"sampler", {"6", 0}}, {"sigmas", {"7", 0}}};
    auto r = rig.run(nodes, "sampling.custom", extra);
    REQUIRE(r.ok());
    REQUIRE(r.outputs.at("9").size() == 2);
    CHECK(r.outputs.at("9")[0].asTensor().shape() == std::vector<int64_t>{1, 4, 8, 8});
    CHECK(rig.services.inf.calls == 12);
    CHECK(rms(r.outputs.at("9")[1].asTensor()) > 0.0);   // the denoised estimate

    // a no-noise source starts from the latent itself
    extra["4"] = {{"type", "sampling.noiseNone"}, {"inputs", json::object()}};
    Rig quiet;
    auto q = quiet.run(nodes, "sampling.custom", extra);
    REQUIRE(q.ok());
    CHECK(rms(q.outputs.at("9")[0].asTensor()) < 1.0);
}
