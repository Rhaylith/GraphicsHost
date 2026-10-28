#include "graph/Executor.h"
#include "graph/Handles.h"
#include "sampling/Guidance.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace gh::sampling;
using Catch::Approx;
using nlohmann::json;

namespace {
double dotp(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * b[i];
    return s;
}
std::vector<float> sub(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}
double stddev(const std::vector<float>& a) {
    double m = 0;
    for (float v : a) m += v;
    m /= static_cast<double>(a.size());
    double s = 0;
    for (float v : a) s += (v - m) * (v - m);
    return std::sqrt(s / static_cast<double>(a.size() - 1));
}
const std::vector<float> kX = {0.5f, -1.0f, 0.3f, 2.0f, -0.7f, 1.1f};
const std::vector<float> kCond = {0.2f, -0.4f, 0.9f, 1.5f, -0.2f, 0.6f};
const std::vector<float> kUncond = {0.1f, -0.1f, 0.4f, 0.8f, 0.3f, 0.2f};
}  // namespace

TEST_CASE("guidance: classifier-free guidance", "[unit][sampling][guidance]") {
    std::vector<float> out(6);
    cfgCombine(kCond.data(), kUncond.data(), 7.5, out.data(), 6);
    for (size_t i = 0; i < 6; ++i) CHECK(out[i] == Approx(kUncond[i] + 7.5 * (kCond[i] - kUncond[i])));
    cfgCombine(kCond.data(), kUncond.data(), 1.0, out.data(), 6);   // scale 1 is the positive prediction
    for (size_t i = 0; i < 6; ++i) CHECK(out[i] == Approx(kCond[i]));
    cfgCombine(kCond.data(), kUncond.data(), 0.0, out.data(), 6);   // scale 0 the negative
    for (size_t i = 0; i < 6; ++i) CHECK(out[i] == Approx(kUncond[i]));

    GuidanceTweaks none;
    GuidanceState state;
    std::vector<float> g(6), plain(6);
    guide(none, state, kX.data(), kCond.data(), kUncond.data(), 4.0, 6, 1, g.data());
    cfgCombine(kCond.data(), kUncond.data(), 4.0, plain.data(), 6);
    for (size_t i = 0; i < 6; ++i) CHECK(g[i] == Approx(plain[i]));
    CHECK(state.step == 1);
}

TEST_CASE("guidance: CFG-Zero* and its zero-initialised steps", "[unit][sampling][guidance]") {
    GuidanceTweaks t;
    t.zeroStar = true;
    t.zeroInitSteps = 2;
    GuidanceState state;
    std::vector<float> g(6);
    guide(t, state, kX.data(), kCond.data(), kUncond.data(), 5.0, 6, 1, g.data());
    for (size_t i = 0; i < 6; ++i) CHECK(g[i] == kX[i]);   // step 0: a zero noise prediction leaves the input as it is
    guide(t, state, kX.data(), kCond.data(), kUncond.data(), 5.0, 6, 1, g.data());
    for (size_t i = 0; i < 6; ++i) CHECK(g[i] == kX[i]);   // step 1
    guide(t, state, kX.data(), kCond.data(), kUncond.data(), 1.0, 6, 1, g.data());   // step 2, scale 1: the positive prediction
    for (size_t i = 0; i < 6; ++i) CHECK(g[i] == Approx(kCond[i]).margin(1e-5));

    // equal predictions stay equal whatever the scale
    GuidanceState fresh;
    GuidanceTweaks z;
    z.zeroStar = true;
    guide(z, fresh, kX.data(), kCond.data(), kCond.data(), 9.0, 6, 1, g.data());
    for (size_t i = 0; i < 6; ++i) CHECK(g[i] == Approx(kCond[i]).margin(1e-5));
    // the optimised scale: when the negative noise prediction is exactly half the positive one, the combination ignores the difference to it
    std::vector<float> condEps = {1, 2, 3, 4, 5, 6}, uncondEps(6), cond(6), uncond(6), x(6, 0.0f);
    for (size_t i = 0; i < 6; ++i) {
        uncondEps[i] = condEps[i] / 2;
        cond[i] = x[i] - condEps[i];
        uncond[i] = x[i] - uncondEps[i];
    }
    GuidanceState s2;
    guide(z, s2, x.data(), cond.data(), uncond.data(), 7.0, 6, 1, g.data());
    for (size_t i = 0; i < 6; ++i) CHECK(g[i] == Approx(-uncondEps[i] * 2.0).margin(1e-4));   // st = 2: e = 2 * eu + 7 * (ec - 2 * eu) = 2 * eu
}

TEST_CASE("guidance: rescale brings the spread of the result back", "[unit][sampling][guidance]") {
    GuidanceTweaks t;
    t.rescale = 1.0;
    GuidanceState state;
    std::vector<float> g(6);
    guide(t, state, kX.data(), kCond.data(), kUncond.data(), 12.0, 6, 1, g.data());
    CHECK(stddev(sub(kX, g)) == Approx(stddev(sub(kX, kCond))).epsilon(1e-5));   // the noise prediction has the spread of the positive one
    t.rescale = 0.0;
    GuidanceState s2;
    std::vector<float> plain(6);
    guide(t, s2, kX.data(), kCond.data(), kUncond.data(), 12.0, 6, 1, plain.data());
    CHECK(stddev(sub(kX, plain)) > stddev(sub(kX, kCond)) * 1.5);                  // plain high CFG overshoots
    t.rescale = 0.5;
    GuidanceState s3;
    std::vector<float> half(6);
    guide(t, s3, kX.data(), kCond.data(), kUncond.data(), 12.0, 6, 1, half.data());
    for (size_t i = 0; i < 6; ++i) CHECK(half[i] == Approx(0.5 * g[i] + 0.5 * plain[i]).margin(1e-4));
}

TEST_CASE("guidance: adaptive projected guidance", "[unit][sampling][guidance]") {
    GuidanceTweaks t;
    t.apg = true;
    t.apgNormThreshold = 0.0;
    GuidanceState s1;
    std::vector<float> g(6), plain(6);
    guide(t, s1, kX.data(), kCond.data(), kUncond.data(), 4.0, 6, 1, g.data());
    cfgCombine(kCond.data(), kUncond.data(), 4.0, plain.data(), 6);
    for (size_t i = 0; i < 6; ++i) CHECK(g[i] == Approx(plain[i]).margin(1e-5));   // eta 1, no cap, no momentum: ordinary CFG

    t.apgEta = 0.0;   // the update loses its part along the positive prediction
    GuidanceState s2;
    guide(t, s2, kX.data(), kCond.data(), kUncond.data(), 4.0, 6, 1, g.data());
    CHECK(dotp(sub(g, kCond), kCond) == Approx(0.0).margin(1e-5));

    t.apgEta = 1.0;
    t.apgNormThreshold = 0.3;   // the update is capped before it is scaled
    GuidanceState s3;
    guide(t, s3, kX.data(), kCond.data(), kUncond.data(), 4.0, 6, 1, g.data());
    CHECK(std::sqrt(dotp(sub(g, kCond), sub(g, kCond))) == Approx(0.3 * 3.0).epsilon(1e-4));

    // momentum: the second step adds the average of the first
    t.apgNormThreshold = 0.0;
    t.apgMomentum = -0.5;
    GuidanceState s4;
    std::vector<float> first(6), second(6);
    guide(t, s4, kX.data(), kCond.data(), kUncond.data(), 2.0, 6, 1, first.data());
    guide(t, s4, kX.data(), kCond.data(), kUncond.data(), 2.0, 6, 1, second.data());
    for (size_t i = 0; i < 6; ++i) {
        const double d = kCond[i] - kUncond[i];
        CHECK(second[i] == Approx(kCond[i] + (d - 0.5 * d)).margin(1e-5));   // the update is d + (-0.5) * d
    }
}

TEST_CASE("guidance: dual CFG and perpendicular negatives", "[unit][sampling][guidance]") {
    const std::vector<float> a = {1, 2, 3}, b = {0.5f, 1, 1}, n = {0, 0.5f, 0};
    std::vector<float> out(3);
    dualCfg(a.data(), b.data(), n.data(), 2.0, 3.0, false, out.data(), 3);
    for (size_t i = 0; i < 3; ++i) CHECK(out[i] == Approx(n[i] + 3.0 * (b[i] - n[i]) + 2.0 * (a[i] - b[i])));
    dualCfg(a.data(), b.data(), n.data(), 2.0, 3.0, true, out.data(), 3);
    for (size_t i = 0; i < 3; ++i) CHECK(out[i] == Approx(n[i] + 3.0 * ((b[i] + 2.0 * (a[i] - b[i])) - n[i])));
    std::vector<float> reg(3), nes(3);
    dualCfg(a.data(), b.data(), n.data(), 0.0, 4.0, false, reg.data(), 3);
    dualCfg(a.data(), b.data(), n.data(), 0.0, 4.0, true, nes.data(), 3);
    CHECK(reg == nes);   // with no first-prompt guidance both styles are one CFG against the negative

    const std::vector<float> empty = {0, 0, 0, 0}, pos = {1, 0, 0, 0};
    std::vector<float> o(4);
    const std::vector<float> parallelNeg = {2, 0, 0, 0};   // along the positive prompt: nothing is perpendicular
    perpNegCfg(pos.data(), parallelNeg.data(), empty.data(), 5.0, 1.0, 4, 1, o.data());
    CHECK(o[0] == Approx(5.0));
    CHECK(o[1] == Approx(0.0));
    const std::vector<float> orthogonalNeg = {0, 1, 0, 0};   // fully perpendicular: removed at the negative scale
    perpNegCfg(pos.data(), orthogonalNeg.data(), empty.data(), 5.0, 1.5, 4, 1, o.data());
    CHECK(o[0] == Approx(5.0));
    CHECK(o[1] == Approx(-5.0 * 1.5));
    const std::vector<float> mixedNeg = {3, 1, 0, 0};   // only the 1 in the second slot is perpendicular
    perpNegCfg(pos.data(), mixedNeg.data(), empty.data(), 2.0, 1.0, 4, 1, o.data());
    CHECK(o[0] == Approx(2.0));
    CHECK(o[1] == Approx(-2.0));
    // a batch is treated item by item
    const std::vector<float> p2 = {1, 0, 0, 1}, n2 = {0, 1, 1, 0}, e2 = {0, 0, 0, 0};
    std::vector<float> o2(4);
    perpNegCfg(p2.data(), n2.data(), e2.data(), 1.0, 1.0, 2, 2, o2.data());
    CHECK(o2[0] == Approx(1.0));
    CHECK(o2[1] == Approx(-1.0));
    CHECK(o2[2] == Approx(-1.0));
    CHECK(o2[3] == Approx(1.0));
}

TEST_CASE("guider nodes: they record how to guide, patches add up", "[unit][sampling][nodes]") {
    using namespace gh::graph;
    OutputCache cache(*CachePolicy::parse("none"));
    Registry reg;
    for (const NodeDef* d : Registry::builtin().all()) reg.add(*d);
    reg.add(NodeDef("t.model").output(Out("model", ValueType::Model)).run([](NodeContext&) { return NodeResult{{handleValue(ValueType::Model, ModelHandle{})}}; }));
    reg.add(NodeDef("t.cond").input(Int("tag").def(0)).output(Out("c", ValueType::Conditioning)).run([](NodeContext& c) { return NodeResult{{c.get("tag")}}; }));
    const auto go = [&](const json& nodes, const char* out = "9") {
        const JobGraph job = jobFromJson({{"format", "graphicshost.job"}, {"version", 1}, {"nodes", nodes}, {"outputs", json::array({out})}});
        return Executor(reg, cache).run(job);
    };
    const json base = {{"0", {{"type", "t.model"}, {"inputs", json::object()}}},
                       {"1", {{"type", "t.cond"}, {"inputs", {{"tag", 1}}}}},
                       {"2", {{"type", "t.cond"}, {"inputs", {{"tag", 2}}}}},
                       {"3", {{"type", "t.cond"}, {"inputs", {{"tag", 3}}}}}};
    const auto with = [&](json extra) {
        json n = base;
        for (const auto& [k, v] : extra.items()) n[k] = v;
        return n;
    };
    auto r = go(with({{"9", {{"type", "sampling.guiderCfg"}, {"inputs", {{"model", {"0", 0}}, {"positive", {"1", 0}}, {"negative", {"2", 0}}, {"cfg", 6.5}}}}}}));
    REQUIRE(r.ok());
    auto g = r.outputs.at("9")[0].asObject<GuiderHandle>();
    CHECK(g->kind == GuiderHandle::Kind::Cfg);
    CHECK(g->cfg == 6.5);
    REQUIRE(g->conditioning.size() == 2);
    CHECK(g->conditioning[0].asInt() == 1);
    CHECK(g->conditioning[1].asInt() == 2);
    CHECK(r.outputs.at("9")[0].type == ValueType::Guider);

    r = go(with({{"9", {{"type", "sampling.guiderBasic"}, {"inputs", {{"model", {"0", 0}}, {"conditioning", {"1", 0}}}}}}}));
    CHECK(r.outputs.at("9")[0].asObject<GuiderHandle>()->kind == GuiderHandle::Kind::Basic);

    r = go(with({{"9", {{"type", "sampling.guiderDualCfg"}, {"inputs", {{"model", {"0", 0}}, {"condA", {"1", 0}}, {"condB", {"2", 0}}, {"negative", {"3", 0}}, {"cfgA", 6}, {"cfgB", 1.5}, {"style", "nested"}}}}}}));
    g = r.outputs.at("9")[0].asObject<GuiderHandle>();
    CHECK(g->kind == GuiderHandle::Kind::DualCfg);
    CHECK(g->nested);
    CHECK(g->cfgA == 6.0);
    CHECK(g->conditioning.size() == 3);

    r = go(with({{"9", {{"type", "sampling.guiderPerpNeg"}, {"inputs", {{"model", {"0", 0}}, {"positive", {"1", 0}}, {"negative", {"2", 0}}, {"emptyConditioning", {"3", 0}}, {"negativeScale", 1.5}}}}}}));
    g = r.outputs.at("9")[0].asObject<GuiderHandle>();
    CHECK(g->kind == GuiderHandle::Kind::PerpNeg);
    CHECK(g->negativeScale == 1.5);

    // the patches stack: rescale, then CFG-Zero*, then APG
    r = go(with({{"4", {{"type", "model.rescaleCfg"}, {"inputs", {{"model", {"0", 0}}, {"multiplier", 0.6}}}}},
                 {"5", {{"type", "model.cfgZeroStar"}, {"inputs", {{"model", {"4", 0}}, {"zeroInitSteps", 1}}}}},
                 {"9", {{"type", "model.apg"}, {"inputs", {{"model", {"5", 0}}, {"eta", 0.5}, {"normThreshold", 4}, {"momentum", -0.25}}}}}}));
    REQUIRE(r.ok());
    const auto m = r.outputs.at("9")[0].asObject<ModelHandle>();
    CHECK(m->guidance.rescale == 0.6);
    CHECK(m->guidance.zeroStar);
    CHECK(m->guidance.zeroInitSteps == 1);
    CHECK(m->guidance.apg);
    CHECK(m->guidance.apgEta == 0.5);
    CHECK(m->guidance.apgNormThreshold == 4.0);
    CHECK(m->guidance.apgMomentum == -0.25);
}
