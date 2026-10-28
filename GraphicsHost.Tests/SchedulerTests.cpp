#include "graph/Executor.h"
#include "graph/Handles.h"
#include "sampling/Schedulers.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace gh::sampling;
using Catch::Approx;
using nlohmann::json;

namespace {
bool descending(const std::vector<float>& s) {
    for (size_t i = 1; i < s.size(); ++i)
        if (!(s[i] < s[i - 1])) return false;
    return true;
}
}  // namespace

TEST_CASE("schedulers: every named schedule runs from the top of the model's range to zero", "[unit][sampling][schedulers]") {
    const ModelSampling sd;
    for (const Scheduler s : {Scheduler::Normal, Scheduler::Karras, Scheduler::Exponential, Scheduler::SgmUniform, Scheduler::Simple, Scheduler::DdimUniform, Scheduler::Beta,
                              Scheduler::LinearQuadratic, Scheduler::KlOptimal}) {
        INFO(toString(s));
        const auto sigmas = makeSchedule(s, sd, 20);
        REQUIRE(sigmas.size() >= 2);
        CHECK(sigmas.back() == 0.0f);
        CHECK(descending(sigmas));
        CHECK(sigmas.front() <= sd.sigmaMax() * 1.0001);
        CHECK(sigmas.front() > sd.sigmaMax() * 0.5);   // starts near the top
        CHECK(parseScheduler(toString(s)) == s);
        if (s != Scheduler::Beta && s != Scheduler::DdimUniform) CHECK(sigmas.size() == 21);   // beta drops duplicate steps, ddim follows the stride
    }
    CHECK_FALSE(parseScheduler("fast").has_value());
}

TEST_CASE("schedulers: the formulas", "[unit][sampling][schedulers]") {
    const ModelSampling sd;
    const double smax = sd.sigmaMax(), smin = sd.sigmaMin();

    const auto karras = makeSchedule(Scheduler::Karras, sd, 10);
    CHECK(karras.front() == Approx(smax));
    CHECK(karras[9] == Approx(smin).epsilon(1e-5));
    const double a = std::pow(smax, 1.0 / 7), b = std::pow(smin, 1.0 / 7);
    CHECK(karras[3] == Approx(std::pow(a + (3.0 / 9.0) * (b - a), 7)).epsilon(1e-5));

    const auto expo = makeSchedule(Scheduler::Exponential, sd, 10);
    CHECK(expo[1] / expo[0] == Approx(expo[2] / expo[1]).epsilon(1e-5));   // a constant ratio
    CHECK(expo[9] == Approx(smin).epsilon(1e-5));

    const auto simple = makeSchedule(Scheduler::Simple, sd, 10);
    CHECK(simple[0] == Approx(sd.discreteSigmas()[999]));
    CHECK(simple[1] == Approx(sd.discreteSigmas()[899]));   // every 100th step of the model's own schedule

    const auto sgm = makeSchedule(Scheduler::SgmUniform, sd, 10);
    CHECK(sgm.front() == Approx(smax));
    CHECK(sgm[9] > smin * 2);   // the last timestep is left out, so it stops above the lowest level

    const auto kl = makeSchedule(Scheduler::KlOptimal, sd, 10);
    CHECK(kl.front() == Approx(smax).epsilon(1e-5));
    CHECK(kl[9] == Approx(smin).epsilon(1e-4));

    const auto lq = makeSchedule(Scheduler::LinearQuadratic, sd, 20);
    CHECK(lq.front() == Approx(smax));
    CHECK(lq[1] == Approx(smax * (1.0 - 0.025 / 10)));   // the first half falls linearly with the threshold slope

    const auto ddim = makeSchedule(Scheduler::DdimUniform, sd, 10);
    CHECK(ddim.size() == 11);
    CHECK(ddim.front() == Approx(sd.discreteSigmas()[901]));
}

TEST_CASE("schedulers: parametric schedules", "[unit][sampling][schedulers]") {
    const auto poly = polyExponentialSchedule(10, 14.6, 0.03, 1.0);
    const auto expo = exponentialSchedule(10, 14.6, 0.03);
    for (size_t i = 0; i < poly.size(); ++i) CHECK(poly[i] == Approx(expo[i]).epsilon(1e-5));   // rho 1 is exponential
    CHECK(polyExponentialSchedule(10, 14.6, 0.03, 2.0)[5] < expo[5]);                              // steeper

    const auto vp = vpSchedule(20);
    CHECK(vp.size() == 21);
    CHECK(vp.front() == Approx(std::sqrt(std::exp(19.9 / 2 + 0.1) - 1)).epsilon(1e-5));
    CHECK(descending(vp));

    const auto lap = laplaceSchedule(20, 14.6, 0.03);
    CHECK(lap.size() == 21);
    CHECK(descending(lap));
    CHECK(lap.front() <= 14.6f * 1.0001f);
    CHECK(lap[19] >= 0.03f * 0.999f);

    const ModelSampling sd;
    const auto turbo = turboSchedule(sd, 1);
    REQUIRE(turbo.size() == 2);
    CHECK(turbo[0] == Approx(sd.sigma(999)));
    CHECK(turbo[1] == 0.0f);
    const auto four = turboSchedule(sd, 4);
    CHECK(four.size() == 5);
    CHECK(four[1] == Approx(sd.sigma(899)));
    CHECK(turboSchedule(sd, 50).size() == 11);   // at most 10 steps
}

TEST_CASE("beta distribution: cdf and inverse", "[unit][sampling][schedulers]") {
    CHECK(betaCdf(0.5, 2, 2) == Approx(0.5));
    CHECK(betaCdf(0.25, 2, 2) == Approx(0.15625));   // 3x^2 - 2x^3
    CHECK(betaCdf(0.3, 1, 1) == Approx(0.3));
    CHECK(betaCdf(0, 0.6, 0.6) == 0.0);
    CHECK(betaCdf(1, 0.6, 0.6) == 1.0);
    for (const double p : {0.05, 0.3, 0.5, 0.77, 0.99}) {
        CHECK(betaCdf(betaPpf(p, 0.6, 0.6), 0.6, 0.6) == Approx(p).epsilon(1e-9));
        CHECK(betaCdf(betaPpf(p, 3, 1.5), 3, 1.5) == Approx(p).epsilon(1e-9));
    }
    const ModelSampling sd;
    const auto beta = betaSchedule(sd, 20);
    CHECK(descending(beta));
    CHECK(beta.back() == 0.0f);
    // alpha = beta = 1 is uniform in time, like the normal schedule rounded to whole timesteps
    const auto uniform = betaSchedule(sd, 10, 1.0, 1.0);
    CHECK(uniform.front() == Approx(sd.sigmaMax()));
}

TEST_CASE("schedulers: denoise keeps the end of a longer schedule", "[unit][sampling][schedulers]") {
    const ModelSampling sd;
    const auto full = makeSchedule(Scheduler::Karras, sd, 40);
    const auto half = makeScheduleDenoise(Scheduler::Karras, sd, 20, 0.5);
    REQUIRE(half.size() == 21);
    CHECK(std::equal(half.begin(), half.end(), full.end() - 21));
    CHECK(half.front() < sd.sigmaMax() * 0.5f);
    CHECK(makeScheduleDenoise(Scheduler::Karras, sd, 20, 1.0) == makeSchedule(Scheduler::Karras, sd, 20));
    CHECK(makeScheduleDenoise(Scheduler::Karras, sd, 20, 0.0).empty());
}

TEST_CASE("sigma tools: split, flip, set first, extend", "[unit][sampling][schedulers]") {
    const std::vector<float> s = {10, 5, 2, 1, 0.5f, 0};
    std::vector<float> high, low;
    splitSigmas(s, 2, high, low);
    CHECK(high == std::vector<float>{10, 5, 2});
    CHECK(low == std::vector<float>{2, 1, 0.5f, 0});
    splitSigmas(s, 0, high, low);
    CHECK(high == std::vector<float>{10});
    CHECK(low == s);
    splitSigmas(s, 99, high, low);   // beyond the end: everything in the first part
    CHECK(high == s);
    CHECK(low == std::vector<float>{0});
    splitSigmasDenoise(s, 0.4, high, low);   // 5 steps: the first 3 go to the high part
    CHECK(high == std::vector<float>{10, 5, 2, 1});
    CHECK(low == std::vector<float>{1, 0.5f, 0});
    splitSigmasDenoise(s, 1.0, high, low);
    CHECK(high == std::vector<float>{10});

    CHECK(flipSigmas(s) == std::vector<float>{1e-4f, 0.5f, 1, 2, 5, 10});
    CHECK(flipSigmas({}).empty());
    CHECK(setFirstSigma(s, 136)[0] == 136.0f);
    CHECK(setFirstSigma({}, 1).empty());

    const auto extended = extendSigmas({10, 2, 0}, 2, -1, 12, "linear");
    REQUIRE(extended.size() == 7);
    CHECK(extended[1] == Approx(10 - 8.0 / 3));
    CHECK(extended[2] == Approx(10 - 16.0 / 3));
    CHECK(extended[3] == 2.0f);
    CHECK(extendSigmas({10, 2, 0}, 2, -1, 5, "linear").size() == 5);   // only the pair starting at 2 is inside [-1, 5]
    CHECK(extendSigmas({10, 2, 0}, 1, 5, -1, "linear").size() == 4);   // only the pair starting at 10
    const auto cos = extendSigmas({10, 0}, 3, -1, -1, "cosine");
    CHECK(cos[1] > 7.0f);   // clustered near the start
    CHECK(extendSigmas({10, 0}, 1, -1, -1, "sine")[1] == Approx(10 - 10 * std::sin(3.14159265358979 / 4)));
}

TEST_CASE("schedule nodes: the nodes run the same maths", "[unit][sampling][nodes]") {
    using namespace gh::graph;
    OutputCache cache(*CachePolicy::parse("none"));
    Registry reg;
    for (const NodeDef* d : Registry::builtin().all()) reg.add(*d);
    reg.add(NodeDef("t.model").output(Out("model", ValueType::Model)).run([](NodeContext&) { return NodeResult{{handleValue(ValueType::Model, ModelHandle{})}}; }));
    const auto go = [&](const char* type, json inputs, bool withModel = true) {
        if (withModel) inputs["model"] = json::array({"0", 0});
        const JobGraph job = jobFromJson({{"format", "graphicshost.job"},
                                          {"version", 1},
                                          {"nodes", {{"0", {{"type", "t.model"}, {"inputs", json::object()}}}, {"1", {{"type", type}, {"inputs", inputs}}}}},
                                          {"outputs", json::array({"1"})}});
        return Executor(reg, cache).run(job);
    };
    auto r = go("sampling.scheduleBasic", {{"scheduler", "karras"}, {"steps", 10}});
    REQUIRE(r.ok());
    CHECK(sigmasOf(r.outputs.at("1")[0]) == makeSchedule(Scheduler::Karras, ModelSampling(), 10));
    CHECK(r.outputs.at("1")[0].type == ValueType::Sigmas);

    r = go("sampling.scheduleBasic", {{"scheduler", "simple"}, {"steps", 20}, {"denoise", 0.5}});
    CHECK(sigmasOf(r.outputs.at("1")[0]).size() == 21);
    r = go("sampling.scheduleBasic", {{"denoise", 0.0}});
    CHECK(sigmasOf(r.outputs.at("1")[0]).empty());

    r = go("sampling.scheduleKarras", {{"steps", 5}, {"sigmaMax", 10}, {"sigmaMin", 0.1}, {"rho", 7}}, false);
    REQUIRE(r.ok());
    CHECK(sigmasOf(r.outputs.at("1")[0]) == karrasSchedule(5, 10, 0.1, 7));
    CHECK_FALSE(go("sampling.scheduleKarras", {{"sigmaMax", 1}, {"sigmaMin", 2}}, false).ok());
    CHECK(go("sampling.scheduleVp", {{"steps", 4}}, false).ok());
    CHECK(go("sampling.scheduleLaplace", {{"steps", 4}}, false).ok());
    CHECK(go("sampling.schedulePolyExp", {{"steps", 4}}, false).ok());
    CHECK(go("sampling.scheduleExponential", {{"steps", 4}}, false).ok());
    CHECK(go("sampling.scheduleBeta", {{"steps", 8}}).ok());
    CHECK(sigmasOf(go("sampling.scheduleTurbo", {{"steps", 1}}).outputs.at("1")[0]).size() == 2);

    // the tools, fed a schedule made by a stub node
    reg.add(NodeDef("t.sigmas").output(Out("sigmas", ValueType::Sigmas)).run([](NodeContext&) { return NodeResult{{sigmasValue({10, 5, 2, 1, 0.5f, 0})}}; }));
    const auto tool = [&](const char* type, json inputs) {
        inputs["sigmas"] = json::array({"0", 0});
        const JobGraph job = jobFromJson({{"format", "graphicshost.job"}, {"version", 1}, {"nodes", {{"0", {{"type", "t.sigmas"}, {"inputs", json::object()}}}, {"1", {{"type", type}, {"inputs", inputs}}}}}, {"outputs", json::array({"1"})}});
        return Executor(reg, cache).run(job);
    };
    r = tool("sampling.splitSigmas", {{"step", 2}});
    REQUIRE(r.ok());
    CHECK(sigmasOf(r.outputs.at("1")[0]) == std::vector<float>{10, 5, 2});
    CHECK(sigmasOf(r.outputs.at("1")[1]) == std::vector<float>{2, 1, 0.5f, 0});
    r = tool("sampling.splitSigmasDenoise", {{"denoise", 0.4}});
    CHECK(sigmasOf(r.outputs.at("1")[1]).size() == 3);
    CHECK(sigmasOf(tool("sampling.flipSigmas", json::object()).outputs.at("1")[0]).front() == 1e-4f);
    CHECK(sigmasOf(tool("sampling.setFirstSigma", {{"sigma", 99}}).outputs.at("1")[0]).front() == 99.0f);
    CHECK(sigmasOf(tool("sampling.extendSigmas", {{"steps", 1}}).outputs.at("1")[0]).size() == 11);
}
