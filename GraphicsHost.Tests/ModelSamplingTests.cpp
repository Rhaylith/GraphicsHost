#include "graph/Executor.h"
#include "graph/Handles.h"
#include "sampling/ModelSampling.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace gh::sampling;
using Catch::Approx;
using nlohmann::json;

TEST_CASE("model sampling: the SD beta schedule has the known noise range", "[unit][sampling][model-sampling]") {
    ModelSampling m;
    CHECK(m.discreteSigmas().size() == 1000);
    CHECK(m.sigmaMax() == Approx(14.6146).epsilon(0.0005));
    CHECK(m.sigmaMin() == Approx(0.0292).epsilon(0.005));
    for (size_t i = 1; i < 1000; ++i) REQUIRE(m.discreteSigmas()[i] > m.discreteSigmas()[i - 1]);
    // time and noise level convert back and forth, between steps too
    for (const double t : {0.0, 1.0, 10.5, 500.25, 998.9, 999.0}) CHECK(m.timestep(m.sigma(t)) == Approx(t).margin(1e-6));
    CHECK(m.sigma(0) == Approx(m.sigmaMin()));
    CHECK(m.sigma(999) == Approx(m.sigmaMax()));
    CHECK(m.timestep(1e9) == 999.0);
    CHECK(m.timestep(1e-9) == 0.0);
    CHECK(m.inputScale(2.0) == Approx(1.0 / std::sqrt(5.0)));
}

TEST_CASE("model sampling: zero terminal SNR ends with almost no signal", "[unit][sampling][model-sampling]") {
    SamplingParams p;
    p.zeroSnr = true;
    const ModelSampling z(p);
    CHECK(z.sigmaMax() == Approx(std::sqrt((1.0 - 4.8973451890853435e-08) / 4.8973451890853435e-08)).epsilon(1e-9));
    CHECK(z.sigmaMax() > 4000);
    CHECK(z.sigmaMin() == Approx(ModelSampling().sigmaMin()).epsilon(1e-6));   // the first step is unchanged
}

TEST_CASE("model sampling: predictions turn the model output into a clean estimate", "[unit][sampling][model-sampling]") {
    const float x[2] = {1.0f, -2.0f}, out[2] = {0.5f, 0.25f};
    float r[2];
    const auto make = [](SamplingKind kind, Prediction pr) {
        SamplingParams p;
        p.kind = kind;
        p.prediction = pr;
        return ModelSampling(p);
    };
    make(SamplingKind::Discrete, Prediction::Eps).denoise(x, out, 2.0, r, 2);
    CHECK(r[0] == Approx(1.0 - 0.5 * 2.0).margin(1e-6));
    CHECK(r[1] == Approx(-2.0 - 0.25 * 2.0));
    make(SamplingKind::Discrete, Prediction::V).denoise(x, out, 2.0, r, 2);
    CHECK(r[0] == Approx(1.0 / 5.0 - 0.5 * 2.0 / std::sqrt(5.0)));
    make(SamplingKind::Discrete, Prediction::X0).denoise(x, out, 2.0, r, 2);
    CHECK(r[0] == 0.5f);
    make(SamplingKind::Edm, Prediction::Edm).denoise(x, out, 2.0, r, 2);
    const double sd2 = 0.25, s2 = 4.0;
    CHECK(r[0] == Approx(0.5 * (2.0 * 0.5 / std::sqrt(s2 + sd2)) + 1.0 * (sd2 / (s2 + sd2))));
    make(SamplingKind::Flow, Prediction::Eps).denoise(x, out, 0.5, r, 2);
    CHECK(r[0] == Approx(1.0 - 0.5 * 0.5));
    float y[2] = {1.0f, -2.0f};   // in place
    make(SamplingKind::Discrete, Prediction::Eps).denoise(y, out, 2.0, y, 2);
    CHECK(y[0] == Approx(0.0).margin(1e-6));
    // a v-prediction model that is exactly right gives back x0
    const double sigma = 1.5, cIn = 1.0 / std::sqrt(sigma * sigma + 1.0);
    const float x0 = 0.8f, eps = -0.3f;
    const float noisy = static_cast<float>(x0 + sigma * eps);
    const float v = static_cast<float>(cIn * eps - sigma * cIn * x0);
    float est;
    make(SamplingKind::Discrete, Prediction::V).denoise(&noisy, &v, sigma, &est, 1);
    CHECK(est == Approx(x0).margin(1e-5));
}

TEST_CASE("model sampling: continuous and flow models", "[unit][sampling][model-sampling]") {
    SamplingParams e;
    e.kind = SamplingKind::Edm;
    e.prediction = Prediction::Edm;
    ModelSampling edm(e);
    CHECK(edm.sigmaMax() == 120.0);
    CHECK(edm.sigmaMin() == 0.002);
    CHECK(edm.sigma(edm.timestep(7.0)) == Approx(7.0));
    e.prediction = Prediction::V;
    CHECK(ModelSampling(e).sigmaMax() == 500.0);
    e.sigmaMax = 80;
    e.sigmaMin = 0.5;
    CHECK(ModelSampling(e).sigmaMax() == 80.0);

    SamplingParams f;
    f.kind = SamplingKind::Flow;
    f.shift = 3.0;
    ModelSampling flow(f);
    CHECK(flow.sigmaMax() == 1.0);
    CHECK(flow.sigma(1000) == Approx(1.0));
    CHECK(flow.sigma(500) == Approx(3.0 * 0.5 / (1.0 + 2.0 * 0.5)));
    CHECK(flow.timestep(flow.sigma(250)) == Approx(250));
    CHECK(flow.inputScale(0.7) == 1.0);
    float noisy[1];
    const float noise[1] = {2.0f}, latent[1] = {1.0f};
    flow.scaleNoise(0.25, noise, latent, noisy, 1);
    CHECK(noisy[0] == Approx(0.25 * 2.0 + 0.75 * 1.0));
    ModelSampling().scaleNoise(2.0, noise, latent, noisy, 1);
    CHECK(noisy[0] == Approx(2.0 * 2.0 + 1.0));
    float back[1];
    flow.unscaleNoise(0.25, noisy, back, 1);
    CHECK(back[0] == Approx(noisy[0] / 0.75));

    CHECK(flowSigma(0.5, 1.0) == Approx(0.5));   // shift 1 is the identity
    CHECK(flowSigma(0.5, 6.0) > flowSigma(0.5, 3.0));
    CHECK(fluxShift(1.15, 0.5, 256, 256) == Approx(std::exp(0.5)));
    CHECK(fluxShift(1.15, 0.5, 1024, 1024) == Approx(std::exp(1.15)));
    CHECK(fluxShift(1.15, 0.5, 512, 512) < fluxShift(1.15, 0.5, 1024, 1024));

    CHECK(parsePrediction("v prediction") == Prediction::V);
    CHECK(parsePrediction("playground v2.5") == Prediction::PlaygroundV25);
    CHECK_FALSE(parsePrediction("nope").has_value());
    CHECK(std::string(toString(Prediction::ImgToImg)) == "img to img");
}

TEST_CASE("model sampling nodes: they put the settings on the model", "[unit][sampling][nodes]") {
    using namespace gh::graph;
    OutputCache cache(*CachePolicy::parse("none"));
    Registry reg;
    for (const NodeDef* d : Registry::builtin().all()) reg.add(*d);
    reg.add(NodeDef("t.model").output(Out("model", ValueType::Model)).run([](NodeContext&) { return NodeResult{{handleValue(ValueType::Model, ModelHandle{})}}; }));
    const auto go = [&](const char* type, json inputs) {
        inputs["model"] = json::array({"0", 0});
        const JobGraph job = jobFromJson({{"format", "graphicshost.job"},
                                          {"version", 1},
                                          {"nodes", {{"0", {{"type", "t.model"}, {"inputs", json::object()}}}, {"1", {{"type", type}, {"inputs", inputs}}}}},
                                          {"outputs", json::array({"1"})}});
        return Executor(reg, cache).run(job);
    };
    auto r = go("model.samplingDiscrete", {{"prediction", "v prediction"}, {"zeroSnr", true}});
    REQUIRE(r.ok());
    auto p = r.outputs.at("1")[0].asObject<ModelHandle>()->sampling;
    REQUIRE(p.has_value());
    CHECK(p->kind == SamplingKind::Discrete);
    CHECK(p->prediction == Prediction::V);
    CHECK(p->zeroSnr);

    r = go("model.samplingEdm", {{"prediction", "playground v2.5"}, {"sigmaMax", 80}, {"sigmaMin", 0.002}});
    p = r.outputs.at("1")[0].asObject<ModelHandle>()->sampling;
    CHECK(p->kind == SamplingKind::Edm);
    CHECK(p->prediction == Prediction::PlaygroundV25);
    CHECK(p->sigmaMax == 80.0);
    CHECK_FALSE(go("model.samplingEdm", {{"sigmaMax", 1}, {"sigmaMin", 2}}).ok());   // inverted range

    r = go("model.samplingV", json::object());
    p = r.outputs.at("1")[0].asObject<ModelHandle>()->sampling;
    CHECK(p->prediction == Prediction::V);
    CHECK(p->sigmaMax == 500.0);

    r = go("model.samplingShift", {{"shift", 5.0}});
    p = r.outputs.at("1")[0].asObject<ModelHandle>()->sampling;
    CHECK(p->kind == SamplingKind::Flow);
    CHECK(p->shift == 5.0);

    r = go("model.samplingFlux", {{"width", 1024}, {"height", 1024}});
    p = r.outputs.at("1")[0].asObject<ModelHandle>()->sampling;
    CHECK(p->shift == Approx(std::exp(1.15)));
}
