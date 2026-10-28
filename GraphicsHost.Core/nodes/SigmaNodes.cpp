#include "graph/Handles.h"
#include "graph/Tips.h"
#include "nodes/Builtin.h"
#include "sampling/Schedulers.h"

namespace gh::nodes {

using namespace graph;
using namespace sampling;

namespace {

const char* const kSigmasTip = "The noise schedule to change. Example: the output of Basic Schedule.";
const char* const kSigmaMaxTip = "Highest noise level. Example: 14.6 for SD 1.5; lowering it gives a gentler start.";
const char* const kSigmaMinTip = "Lowest noise level. Example: 0.03 for SD 1.5.";
const char* const kRhoTip = "Curve shape: higher spends more steps at low noise. Example: 7 (the paper value).";

ModelSampling samplingOf(const NodeContext& c) {
    const auto model = c.get("model").asObject<ModelHandle>();
    return ModelSampling(model->sampling.value_or(SamplingParams{}));
}

int stepsOf(const NodeContext& c) { return static_cast<int>(c.get("steps").asInt()); }

NodeResult schedule(std::vector<float> sigmas) { return NodeResult{{sigmasValue(sigmas)}}; }

Param stepsInput(int def, int max = 10000) { return Int("steps").range(1, max).def(def).tip(tips::steps); }
Param sigmaMax(double def) { return Float("sigmaMax").range(0, 5000).step(0.01).def(def).tip(kSigmaMaxTip); }
Param sigmaMin(double def) { return Float("sigmaMin").range(0, 5000).step(0.001).def(def).tip(kSigmaMinTip); }

void checkRange(const NodeContext& c) {
    if (c.get("sigmaMin").asFloat() >= c.get("sigmaMax").asFloat() || c.get("sigmaMin").asFloat() <= 0)
        throw NodeError("sigmaMin must be above 0 and below sigmaMax", "sigmaMin");
}

void registerSchedules(Registry& r) {
    r.add(NodeDef("sampling.scheduleBasic")
              .display("Basic Schedule")
              .category("sampling")
              .description("A named noise schedule for a model.")
              .searchAliases({"scheduler", "sigmas", "karras", "simple"})
              .input(Model("model").tip(tips::model))
              .input(tips::withOptions(Combo("scheduler").def("simple"), tips::schedulerOptions).tip(tips::scheduler))
              .input(stepsInput(20))
              .input(Float("denoise").range(0, 1).step(0.01).def(1.0).tip(tips::denoise))
              .output(Out("sigmas", ValueType::Sigmas).tip("The noise levels, from the highest to 0."))
              .run([](NodeContext& c) {
                  const auto scheduler = parseScheduler(c.get("scheduler").asString());
                  if (!scheduler) throw NodeError("unknown scheduler", "scheduler");
                  return schedule(makeScheduleDenoise(*scheduler, samplingOf(c), stepsOf(c), c.get("denoise").asFloat()));
              }));

    r.add(NodeDef("sampling.scheduleKarras")
              .display("Karras Schedule")
              .category("sampling")
              .description("The Karras noise schedule with its own range.")
              .input(stepsInput(20))
              .input(sigmaMax(14.61))
              .input(sigmaMin(0.03))
              .input(Float("rho").range(0, 100).step(0.01).def(7.0).tip(kRhoTip))
              .output(Out("sigmas", ValueType::Sigmas).tip("The noise levels, from the highest to 0."))
              .run([](NodeContext& c) {
                  checkRange(c);
                  return schedule(karrasSchedule(stepsOf(c), c.get("sigmaMax").asFloat(), c.get("sigmaMin").asFloat(), c.get("rho").asFloat()));
              }));

    r.add(NodeDef("sampling.scheduleExponential")
              .display("Exponential Schedule")
              .category("sampling")
              .description("Noise falls evenly in log scale.")
              .input(stepsInput(20))
              .input(sigmaMax(14.61))
              .input(sigmaMin(0.03))
              .output(Out("sigmas", ValueType::Sigmas).tip("The noise levels, from the highest to 0."))
              .run([](NodeContext& c) {
                  checkRange(c);
                  return schedule(exponentialSchedule(stepsOf(c), c.get("sigmaMax").asFloat(), c.get("sigmaMin").asFloat()));
              }));

    r.add(NodeDef("sampling.schedulePolyExp")
              .display("Poly-exponential Schedule")
              .category("sampling")
              .description("Between exponential and steeper curves.")
              .input(stepsInput(20))
              .input(sigmaMax(14.61))
              .input(sigmaMin(0.03))
              .input(Float("rho").range(0, 100).step(0.01).def(1.0).tip("Between exponential (1) and steeper curves. Example: 2."))
              .output(Out("sigmas", ValueType::Sigmas).tip("The noise levels, from the highest to 0."))
              .run([](NodeContext& c) {
                  checkRange(c);
                  return schedule(polyExponentialSchedule(stepsOf(c), c.get("sigmaMax").asFloat(), c.get("sigmaMin").asFloat(), c.get("rho").asFloat()));
              }));

    r.add(NodeDef("sampling.scheduleVp")
              .display("VP Schedule")
              .category("sampling")
              .description("The variance-preserving noise schedule.")
              .input(stepsInput(20))
              .input(Float("betaD").range(0, 100).step(0.01).def(19.9).tip("Variance-preserving schedule parameter. Example: 19.9 reproduces the classic VP curve."))
              .input(Float("betaMin").range(0, 100).step(0.01).def(0.1).tip("Minimum of the variance-preserving beta. Example: 0.1."))
              .input(Float("epsS").range(0, 1).step(0.0001).def(0.001).tip("Smallest time of the schedule. Example: 0.001."))
              .output(Out("sigmas", ValueType::Sigmas).tip("The noise levels, from the highest to 0."))
              .run([](NodeContext& c) { return schedule(vpSchedule(stepsOf(c), c.get("betaD").asFloat(), c.get("betaMin").asFloat(), c.get("epsS").asFloat())); }));

    r.add(NodeDef("sampling.scheduleBeta")
              .display("Beta Schedule")
              .category("sampling")
              .description("Timesteps drawn from a beta distribution.")
              .input(Model("model").tip(tips::model))
              .input(stepsInput(20))
              .input(Float("alpha").range(0, 50).step(0.01).def(0.6).tip("Shape of the beta distribution; values under 1 put more steps at both ends. Example: 0.6."))
              .input(Float("beta").range(0, 50).step(0.01).def(0.6).tip("Second shape of the beta distribution. Example: 0.6."))
              .output(Out("sigmas", ValueType::Sigmas).tip("The noise levels, from the highest to 0."))
              .run([](NodeContext& c) {
                  const ModelSampling m = samplingOf(c);
                  if (m.discreteSigmas().empty()) throw NodeError("the beta schedule needs a model with discrete timesteps", "model");
                  return schedule(betaSchedule(m, stepsOf(c), c.get("alpha").asFloat(), c.get("beta").asFloat()));
              }));

    r.add(NodeDef("sampling.scheduleLaplace")
              .display("Laplace Schedule")
              .category("sampling")
              .description("Noise levels spread by a Laplace distribution in log noise.")
              .input(stepsInput(20))
              .input(sigmaMax(14.61))
              .input(sigmaMin(0.03))
              .input(Float("mu").range(-10, 10).step(0.01).def(0.0).tip("Centre of the Laplace distribution in log noise. Example: 0."))
              .input(Float("beta").range(0, 10).step(0.01).def(0.5).tip("Width of the Laplace distribution in log noise. Example: 0.5."))
              .output(Out("sigmas", ValueType::Sigmas).tip("The noise levels, from the highest to 0."))
              .run([](NodeContext& c) {
                  checkRange(c);
                  return schedule(laplaceSchedule(stepsOf(c), c.get("sigmaMax").asFloat(), c.get("sigmaMin").asFloat(), c.get("mu").asFloat(), c.get("beta").asFloat()));
              }));

    r.add(NodeDef("sampling.scheduleTurbo")
              .display("Turbo Schedule")
              .category("sampling")
              .description("Schedule for one-to-few-step Turbo models.")
              .input(Model("model").tip(tips::model))
              .input(stepsInput(1, 10))
              .output(Out("sigmas", ValueType::Sigmas).tip("The noise levels, from the highest to 0."))
              .run([](NodeContext& c) { return schedule(turboSchedule(samplingOf(c), stepsOf(c))); }));
}

void registerSigmaTools(Registry& r) {
    r.add(NodeDef("sampling.splitSigmas")
              .display("Split Sigmas")
              .category("sampling")
              .description("Cuts a schedule in two at a step.")
              .input(Socket("sigmas", ValueType::Sigmas).tip(kSigmasTip))
              .input(Int("step").range(0, 10000).def(0).tip("Cuts a schedule in two at this step, for two-stage sampling. Example: 20 steps split at 15: base model runs 15, refiner runs 5."))
              .output(Out("high", ValueType::Sigmas).tip("The noisy first part, up to and including the step."))
              .output(Out("low", ValueType::Sigmas).tip("The last part, from the step on."))
              .run([](NodeContext& c) {
                  std::vector<float> high, low;
                  splitSigmas(sigmasOf(c.get("sigmas")), static_cast<int>(c.get("step").asInt()), high, low);
                  return NodeResult{{sigmasValue(high), sigmasValue(low)}};
              }));

    r.add(NodeDef("sampling.splitSigmasDenoise")
              .display("Split Sigmas by Denoise")
              .category("sampling")
              .description("Cuts a schedule so the second part covers a fraction of it.")
              .input(Socket("sigmas", ValueType::Sigmas).tip(kSigmasTip))
              .input(Float("denoise").range(0, 1).step(0.01).def(1.0).tip("Cuts so the second part covers this fraction. Example: 0.25 gives the last quarter to the second sampler."))
              .output(Out("high", ValueType::Sigmas).tip("The noisy first part."))
              .output(Out("low", ValueType::Sigmas).tip("The last part."))
              .run([](NodeContext& c) {
                  std::vector<float> high, low;
                  splitSigmasDenoise(sigmasOf(c.get("sigmas")), c.get("denoise").asFloat(), high, low);
                  return NodeResult{{sigmasValue(high), sigmasValue(low)}};
              }));

    r.add(NodeDef("sampling.flipSigmas")
              .display("Flip Sigmas")
              .category("sampling")
              .description("Reverses a schedule to go from image to noise.")
              .input(Socket("sigmas", ValueType::Sigmas).tip("Reverses the schedule to go from image to noise (inversion). Example: find the noise that recreates a photo."))
              .output(Out("sigmas", ValueType::Sigmas).tip("The reversed schedule."))
              .run([](NodeContext& c) { return schedule(flipSigmas(sigmasOf(c.get("sigmas")))); }));

    r.add(NodeDef("sampling.setFirstSigma")
              .display("Set First Sigma")
              .category("sampling")
              .description("Replaces the starting noise level of a schedule.")
              .input(Socket("sigmas", ValueType::Sigmas).tip(kSigmasTip))
              .input(Float("sigma").range(0, 20000).step(0.01).def(136.0).tip("Replaces the starting noise level. Example: 136 for a fully noised start on a high-noise model."))
              .output(Out("sigmas", ValueType::Sigmas).tip("The schedule with the new first level."))
              .run([](NodeContext& c) { return schedule(setFirstSigma(sigmasOf(c.get("sigmas")), static_cast<float>(c.get("sigma").asFloat()))); }));

    r.add(NodeDef("sampling.extendSigmas")
              .display("Extend Intermediate Sigmas")
              .category("sampling")
              .description("Inserts more levels between the existing ones.")
              .input(Socket("sigmas", ValueType::Sigmas).tip(kSigmasTip))
              .input(Int("steps").range(1, 100).def(2).tip("New steps inserted between each pair. Example: 2 turns 10 steps into 28."))
              .input(Float("startAt").range(-1, 1000).step(0.01).def(-1.0).tip("Only extend where the noise level is at least this (-1 = no limit). Example: 0."))
              .input(Float("endAt").range(-1, 1000).step(0.01).def(12.0).tip("Only extend where the noise level is at most this (-1 = no limit). Example: 2 adds steps only in the fine-detail phase."))
              .input(Combo("spacing").options({"linear", "cosine", "sine"}).def("linear").tip("How the new points are spaced. Example: *cosine* clusters them near the ends."))
              .output(Out("sigmas", ValueType::Sigmas).tip("The longer schedule."))
              .run([](NodeContext& c) {
                  return schedule(extendSigmas(sigmasOf(c.get("sigmas")), static_cast<int>(c.get("steps").asInt()), c.get("startAt").asFloat(), c.get("endAt").asFloat(), c.get("spacing").asString()));
              }));
}

}  // namespace

void registerSigmaNodes(Registry& r) {
    registerSchedules(r);
    registerSigmaTools(r);
}

}  // namespace gh::nodes
