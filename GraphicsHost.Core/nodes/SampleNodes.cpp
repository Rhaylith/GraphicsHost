#include "graph/Handles.h"
#include "graph/SampleRun.h"
#include "graph/Tips.h"
#include "nodes/Builtin.h"
#include "sampling/Schedulers.h"

namespace gh::nodes {

using namespace graph;
using sampling::ModelSampling;
using sampling::SamplingParams;
using sampling::makeScheduleDenoise;
using sampling::parseScheduler;
using sampling::samplerAvailable;

namespace {

Param seedInput() { return Int("seed").range(0, 9007199254740991).def(0).seedControl().tip(tips::seed); }
Param stepsInput() { return Int("steps").range(1, 10000).def(20).tip(tips::steps); }
Param cfgInput() { return Float("cfg").range(0, 100).step(0.1).def(7.5).tip(tips::cfg); }

SampleRequest requestFor(const NodeContext& c) {
    SampleRequest r;
    r.inference = c.services ? c.services->inference() : nullptr;
    if (!r.inference) throw NodeError("sampling needs a model runtime: this job runs without one");
    r.defaultRng = c.services->defaultRng();
    r.progress = c.progress;
    const PreviewPolicy preview = c.services->previewPolicy();
    r.previewMethod = preview.method;
    r.previewEvery = preview.everyNSteps;
    r.previewMaxSize = preview.maxSize;
    if (c.preview) r.emitPreview = [preview = c.preview](const tensor::Tensor& image) { preview(Value::makeTensor(ValueType::Image, image)); };
    r.stop = c.stop;
    return r;
}

tensor::Tensor latentOf(const NodeContext& c) {
    const tensor::Tensor& t = c.get("latent").asTensor();
    if (t.ndim() != 4) throw NodeError("a latent has the shape [batch, channels, height, width]", "latent");
    return t;
}

NodeResult outputs(SampleResult&& r) {
    return NodeResult{{Value::makeTensor(ValueType::Latent, std::move(r.output)), Value::makeTensor(ValueType::Latent, std::move(r.denoised))}};
}

GuiderHandle cfgGuider(const NodeContext& c) {
    GuiderHandle g;
    g.kind = GuiderHandle::Kind::Cfg;
    g.model = c.get("model");
    g.conditioning = {c.get("positive"), c.get("negative")};
    g.cfg = c.get("cfg").asFloat();
    return g;
}

SamplerHandle samplerOf(const NodeContext& c) {
    SamplerHandle s;
    s.name = c.get("sampler").asString();
    if (!samplerAvailable(s.name)) throw NodeError("the sampler '" + s.name + "' is not available in this version", "sampler");
    return s;
}

std::vector<float> scheduleOf(const NodeContext& c, double denoise) {
    const auto scheduler = parseScheduler(c.get("scheduler").asString());
    if (!scheduler) throw NodeError("unknown scheduler", "scheduler");
    const auto model = c.get("model").asObject<ModelHandle>();
    return makeScheduleDenoise(*scheduler, ModelSampling(model->sampling.value_or(SamplingParams{})), static_cast<int>(c.get("steps").asInt()), denoise);
}

NoiseHandle noiseOf(const NodeContext& c, bool add) {
    NoiseHandle n;
    n.none = !add;
    n.seed = static_cast<uint64_t>(c.get("seed").asInt());
    n.rng = "setting default";
    return n;
}

void registerSampleNodes(Registry& r) {
    r.add(NodeDef("sampling.sample")
              .display("Sampler")
              .category("sampling")
              .description("Makes the image: starts from noise and removes it step by step, guided by the prompts.")
              .searchAliases({"ksampler", "denoise", "generate", "sample"})
              .input(Model("model").tip(tips::model))
              .input(Conditioning("positive").tip(tips::positive))
              .input(Conditioning("negative").tip(tips::negative))
              .input(Latent("latent").tip(tips::latentIn))
              .input(seedInput())
              .input(stepsInput())
              .input(cfgInput())
              .input(tips::withOptions(Combo("sampler").def("euler"), tips::samplerOptions).tip(tips::sampler))
              .input(tips::withOptions(Combo("scheduler").def("normal"), tips::schedulerOptions).tip(tips::scheduler))
              .input(Float("denoise").range(0, 1).step(0.01).def(1.0).tip(tips::denoise))
              .output(Out("latent", ValueType::Latent).tip("The latent after sampling."))
              .run([](NodeContext& c) {
                  SampleRequest req = requestFor(c);
                  req.guider = cfgGuider(c);
                  req.sampler = samplerOf(c);
                  req.noise = noiseOf(c, true);
                  req.sigmas = scheduleOf(c, c.get("denoise").asFloat());
                  req.latent = latentOf(c);
                  SampleResult res = runSampling(req);
                  return NodeResult{{Value::makeTensor(ValueType::Latent, std::move(res.output))}};
              }));

    r.add(NodeDef("sampling.sampleAdvanced")
              .display("Sampler (advanced)")
              .category("sampling")
              .description("The sampler with control over the noise and the steps it runs, for chaining samplers.")
              .searchAliases({"ksampler advanced", "refiner", "start step", "end step"})
              .input(Model("model").tip(tips::model))
              .input(Conditioning("positive").tip(tips::positive))
              .input(Conditioning("negative").tip(tips::negative))
              .input(Latent("latent").tip(tips::latentIn))
              .input(Bool("addNoise").def(true).tip("Add the seed's noise to the input first. Turn off when the input already carries leftover noise from a previous sampler. Example: off for the second stage of a base \xE2\x86\x92 refiner chain."))
              .input(seedInput())
              .input(stepsInput())
              .input(cfgInput())
              .input(tips::withOptions(Combo("sampler").def("euler"), tips::samplerOptions).tip(tips::sampler))
              .input(tips::withOptions(Combo("scheduler").def("normal"), tips::schedulerOptions).tip(tips::scheduler))
              .input(Int("startStep").range(0, 10000).def(0).tip("First step of the schedule to run. Example: steps 30, start 20 runs only the last 10 steps (like denoise \xE2\x89\x88 0.33 but on the same schedule)."))
              .input(Int("endStep").range(0, 10000).def(10000).tip("Step to stop at (exclusive). Example: steps 30, end 24 hands the last 6 steps to an SDXL refiner."))
              .input(Bool("returnWithNoise").def(false).tip("Return the latent before the final noise is removed, for chaining into another sampler. Example: on for the base pass when a refiner continues from step 24."))
              .output(Out("latent", ValueType::Latent).tip("The latent after sampling."))
              .run([](NodeContext& c) {
                  SampleRequest req = requestFor(c);
                  req.guider = cfgGuider(c);
                  req.sampler = samplerOf(c);
                  req.noise = noiseOf(c, c.get("addNoise").asBool());
                  req.sigmas = scheduleOf(c, 1.0);
                  req.firstStep = static_cast<int>(c.get("startStep").asInt());
                  req.lastStep = static_cast<int>(c.get("endStep").asInt());
                  req.returnWithLeftoverNoise = c.get("returnWithNoise").asBool();
                  req.latent = latentOf(c);
                  SampleResult res = runSampling(req);
                  return NodeResult{{Value::makeTensor(ValueType::Latent, std::move(res.output))}};
              }));

    r.add(NodeDef("sampling.custom")
              .display("Sample (custom)")
              .category("sampling")
              .description("Runs any sampler with any schedule and any guidance method. Outputs the result and the denoised estimate. Example: Random Noise + CFG Guider + dpm++ 2M + Karras Schedule.")
              .searchAliases({"sampler custom", "sample custom"})
              .input(Socket("noise", ValueType::Noise).tip("The noise source. Example: Random Noise."))
              .input(Socket("guider", ValueType::Guider).tip("How the model is called and its predictions combined. Example: CFG Guider."))
              .input(Socket("sampler", ValueType::Sampler).tip("The sampling method. Example: Choose Sampler."))
              .input(Socket("sigmas", ValueType::Sigmas).tip("The noise schedule. Example: Basic Schedule."))
              .input(Latent("latent").tip(tips::latentIn))
              .output(Out("output", ValueType::Latent).tip("The latent after sampling."))
              .output(Out("denoised", ValueType::Latent).tip("The last estimate of the clean latent."))
              .run([](NodeContext& c) {
                  SampleRequest req = requestFor(c);
                  req.guider = *c.get("guider").asObject<GuiderHandle>();
                  req.sampler = *c.get("sampler").asObject<SamplerHandle>();
                  req.noise = *c.get("noise").asObject<NoiseHandle>();
                  req.sigmas = sigmasOf(c.get("sigmas"));
                  req.latent = latentOf(c);
                  return outputs(runSampling(req));
              }));
}

}  // namespace

void registerSampleNodes2(Registry& r) {
    registerSampleNodes(r);
}

}  // namespace gh::nodes
