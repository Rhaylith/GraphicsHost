#include "graph/Handles.h"
#include "graph/Tips.h"
#include "nodes/Builtin.h"

namespace gh::nodes {

using namespace graph;
using sampling::Prediction;
using sampling::SamplingKind;
using sampling::SamplingParams;

namespace {

// A model with its sampling parameters replaced: the node gives the model a way to relate noise levels to its output.
NodeResult withSampling(const NodeContext& c, SamplingParams params) {
    auto model = *c.get("model").asObject<ModelHandle>();
    model.sampling = params;
    return NodeResult{{handleValue(ValueType::Model, std::move(model))}};
}

void registerModelSampling(Registry& r) {
    r.add(NodeDef("model.samplingDiscrete")
              .display("Model Sampling (discrete)")
              .category("model")
              .description("Sets what the model predicts and whether its schedule reaches zero signal.")
              .searchAliases({"v prediction", "zero snr", "eps", "lcm"})
              .input(Model("model").tip(tips::model))
              .input(tips::withOptions(Combo("prediction").def("eps"), tips::predictionTypeOptions)
                         .tip("Overrides what the model predicts when the file does not say. Example: *v prediction* for an SD 2.1 768 checkpoint that renders grey."))
              .input(Bool("zeroSnr").def(false).tip("Rescale the schedule to reach true zero signal at the end; for models trained with zero terminal SNR. Example: on for zero-SNR v-prediction SDXL fine-tunes; gives real blacks."))
              .output(Out("model", ValueType::Model).tip("The model with these sampling settings."))
              .run([](NodeContext& c) {
                  SamplingParams p;
                  p.kind = SamplingKind::Discrete;
                  const auto prediction = sampling::parsePrediction(c.get("prediction").asString());
                  if (!prediction) throw NodeError("unknown prediction type", "prediction");
                  p.prediction = *prediction;
                  p.zeroSnr = c.get("zeroSnr").asBool();
                  return withSampling(c, p);
              }));

    r.add(NodeDef("model.samplingEdm")
              .display("Model Sampling (continuous EDM)")
              .category("model")
              .description("Continuous noise levels for EDM-style models.")
              .searchAliases({"edm", "playground", "continuous"})
              .input(Model("model").tip(tips::model))
              .input(Combo("prediction").options({"v prediction", "edm", "eps", "playground v2.5"}).def("v prediction").tip("Prediction type for continuous-noise models. Example: *playground v2.5* for that model."))
              .input(Float("sigmaMax").range(0.001, 100000).step(0.001).def(120.0).tip("Highest noise level of the model. Example: use the value from the model card."))
              .input(Float("sigmaMin").range(0.0001, 1000).step(0.0001).def(0.002).tip("Lowest noise level of the model. Example: use the value from the model card."))
              .output(Out("model", ValueType::Model).tip("The model with these sampling settings."))
              .run([](NodeContext& c) {
                  SamplingParams p;
                  p.kind = SamplingKind::Edm;
                  const auto prediction = sampling::parsePrediction(c.get("prediction").asString());
                  if (!prediction) throw NodeError("unknown prediction type", "prediction");
                  p.prediction = *prediction;
                  p.sigmaMax = c.get("sigmaMax").asFloat();
                  p.sigmaMin = c.get("sigmaMin").asFloat();
                  if (p.sigmaMin >= p.sigmaMax) throw NodeError("sigmaMin must be lower than sigmaMax", "sigmaMin");
                  return withSampling(c, p);
              }));

    r.add(NodeDef("model.samplingV")
              .display("Model Sampling (continuous V)")
              .category("model")
              .description("Continuous noise levels for v-prediction models.")
              .searchAliases({"v prediction", "continuous"})
              .input(Model("model").tip(tips::model))
              .input(Float("sigmaMax").range(0.001, 100000).step(0.001).def(500.0).tip("Highest noise level of the model. Example: use the value from the model card."))
              .input(Float("sigmaMin").range(0.0001, 1000).step(0.0001).def(0.03).tip("Lowest noise level of the model. Example: use the value from the model card."))
              .output(Out("model", ValueType::Model).tip("The model with these sampling settings."))
              .run([](NodeContext& c) {
                  SamplingParams p;
                  p.kind = SamplingKind::Edm;
                  p.prediction = Prediction::V;
                  p.sigmaMax = c.get("sigmaMax").asFloat();
                  p.sigmaMin = c.get("sigmaMin").asFloat();
                  if (p.sigmaMin >= p.sigmaMax) throw NodeError("sigmaMin must be lower than sigmaMax", "sigmaMin");
                  return withSampling(c, p);
              }));

    r.add(NodeDef("model.samplingShift")
              .display("Model Sampling (flow shift)")
              .category("model")
              .description("Shift of the noise schedule of flow models.")
              .searchAliases({"shift", "sd3", "auraflow", "flow"})
              .input(Model("model").tip(tips::model))
              .input(Float("shift").range(0, 100).step(0.01).def(3.0).tip("For flow models (SD3, AuraFlow, Wan, HiDream ...): higher shift spends more steps at high noise, which helps composition at high resolution. Example: 3.0 for SD3.5 at 1024\xC2\xB2; 5\xE2\x80\x93" "8 for Wan video."))
              .output(Out("model", ValueType::Model).tip("The model with this shift."))
              .run([](NodeContext& c) {
                  SamplingParams p;
                  p.kind = SamplingKind::Flow;
                  p.shift = c.get("shift").asFloat();
                  return withSampling(c, p);
              }));

    r.add(NodeDef("model.samplingFlux")
              .display("Model Sampling (FLUX)")
              .category("model")
              .description("Shift of the flow schedule that grows with the image size.")
              .searchAliases({"flux", "shift", "flow"})
              .input(Model("model").tip(tips::model))
              .input(Float("maxShift").range(0, 100).step(0.01).def(1.15).tip("Largest shift: used for images of 4096 tokens (1024\xC2\xB2 and up). Example: raise to 1.5 for very large images."))
              .input(Float("baseShift").range(0, 100).step(0.01).def(0.5).tip("Shift for the smallest images (256 tokens, 256\xC2\xB2). Example: 0.5."))
              .input(Int("width").range(16, 16384).step(8).def(1024).tip("The image size the shift is computed for. Example: set to the render size."))
              .input(Int("height").range(16, 16384).step(8).def(1024).tip("Height of that image size. Example: 768 for a landscape render."))
              .output(Out("model", ValueType::Model).tip("The model with the shift for this size."))
              .run([](NodeContext& c) {
                  SamplingParams p;
                  p.kind = SamplingKind::Flow;
                  p.shift = sampling::fluxShift(c.get("maxShift").asFloat(), c.get("baseShift").asFloat(), static_cast<int>(c.get("width").asInt()), static_cast<int>(c.get("height").asInt()));
                  return withSampling(c, p);
              }));
}

}  // namespace

void registerModelSamplingNodes(Registry& r) {
    registerModelSampling(r);
}

}  // namespace gh::nodes
