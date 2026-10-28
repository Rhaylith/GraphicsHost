#include "graph/Handles.h"
#include "graph/Tips.h"
#include "nodes/Builtin.h"

namespace gh::nodes {

using namespace graph;

namespace {

NodeResult guider(GuiderHandle h) { return NodeResult{{handleValue(ValueType::Guider, std::move(h))}}; }

Param cfgInput() { return Float("cfg").range(0, 100).step(0.1).def(8.0).tip(tips::cfg); }

NodeResult patched(const NodeContext& c, const sampling::GuidanceTweaks& tweaks) {
    auto model = *c.get("model").asObject<ModelHandle>();
    model.guidance = tweaks;
    return NodeResult{{handleValue(ValueType::Model, std::move(model))}};
}

// The tweaks of the incoming model, kept when a patch node adds another kind.
sampling::GuidanceTweaks tweaksOf(const NodeContext& c) { return c.get("model").asObject<ModelHandle>()->guidance; }

void registerGuiders(Registry& r) {
    r.add(NodeDef("sampling.guiderBasic")
              .display("Basic Guider")
              .category("sampling")
              .description("Guidance with only the positive prompt.")
              .input(Model("model").tip(tips::model))
              .input(Conditioning("conditioning").tip("Uses only the positive prompt (no CFG). Example: FLUX dev with Flux Guidance."))
              .output(Out("guider", ValueType::Guider).tip("The guider."))
              .run([](NodeContext& c) {
                  GuiderHandle h;
                  h.kind = GuiderHandle::Kind::Basic;
                  h.model = c.get("model");
                  h.conditioning = {c.get("conditioning")};
                  return guider(std::move(h));
              }));

    r.add(NodeDef("sampling.guiderCfg")
              .display("CFG Guider")
              .category("sampling")
              .description("Standard classifier-free guidance.")
              .input(Model("model").tip(tips::model))
              .input(Conditioning("positive").tip(tips::positive))
              .input(Conditioning("negative").tip(tips::negative))
              .input(cfgInput())
              .output(Out("guider", ValueType::Guider).tip("The guider."))
              .run([](NodeContext& c) {
                  GuiderHandle h;
                  h.kind = GuiderHandle::Kind::Cfg;
                  h.model = c.get("model");
                  h.conditioning = {c.get("positive"), c.get("negative")};
                  h.cfg = c.get("cfg").asFloat();
                  return guider(std::move(h));
              }));

    r.add(NodeDef("sampling.guiderDualCfg")
              .display("Dual CFG Guider")
              .category("sampling")
              .description("Guides with two positive prompts at separate strengths (e.g. an edit instruction and an image).")
              .input(Model("model").tip(tips::model))
              .input(Conditioning("condA").tip("The first positive prompt, for example the edit instruction."))
              .input(Conditioning("condB").tip("The second positive prompt, for example the reference image."))
              .input(Conditioning("negative").tip(tips::negative))
              .input(Float("cfgA").range(0, 100).step(0.1).def(8.0).tip("Guidance from the first prompt against the second. Example: 6 for the text instruction."))
              .input(Float("cfgB").range(0, 100).step(0.1).def(8.0).tip("Guidance from the second prompt against the negative. Example: 1.5 for the reference image."))
              .input(Combo("style").options({"regular", "nested"}).def("regular").tip("*nested* applies cfgA on top of the result of cfgB, which some edit models expect. Example: *nested* for instruct-edit SDXL models."))
              .output(Out("guider", ValueType::Guider).tip("The guider."))
              .run([](NodeContext& c) {
                  GuiderHandle h;
                  h.kind = GuiderHandle::Kind::DualCfg;
                  h.model = c.get("model");
                  h.conditioning = {c.get("condA"), c.get("condB"), c.get("negative")};
                  h.cfgA = c.get("cfgA").asFloat();
                  h.cfgB = c.get("cfgB").asFloat();
                  h.nested = c.get("style").asString() == "nested";
                  return guider(std::move(h));
              }));

    r.add(NodeDef("sampling.guiderPerpNeg")
              .display("Perpendicular Negative Guider")
              .category("sampling")
              .description("CFG where the negative only removes what does not overlap with the positive, so \"no glasses\" does not erase the face.")
              .input(Model("model").tip(tips::model))
              .input(Conditioning("positive").tip(tips::positive))
              .input(Conditioning("negative").tip(tips::negative))
              .input(Conditioning("emptyConditioning").tip("The conditioning of an empty prompt: the reference the perpendicular part is measured from. Example: a Text Encode node with empty text."))
              .input(cfgInput())
              .input(Float("negativeScale").range(0, 100).step(0.01).def(1.0).tip("Strength of the perpendicular negative. Example: 1.5 to push harder away from the negative."))
              .output(Out("guider", ValueType::Guider).tip("The guider."))
              .run([](NodeContext& c) {
                  GuiderHandle h;
                  h.kind = GuiderHandle::Kind::PerpNeg;
                  h.model = c.get("model");
                  h.conditioning = {c.get("positive"), c.get("negative"), c.get("emptyConditioning")};
                  h.cfg = c.get("cfg").asFloat();
                  h.negativeScale = c.get("negativeScale").asFloat();
                  return guider(std::move(h));
              }));
}

void registerGuidancePatches(Registry& r) {
    r.add(NodeDef("model.rescaleCfg")
              .display("Rescale CFG")
              .category("model")
              .description("Pulls the guided prediction back toward the un-guided brightness.")
              .input(Model("model").tip(tips::model))
              .input(Float("multiplier").range(0, 1).step(0.01).def(0.7).tip("Pulls the guided prediction back toward the un-guided brightness, fixing over-saturation at high CFG and with zero-SNR models. Example: 0.7."))
              .output(Out("model", ValueType::Model).tip("The model with the rescale."))
              .run([](NodeContext& c) {
                  auto t = tweaksOf(c);
                  t.rescale = c.get("multiplier").asFloat();
                  return patched(c, t);
              }));

    r.add(NodeDef("model.cfgZeroStar")
              .display("CFG-Zero*")
              .category("model")
              .description("Optimised scale of the negative prediction.")
              .input(Model("model").tip(tips::model))
              .input(Int("zeroInitSteps").range(0, 10).def(0).tip("Steps at the start where the prediction is zeroed (skips unreliable early guidance). Example: 1 for FLUX/Wan."))
              .output(Out("model", ValueType::Model).tip("The model with CFG-Zero*."))
              .run([](NodeContext& c) {
                  auto t = tweaksOf(c);
                  t.zeroStar = true;
                  t.zeroInitSteps = static_cast<int>(c.get("zeroInitSteps").asInt());
                  return patched(c, t);
              }));

    r.add(NodeDef("model.apg")
              .display("Adaptive Projected Guidance")
              .category("model")
              .description("Guidance that keeps the update from over-saturating the image.")
              .input(Model("model").tip(tips::model))
              .input(Float("eta").range(-10, 10).step(0.01).def(1.0).tip("How much of the guidance parallel to the prediction is kept; lower reduces over-saturation. Example: 0 removes the parallel part completely."))
              .input(Float("normThreshold").range(0, 50).step(0.1).def(5.0).tip("Caps the size of the guidance update. Example: 5; lower for gentler colours."))
              .input(Float("momentum").range(-5, 1).step(0.01).def(0.0).tip("Averages guidance over steps; negative values sharpen. Example: -0.5."))
              .output(Out("model", ValueType::Model).tip("The model with adaptive projected guidance."))
              .run([](NodeContext& c) {
                  auto t = tweaksOf(c);
                  t.apg = true;
                  t.apgEta = c.get("eta").asFloat();
                  t.apgNormThreshold = c.get("normThreshold").asFloat();
                  t.apgMomentum = c.get("momentum").asFloat();
                  return patched(c, t);
              }));
}

}  // namespace

void registerGuiderNodes(Registry& r) {
    registerGuiders(r);
    registerGuidancePatches(r);
}

}  // namespace gh::nodes
