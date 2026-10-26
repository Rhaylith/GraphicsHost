#include "graph/Handles.h"
#include "graph/Tips.h"
#include "models/Precision.h"
#include "nodes/Builtin.h"

namespace gh::nodes {

using namespace graph;

namespace {

NodeResult results(std::vector<Value> values) { return NodeResult{std::move(values)}; }

const NodeServices& servicesOf(const NodeContext& c) {
    if (!c.services) throw NodeError("models cannot be looked up here: the job runs without a model library");
    return *c.services;
}

// A model file of a kind, looked up by the combo value. A missing file is a mistake in that input.
ModelFileRef modelFile(const NodeContext& c, const char* kind, const char* input) {
    const std::string name = c.get(input).asString();
    std::string problem;
    const auto found = servicesOf(c).findModel(kind, name, &problem);
    if (!found) throw NodeError(problem.empty() ? "the model '" + name + "' was not found" : problem, input);
    return *found;
}

// The dtype a handle carries: what the node says, else what the setting says; "default" when it is the file's own format.
std::string weightDtype(const std::string& nodeValue, const std::string& settingValue) {
    const models::DtypeChoice choice = models::resolveDtype(nodeValue, settingValue);
    if (choice.precision == models::Precision::Auto) return "default";
    return std::string(models::toString(choice.precision)) + (choice.fast ? " fast" : "");
}

const std::vector<std::string> kDevices = {"auto", "vulkan0", "vulkan1", "vulkan2", "vulkan3"};
const std::vector<std::string> kTriState = {"auto", "on", "off"};
const std::vector<std::string> kWeightDtypes = {"default", "fp8 e4m3", "fp8 e4m3 fast", "fp8 e5m2"};

const std::string kFlashTip =
    "A faster, memory-saving attention kernel. *auto* turns it on for SD 1.x above 512\xC2\xB2 on Vulkan, where it was measured 5\xC3\x97 faster. Example: on for SD 1.5 at 768\xC2\xB2.";
const std::string kOffloadTip =
    "Keep the weights in system RAM and stream them to the GPU, for cards with little VRAM. Measured almost free on SDXL (3.04 vs 2.89 s/step). Example: on for SDXL on an 8 GB card.";

const std::string kLoraTip = "A small add-on that teaches a style, character or concept. Example: `watercolor_v2.safetensors`; add its trigger word to the prompt if it has one.";
const std::string kLoraModelStrengthTip =
    "How strongly the LoRA changes the diffusion model; negative values push away from it. Example: 0.7 for a softer style; \xE2\x88\x92" "0.5 to suppress what the LoRA adds.";
const std::string kLoraTextStrengthTip =
    "How strongly the LoRA changes the text encoder (how words are understood). Example: 0 to apply only the visual part; 1.0 when the trigger word matters.";

void registerCheckpointLoaders(Registry& r) {
    r.add(NodeDef("loader.checkpoint")
              .display("Load Checkpoint")
              .category("loaders")
              .description("Loads a checkpoint: the diffusion model, its text encoder(s) and its VAE.")
              .searchAliases({"model", "load model", "safetensors"})
              .input(Combo("checkpoint").modelFolder("checkpoints").tip("A single file holding the diffusion model, text encoder(s) and VAE. Example: `sd_xl_base_1.0.safetensors` gives MODEL, TEXT_ENCODER and VAE outputs for SDXL."))
              .input(Combo("device").options(kDevices).def("auto").tip("Which Vulkan GPU runs the model. *auto* uses the GPU chosen in Settings (`gpuDevice`). Example: *vulkan1* to render on a second card while the first drives the desktop."))
              .input(Combo("flashAttention").options(kTriState).def("auto").tip(kFlashTip))
              .input(Combo("offloadToRam").options(kTriState).def("auto").tip(kOffloadTip))
              .input(Int("idleUnloadMinutes").range(0, 1440).def(10).tip("Unload the model after this many idle minutes; 0 keeps it loaded. Example: 0 on a dedicated server."))
              .output(Out("model", ValueType::Model).tip("The diffusion model."))
              .output(Out("textEncoder", ValueType::TextEncoder).tip("The text encoder(s) inside the checkpoint."))
              .output(Out("vae", ValueType::Vae).tip("The VAE inside the checkpoint."))
              .run([](NodeContext& c) {
                  const ModelFileRef file = modelFile(c, "checkpoint", "checkpoint");
                  LoadSettings load;
                  load.device = c.get("device").asString();
                  load.flashAttention = c.get("flashAttention").asString();
                  load.offloadToRam = c.get("offloadToRam").asString();
                  load.idleUnloadMinutes = static_cast<int>(c.get("idleUnloadMinutes").asInt());
                  const PrecisionPolicy policy = servicesOf(c).precisionPolicy();
                  ModelHandle model{file, load, {}};
                  model.load.weightDtype = weightDtype("default", policy.diffusion);
                  TextEncoderHandle encoders;
                  encoders.files = {file};
                  encoders.fromCheckpoint = true;
                  encoders.load = load;
                  encoders.load.weightDtype = weightDtype("default", policy.textEncoder);
                  VaeHandle vae;
                  vae.file = file;
                  vae.fromCheckpoint = true;
                  vae.load = load;
                  vae.load.weightDtype = weightDtype("default", policy.vae);
                  vae.fallbackFp32 = policy.vaeFallbackFp32;
                  return results({handleValue(ValueType::Model, std::move(model)), handleValue(ValueType::TextEncoder, std::move(encoders)), handleValue(ValueType::Vae, std::move(vae))});
              }));

    r.add(NodeDef("loader.diffusionModel")
              .display("Load Diffusion Model")
              .category("loaders")
              .description("Loads a diffusion model that comes without text encoder and VAE.")
              .searchAliases({"unet", "flux", "dit"})
              .input(Combo("model").modelFolder("diffusion_models").tip("A diffusion model without text encoder or VAE (common for FLUX, SD3.5, Wan). Load the encoders and VAE separately. Example: `flux1-dev.safetensors`."))
              .input(Combo("weightDtype").options(kWeightDtypes).def("default").tip(tips::common("weightDtype", "*fp8 e4m3 fast* also computes in fp8 where the GPU supports it.")))
              .output(Out("model", ValueType::Model).tip("The diffusion model."))
              .run([](NodeContext& c) {
                  ModelHandle model;
                  model.file = modelFile(c, "diffusion_model", "model");
                  model.load.weightDtype = weightDtype(c.get("weightDtype").asString(), servicesOf(c).precisionPolicy().diffusion);
                  return results({handleValue(ValueType::Model, std::move(model))});
              }));
}

void registerEncoderAndVae(Registry& r) {
    const std::string encoderTip =
        "One file per encoder; the family decides how many are needed. Example: FLUX needs `clip_l.safetensors` and `t5xxl_fp16.safetensors`; SD3.5 adds `clip_g.safetensors`; HiDream uses four.";
    r.add(NodeDef("loader.textEncoders")
              .display("Load Text Encoders")
              .category("loaders")
              .description("Loads one to four separate text encoder files.")
              .searchAliases({"clip", "t5", "text encoder"})
              .input(Combo("encoder1").modelFolder("text_encoders").tip(encoderTip))
              .input(Combo("encoder2").modelFolder("text_encoders").optional().tip(encoderTip))
              .input(Combo("encoder3").modelFolder("text_encoders").optional().tip(encoderTip))
              .input(Combo("encoder4").modelFolder("text_encoders").optional().tip(encoderTip))
              .input(Combo("family")
                         .options({"auto", "sd1", "sdxl", "sd3", "flux", "hunyuan video", "hidream", "stable cascade", "stable audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "chroma", "ace", "omnigen2", "qwen image"})
                         .def("auto")
                         .tip("Which model family these encoders are for; *auto* guesses from the files. Example: set *flux* when a T5 file alone is ambiguous."))
              .input(Combo("device").options({"auto", "GPU weights in VRAM", "GPU weights in RAM"}).def("auto").tip(tips::device))
              .output(Out("textEncoder", ValueType::TextEncoder).tip("The loaded encoders, as one text encoder."))
              .run([](NodeContext& c) {
                  TextEncoderHandle h;
                  for (const char* input : {"encoder1", "encoder2", "encoder3", "encoder4"}) {
                      if (!c.has(input) || c.get(input).asString().empty()) continue;
                      h.files.push_back(modelFile(c, "text_encoder", input));
                  }
                  if (h.files.empty()) throw NodeError("choose at least one text encoder file", "encoder1");
                  h.family = c.get("family").asString();
                  h.load.device = c.get("device").asString();
                  h.load.weightDtype = weightDtype("default", servicesOf(c).precisionPolicy().textEncoder);
                  return results({handleValue(ValueType::TextEncoder, std::move(h))});
              }));

    r.add(NodeDef("loader.vae")
              .display("Load VAE")
              .category("loaders")
              .description("Loads a VAE, a tiny preview decoder, or selects pixel space.")
              .searchAliases({"autoencoder", "taesd"})
              .input(Combo("vae").modelFolder("vae").tip("The autoencoder to use instead of the checkpoint's. Example: `sdxl_vae_fp16fix.safetensors`; *taesdxl* decodes very fast at lower quality; *pixel space* for models that work directly on pixels."))
              .output(Out("vae", ValueType::Vae).tip("The autoencoder."))
              .run([](NodeContext& c) {
                  VaeHandle h;
                  const PrecisionPolicy policy = servicesOf(c).precisionPolicy();
                  h.load.weightDtype = weightDtype("default", policy.vae);
                  h.fallbackFp32 = policy.vaeFallbackFp32;
                  if (c.get("vae").asString() == "pixel space") {
                      h.pixelSpace = true;
                  } else {
                      std::string problem;
                      const auto found = servicesOf(c).findModel("vae", c.get("vae").asString(), &problem);   // a VAE, else a preview decoder
                      if (!found) throw NodeError(problem.empty() ? "the VAE '" + c.get("vae").asString() + "' was not found" : problem, "vae");
                      h.file = *found;
                  }
                  return results({handleValue(ValueType::Vae, std::move(h))});
              }));
}

PatchRef patchFrom(const NodeContext& c, bool withText) {
    PatchRef p;
    p.lora = modelFile(c, "lora", "lora");
    p.strengthModel = c.get("strengthModel").asFloat();
    p.strengthText = withText ? c.get("strengthText").asFloat() : 0.0;
    return p;
}

void registerLoras(Registry& r) {
    r.add(NodeDef("loader.lora")
              .display("Load LoRA")
              .category("loaders")
              .description("Adds a LoRA to a diffusion model and its text encoder.")
              .searchAliases({"lora", "lycoris", "adapter", "style"})
              .input(Model("model").tip(tips::model))
              .input(TextEncoder("textEncoder").tip(tips::textEncoder))
              .input(Combo("lora").modelFolder("loras").tip(kLoraTip))
              .input(Float("strengthModel").range(-100, 100).step(0.01).def(1.0).tip(kLoraModelStrengthTip))
              .input(Float("strengthText").range(-100, 100).step(0.01).def(1.0).tip(kLoraTextStrengthTip))
              .output(Out("model", ValueType::Model).tip("The diffusion model with the LoRA added."))
              .output(Out("textEncoder", ValueType::TextEncoder).tip("The text encoder with the LoRA added."))
              .run([](NodeContext& c) {
                  const PatchRef patch = patchFrom(c, true);
                  auto model = *c.get("model").asObject<ModelHandle>();
                  auto encoder = *c.get("textEncoder").asObject<TextEncoderHandle>();
                  model.patches.push_back(patch);
                  encoder.patches.push_back(patch);
                  return results({handleValue(ValueType::Model, std::move(model)), handleValue(ValueType::TextEncoder, std::move(encoder))});
              }));

    r.add(NodeDef("loader.loraModel")
              .display("Load LoRA (model only)")
              .category("loaders")
              .description("As Load LoRA, for models whose LoRAs do not touch the text encoder. Example: a FLUX style LoRA at 0.9.")
              .searchAliases({"lora", "lycoris", "adapter", "style"})
              .input(Model("model").tip(tips::model))
              .input(Combo("lora").modelFolder("loras").tip(kLoraTip))
              .input(Float("strengthModel").range(-100, 100).step(0.01).def(1.0).tip(kLoraModelStrengthTip))
              .output(Out("model", ValueType::Model).tip("The diffusion model with the LoRA added."))
              .run([](NodeContext& c) {
                  auto model = *c.get("model").asObject<ModelHandle>();
                  model.patches.push_back(patchFrom(c, false));
                  return results({handleValue(ValueType::Model, std::move(model))});
              }));
}

}  // namespace

void registerLoaderNodes(Registry& r) {
    registerCheckpointLoaders(r);
    registerEncoderAndVae(r);
    registerLoras(r);
}

}  // namespace gh::nodes
