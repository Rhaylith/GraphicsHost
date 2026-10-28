#include "graph/Handles.h"
#include "graph/Inference.h"
#include "graph/Tips.h"
#include "nodes/Builtin.h"
#include "text/PromptParser.h"

namespace gh::nodes {

using namespace graph;

namespace {

void registerTextEncode(Registry& r) {
    r.add(NodeDef("text.encode")
              .display("Text Encode")
              .category("conditioning")
              .description("Turns a prompt into the conditioning the model is guided by.")
              .searchAliases({"prompt", "clip text encode", "conditioning"})
              .input(String("text").multiline().dynamicPrompts().def("").tip("The prompt. `(word:1.3)` weights a word, `((word))` is \xC3\x97" "1.21, `\\(` types a bracket, `embedding:name` uses a textual inversion, `{a|b}` picks one option per seed, `//` starts a comment, `BREAK` starts a new 75-token chunk. Example: `a (red:1.2) fox, {winter|autumn} forest, embedding:goodhands`."))
              .input(TextEncoder("textEncoder").tip(tips::textEncoder))
              .input(Int("seed").range(0, 9007199254740991).def(0).tip("Picks which option of each `{a|b}` choice in the prompt is used. Example: the seed of the sampler, so every run draws another option."))
              .output(Out("conditioning", ValueType::Conditioning).tip("The encoded prompt."))
              .run([](NodeContext& c) {
                  const std::string text = c.get("text").asString();
                  if (c.services) {
                      const std::string why = c.services->refusePrompt(text);
                      if (!why.empty()) throw NodeError("the safety filter refused this prompt: " + why, "text");
                  }
                  InferenceService* inference = c.services ? c.services->inference() : nullptr;
                  if (!inference) throw NodeError("encoding text needs a model runtime: this job runs without one");
                  const auto encoder = c.get("textEncoder").asObject<TextEncoderHandle>();
                  const text::ParsedPrompt parsed = text::parsePrompt(text, static_cast<uint64_t>(c.get("seed").asInt()));
                  // an empty negative prompt of an SDXL checkpoint is encoded as zeros, as the engine does
                  const bool sdxl = !encoder->files.empty() && encoder->files.front().family == "SDXL";
                  const bool zeroOut = sdxl && text::plainText(parsed).find_first_not_of(" \t\r\n") == std::string::npos;
                  return NodeResult{{inference->encodeText(*encoder, parsed, encoder->clipSkip, 1024, 1024, zeroOut)}};
              }));

    r.add(NodeDef("text.clipSkip")
              .display("Set Text Encoder Last Layer")
              .category("conditioning")
              .description("Uses an earlier layer of the text encoder.")
              .searchAliases({"clip skip", "stop at clip layer"})
              .input(Int("stopAtLayer").range(-24, -1).step(1).def(-1).tip("Use an earlier layer of the text encoder (\"clip skip\"). \xE2\x88\x92" "1 is the last layer. Example: \xE2\x88\x92" "2 for many anime SD 1.5 models trained that way."))
              .input(TextEncoder("textEncoder").tip(tips::textEncoder))
              .output(Out("textEncoder", ValueType::TextEncoder).tip("The text encoder that stops at that layer."))
              .run([](NodeContext& c) {
                  auto encoder = *c.get("textEncoder").asObject<TextEncoderHandle>();
                  encoder.clipSkip = static_cast<int>(-c.get("stopAtLayer").asInt());
                  return NodeResult{{handleValue(ValueType::TextEncoder, std::move(encoder))}};
              }));
}

}  // namespace

void registerTextNodes(Registry& r) {
    registerTextEncode(r);
}

}  // namespace gh::nodes
