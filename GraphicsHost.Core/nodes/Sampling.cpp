#include "graph/Handles.h"
#include "graph/Tips.h"
#include "nodes/Builtin.h"
#include "sampling/Rng.h"

namespace gh::nodes {

using namespace graph;

namespace {

NodeResult one(Value v) { return NodeResult{{std::move(v)}}; }

// "setting default" resolves through the services; an unknown name is a mistake in that input.
std::string resolveRng(const NodeContext& c, const char* input) {
    const std::string wanted = c.get(input).asString();
    if (wanted != "setting default") {
        if (!sampling::parseRng(wanted)) throw NodeError("unknown random generator '" + wanted + "'", input);
        return wanted;
    }
    return c.services ? c.services->defaultRng() : "cpu-compatible";
}

const std::string kNoiseGeneratorTip =
    "Random generator for the SDE noise (made on the host, so it is identical on every GPU). *cpu-compatible* matches the generator most tools use for seeds. Example: *cpu-compatible* to reproduce a seed shared from another tool.";

void registerNoise(Registry& r) {
    r.add(NodeDef("sampling.noiseRandom")
              .display("Random Noise")
              .category("sampling")
              .description("Noise made from a seed, for the custom sampling nodes.")
              .searchAliases({"seed", "noise"})
              .input(Int("seed").range(0, 9007199254740991).def(0).seedControl().tip(tips::seed))
              .input(Combo("generator").options({"setting default", "cpu-compatible", "philox"}).def("setting default").tip(kNoiseGeneratorTip))
              .output(Out("noise", ValueType::Noise).tip("The noise source."))
              .run([](NodeContext& c) {
                  NoiseHandle h;
                  h.seed = static_cast<uint64_t>(c.get("seed").asInt());
                  h.rng = resolveRng(c, "generator");
                  return one(handleValue(ValueType::Noise, std::move(h)));
              }));
    r.add(NodeDef("sampling.noiseNone")
              .display("No Noise")
              .category("sampling")
              .description("Supplies zero noise, for continuing from a latent that already has noise.")
              .output(Out("noise", ValueType::Noise).tip("A noise source that adds nothing."))
              .run([](NodeContext&) {
                  NoiseHandle h;
                  h.none = true;
                  return one(handleValue(ValueType::Noise, std::move(h)));
              }));
}

}  // namespace

void registerSamplingNodes(Registry& r) {
    registerNoise(r);
}

}  // namespace gh::nodes
