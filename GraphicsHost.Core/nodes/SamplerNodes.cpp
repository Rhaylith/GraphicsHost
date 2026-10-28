#include "graph/Handles.h"
#include "graph/Tips.h"
#include "nodes/Builtin.h"
#include "sampling/Rng.h"

namespace gh::nodes {

using namespace graph;
using namespace sampling;

namespace {

NodeResult sampler(std::string name, SamplerOptions options, std::string generator = "setting default") {
    SamplerHandle h;
    h.name = std::move(name);
    h.options = std::move(options);
    h.generator = std::move(generator);
    return NodeResult{{handleValue(ValueType::Sampler, std::move(h))}};
}

std::string generatorOf(const NodeContext& c) {
    const std::string g = c.get("noiseGenerator").asString();
    if (g != "setting default" && !parseRng(g)) throw NodeError("unknown random generator '" + g + "'", "noiseGenerator");
    return g;
}

Param etaInput(double def) {
    return Float("eta").range(0, 100).step(0.01).def(def).tip("Amount of fresh noise per step; 0 turns it into the plain deterministic sampler. Example: 0.5 for a middle ground.");
}
Param noiseScaleInput() { return Float("noiseScale").range(0, 100).step(0.01).def(1.0).tip("Multiplier on that fresh noise. Example: 1.02 slightly increases texture."); }
Param generatorInput() {
    return Combo("noiseGenerator")
        .options({"setting default", "cpu-compatible", "philox"})
        .def("setting default")
        .tip("Random generator for the SDE noise (made on the host, so it is identical on every GPU). *cpu-compatible* matches the generator most tools use for seeds. Example: *cpu-compatible* to reproduce a seed shared from another tool.");
}

void registerSamplerNodes(Registry& r) {
    r.add(NodeDef("sampling.samplerSelect")
              .display("Choose Sampler")
              .category("sampling")
              .description("Picks a sampler by name.")
              .searchAliases({"euler", "dpm++", "ddim", "heun"})
              .input(tips::withOptions(Combo("sampler").def("euler"), tips::samplerOptions).tip(tips::sampler))
              .output(Out("sampler", ValueType::Sampler).tip("The chosen sampler."))
              .run([](NodeContext& c) {
                  const std::string name = c.get("sampler").asString();
                  if (!samplerAvailable(name)) throw NodeError("the sampler '" + name + "' is not available in this version", "sampler");
                  return sampler(name, {});
              }));

    r.add(NodeDef("sampling.samplerEulerA")
              .display("Euler Ancestral Sampler")
              .category("sampling")
              .description("Euler ancestral with its noise settings.")
              .input(etaInput(1.0))
              .input(noiseScaleInput())
              .output(Out("sampler", ValueType::Sampler).tip("The sampler."))
              .run([](NodeContext& c) {
                  SamplerOptions o;
                  o.eta = c.get("eta").asFloat();
                  o.sNoise = c.get("noiseScale").asFloat();
                  return sampler("euler ancestral", o);
              }));

    r.add(NodeDef("sampling.samplerDpmpp2mSde")
              .display("DPM++ 2M SDE Sampler")
              .category("sampling")
              .description("DPM++ 2M SDE with its settings.")
              .input(Combo("solver").options({"midpoint", "heun"}).def("midpoint").tip("Second-order correction method. Example: *heun* is a little sharper."))
              .input(etaInput(1.0))
              .input(noiseScaleInput())
              .input(generatorInput())
              .output(Out("sampler", ValueType::Sampler).tip("The sampler."))
              .run([](NodeContext& c) {
                  SamplerOptions o;
                  o.solver = c.get("solver").asString();
                  o.eta = c.get("eta").asFloat();
                  o.sNoise = c.get("noiseScale").asFloat();
                  return sampler("dpm++ 2M SDE", o, generatorOf(c));
              }));

    r.add(NodeDef("sampling.samplerDpmpp3mSde")
              .display("DPM++ 3M SDE Sampler")
              .category("sampling")
              .description("DPM++ 3M SDE with its settings.")
              .input(etaInput(1.0))
              .input(noiseScaleInput())
              .input(generatorInput())
              .output(Out("sampler", ValueType::Sampler).tip("The sampler."))
              .run([](NodeContext& c) {
                  SamplerOptions o;
                  o.eta = c.get("eta").asFloat();
                  o.sNoise = c.get("noiseScale").asFloat();
                  return sampler("dpm++ 3M SDE", o, generatorOf(c));
              }));

    r.add(NodeDef("sampling.samplerDpmppSde")
              .display("DPM++ SDE Sampler")
              .category("sampling")
              .description("DPM++ SDE with its settings.")
              .input(etaInput(1.0))
              .input(noiseScaleInput())
              .input(generatorInput())
              .input(Float("r").range(0, 100).step(0.01).def(0.5).tip("Position of the intermediate step. Example: 0.5 (midpoint)."))
              .output(Out("sampler", ValueType::Sampler).tip("The sampler."))
              .run([](NodeContext& c) {
                  SamplerOptions o;
                  o.eta = c.get("eta").asFloat();
                  o.sNoise = c.get("noiseScale").asFloat();
                  o.r = c.get("r").asFloat();
                  if (o.r <= 0) throw NodeError("r must be above 0", "r");
                  return sampler("dpm++ SDE", o, generatorOf(c));
              }));

    r.add(NodeDef("sampling.samplerLms")
              .display("LMS Sampler")
              .category("sampling")
              .description("Linear multistep with a chosen order.")
              .input(Int("order").range(1, 100).def(4).tip("Number of previous steps combined. Example: 4."))
              .output(Out("sampler", ValueType::Sampler).tip("The sampler."))
              .run([](NodeContext& c) {
                  SamplerOptions o;
                  o.order = static_cast<int>(c.get("order").asInt());
                  return sampler("lms", o);
              }));
}

}  // namespace

void registerSamplerListNodes(Registry& r) {
    registerSamplerNodes(r);
}

}  // namespace gh::nodes
