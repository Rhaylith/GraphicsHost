#include "graph/Executor.h"
#include "graph/Handles.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::graph;
using nlohmann::json;

TEST_CASE("sampler nodes: they describe a sampler with its settings", "[unit][sampling][nodes]") {
    OutputCache cache(*CachePolicy::parse("none"));
    const auto run = [&](const char* type, json inputs) {
        const JobGraph job = jobFromJson({{"format", "graphicshost.job"}, {"version", 1}, {"nodes", {{"1", {{"type", type}, {"inputs", inputs}}}}}, {"outputs", json::array({"1"})}});
        return Executor(Registry::builtin(), cache).run(job);
    };
    auto r = run("sampling.samplerSelect", {{"sampler", "dpm++ 2M"}});
    REQUIRE(r.ok());
    CHECK(r.outputs.at("1")[0].type == ValueType::Sampler);
    CHECK(r.outputs.at("1")[0].asObject<SamplerHandle>()->name == "dpm++ 2M");
    r = run("sampling.samplerSelect", {{"sampler", "uni pc"}});   // listed in the reference, not available yet
    CHECK_FALSE(r.ok());
    CHECK(r.errorInput == "sampler");
    CHECK(r.errorMessage.find("not available") != std::string::npos);

    r = run("sampling.samplerEulerA", {{"eta", 0.5}, {"noiseScale", 1.02}});
    auto h = r.outputs.at("1")[0].asObject<SamplerHandle>();
    CHECK(h->name == "euler ancestral");
    CHECK(h->options.eta == 0.5);
    CHECK(h->options.sNoise == 1.02);

    r = run("sampling.samplerDpmpp2mSde", {{"solver", "heun"}, {"noiseGenerator", "philox"}});
    h = r.outputs.at("1")[0].asObject<SamplerHandle>();
    CHECK(h->name == "dpm++ 2M SDE");
    CHECK(h->options.solver == "heun");
    CHECK(h->generator == "philox");

    r = run("sampling.samplerDpmppSde", {{"r", 0.4}});
    h = r.outputs.at("1")[0].asObject<SamplerHandle>();
    CHECK(h->name == "dpm++ SDE");
    CHECK(h->options.r == 0.4);
    CHECK(h->generator == "setting default");
    CHECK_FALSE(run("sampling.samplerDpmppSde", {{"r", 0}}).ok());

    CHECK(run("sampling.samplerDpmpp3mSde", json::object()).outputs.at("1")[0].asObject<SamplerHandle>()->name == "dpm++ 3M SDE");
    CHECK(run("sampling.samplerLms", {{"order", 6}}).outputs.at("1")[0].asObject<SamplerHandle>()->options.order == 6);
}
