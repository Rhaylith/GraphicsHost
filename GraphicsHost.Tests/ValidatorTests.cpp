#include "graph/Validator.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::graph;
using nlohmann::json;

namespace {
Registry registry() {
    Registry r;
    r.add(NodeDef("t.load").output(Out("model", ValueType::Model)).input(Combo("name").modelFolder("checkpoints")));
    r.add(NodeDef("t.sample")
              .input(Model("model"))
              .input(Latent("latent").optional())
              .input(Int("steps").range(1, 100).def(20))
              .input(Float("cfg").range(0, 30).step(0.5).def(7.0))
              .input(Combo("sampler").options({"euler", "heun"}).def("euler"))
              .input(Bool("flag").def(false))
              .input(String("text").def(""))
              .input(Int("seed"))
              .output(Out("latent", ValueType::Latent))
              .validate([](const auto& lit) {
                  std::vector<std::string> bad;
                  if (lit.count("steps") && lit.at("steps") == 13) bad.push_back("Thirteen steps is unlucky.");
                  return bad;
              }));
    r.add(NodeDef("t.save").input(Latent("latent")).isOutput());
    return r;
}

JobGraph job(const json& nodes, json outputs = json::array()) {
    json d = {{"format", "graphicshost.job"}, {"version", 1}, {"nodes", nodes}, {"outputs", outputs}};
    return jobFromJson(d);
}

bool has(const ValidationResult& r, const std::string& node, const std::string& input, const std::string& code) {
    for (const auto& i : r.issues)
        if (i.node == node && i.input == input && i.code == code) return true;
    return false;
}

json goodNodes() {
    return {{"1", {{"type", "t.load"}, {"inputs", {{"name", "a.safetensors"}}}}},
            {"2", {{"type", "t.sample"}, {"inputs", {{"model", json::array({"1", 0})}, {"seed", 1}}}}},
            {"3", {{"type", "t.save"}, {"inputs", {{"latent", json::array({"2", 0})}}}}}};
}
}  // namespace

TEST_CASE("validator: a good job has no issues", "[unit][validator]") {
    const auto r = validate(job(goodNodes()), registry());
    CHECK(r.ok());
    CHECK(r.toJson()["ok"] == true);
}

TEST_CASE("validator: reports every problem, each on its node and input", "[unit][validator]") {
    json n = goodNodes();
    n["2"]["inputs"].erase("seed");                              // required, no default
    n["2"]["inputs"]["steps"] = 500;                             // out of range
    n["2"]["inputs"]["cfg"] = "high";                            // wrong kind
    n["2"]["inputs"]["sampler"] = "rk4";                         // not offered
    n["2"]["inputs"]["flag"] = 1;
    n["2"]["inputs"]["surprise"] = 1;                            // no such input
    n["3"]["inputs"]["latent"] = json::array({"9", 0});          // node missing
    const auto r = validate(job(n), registry());
    CHECK(has(r, "2", "seed", "missing_input"));
    CHECK(has(r, "2", "steps", "out_of_range"));
    CHECK(has(r, "2", "cfg", "bad_value"));
    CHECK(has(r, "2", "sampler", "unknown_option"));
    CHECK(has(r, "2", "flag", "bad_value"));
    CHECK(has(r, "2", "surprise", "unknown_input"));
    CHECK(has(r, "3", "latent", "bad_link"));
    CHECK(r.issues.size() == 7);
    const json j = r.toJson();
    CHECK(j["ok"] == false);
    for (const auto& i : j["issues"])
        if (i["code"] == "out_of_range") {
            CHECK(i["details"]["min"] == 1);
            CHECK(i["details"]["max"] == 100);
        }
}

TEST_CASE("validator: numbers are checked after rounding to the step", "[unit][validator]") {
    json n = goodNodes();
    n["2"]["inputs"]["cfg"] = 30.2;   // rounds to 30.0: fine
    CHECK(validate(job(n), registry()).ok());
    n["2"]["inputs"]["cfg"] = 30.4;   // rounds to 30.5: too high
    CHECK(has(validate(job(n), registry()), "2", "cfg", "out_of_range"));
    n["2"]["inputs"]["cfg"] = 7;
    n["2"]["inputs"]["steps"] = 2.5;
    CHECK(has(validate(job(n), registry()), "2", "steps", "bad_value"));   // not whole
}

TEST_CASE("validator: link types, output indexes and unknown types", "[unit][validator]") {
    json n = goodNodes();
    n["3"]["inputs"]["latent"] = json::array({"1", 0});   // MODEL into LATENT
    auto r = validate(job(n), registry());
    REQUIRE(has(r, "3", "latent", "type_mismatch"));
    CHECK(r.issues[0].details["expected"] == "LATENT");
    CHECK(r.issues[0].details["got"] == "MODEL");

    n = goodNodes();
    n["3"]["inputs"]["latent"] = json::array({"2", 4});
    CHECK(has(validate(job(n), registry()), "3", "latent", "bad_link"));

    n = goodNodes();
    n["4"] = {{"type", "plugin.mystery"}, {"inputs", json::object()}};
    CHECK(has(validate(job(n), registry()), "4", "", "unknown_type"));

    n = goodNodes();
    n["3"]["inputs"]["latent"] = 5;   // a typed value where a link is needed
    CHECK(has(validate(job(n), registry()), "3", "latent", "bad_value"));
}

TEST_CASE("validator: missing models, custom checks, cycles and outputs", "[unit][validator]") {
    ValidateOptions options;
    options.modelExists = [](const std::string& folder, const std::string& name) { return folder == "checkpoints" && name == "a.safetensors"; };
    CHECK(validate(job(goodNodes()), registry(), options).ok());

    json n = goodNodes();
    n["1"]["inputs"]["name"] = "gone.safetensors";
    auto r = validate(job(n), registry(), options);
    REQUIRE(has(r, "1", "name", "missing_model"));
    CHECK(r.issues[0].details["folder"] == "checkpoints");
    CHECK(r.issues[0].details["name"] == "gone.safetensors");

    n = goodNodes();
    n["2"]["inputs"]["steps"] = 13;
    CHECK(has(validate(job(n), registry()), "2", "", "custom"));

    n = goodNodes();
    n["2"]["inputs"]["latent"] = json::array({"2", 0});   // sampler feeds itself
    CHECK(has(validate(job(n), registry()), "2", "", "cycle"));

    CHECK(has(validate(job(goodNodes(), json::array({"77"})), registry()), "77", "", "unknown_output"));

    n = goodNodes();
    n.erase("3");
    CHECK(has(validate(job(n), registry()), "", "", "no_outputs"));
}

TEST_CASE("validator: very long chains are checked without exhausting the stack", "[unit][validator]") {
    json nodes = {{"0", {{"type", "t.load"}, {"inputs", {{"name", "a.safetensors"}}}}}};
    nodes["1"] = {{"type", "t.sample"}, {"inputs", {{"model", json::array({"0", 0})}, {"seed", 1}}}};
    // a very long run of samplers feeding each other's optional latent input
    for (int i = 2; i <= 5000; ++i)
        nodes[std::to_string(i)] = {{"type", "t.sample"}, {"inputs", {{"model", json::array({"0", 0})}, {"seed", 1}, {"latent", json::array({std::to_string(i - 1), 0})}}}};
    nodes["end"] = {{"type", "t.save"}, {"inputs", {{"latent", json::array({"5000", 0})}}}};
    CHECK(validate(job(nodes), registry()).ok());
    nodes["1"]["inputs"]["latent"] = json::array({"5000", 0});   // close the loop
    CHECK(has(validate(job(nodes), registry()), "2500", "", "cycle"));
}
