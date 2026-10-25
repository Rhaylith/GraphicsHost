#include "graph/NodeDef.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::graph;
using nlohmann::json;

namespace {
NodeDef sampleNode() {
    return NodeDef("test.sample")
        .display("Sample")
        .category("test")
        .description("Denoises.")
        .input(Model("model").tip("The model. Example: Load Checkpoint output."))
        .input(Int("seed").range(0, 1e18).def(0).seedControl().tip("Seed. Example: 42."))
        .input(Int("steps").range(1, 10000).def(20).tip("Steps. Example: 20."))
        .input(Float("cfg").range(0, 100).step(0.1).round(0.01).def(7.0).slider().tip("Guidance. Example: 7."))
        .input(Combo("sampler").options({"euler", "heun"}).optionTips({{"euler", "Fast. Example: 20 steps."}}).def("euler").tip("Method. Example: euler."))
        .input(String("note").multiline().dynamicPrompts().optional().tip("Text. Example: a fox."))
        .input(Combo("lora").modelFolder("loras").tip("A LoRA. Example: style.safetensors."))
        .input(AnyIn("later").lazy().optional().tip("Only if needed. Example: a mask."))
        .output(Out("latent", ValueType::Latent).tip("The result."))
        .searchAliases({"denoise", "generate"})
        .isOutput();
}
}  // namespace

TEST_CASE("graph: value types have ids, colours and link rules", "[unit][graph]") {
    for (const ValueType t : allValueTypes()) {
        ValueType back = ValueType::Any;
        REQUIRE(parseTypeId(typeId(t), back));
        CHECK(back == t);
        const std::string color = typeColor(t);
        CHECK(color.size() == 7);
        CHECK(color[0] == '#');
    }
    ValueType unused;
    CHECK_FALSE(parseTypeId("NOT_A_TYPE", unused));
    CHECK(std::string(typeId(ValueType::Any)) == "*");

    CHECK(linkCompatible(ValueType::Model, ValueType::Model));
    CHECK_FALSE(linkCompatible(ValueType::Model, ValueType::Vae));
    CHECK(linkCompatible(ValueType::Image, ValueType::Any));
    CHECK(linkCompatible(ValueType::Any, ValueType::Latent));
    CHECK(linkCompatible(ValueType::Int, ValueType::Float));
    CHECK_FALSE(linkCompatible(ValueType::Float, ValueType::Int));
    CHECK(linkCompatible(ValueType::Combo, ValueType::String));
}

TEST_CASE("graph: values coerce between the scalar types", "[unit][graph]") {
    CHECK(Value::makeInt(3).asFloat() == 3.0);
    CHECK(Value::makeFloat(2.6).asInt() == 3);
    CHECK(Value::makeString("42").asInt() == 42);
    CHECK(Value::makeInt(7).asString() == "7");
    CHECK(Value::makeFloat(1.5).asString() == "1.5");
    CHECK(Value::makeFloat(2.0).asString() == "2");
    CHECK(Value::makeBool(true).asInt() == 1);
    CHECK(Value::makeString("on").asBool());
    CHECK_THROWS(Value::makeString("many").asInt());
    CHECK_THROWS(Value::makeInt(1).asTensor());
    const Value image = Value::makeTensor(ValueType::Image, gh::tensor::Tensor({1, 4, 4, 3}));
    CHECK(image.summary() == "IMAGE 1x4x4x3");
    CHECK(image.approxBytes() == 4 * 4 * 3 * sizeof(float));
    const Value list = Value::makeList(ValueType::Int, {Value::makeInt(1), Value::makeInt(2)});
    CHECK(list.isList());
    CHECK(list.list().size() == 2);
    CHECK(list.summary() == "INT list of 2");
}

TEST_CASE("graph: the registry stores definitions and refuses mistakes", "[unit][graph]") {
    Registry r;
    r.add(sampleNode());
    REQUIRE(r.find("test.sample") != nullptr);
    CHECK(r.find("nope") == nullptr);
    CHECK(r.size() == 1);
    CHECK_THROWS_AS(r.add(sampleNode()), std::invalid_argument);                                     // same id twice
    CHECK_THROWS_AS(r.add(NodeDef("x").input(Int("a")).input(Int("a"))), std::invalid_argument);    // input name twice
    CHECK_THROWS_AS(r.add(NodeDef("")), std::invalid_argument);
    CHECK_THROWS_AS(r.add(NodeDef("y").input(Int(""))), std::invalid_argument);
    const NodeDef* d = r.find("test.sample");
    CHECK(d->findInput("steps")->def == 20);
    CHECK(d->findInput("nothing") == nullptr);
}

TEST_CASE("graph: the catalogue describes every node completely", "[unit][graph]") {
    Registry r;
    r.add(sampleNode());
    r.add(NodeDef("a.first").category("a").display("First"));
    const json c = r.catalogue();
    CHECK(c["version"] == 1);
    CHECK(c["types"].size() == allValueTypes().size());
    REQUIRE(c["nodes"].size() == 2);
    CHECK(c["nodes"][0]["id"] == "a.first");   // sorted by category

    const json& n = c["nodes"][1];
    CHECK(n["id"] == "test.sample");
    CHECK(n["outputNode"] == true);
    CHECK(n["searchAliases"].size() == 2);
    CHECK(n["outputs"][0]["type"] == "LATENT");
    const json& model = n["inputs"][0];
    CHECK(model["type"] == "MODEL");
    CHECK(model["widget"] == false);
    CHECK(model["kind"] == "required");
    const json& seed = n["inputs"][1];
    CHECK(seed["type"] == "INT");
    CHECK(seed["widget"] == true);
    CHECK(seed["seedControl"] == true);
    CHECK(seed["max"] == 1e18);
    const json& cfg = n["inputs"][3];
    CHECK(cfg["display"] == "slider");
    CHECK(cfg["step"] == 0.1);
    CHECK(cfg["round"] == 0.01);
    CHECK(cfg["default"] == 7.0);
    const json& sampler = n["inputs"][4];
    CHECK(sampler["options"][1] == "heun");
    CHECK(sampler["optionTips"]["euler"] == "Fast. Example: 20 steps.");
    const json& note = n["inputs"][5];
    CHECK(note["kind"] == "optional");
    CHECK(note["multiline"] == true);
    CHECK(note["dynamicPrompts"] == true);
    CHECK(n["inputs"][6]["modelFolder"] == "loras");
    CHECK(n["inputs"][7]["lazy"] == true);
}

TEST_CASE("graph: node ids sort the way people read them", "[unit][graph]") {
    NodeIdLess less;
    CHECK(less("2", "10"));
    CHECK_FALSE(less("10", "2"));
    CHECK(less("7", "7:3"));
    CHECK(less("7:3", "7:12"));
    CHECK(less("7:12", "8"));
    CHECK(less("007", "8"));
    CHECK_FALSE(less("5", "5"));
}

TEST_CASE("graph: the job format reads and writes links and literals", "[unit][graph]") {
    const json doc = json::parse(R"({"format":"graphicshost.job","version":1,
      "nodes":{"4":{"type":"loader.checkpoint","inputs":{"checkpoint":"sd15.safetensors"}},
               "7":{"type":"sampling.sample","inputs":{"model":["4",0],"seed":42,"cfg":7.5,"sampler":"euler","flag":true}}},
      "outputs":["7"],"client":{"id":"abc","label":"run"}})");
    const JobGraph g = jobFromJson(doc);
    REQUIRE(g.nodes.size() == 2);
    CHECK(g.nodes.at("7").type == "sampling.sample");
    const auto& model = std::get<NodeRef>(g.nodes.at("7").inputs.at("model"));
    CHECK(model.node == "4");
    CHECK(model.output == 0);
    CHECK(std::get<json>(g.nodes.at("7").inputs.at("seed")) == 42);
    CHECK(std::get<json>(g.nodes.at("7").inputs.at("flag")) == true);
    CHECK(g.outputs == std::vector<std::string>{"7"});
    CHECK(g.clientLabel == "run");

    const json back = toJson(g);
    CHECK(back["nodes"]["7"]["inputs"]["model"] == json::array({"4", 0}));
    CHECK(back["format"] == "graphicshost.job");
    CHECK(jobFromJson(back).nodes.size() == 2);

    // ids given as numbers are accepted in links
    const JobGraph numeric = jobFromJson(json::parse(R"({"format":"graphicshost.job","version":1,"nodes":{"2":{"type":"t","inputs":{"a":[1,0]}}}})"));
    CHECK(std::get<NodeRef>(numeric.nodes.at("2").inputs.at("a")).node == "1");

    CHECK_THROWS(jobFromJson(json::parse(R"({"format":"other","version":1,"nodes":{}})")));
    CHECK_THROWS(jobFromJson(json::parse(R"({"format":"graphicshost.job","version":99,"nodes":{}})")));
    CHECK_THROWS(jobFromJson(json::parse(R"({"format":"graphicshost.job","version":1})")));
    CHECK_THROWS(jobFromJson(json::parse(R"({"format":"graphicshost.job","version":1,"nodes":{"1":{"inputs":{}}}})")));
}
