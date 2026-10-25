#include "graph/Workflow.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace gh::graph;
using nlohmann::json;

namespace {
Registry testRegistry() {
    Registry r;
    r.add(NodeDef("t.load").output(Out("model", ValueType::Model)).input(String("name").def("a.safetensors")));
    r.add(NodeDef("t.lora").input(Model("model")).input(Float("strength").def(1.0)).output(Out("model", ValueType::Model)));
    r.add(NodeDef("t.sample").input(Model("model")).input(Int("seed").def(0)).input(Int("steps").def(20)).output(Out("latent", ValueType::Latent)));
    r.add(NodeDef("t.save").input(Socket("latent", ValueType::Latent)).isOutput());
    return r;
}

json node(int id, const char* type, json widgets = json::object(), const char* mode = "active") {
    return {{"id", id}, {"type", type}, {"mode", mode}, {"widgets", widgets}};
}
json link(int from, int out, int to, const char* input) {
    return {{"from", json::array({from, out})}, {"to", json::array({to, input})}};
}
json document(json nodes, json links, json extra = json::object()) {
    json d = {{"format", "graphicshost.workflow"}, {"version", 1}, {"id", "wf"}, {"nodes", nodes}, {"links", links}};
    for (const auto& [k, v] : extra.items()) d[k] = v;
    return d;
}
JobGraph convert(const json& doc) { return toJob(workflowFromJson(doc), testRegistry()); }
}  // namespace

TEST_CASE("workflow: a document round trips and keeps what it does not understand", "[unit][workflow]") {
    json d = document(json::array({node(1, "t.load"), node(2, "t.sample", {{"seed", 5}, {"seedControl", "randomize"}})}),
                      json::array({link(1, 0, 2, "model")}),
                      {{"revision", 12},
                       {"future", {{"a", 1}}},
                       {"extra", {{"plugin", "data"}}},
                       {"view", {{"offset", json::array({3, 4})}, {"scale", 1.5}}},
                       {"groups", json::array({{{"id", 1}, {"title", "Upscale"}, {"bounds", json::array({0, 0, 800, 400})}, {"color", "#335"}}})},
                       {"reroutes", json::array({{{"id", 1}, {"pos", json::array({300, 200})}, {"link", 0}}})}});
    d["nodes"][1]["title"] = "Sampler";
    d["nodes"][1]["pos"] = json::array({400, 120});
    d["nodes"][1]["futureNodeKey"] = "kept";
    const Workflow wf = workflowFromJson(d);
    CHECK(wf.revision == 12);
    CHECK(wf.body.nodes.size() == 2);
    CHECK(wf.body.nodes[1].title == "Sampler");
    CHECK(wf.body.nodes[1].widgets["seedControl"] == "randomize");
    CHECK(wf.body.links[0].toInput == "model");
    CHECK(wf.groups[0].bounds[2] == 800);
    CHECK(wf.reroutes[0].pos[1] == 200);

    const json back = toJson(wf);
    CHECK(back["future"]["a"] == 1);
    CHECK(back["extra"]["plugin"] == "data");
    CHECK(back["view"]["scale"] == 1.5);
    CHECK(back["nodes"][1]["futureNodeKey"] == "kept");
    CHECK(back["nodes"][1]["widgets"]["seed"] == 5);
    CHECK(back["nodes"][1]["mode"] == "active");
    CHECK(back["links"][0]["from"] == json::array({1, 0}));
    CHECK(back["links"][0]["to"] == json::array({2, "model"}));
    CHECK(toJson(workflowFromJson(back)) == back);
}

TEST_CASE("workflow: old and foreign documents are migrated or refused", "[unit][workflow]") {
    // version 0: no format, no version, booleans instead of modes
    const json old = {{"nodes", json::array({{{"id", 1}, {"type", "t.load"}, {"muted", true}}, {{"id", 2}, {"type", "t.lora"}, {"bypassed", true}}})}, {"links", json::array()}};
    const Workflow wf = workflowFromJson(old);
    CHECK(wf.body.nodes[0].mode == NodeMode::Muted);
    CHECK(wf.body.nodes[1].mode == NodeMode::Bypassed);
    CHECK(toJson(wf)["version"] == kWorkflowVersion);

    CHECK_THROWS_WITH(workflowFromJson({{"format", "other"}, {"version", 1}}), Catch::Matchers::ContainsSubstring("not graphicshost.workflow"));
    CHECK_THROWS_WITH(workflowFromJson({{"format", "graphicshost.workflow"}, {"version", 99}}), Catch::Matchers::ContainsSubstring("newer"));
    CHECK_THROWS(workflowFromJson(json::array()));
    CHECK_THROWS(workflowFromJson(document(json::array({{{"id", 1}}}), json::array())));                                   // no type
    CHECK_THROWS(workflowFromJson(document(json::array({node(1, "t.load", json::array())}), json::array())));            // widgets not an object
    CHECK_THROWS(workflowFromJson(document(json::array({node(1, "t.load", json::object(), "sleeping")}), json::array())));   // unknown mode
    CHECK_THROWS(workflowFromJson(document(json::array(), json::array({{{"from", json::array({1})}, {"to", json::array({2, "x"})}}}))));
}

TEST_CASE("workflow: widgets and links become plain job inputs", "[unit][workflow]") {
    const JobGraph job = convert(document(json::array({node(1, "t.load", {{"name", "sd15.safetensors"}}), node(2, "t.sample", {{"seed", 5}, {"steps", 8}, {"seedControl", "randomize"}})}),
                                          json::array({link(1, 0, 2, "model")})));
    REQUIRE(job.nodes.size() == 2);
    const JobNode& s = job.nodes.at("2");
    CHECK(s.type == "t.sample");
    CHECK(std::get<json>(s.inputs.at("seed")) == 5);
    CHECK(std::get<json>(s.inputs.at("steps")) == 8);
    CHECK(s.inputs.count("seedControl") == 0);   // editor-only
    CHECK(std::get<NodeRef>(s.inputs.at("model")) == NodeRef{"1", 0});
    CHECK(job.clientId == "wf");
}

TEST_CASE("workflow: muted nodes vanish and bypassed nodes are short-circuited", "[unit][workflow]") {
    // load -> lora(bypassed) -> sample
    JobGraph job = convert(document(json::array({node(1, "t.load"), node(2, "t.lora", json::object(), "bypassed"), node(3, "t.sample")}),
                                    json::array({link(1, 0, 2, "model"), link(2, 0, 3, "model")})));
    CHECK(job.nodes.count("2") == 0);
    CHECK(std::get<NodeRef>(job.nodes.at("3").inputs.at("model")) == NodeRef{"1", 0});

    // a muted source leaves the input empty (the validator reports it)
    job = convert(document(json::array({node(1, "t.load", json::object(), "muted"), node(3, "t.sample", {{"model", "stale"}})}), json::array({link(1, 0, 3, "model")})));
    CHECK(job.nodes.size() == 1);
    CHECK(job.nodes.at("3").inputs.count("model") == 0);

    // a bypassed node with nothing of its output type linked cannot be rewired
    CHECK_THROWS_AS(convert(document(json::array({node(2, "t.lora", json::object(), "bypassed"), node(3, "t.sample")}), json::array({link(2, 0, 3, "model")}))), ConvertError);

    // two bypassed nodes in a row
    job = convert(document(json::array({node(1, "t.load"), node(2, "t.lora", json::object(), "bypassed"), node(4, "t.lora", json::object(), "bypassed"), node(3, "t.sample")}),
                           json::array({link(1, 0, 2, "model"), link(2, 0, 4, "model"), link(4, 0, 3, "model")})));
    CHECK(std::get<NodeRef>(job.nodes.at("3").inputs.at("model")) == NodeRef{"1", 0});
}

TEST_CASE("workflow: reroutes and primitive constants are folded", "[unit][workflow]") {
    const JobGraph job = convert(document(
        json::array({node(1, "t.load"), node(5, "util.reroute"), node(6, "util.reroute"), node(7, "util.primitive", {{"value", 30}}), node(3, "t.sample")}),
        json::array({link(1, 0, 5, "value"), link(5, 0, 6, "value"), link(6, 0, 3, "model"), link(7, 0, 3, "steps")})));
    REQUIRE(job.nodes.size() == 2);
    CHECK(std::get<NodeRef>(job.nodes.at("3").inputs.at("model")) == NodeRef{"1", 0});
    CHECK(std::get<json>(job.nodes.at("3").inputs.at("steps")) == 30);

    // links that loop through reroutes are an error, not a hang
    CHECK_THROWS_AS(convert(document(json::array({node(5, "util.reroute"), node(6, "util.reroute"), node(3, "t.sample")}),
                                     json::array({link(5, 0, 6, "value"), link(6, 0, 5, "value"), link(6, 0, 3, "model")}))),
                    ConvertError);
}

TEST_CASE("workflow: subgraphs are flattened with stable inner ids", "[unit][workflow]") {
    // subgraph "pair": input "m" (MODEL) -> lora -> output "out"; instance 10 in the parent
    json inner = {{"id", "pair"},
                  {"name", "Pair"},
                  {"inputs", json::array({{{"name", "m"}, {"type", "MODEL"}}})},
                  {"outputs", json::array({{{"name", "out"}, {"type", "MODEL"}}})},
                  {"nodes", json::array({node(1, "t.lora", {{"strength", 0.5}})})},
                  {"links", json::array({link(kSubgraphInput, 0, 1, "model"), link(1, 0, kSubgraphOutput, "out")})}};
    json d = document(json::array({node(2, "t.load"), node(10, "subgraph:pair"), node(3, "t.sample")}),
                      json::array({link(2, 0, 10, "m"), link(10, 0, 3, "model")}), {{"subgraphs", json::array({inner})}});
    JobGraph job = convert(d);
    REQUIRE(job.nodes.count("10:1") == 1);
    CHECK(job.nodes.count("10") == 0);
    CHECK(std::get<json>(job.nodes.at("10:1").inputs.at("strength")) == 0.5);
    CHECK(std::get<NodeRef>(job.nodes.at("10:1").inputs.at("model")) == NodeRef{"2", 0});
    CHECK(std::get<NodeRef>(job.nodes.at("3").inputs.at("model")) == NodeRef{"10:1", 0});

    // bypassing the instance passes the model straight through
    d["nodes"][1]["mode"] = "bypassed";
    job = convert(d);
    CHECK(std::get<NodeRef>(job.nodes.at("3").inputs.at("model")) == NodeRef{"2", 0});

    // a widget on the instance feeds an unlinked subgraph input as a literal
    json lit = inner;
    lit["inputs"] = json::array({{{"name", "strength"}, {"type", "FLOAT"}}});
    lit["links"] = json::array({link(kSubgraphInput, 0, 1, "strength")});
    lit["outputs"] = json::array();
    json d2 = document(json::array({node(10, "subgraph:pair", {{"strength", 0.25}})}), json::array(), {{"subgraphs", json::array({lit})}});
    CHECK(std::get<json>(convert(d2).nodes.at("10:1").inputs.at("strength")) == 0.25);

    // missing and self-containing subgraphs are errors
    CHECK_THROWS_AS(convert(document(json::array({node(1, "subgraph:nope")}), json::array())), ConvertError);
    json loop = inner;
    loop["nodes"] = json::array({node(1, "subgraph:pair")});
    loop["links"] = json::array();
    CHECK_THROWS_AS(convert(document(json::array({node(10, "subgraph:pair")}), json::array(), {{"subgraphs", json::array({loop})}})), ConvertError);
}

TEST_CASE("workflow: unknown node types pass through for the validator to report", "[unit][workflow]") {
    const JobGraph job = convert(document(json::array({node(1, "plugin.mystery", {{"x", 1}})}), json::array()));
    REQUIRE(job.nodes.count("1") == 1);
    CHECK(job.nodes.at("1").type == "plugin.mystery");
    CHECK(std::get<json>(job.nodes.at("1").inputs.at("x")) == 1);
}
