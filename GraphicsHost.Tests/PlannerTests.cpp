#include "graph/Planner.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace gh::graph;
using nlohmann::json;

namespace {
Registry registry() {
    Registry r;
    r.add(NodeDef("t.src").output(Out("v", ValueType::Int)));
    r.add(NodeDef("t.add").input(Socket("a", ValueType::Int)).input(Socket("b", ValueType::Int).optional()).output(Out("v", ValueType::Int)));
    r.add(NodeDef("t.pick").input(Socket("cond", ValueType::Int)).input(Socket("x", ValueType::Int).lazy().optional()).input(Socket("y", ValueType::Int).lazy().optional()).output(Out("v", ValueType::Int)));
    r.add(NodeDef("t.out").input(Socket("v", ValueType::Int)).isOutput());
    return r;
}
JobGraph job(const json& nodes, json outputs = json::array()) {
    return jobFromJson({{"format", "graphicshost.job"}, {"version", 1}, {"nodes", nodes}, {"outputs", outputs}});
}
json n(const char* type, json inputs = json::object()) { return {{"type", type}, {"inputs", inputs}}; }
json ref(const char* id) { return json::array({id, 0}); }
}  // namespace

TEST_CASE("planner: only the ancestors of the outputs run, dependencies first", "[unit][planner]") {
    const auto plan = makePlan(job({{"1", n("t.src")},
                                    {"2", n("t.src")},
                                    {"3", n("t.add", {{"a", ref("1")}, {"b", ref("2")}})},
                                    {"4", n("t.out", {{"v", ref("3")}})},
                                    {"5", n("t.src")},                                // not needed by anything
                                    {"10", n("t.out", {{"v", ref("5")}})}}),           // a second output node
                               registry());
    CHECK(plan.roots == std::vector<std::string>{"4", "10"});
    CHECK(plan.order == std::vector<std::string>{"1", "2", "3", "4", "5", "10"});

    // a partial run: only the requested output and what it needs
    const auto partial = makePlan(job({{"1", n("t.src")}, {"2", n("t.src")}, {"3", n("t.add", {{"a", ref("1")}})}, {"4", n("t.out", {{"v", ref("3")}})}, {"5", n("t.out", {{"v", ref("2")}})}},
                                      json::array({"5"})),
                                  registry());
    CHECK(partial.order == std::vector<std::string>{"2", "5"});

    // an output that is not an output node can be requested too
    const auto mid = makePlan(job({{"1", n("t.src")}, {"3", n("t.add", {{"a", ref("1")}})}}, json::array({"3"})), registry());
    CHECK(mid.order == std::vector<std::string>{"1", "3"});

    // the order is stable: roots by id, then each node's inputs by name
    const auto many = makePlan(job({{"2", n("t.src")}, {"10", n("t.src")}, {"11", n("t.add", {{"a", ref("10")}, {"b", ref("2")}})}, {"12", n("t.out", {{"v", ref("11")}})}}), registry());
    CHECK(many.order == std::vector<std::string>{"10", "2", "11", "12"});   // input "a" is followed before "b"

    CHECK(makePlan(job({{"1", n("t.src")}}), registry()).order.empty());   // no output node: nothing to run
}

TEST_CASE("planner: loops and dangling links are refused", "[unit][planner]") {
    CHECK_THROWS_WITH(makePlan(job({{"1", n("t.add", {{"a", ref("2")}})}, {"2", n("t.add", {{"a", ref("1")}})}, {"3", n("t.out", {{"v", ref("1")}})}}), registry()),
                      Catch::Matchers::ContainsSubstring("loop"));
    CHECK_THROWS_AS(makePlan(job({{"1", n("t.add", {{"a", ref("1")}})}, {"3", n("t.out", {{"v", ref("1")}})}}), registry()), PlanError);   // a node feeding itself
    CHECK_THROWS_AS(makePlan(job({{"3", n("t.out", {{"v", ref("9")}})}}), registry()), PlanError);
    CHECK_THROWS_AS(makePlan(job({{"3", n("t.out")}}, json::array({"8"})), registry()), PlanError);
}

TEST_CASE("planner: lazy inputs mark the nodes only they need", "[unit][planner]") {
    const auto plan = makePlan(job({{"1", n("t.src")},
                                    {"2", n("t.src")},
                                    {"3", n("t.src")},
                                    {"4", n("t.pick", {{"cond", ref("1")}, {"x", ref("2")}, {"y", ref("3")}})},
                                    {"5", n("t.out", {{"v", ref("4")}})}}),
                               registry());
    CHECK(plan.order.size() == 5);
    CHECK(plan.isEager("1"));
    CHECK(plan.isEager("4"));
    CHECK_FALSE(plan.isEager("2"));
    CHECK_FALSE(plan.isEager("3"));

    // a node that is also needed through a normal input stays eager
    const auto both = makePlan(job({{"2", n("t.src")}, {"4", n("t.pick", {{"cond", ref("2")}, {"x", ref("2")}})}, {"5", n("t.out", {{"v", ref("4")}})}}), registry());
    CHECK(both.isEager("2"));
}
