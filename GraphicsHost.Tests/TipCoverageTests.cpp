#include "core/Settings.h"
#include "graph/TipCoverage.h"
#include "graph/Tips.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>

using namespace gh::graph;

TEST_CASE("coverage: the checker finds missing tooltips and examples", "[unit][tips][coverage]") {
    CHECK(hasExample("Does a thing. Example: 5."));
    CHECK(hasExample("Does a thing, e.g. five."));
    CHECK_FALSE(hasExample("Does a thing."));

    Registry r;
    r.add(NodeDef("t.good").input(Int("a").tip("Counts. Example: 3.")).input(Model("m").tip("The model.")).input(Int("h").hidden()));
    CHECK(tooltipProblems(r).empty());

    r.add(NodeDef("t.bad")
              .input(Int("a"))                                                // no tooltip
              .input(Int("b").tip("Counts."))                                 // widget without example
              .input(Combo("c").options({"x", "y"}).optionTips({{"x", "Explains x."}, {"z", "Example: z."}}).tip("Picks. Example: x."))   // option tip without example; tip for a missing option
              .input(Model("m")));                                            // socket without tooltip
    const auto problems = tooltipProblems(r);
    const auto has = [&](const std::string& s) { return std::find(problems.begin(), problems.end(), s) != problems.end(); };
    CHECK(has("t.bad.a: no tooltip"));
    CHECK(has("t.bad.b: the tooltip has no example"));
    CHECK(has("t.bad.c: the tooltip of option 'x' has no example"));
    CHECK(has("t.bad.c: a tooltip for 'z', which is not an option"));
    CHECK(has("t.bad.m: no tooltip"));
    CHECK(problems.size() == 5);
}

TEST_CASE("coverage: every built-in node input has a tooltip, every widget and option an example", "[unit][tips][coverage]") {
    const auto problems = tooltipProblems(Registry::builtin());
    const std::string first = problems.empty() ? std::string("none") : problems.front();
    INFO(first);
    CHECK(problems.empty());
}

TEST_CASE("coverage: every shared, option and setting tooltip has an example", "[unit][tips][coverage]") {
    for (const auto& tip : tips::allCommon) {
        INFO(tip.key);
        CHECK(hasExample(tip.text));
    }
    for (const auto& table : tips::allOptionTables)
        for (const auto& o : table.options) {
            INFO(std::string(table.name) + "." + o.option);
            CHECK(hasExample(o.tip));
        }
    for (const auto& tip : tips::allSettingTips) {
        INFO(tip.key);
        CHECK(hasExample(tip.text));
    }
}

TEST_CASE("coverage: every key of settings.json has a tooltip", "[unit][tips][coverage]") {
    const auto keys = settingKeys(gh::core::toJson(gh::core::Settings{}));
    CHECK(keys.size() > 15);
    const std::vector<std::string> internal = {"schemaVersion", "firstRunDone"};   // bookkeeping, not options
    for (const auto& key : keys) {
        if (std::find(internal.begin(), internal.end(), key) != internal.end()) continue;
        INFO(key);
        const char* tip = tips::setting(key);
        REQUIRE(tip != nullptr);
        CHECK(hasExample(tip));
    }
    CHECK(settingKeys(nlohmann::json{{"a", {{"b", 1}, {"c", nlohmann::json::array({1})}}}, {"d", true}}) == std::vector<std::string>{"a.b", "a.c", "d"});
}
