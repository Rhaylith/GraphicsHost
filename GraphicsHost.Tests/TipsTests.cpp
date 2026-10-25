#include "RepoRoot.h"
#include "graph/ReferenceDoc.h"
#include "graph/Tips.h"

#include <catch2/catch_test_macros.hpp>

using namespace gh::graph;

TEST_CASE("tips: the generated tooltip constants match the reference document", "[unit][tips]") {
    const auto root = repoRoot();
    if (root.empty()) SKIP("the repository is not above the test executable");
    const ReferenceDoc doc = parseReferenceDoc(readTextFile(root / "docs" / "workflow-node-reference.md"));
    REQUIRE(doc.common.size() > 30);

    // regenerate with: awk -f tools/gen_tips.awk docs/workflow-node-reference.md > GraphicsHost.Core/graph/TipsData.h
    INFO("TipsData.h is out of date: run tools/gen_tips.awk");
    CHECK(std::size(tips::allCommon) == doc.common.size());
    for (const auto& tip : tips::allCommon) {
        INFO(tip.key);
        REQUIRE(doc.common.count(tip.key) == 1);
        CHECK(doc.common.at(tip.key) == tip.text);
    }

    REQUIRE(doc.options.size() == std::size(tips::allOptionTables));
    for (const auto& table : tips::allOptionTables) {
        INFO(table.name);
        REQUIRE(doc.options.count(table.name) == 1);
        const auto& rows = doc.options.at(table.name);
        REQUIRE(rows.size() == table.options.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            CHECK(rows[i].first == table.options[i].option);
            CHECK(rows[i].second == table.options[i].tip);
        }
    }
}

TEST_CASE("tips: lookups, option helpers and the JSON export", "[unit][tips]") {
    REQUIRE(tips::common("seed") != nullptr);
    CHECK(std::string(tips::common("seed")) == tips::seed);
    CHECK(tips::common("no such key") == nullptr);
    CHECK(tips::common("strength", "Applies to the model.").ends_with(" Applies to the model."));
    CHECK(tips::common("strength", "").find("Example:") != std::string::npos);

    const auto* sampler = tips::optionTable("sampler");
    REQUIRE(sampler != nullptr);
    CHECK(tips::optionTable("nope") == nullptr);
    const InputDef def = tips::withOptions(Combo("sampler").tip(tips::sampler), tips::samplerOptions);
    CHECK(def.options.front() == "euler");
    CHECK(def.options.size() == sampler->options.size());
    CHECK(def.optionTips.at("euler").find("Example:") != std::string::npos);
    CHECK(def.optionTips.count("seeds 3") == 1);   // "seeds 2 / seeds 3" is two options

    const auto j = tips::toJson();
    CHECK(j["common"]["cfg"] == tips::cfg);
    CHECK(j["options"]["scheduler"]["karras"].is_string());
}
