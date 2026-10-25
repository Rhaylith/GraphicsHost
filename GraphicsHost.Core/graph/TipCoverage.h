#pragma once

#include "graph/NodeDef.h"

#include <string>
#include <vector>

namespace gh::graph {

// True when a tooltip carries an example: the marker "Example:" or "e.g." (workflow plan, tooltip rules).
bool hasExample(const std::string& tooltip);

// Everything wrong with the tooltips of a registry, one line per problem, "<node>.<input>: what":
// every input has a tooltip; every widget's tooltip has an example; every option tooltip has an example.
std::vector<std::string> tooltipProblems(const Registry& registry);

// The keys of a settings document as dotted paths ("server.port"), for checking that each has a tooltip. Objects are
// walked; arrays and scalars are leaves.
std::vector<std::string> settingKeys(const nlohmann::json& settings);

}  // namespace gh::graph
