#pragma once

#include "graph/NodeDef.h"

namespace gh::nodes {

// Adds every built-in node to the registry (one function per category, defined in the .cpp of that category).
void registerBuiltinNodes(graph::Registry& registry);

void registerUtilityNodes(graph::Registry& registry);   // Utility.cpp
void registerLoaderNodes(graph::Registry& registry);    // Loaders.cpp
void registerSamplingNodes(graph::Registry& registry);  // Sampling.cpp

}  // namespace gh::nodes
