#include "nodes/Builtin.h"

namespace gh::nodes {

void registerBuiltinNodes(graph::Registry& registry) {
    // the other categories register themselves here as they are written: loaders (W3.2), ...
    registerUtilityNodes(registry);
}

}  // namespace gh::nodes
