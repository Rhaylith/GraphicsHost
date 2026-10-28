#include "nodes/Builtin.h"

namespace gh::nodes {

void registerBuiltinNodes(graph::Registry& registry) {
    // the other categories register themselves here as they are written: the rest of the node families (W4 ...)
    registerUtilityNodes(registry);
    registerLoaderNodes(registry);
    registerSamplingNodes(registry);
    registerModelSamplingNodes(registry);
}

}  // namespace gh::nodes
