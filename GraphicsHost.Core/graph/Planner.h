#pragma once

#include "graph/JobGraph.h"
#include "graph/NodeDef.h"

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace gh::graph {

// The job cannot be planned: a cycle, or a link to a node that is not in the job. (Run the validator first for messages
// tied to inputs; the planner only protects the executor.)
class PlanError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// What a job will run (workflow plan, section 6.5, steps 3 and 4).
struct Plan {
    std::vector<std::string> roots;   // the requested outputs, in id order
    std::vector<std::string> order;   // every node that may run, dependencies first (roots in id order, a node's inputs in name order)
    // Nodes reached from the roots without crossing a lazy input. The others run only if a node asks for the lazy input
    // they feed.
    std::set<std::string, NodeIdLess> eager;

    bool isEager(const std::string& id) const { return eager.count(id) != 0; }
};

// Plans the ancestors of the requested outputs: the job's `outputs` when given (a partial run), otherwise every output
// node. Nodes nothing asked for are left out. Throws PlanError.
Plan makePlan(const JobGraph& job, const Registry& registry);

}  // namespace gh::graph
