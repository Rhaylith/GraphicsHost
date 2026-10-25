#pragma once

#include "graph/Events.h"
#include "graph/NodeApi.h"
#include "graph/OutputCache.h"
#include "graph/Planner.h"

#include <map>
#include <stop_token>
#include <string>
#include <vector>

namespace gh::graph {

struct RunOptions {
    std::string jobId;
    std::stop_token stop;   // set to interrupt the job: checked before every node and handed to the nodes, which check it in their own loops (a sampler at every step)
    EventBus* events = nullptr;   // optional: told about everything that happens
};

struct ExecResult {
    enum class Status { Done, Failed, Interrupted };
    Status status = Status::Done;
    // The outputs of the requested nodes (the job's `outputs`, or every output node).
    std::map<std::string, NodeOutputs, NodeIdLess> outputs;
    std::string errorNode;      // Failed: the node that failed ...
    std::string errorInput;     // ... the input to blame, when known ...
    std::string errorMessage;   // ... and why
    std::vector<std::string> ran;      // nodes that ran, in order (expanded nodes have ids such as "7:3")
    std::vector<std::string> cached;   // nodes whose result came from the cache
    std::map<std::string, double, NodeIdLess> millis;   // wall time of each node that ran, without the time of the nodes it waited for
    bool ok() const { return status == Status::Done; }
};

// Runs a validated job (workflow plan, section 6.5, steps 4 to 6):
//  - nodes are evaluated on demand from the requested outputs, each at most once per job;
//  - a node whose cache key is in the OutputCache is skipped;
//  - a lazy input is evaluated only when the node asks for it (NodeResult::needInputs), and the node is then run again;
//  - a node that does not accept lists runs once per item of a list input (equal lengths, or a single item repeated), its
//    outputs becoming lists;
//  - a node that returns an expansion has the sub-graph spliced in with ids "<node>:<inner>", validated and evaluated;
//    its outputs are the expansion's outputs.
// Failures never throw: they end the job with status Failed and name the node.
class Executor {
public:
    Executor(const Registry& registry, OutputCache& cache) : registry_(registry), cache_(cache) {}
    ExecResult run(const JobGraph& job, const RunOptions& options = {});

private:
    const Registry& registry_;
    OutputCache& cache_;
};

}  // namespace gh::graph
