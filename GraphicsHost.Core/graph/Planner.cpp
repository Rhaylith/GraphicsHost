#include "graph/Planner.h"

#include <algorithm>
#include <functional>
#include <map>

namespace gh::graph {

namespace {

bool isLazyInput(const Registry& registry, const JobNode& node, const std::string& input) {
    const NodeDef* def = registry.find(node.type);
    const InputDef* in = def ? def->findInput(input) : nullptr;
    return in && in->lazy;
}

}  // namespace

Plan makePlan(const JobGraph& job, const Registry& registry) {
    Plan plan;
    if (!job.outputs.empty()) {
        for (const auto& id : job.outputs) {
            if (!job.nodes.count(id)) throw PlanError("the job asks for the output of node " + id + ", which does not exist");
            plan.roots.push_back(id);
        }
    } else {
        for (const auto& [id, node] : job.nodes) {
            const NodeDef* def = registry.find(node.type);
            if (def && def->outputNode()) plan.roots.push_back(id);
        }
    }
    std::sort(plan.roots.begin(), plan.roots.end(), NodeIdLess());
    plan.roots.erase(std::unique(plan.roots.begin(), plan.roots.end()), plan.roots.end());

    enum Mark { Open = 1, Done = 2 };
    std::map<std::string, int> mark;
    std::vector<std::string> path;
    const std::function<void(const std::string&)> visit = [&](const std::string& id) {
        mark[id] = Open;
        path.push_back(id);
        const JobNode& node = job.nodes.at(id);
        for (const auto& [name, input] : node.inputs) {   // std::map: inputs in name order, so the plan is deterministic
            const NodeRef* ref = std::get_if<NodeRef>(&input);
            if (!ref) continue;
            if (!job.nodes.count(ref->node)) throw PlanError("node " + id + ", input '" + name + "' is linked to node " + ref->node + ", which does not exist");
            const int state = mark[ref->node];
            if (state == Open) {
                std::string loop;
                for (auto it = std::find(path.begin(), path.end(), ref->node); it != path.end(); ++it) loop += *it + " -> ";
                throw PlanError("the graph has a loop: " + loop + ref->node);
            }
            if (state == 0) visit(ref->node);
        }
        path.pop_back();
        mark[id] = Done;
        plan.order.push_back(id);
    };
    for (const auto& root : plan.roots)
        if (mark[root] == 0) visit(root);

    // eager = reachable from the roots over non-lazy links
    std::vector<std::string> work(plan.roots.begin(), plan.roots.end());
    while (!work.empty()) {
        const std::string id = work.back();
        work.pop_back();
        if (!plan.eager.insert(id).second) continue;
        const JobNode& node = job.nodes.at(id);
        for (const auto& [name, input] : node.inputs)
            if (const NodeRef* ref = std::get_if<NodeRef>(&input))
                if (!isLazyInput(registry, node, name)) work.push_back(ref->node);
    }
    return plan;
}

}  // namespace gh::graph
