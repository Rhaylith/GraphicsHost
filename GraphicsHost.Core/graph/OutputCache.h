#pragma once

#include "graph/JobGraph.h"
#include "graph/NodeDef.h"
#include "graph/Value.h"

#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace gh::graph {

using NodeOutputs = std::vector<Value>;

// ---- cache keys -------------------------------------------------------------------------------------------------------
// The cache key of a node (workflow plan, section 6.5, step 4): its type, its literal inputs (defaults filled in, numbers
// normalised, so an omitted default and an explicit one agree), the keys of the nodes feeding it with the output index
// used, and the node's own fingerprint() text. Returns "" when the node must always run (fingerprint "*" or a type the
// registry does not know).
struct Upstream {
    std::string key;   // the key of the source node
    int output = 0;
};
std::string nodeCacheKey(const JobNode& node, const Registry& registry, const std::map<std::string, Upstream>& upstream);

// ---- the policy -------------------------------------------------------------------------------------------------------
struct CachePolicy {
    enum class Kind { Last, Lru, Ram, None };
    Kind kind = Kind::Last;
    int entries = 10;         // Lru: how many results are kept
    double freeRamGb = 10.0;  // Ram: results are evicted (least recently used first) while free RAM is below this

    // "last", "lru:20", "ram:16", "none"; nullopt for anything else.
    static std::optional<CachePolicy> parse(const std::string& text);
    std::string toString() const;
    bool operator==(const CachePolicy&) const = default;
};

// Keeps node outputs between jobs so unchanged parts of a workflow are skipped.
//   last  the results of the most recent job (default): an entry survives until a job finishes without using it
//   lru   at most N results, least recently used evicted first
//   ram   as lru without a count, evicting while free RAM is under the threshold
//   none  nothing is kept (a job still reuses a node's result inside itself)
// Eviction happens when a job ends, never while one runs, so a running job never loses a result it still needs.
class OutputCache {
public:
    using FreeRamFn = std::function<uint64_t()>;
    explicit OutputCache(CachePolicy policy = {}, FreeRamFn freeRam = {});

    void setPolicy(CachePolicy policy);
    CachePolicy policy() const;

    void beginJob();
    std::shared_ptr<const NodeOutputs> find(const std::string& key);   // marks the entry as used by the running job
    void put(const std::string& key, NodeOutputs outputs);
    void endJob();

    void clear();
    size_t size() const;
    size_t bytes() const;

    static uint64_t systemFreeRamBytes();

private:
    struct Entry {
        std::shared_ptr<const NodeOutputs> outputs;
        size_t bytes = 0;
        uint64_t lastUse = 0;
        uint64_t usedInJob = 0;
    };
    void evict();

    mutable std::mutex mutex_;
    CachePolicy policy_;
    FreeRamFn freeRam_;
    std::map<std::string, Entry> entries_;
    uint64_t clock_ = 0;
    uint64_t job_ = 0;
};

}  // namespace gh::graph
