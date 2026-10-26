#pragma once

#include "graph/Value.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gh::graph {

// What happens while a job runs (workflow plan, section 6.5, step 6). The server turns these into Server-Sent Events and
// the editor into canvas updates; the history keeps the finished ones.
struct Event {
    enum class Kind {
        JobStarted,     // `total` nodes may run
        NodeStarted,
        NodeCached,     // the result came from the cache
        NodeProgress,   // `step` of `steps`
        NodePreview,    // `preview` holds an IMAGE value
        NodeFinished,   // `summaries` describes each output; `millis` is the node's own time
        NodeError,      // `message`, and `input` when one input is to blame
        JobFinished,    // `status` is "done", "failed" or "interrupted"
    };
    Kind kind = Kind::JobStarted;
    std::string jobId;
    std::string node;
    std::string type;                     // the node's type
    int step = 0, steps = 0;
    int total = 0;
    int done = 0;                         // nodes finished (ran or cached) so far
    double millis = 0;
    std::vector<std::string> summaries;
    std::string message;
    std::string input;
    std::string status;
    std::shared_ptr<const Value> preview;

    // {"type": "node.progress", "job": ..., "node": ..., ...}; previews are not included (they are binary: the server
    // sends them separately).
    nlohmann::json toJson() const;
};

const char* eventName(Event::Kind kind);   // "job.started", "node.started", "node.cached", "node.progress", ...

// Many subscribers, many publishers. Callbacks run on the publishing thread, outside the bus's lock, and must not throw.
class EventBus {
public:
    using Handler = std::function<void(const Event&)>;
    using Id = int;

    Id subscribe(Handler handler);
    void unsubscribe(Id id);
    void publish(const Event& event) const;

private:
    mutable std::mutex mutex_;
    std::map<Id, Handler> handlers_;
    Id next_ = 1;
};

}  // namespace gh::graph
