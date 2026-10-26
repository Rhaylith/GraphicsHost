#include "graph/Events.h"

namespace gh::graph {

const char* eventName(Event::Kind kind) {
    switch (kind) {
        case Event::Kind::JobStarted: return "job.started";
        case Event::Kind::NodeStarted: return "node.started";
        case Event::Kind::NodeCached: return "node.cached";
        case Event::Kind::NodeProgress: return "node.progress";
        case Event::Kind::NodePreview: return "node.preview";
        case Event::Kind::NodeFinished: return "node.finished";
        case Event::Kind::NodeError: return "node.error";
        case Event::Kind::JobFinished: return "job.finished";
    }
    return "unknown";
}

nlohmann::json Event::toJson() const {
    nlohmann::json j = {{"type", eventName(kind)}, {"job", jobId}};
    if (!node.empty()) {
        j["node"] = node;
        if (!type.empty()) j["nodeType"] = type;
    }
    switch (kind) {
        case Kind::JobStarted: j["total"] = total; break;
        case Kind::NodeStarted: j["done"] = done; j["total"] = total; break;
        case Kind::NodeCached: j["done"] = done; j["total"] = total; j["outputs"] = summaries; break;
        case Kind::NodeProgress: j["step"] = step; j["steps"] = steps; break;
        case Kind::NodePreview: break;
        case Kind::NodeFinished: j["done"] = done; j["total"] = total; j["millis"] = millis; j["outputs"] = summaries; break;
        case Kind::NodeError:
            j["message"] = message;
            if (!input.empty()) j["input"] = input;
            break;
        case Kind::JobFinished:
            j["status"] = status;
            j["done"] = done;
            j["total"] = total;
            if (!message.empty()) j["message"] = message;
            break;
    }
    return j;
}

EventBus::Id EventBus::subscribe(Handler handler) {
    std::lock_guard lock(mutex_);
    handlers_[next_] = std::move(handler);
    return next_++;
}

void EventBus::unsubscribe(Id id) {
    std::lock_guard lock(mutex_);
    handlers_.erase(id);
}

void EventBus::publish(const Event& event) const {
    std::vector<Handler> copy;
    {
        std::lock_guard lock(mutex_);
        for (const auto& [id, h] : handlers_) copy.push_back(h);
    }
    for (const auto& h : copy) h(event);
}

}  // namespace gh::graph
