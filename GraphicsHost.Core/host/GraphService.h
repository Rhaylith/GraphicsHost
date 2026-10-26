#pragma once

#include "core/Settings.h"
#include "engine/JobQueue.h"
#include "gallery/GalleryStore.h"
#include "graph/Events.h"
#include "graph/Executor.h"
#include "graph/Validator.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace gh::host {

class AppHost;

// One graph job as the outside world asks for it: the HTTP API, the CLI and the editor.
struct GraphJobCall {
    nlohmann::json job;                                   // the job format
    nlohmann::json workflow;                              // the editor format it came from, or null; kept in the history
    bool front = false;                                   // jump the queue
    std::string clientId;
    std::string label;
};

struct GraphSubmission {
    std::shared_ptr<engine::Job> job;                     // null when the job did not validate
    graph::ValidationResult validation;                   // empty when it did
};

// Validates, queues and runs node graphs, and keeps their history (workflow plan, sections 6.5 and 6.6). Graph jobs go
// through the same JobQueue as generations, so they share the GPU; the executor runs on the queue's worker.
class GraphService {
public:
    struct Options {
        graph::CachePolicy cache;
        size_t historyMax = 10000;   // 0 = keep everything
    };

    GraphService(const graph::Registry& registry, engine::JobQueue& queue, gallery::GalleryStore& gallery, Options options, graph::ValidateOptions validateOptions = {});
    explicit GraphService(AppHost& host);   // the built-in nodes on the host's queue and gallery, configured by its settings
    // Cancels the graph jobs still queued or running and waits for them.
    ~GraphService();
    GraphService(const GraphService&) = delete;
    GraphService& operator=(const GraphService&) = delete;

    // Checks a job document. A document that is not a job at all gives one issue with the code "invalid_job".
    graph::ValidationResult validate(const nlohmann::json& job) const;

    // Validates and queues. When the job is invalid nothing is queued and `validation` says why.
    GraphSubmission submit(const GraphJobCall& call);

    // Applies the settings that concern graphs (cache policy and size, history length).
    void applySettings(const core::GraphSettings& settings);

    // Node catalogue plus tooltips, as served to editors: {"catalogue": {...}, "tips": {...}}.
    nlohmann::json nodeCatalogue() const;

    // Drops every cached node result (the "free memory" action).
    void freeCaches();

    // History of finished graph jobs.
    std::vector<gallery::HistoryEntry> history(int limit = 50, int offset = 0) const;
    std::optional<gallery::HistoryEntry> historyItem(int64_t id) const;
    bool removeHistory(int64_t id);

    graph::EventBus& events() { return events_; }
    graph::OutputCache& cache() { return cache_; }
    const graph::Registry& registry() const { return registry_; }

    // The cache policy and history length a settings section asks for.
    static Options optionsFrom(const core::GraphSettings& settings);

private:
    nlohmann::json execute(engine::JobControl& control, const graph::JobGraph& job, const nlohmann::json& workflow, const GraphJobCall& call);

    const graph::Registry& registry_;
    engine::JobQueue& queue_;
    gallery::GalleryStore& gallery_;
    graph::ValidateOptions validateOptions_;
    graph::OutputCache cache_;
    graph::EventBus events_;

    mutable std::mutex mutex_;
    size_t historyMax_;
    std::vector<std::weak_ptr<engine::Job>> jobs_;
};

}  // namespace gh::host
