#pragma once

#include "engine/ImageEngine.h"
#include "engine/JobQueue.h"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

namespace gh::models {
class ComponentCache;
}

namespace gh::engine {

struct LoadedModelInfo {
    ModelInfo info;
    LoadOptions options;
    std::chrono::steady_clock::time_point loadedAt;
    std::chrono::steady_clock::time_point lastUsed;
};

// Keeps at most one model in memory (image models are large), reuses it when asked for the same options, evicts it
// when another one is requested or when it has been idle longer than its ttlMinutes, and owns the job queue that
// runs the generations.
class ModelManager {
public:
    using Clock = std::function<std::chrono::steady_clock::time_point()>;
    // Gives the engine to load with; called for every load, so it may create the runtime lazily.
    using EngineProvider = std::function<std::shared_ptr<IImageEngine>()>;

    // sweepInterval: how often the background thread checks for idle models; zero disables the thread (tests call
    // sweepIdle themselves).
    explicit ModelManager(EngineProvider engine, Clock clock = {},
                          std::chrono::seconds sweepInterval = std::chrono::seconds(15));
    ~ModelManager();
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

    // The cache of opened components: the loaded model is registered there so the memory plan counts it, and a file the cache
    // holds is closed there before this manager opens it, so a file is never held twice. Call before the first load.
    void setComponentCache(models::ComponentCache* cache) { components_ = cache; }

    // Loads the model, or returns the loaded one when the options are identical. A different model replaces the
    // loaded one, but only while no job is queued or running (EngineError(Failed) otherwise).
    std::shared_ptr<ILoadedModel> load(const LoadOptions& options);
    void unload();

    std::optional<LoadedModelInfo> loaded() const;
    // The loaded model (and counts as use), or nullptr.
    std::shared_ptr<ILoadedModel> current();

    // Queues a generation on the loaded model. Throws EngineError(NotFound) if no model is loaded.
    std::shared_ptr<Job> submit(GenerationRequest request);
    JobQueue& queue() { return queue_; }

    // Unloads the model when it has been idle for its ttl and no job needs it. Returns true if it unloaded.
    bool sweepIdle();

private:
    void sweeperLoop(std::chrono::seconds interval);
    std::chrono::steady_clock::time_point now() const;

    EngineProvider engine_;
    models::ComponentCache* components_ = nullptr;
    Clock clock_;
    JobQueue queue_;

    mutable std::mutex mutex_;
    std::shared_ptr<ILoadedModel> model_;
    std::optional<LoadedModelInfo> info_;

    std::mutex sweepMutex_;
    std::condition_variable sweepWake_;
    bool stopping_ = false;
    std::thread sweeper_;
};

}  // namespace gh::engine
