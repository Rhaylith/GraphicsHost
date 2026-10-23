#pragma once

#include "engine/ImageEngine.h"
#include "engine/JobQueue.h"
#include "gallery/GalleryStore.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace gh::host {

class AppHost;

// One generation as the outside world asks for it: the HTTP API, and later the UIs.
struct GenerateCall {
    std::string model;                      // checkpoint name, file name or path; empty = the model that is loaded now
    std::string preset;                     // optional preset name
    nlohmann::json request = nlohmann::json::object();       // only the parameters the caller gave (partial)
    nlohmann::json loadOptions = nlohmann::json::object();   // only the load options the caller gave (partial)
    bool addToGallery = true;
    std::shared_ptr<const engine::Image> initImage;   // image to image: the picture to start from (strength is a request parameter)
    std::shared_ptr<const engine::Image> maskImage;   // inpainting: white = repaint, black = keep (needs initImage)
};

struct Submission {
    std::shared_ptr<engine::Job> job;
    std::string model;                      // file name of the checkpoint in use
    std::vector<std::string> notes;         // adjustments made on the way ("weights offloaded to RAM", ...)
};

struct GenerationCounters {
    uint64_t submitted = 0, succeeded = 0, failed = 0, cancelled = 0, images = 0;
    double seconds = 0;                     // time spent generating, summed over jobs
};

// Turns a GenerateCall into a job: finds the model, loads it just in time (replacing the loaded one when the queue is
// idle), layers model defaults < preset < the caller's values, checks that the model fits, submits, and stores the
// finished images in the gallery. Used by every front end except the CLI, which does the same on its own.
class GenerationService {
public:
    explicit GenerationService(AppHost& host);
    ~GenerationService();

    // Throws engine::EngineError: InvalidRequest (parameters, unsupported or wrong kind of model), NotFound (no such
    // model, nothing loaded), OutOfMemory (does not fit), Failed (busy with another model), LoadFailed.
    Submission submit(const GenerateCall& call);

    // Resolves, checks and loads the model like submit() does, without generating (result.job stays empty).
    Submission loadModel(const GenerateCall& call);

    // The gallery entries of a job once it has succeeded; empty before that or when it was not stored.
    std::vector<gallery::GalleryEntry> entries(uint64_t jobId) const;

    // Blocks until the gallery entries of a finished job are stored (or the job turns out not to need them).
    void waitHandled(uint64_t jobId, std::chrono::milliseconds timeout = std::chrono::seconds(30)) const;

    GenerationCounters counters() const;

private:
    void onJob(const engine::Job& job);

    struct Prepared {
        Submission submission;
        engine::GenerationRequest request;
        engine::LoadOptions load;
    };
    Prepared prepare(const GenerateCall& call);

    AppHost& host_;
    mutable std::mutex mutex_;
    std::map<uint64_t, std::vector<gallery::GalleryEntry>> stored_;
    std::map<uint64_t, bool> wantsGallery_;
    std::map<uint64_t, engine::ModelInfo> modelOf_;
    GenerationCounters counters_;
    std::map<uint64_t, bool> counted_;
    std::map<uint64_t, bool> handled_;
    mutable std::condition_variable handledCv_;
    void trim();  // callers hold mutex_
};

}  // namespace gh::host
