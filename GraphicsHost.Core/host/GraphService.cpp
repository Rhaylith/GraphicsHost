#include "host/GraphService.h"

#include "core/Log.h"
#include "graph/Tips.h"
#include "host/AppHost.h"
#include "nodes/Builtin.h"

#include <algorithm>
#include <chrono>

namespace gh::host {

using Json = nlohmann::json;

std::optional<graph::ModelFileRef> LibraryNodeServices::findModel(const std::string& kind, const std::string& name, std::string* problem) const {
    const auto k = models::kindFromFolder(kind);
    if (!k) {
        if (problem) *problem = "unknown kind of model '" + kind + "'";
        return std::nullopt;
    }
    auto found = library_.resolveLoose(*k, name);
    if (found.status != models::ModelLibrary::Resolution::Status::Found && *k == models::Kind::Vae) {
        const auto decoder = library_.resolveLoose(models::Kind::PreviewDecoder, name);   // a VAE combo also lists the tiny preview decoders
        if (decoder.status == models::ModelLibrary::Resolution::Status::Found) found = decoder;
    }
    if (found.status != models::ModelLibrary::Resolution::Status::Found) {
        if (problem) *problem = found.problem;
        return std::nullopt;
    }
    const models::ModelCard& c = *found.card;
    return graph::ModelFileRef{models::toString(c.kind), c.relative, c.path.string(), c.familyName, c.supported};
}

std::vector<std::string> LibraryNodeServices::modelNames(const std::string& kind) const {
    const auto k = models::kindFromFolder(kind);
    if (!k) return {};
    auto names = library_.names(*k);
    if (*k == models::Kind::Vae) {
        for (auto& n : library_.names(models::Kind::PreviewDecoder)) names.push_back(std::move(n));
        names.push_back("pixel space");
    }
    return names;
}

std::vector<std::string> LibraryNodeServices::embeddingNames() const {
    std::vector<std::string> names;
    for (const auto& c : library_.ofKind(models::Kind::Embedding)) names.push_back(c.name);
    std::sort(names.begin(), names.end());
    return names;
}

GraphService::Options GraphService::optionsFrom(const core::GraphSettings& settings) {
    std::string text = settings.cachePolicy;
    if (text == "lru" || text == "ram") text += ":" + std::to_string(settings.cacheSize);
    Options o;
    o.cache = graph::CachePolicy::parse(text).value_or(graph::CachePolicy{});
    o.historyMax = static_cast<size_t>(std::max(settings.historyMaxItems, 0));
    return o;
}

GraphService::GraphService(const graph::Registry& registry, engine::JobQueue& queue, gallery::GalleryStore& gallery, Options options, graph::ValidateOptions validateOptions)
    : registry_(registry), queue_(queue), gallery_(gallery), validateOptions_(std::move(validateOptions)), services_(options.services), cache_(options.cache), historyMax_(options.historyMax) {
    if (services_ && !validateOptions_.modelExists) {
        const auto services = services_;
        validateOptions_.modelExists = [services](const std::string& folder, const std::string& name) {
            if (folder == "vae" && name == "pixel space") return true;
            return services->findModel(folder, name).has_value();
        };
    }
}

GraphService::GraphService(AppHost& host)
    : GraphService(graph::Registry::builtin(), host.models().queue(), host.gallery(), [&host] {
          Options o = optionsFrom(host.settings().graph);
          o.services = std::make_shared<LibraryNodeServices>(host.library(), [&host] {
              const auto& p = host.settings().precision;
              return graph::PrecisionPolicy{p.diffusion, p.textEncoder, p.vae, p.vaeFallbackFp32};
          });
          return o;
      }()) {}

GraphService::~GraphService() {
    std::vector<std::shared_ptr<engine::Job>> live;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& weak : jobs_)
            if (auto job = weak.lock()) live.push_back(std::move(job));
    }
    for (const auto& job : live) job->cancel();
    for (const auto& job : live) job->wait();
}

graph::ValidationResult GraphService::validate(const Json& job) const {
    graph::JobGraph parsed;
    try {
        parsed = graph::jobFromJson(job);
    } catch (const std::exception& e) {
        graph::ValidationResult r;
        r.issues.push_back({"", "", "invalid_job", e.what(), Json::object()});
        return r;
    }
    return graph::validate(parsed, registry_, validateOptions_);
}

GraphSubmission GraphService::submit(const GraphJobCall& call) {
    GraphSubmission out;
    out.validation = validate(call.job);
    if (!out.validation.ok()) return out;
    const graph::JobGraph parsed = graph::jobFromJson(call.job);

    engine::SubmitOptions options;
    options.front = call.front;
    options.clientId = call.clientId;
    options.label = call.label.empty() ? parsed.clientLabel : call.label;
    out.job = queue_.submitGraph([this, parsed, call](engine::JobControl& control) { return execute(control, parsed, call.workflow, call); }, options);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::erase_if(jobs_, [](const std::weak_ptr<engine::Job>& w) { return w.expired(); });
        jobs_.push_back(out.job);
    }
    return out;
}

Json GraphService::execute(engine::JobControl& control, const graph::JobGraph& job, const Json& workflow, const GraphJobCall& call) {
    const auto started = std::chrono::steady_clock::now();
    graph::RunOptions run;
    run.jobId = std::to_string(control.jobId);
    run.stop = control.stop;
    run.events = &events_;
    run.services = services_.get();
    // node progress also reaches the queue's observers (the desktop UI and the older job routes watch the queue)
    const auto subscription = events_.subscribe([&](const graph::Event& e) {
        if (e.kind == graph::Event::Kind::NodeProgress && e.jobId == run.jobId && control.progress) control.progress(engine::StepProgress{e.step, e.steps, 0.0f, e.node});
    });
    graph::ExecResult result = graph::Executor(registry_, cache_).run(job, run);
    events_.unsubscribe(subscription);

    Json outputs = Json::array();
    for (const auto& [node, values] : result.outputs) {
        Json summaries = Json::array();
        for (const graph::Value& v : values) summaries.push_back(v.summary());
        outputs.push_back({{"node", node}, {"values", summaries}});
    }
    Json timings = Json::object();
    for (const auto& [node, ms] : result.millis) timings[node] = ms;

    gallery::HistoryEntry entry;
    entry.queueNumber = static_cast<int64_t>(control.jobId);
    entry.clientId = call.clientId;
    entry.label = call.label.empty() ? job.clientLabel : call.label;
    entry.status = result.ok() ? "done" : result.status == graph::ExecResult::Status::Interrupted ? "interrupted" : "failed";
    entry.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    entry.error = result.errorMessage;
    entry.errorNode = result.errorNode;
    entry.job = graph::toJson(job);
    entry.workflow = workflow;
    entry.outputs = outputs;
    entry.timings = timings;
    size_t keep = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        keep = historyMax_;
    }
    try {
        gallery_.addHistory(entry);
        gallery_.pruneHistory(keep);
    } catch (const std::exception& e) {
        log::warn(std::string("graph history: ") + e.what());
    }

    if (result.status == graph::ExecResult::Status::Interrupted) throw engine::EngineError(engine::EngineErrorCode::Cancelled, "interrupted");
    if (!result.ok()) {
        const std::string where = result.errorNode.empty() ? std::string() : "node " + result.errorNode + (result.errorInput.empty() ? "" : ", input '" + result.errorInput + "'") + ": ";
        throw engine::EngineError(engine::EngineErrorCode::Failed, where + result.errorMessage);
    }
    return {{"outputs", outputs}, {"ran", result.ran}, {"cached", result.cached}, {"millis", timings}};
}

void GraphService::applySettings(const core::GraphSettings& settings) {
    const Options o = optionsFrom(settings);
    cache_.setPolicy(o.cache);
    std::lock_guard<std::mutex> lock(mutex_);
    historyMax_ = o.historyMax;
}

Json GraphService::nodeCatalogue() const {
    return {{"catalogue", registry_.catalogue()}, {"tips", graph::tips::toJson()}};
}

std::vector<std::string> GraphService::modelNames(const std::string& kind) const {
    return services_ ? services_->modelNames(kind) : std::vector<std::string>{};
}

std::vector<std::string> GraphService::embeddingNames() const {
    return services_ ? services_->embeddingNames() : std::vector<std::string>{};
}

void GraphService::freeCaches() {
    cache_.clear();
}

std::vector<gallery::HistoryEntry> GraphService::history(int limit, int offset) const {
    return gallery_.history(limit, offset);
}

std::optional<gallery::HistoryEntry> GraphService::historyItem(int64_t id) const {
    return gallery_.historyItem(id);
}

bool GraphService::removeHistory(int64_t id) {
    return gallery_.removeHistory(id);
}

}  // namespace gh::host
