#include "models/ModelService.h"

#include "core/AtomicFile.h"
#include "core/Log.h"

#include <algorithm>

namespace gh::models {

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

std::string urlFor(const std::string& base, const std::string& repo, const std::string& revision, const std::string& file) {
    std::string b = base;
    while (!b.empty() && b.back() == '/') b.pop_back();
    return b + "/" + repo + "/resolve/" + (revision.empty() ? "main" : revision) + "/" + file;
}

bool presentOnDisk(const fs::path& file, uint64_t size) {
    std::error_code ec;
    return fs::is_regular_file(file, ec) && size > 0 && fs::file_size(file, ec) == size;
}

}  // namespace

Json toJson(const DownloadPlan& plan) {
    Json items = Json::array();
    for (const auto& i : plan.items) {
        items.push_back({{"role", i.role}, {"key", i.key}, {"repo", i.repo}, {"file", i.file}, {"url", i.url},
                         {"destination", i.destination.string()}, {"sizeBytes", i.sizeBytes}, {"sha256", i.sha256},
                         {"present", i.present}});
    }
    return {{"items", items}, {"bytesToDownload", plan.bytesToDownload}, {"notes", plan.notes}};
}

ModelService::ModelService(Catalog catalog, ModelLibrary& library, DownloadManager& downloads,
                           presets::ModelDefaultsStore& defaults, ModelServiceConfig config)
    : catalog_(std::move(catalog)), library_(library), downloads_(downloads), defaults_(defaults), config_(std::move(config)) {
    downloads_.setObserver([this](const Download& d) { onDownload(d); });
}

void ModelService::setObserver(DownloadManager::Observer observer) {
    std::lock_guard<std::mutex> lock(mutex_);
    observer_ = std::move(observer);
}

PlannedDownload ModelService::planFor(const CatalogEntry& entry, const std::string& role) const {
    PlannedDownload p;
    p.role = role;
    p.key = entry.key;
    p.repo = entry.repo;
    p.file = entry.file;
    p.revision = entry.revision;
    p.url = urlFor(config_.hubBaseUrl, entry.repo, entry.revision, entry.file);
    p.destination = (config_.modelsDir / entry.relativePath()).make_preferred();
    p.sizeBytes = entry.sizeBytes;
    p.sha256 = entry.sha256;
    p.present = presentOnDisk(p.destination, entry.sizeBytes);
    p.source = {{"key", entry.key}, {"repo", entry.repo}, {"file", entry.file}, {"revision", entry.revision}, {"name", entry.name}};
    p.defaults = entry.defaults;
    return p;
}

void ModelService::requireAccess(const CatalogEntry& entry) const {
    if (!entry.gated) return;
    const std::string token = config_.token ? config_.token() : std::string();
    if (token.empty()) {
        throw ModelServiceError(ModelServiceError::Code::NeedsToken,
                                entry.name + " is a gated model: accept its terms on https://huggingface.co/" + entry.repo +
                                    " and give GraphicsHost an access token (graphicshost login)");
    }
    try {
        HuggingFace(token, config_.hubBaseUrl).checkAccess(entry.repo, entry.file, entry.revision);
    } catch (const HfError& e) {
        throw ModelServiceError(e.code() == HfErrorCode::Network ? ModelServiceError::Code::Network : ModelServiceError::Code::Gated, e.what());
    }
}

DownloadPlan ModelService::planCatalog(const std::string& key, bool withCompanions) const {
    const CatalogEntry* entry = catalog_.find(key);
    if (!entry) {
        throw ModelServiceError(ModelServiceError::Code::UnknownEntry, "no catalog entry named '" + key + "' (see: graphicshost catalog)");
    }
    requireAccess(*entry);

    DownloadPlan plan;
    PlannedDownload main = planFor(*entry, entry->kind);
    std::vector<PlannedDownload> companions;
    if (withCompanions) {
        for (const auto& c : entry->companions) {
            if (!c.autoDownload) continue;
            const CatalogEntry* companion = catalog_.find(c.key);
            if (!companion) continue;
            requireAccess(*companion);
            PlannedDownload cp = planFor(*companion, c.role.empty() ? companion->kind : c.role);
            // the checkpoint's record names its companions by a path relative to the checkpoint, so the library can link them
            Json link = {{"file", fs::relative(cp.destination, main.destination.parent_path()).generic_string()},
                         {"role", cp.role}, {"key", cp.key}};
            if (!main.source.contains("companions")) main.source["companions"] = Json::array();
            main.source["companions"].push_back(link);
            plan.notes.push_back("includes " + companion->name + (c.note.empty() ? std::string() : " (" + c.note + ")"));
            companions.push_back(std::move(cp));
        }
    }
    plan.items.push_back(std::move(main));
    for (auto& c : companions) plan.items.push_back(std::move(c));
    for (const auto& i : plan.items) {
        if (!i.present) plan.bytesToDownload += i.sizeBytes;
    }
    if (!entry->license.empty()) plan.notes.push_back("license: " + entry->license);
    return plan;
}

DownloadPlan ModelService::planRepoFile(const std::string& repo, const std::string& file) const {
    HfModelInfo info;
    try {
        info = HuggingFace(config_.token ? config_.token() : std::string(), config_.hubBaseUrl).info(repo);
    } catch (const HfError& e) {
        using C = ModelServiceError::Code;
        throw ModelServiceError(e.code() == HfErrorCode::Network ? C::Network : e.code() == HfErrorCode::NotFound ? C::NotFound : e.code() == HfErrorCode::Gated || e.code() == HfErrorCode::Unauthorized ? C::Gated : C::Other, e.what());
    }
    const auto it = std::find_if(info.files.begin(), info.files.end(), [&](const HfFile& f) { return f.path == file; });
    if (it == info.files.end()) {
        throw ModelServiceError(ModelServiceError::Code::NotFound, repo + " has no file '" + file + "'");
    }
    DownloadPlan plan;
    PlannedDownload p;
    p.role = guessFileKind(file);
    p.repo = repo;
    p.file = file;
    p.revision = info.model.commit;
    p.url = urlFor(config_.hubBaseUrl, repo, info.model.commit, file);
    p.destination = (config_.modelsDir / fs::path(repo) / fs::path(file).filename()).make_preferred();
    p.sizeBytes = it->sizeBytes;
    p.sha256 = it->sha256;
    p.present = presentOnDisk(p.destination, it->sizeBytes);
    p.source = {{"repo", repo}, {"file", file}, {"revision", info.model.commit}};
    if (it->sha256.empty()) plan.notes.push_back("the repository gives no checksum for this file, so it cannot be verified against one");
    if (info.model.isGated()) plan.notes.push_back("gated model: a token with access is needed");
    if (!info.model.license.empty()) plan.notes.push_back("license: " + info.model.license);
    plan.items.push_back(std::move(p));
    if (!plan.items[0].present) plan.bytesToDownload = plan.items[0].sizeBytes;
    return plan;
}

std::vector<std::shared_ptr<Download>> ModelService::start(const DownloadPlan& plan) {
    std::vector<std::shared_ptr<Download>> started;
    for (const auto& item : plan.items) {
        if (item.present) {
            // already there: make sure it has its record (the library links companions through it) and its defaults
            Json record = item.source.is_object() ? item.source : Json::object();
            record["url"] = item.url;
            if (!record.contains("file")) record["file"] = item.destination.filename().string();
            record["localFile"] = item.destination.filename().string();
            record["sha256"] = item.sha256;
            record["sizeBytes"] = item.sizeBytes;
            try {
                core::writeFileAtomic(sourceSidecarFor(item.destination), record.dump(2));
                if (item.defaults.is_object() && !defaults_.get(item.destination.filename().string())) {
                    defaults_.set(item.destination.filename().string(), {item.defaults.value("request", Json::object()), item.defaults.value("load", Json::object())});
                }
            } catch (const std::exception& e) {
                log::warn(std::string("could not record an existing model: ") + e.what());
            }
            continue;
        }
        DownloadRequest r;
        r.url = item.url;
        r.destination = item.destination;
        r.label = item.destination.filename().string();
        r.sha256 = item.sha256;
        r.source = item.source;
        r.bearerToken = config_.token ? config_.token() : std::string();
        const auto d = downloads_.enqueue(std::move(r));
        if (item.defaults.is_object()) {
            std::lock_guard<std::mutex> lock(mutex_);
            pendingDefaults_[d->id()] = {{"file", item.destination.filename().string()}, {"defaults", item.defaults}};
        }
        started.push_back(d);
    }
    return started;
}

void ModelService::onDownload(const Download& d) {
    if (d.state() == DownloadState::Completed) {
        Json pending;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (const auto it = pendingDefaults_.find(d.id()); it != pendingDefaults_.end()) {
                pending = it->second;
                pendingDefaults_.erase(it);
            }
        }
        if (pending.is_object()) {
            try {
                const Json& defaults = pending["defaults"];
                defaults_.set(pending["file"].get<std::string>(), {defaults.value("request", Json::object()), defaults.value("load", Json::object())});
            } catch (const std::exception& e) {
                log::warn(std::string("could not apply the catalog defaults: ") + e.what());
            }
        }
    }
    DownloadManager::Observer observer;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        observer = observer_;
    }
    if (observer) observer(d);
}

}  // namespace gh::models
