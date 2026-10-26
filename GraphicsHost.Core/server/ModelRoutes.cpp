#include "core/SecretStore.h"
#include "models/FitHints.h"
#include "models/HuggingFace.h"
#include "models/ModelUpdates.h"
#include "server/GenerateMapping.h"
#include "server/HttpCommon.h"
#include "server/RouteContext.h"

#include <algorithm>
#include <cstdlib>

namespace gh::server {

using Json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

models::HuggingFace makeHub() { return models::makeConfiguredHub(); }

models::HardwareBudget budget(host::AppHost& host) {
    const auto* runtime = host.selectedRuntime();
    return models::budgetFrom(host.gpu(), runtime && runtime->backend == "vulkan");
}

void sendHfError(httplib::Response& res, const models::HfError& e) {
    using C = models::HfErrorCode;
    switch (e.code()) {
        case C::NotFound: sendError(res, 404, "not_found", e.what(), "hub_not_found"); break;
        case C::Unauthorized: sendError(res, 403, "hub_error", e.what(), "hub_unauthorized"); break;
        case C::Gated: sendError(res, 403, "hub_error", e.what(), "hub_gated"); break;
        case C::RateLimited: sendError(res, 429, "hub_error", e.what(), "hub_rate_limited"); break;
        default: sendError(res, 502, "hub_error", e.what(), "hub_unreachable"); break;
    }
}

Json downloadJson(const models::Download& d) {
    Json out = {{"id", d.id()},
                {"label", d.request().label},
                {"state", models::toString(d.state())},
                {"bytesDone", d.bytesDone()},
                {"bytesTotal", d.bytesTotal()},
                {"bytesPerSecond", d.bytesPerSecond()},
                {"destination", d.request().destination.string()}};
    if (d.state() == models::DownloadState::Failed) {
        out["error"] = {{"kind", models::toString(d.error())}, {"message", d.message()}, {"httpStatus", d.httpStatus()}};
    }
    return out;
}

bool isInstalled(host::AppHost& host, const models::CatalogEntry& e) {
    std::error_code ec;
    const fs::path p = host.modelsDir() / e.relativePath();
    return fs::is_regular_file(p, ec) && fs::file_size(p, ec) == e.sizeBytes;
}

Json fitJson(const models::FitHint& h) {
    return {{"label", h.label}, {"detail", h.detail}, {"device", h.device}, {"status", engine::toString(h.status)}};
}

Json presetJson(const presets::Preset& p) {
    return {{"name", p.name}, {"description", p.description}, {"request", p.request}, {"load", p.load}};
}

bool decodeId(const httplib::Request& req, uint64_t& id) {
    try {
        id = std::stoull(req.matches[1]);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace

void registerModelRoutes(httplib::Server& server, RouteContext& ctx) {
    // ---- the models folder ------------------------------------------------------------------------------------------
    server.Get("/api/v1/models", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        const auto loaded = ctx.host.models().loaded();
        Json list = Json::array();
        for (const auto& card : ctx.host.library().scan()) {
            Json entry = models::toJson(card);
            entry["hasDefaults"] = ctx.host.modelDefaults().get(card.file).has_value();
            entry["loaded"] = loaded && fs::path(loaded->options.modelPath) == card.path;
            list.push_back(std::move(entry));
        }
        Json current = nullptr;
        if (loaded) {
            current = {{"path", loaded->options.modelPath}, {"family", loaded->info.family}, {"runtime", loaded->info.runtimeId},
                       {"device", loaded->info.device}, {"options", engine::toJson(loaded->options)}};
        }
        sendJson(res, {{"modelsDir", ctx.host.modelsDir().string()}, {"loaded", current}, {"models", list}});
    });

    server.Post("/api/v1/models/load", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "manage")) return;
        Json body;
        if (!readJsonObject(req, res, body)) return;
        try {
            const host::Submission s = ctx.host.generation().loadModel(callFromBody(body));
            sendJson(res, {{"model", s.model}, {"notes", s.notes}});
        } catch (const engine::EngineError& e) {
            sendEngineError(res, e);
        } catch (const std::exception& e) {
            sendError(res, 400, "invalid_request_error", e.what(), "invalid_parameters");
        }
    });

    server.Post("/api/v1/models/unload", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "manage")) return;
        if (ctx.host.models().queue().active() > 0) {
            sendError(res, 409, "server_error", "a generation is running or queued: cancel it first", "busy");
            return;
        }
        const auto report = ctx.host.freeMemory({true, false});   // also closes the opened components of node graphs
        sendJson(res, {{"loaded", nullptr}, {"componentsClosed", report.componentsClosed}});
    });

    // "free memory": unload models and/or drop cached node results; what is in use is left alone and listed
    server.Post("/api/v1/graph/free", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "manage")) return;
        host::AppHost::FreeMemoryRequest request;
        if (!req.body.empty()) {
            const Json body = Json::parse(req.body, nullptr, false);
            if (!body.is_object()) {
                sendError(res, 400, "invalid_request_error", "the body must be a JSON object", "invalid_parameters");
                return;
            }
            request.unloadModels = body.value("unloadModels", true);
            request.freeCaches = body.value("freeCaches", true);
        }
        const auto report = ctx.host.freeMemory(request);
        sendJson(res, {{"componentsClosed", report.componentsClosed},
                       {"componentBytes", report.componentBytes},
                       {"generateModelUnloaded", report.generateModelUnloaded},
                       {"resultsDropped", report.resultsDropped},
                       {"resultBytes", report.resultBytes},
                       {"kept", report.kept}});
    });

    // ---- per model defaults --------------------------------------------------------------------------------------
    server.Get("/api/v1/defaults", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        Json list = Json::array();
        for (const auto& file : ctx.host.modelDefaults().models()) {
            if (const auto d = ctx.host.modelDefaults().get(file)) list.push_back({{"model", file}, {"request", d->request}, {"load", d->load}});
        }
        sendJson(res, {{"defaults", list}});
    });
    server.Get(R"(/api/v1/defaults/([^/]+))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        const auto d = ctx.host.modelDefaults().get(req.matches[1]);
        if (!d) {
            sendError(res, 404, "not_found", "no defaults for that model", "defaults_not_found");
            return;
        }
        sendJson(res, {{"model", req.matches[1].str()}, {"request", d->request}, {"load", d->load}});
    });
    server.Put(R"(/api/v1/defaults/([^/]+))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "manage")) return;
        Json body;
        if (!readJsonObject(req, res, body)) return;
        try {
            presets::ModelDefaultsStore::Defaults d;
            if (body.contains("request")) d.request = body["request"];
            if (body.contains("load")) d.load = body["load"];
            ctx.host.modelDefaults().set(req.matches[1], d);
            sendJson(res, {{"model", req.matches[1].str()}, {"request", d.request}, {"load", d.load}});
        } catch (const std::exception& e) {
            sendError(res, 400, "invalid_request_error", e.what(), "invalid_defaults");
        }
    });
    server.Delete(R"(/api/v1/defaults/([^/]+))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "manage")) return;
        if (!ctx.host.modelDefaults().remove(req.matches[1])) {
            sendError(res, 404, "not_found", "no defaults for that model", "defaults_not_found");
            return;
        }
        res.status = 204;
    });

    // ---- presets -----------------------------------------------------------------------------------------------------
    server.Get("/api/v1/presets", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        Json list = Json::array();
        for (const auto& name : ctx.host.presets().list()) {
            if (const auto p = ctx.host.presets().get(name)) list.push_back(presetJson(*p));
        }
        sendJson(res, {{"presets", list}});
    });
    server.Get(R"(/api/v1/presets/([^/]+))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        const auto p = ctx.host.presets().get(req.matches[1]);
        if (!p) {
            sendError(res, 404, "not_found", "no such preset", "preset_not_found");
            return;
        }
        sendJson(res, presetJson(*p));
    });
    server.Put(R"(/api/v1/presets/([^/]+))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "manage")) return;
        Json body;
        if (!readJsonObject(req, res, body)) return;
        try {
            presets::Preset p;
            p.name = req.matches[1];
            p.description = body.value("description", std::string());
            if (body.contains("request")) p.request = body["request"];
            if (body.contains("load")) p.load = body["load"];
            ctx.host.presets().save(p);
            sendJson(res, presetJson(p));
        } catch (const std::exception& e) {
            sendError(res, 400, "invalid_request_error", e.what(), "invalid_preset");
        }
    });
    server.Delete(R"(/api/v1/presets/([^/]+))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "manage")) return;
        if (!ctx.host.presets().remove(req.matches[1])) {
            sendError(res, 404, "not_found", "no such preset", "preset_not_found");
            return;
        }
        res.status = 204;
    });

    // ---- discovery ---------------------------------------------------------------------------------------------------
    server.Get("/api/v1/catalog", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        const models::Catalog& catalog = ctx.host.catalog();
        models::CatalogFilter filter;
        filter.text = req.get_param_value("text");
        filter.family = req.get_param_value("family");
        filter.kind = req.get_param_value("kind");
        const auto hardware = budget(ctx.host);
        Json list = Json::array();
        for (const auto* e : catalog.search(filter)) {
            list.push_back({{"key", e->key}, {"name", e->name}, {"kind", e->kind}, {"family", e->family}, {"repo", e->repo},
                            {"file", e->file}, {"sizeBytes", e->sizeBytes}, {"sha256", e->sha256}, {"license", e->license},
                            {"gated", e->gated}, {"recommended", e->recommended}, {"notes", e->notes}, {"tags", e->tags},
                            {"installed", isInstalled(ctx.host, *e)}, {"fit", fitJson(models::fitHintFor(e->sizeBytes, e->family, e->kind, hardware))}});
        }
        sendJson(res, {{"entries", list}, {"warnings", catalog.warnings()}});
    });

    server.Get("/api/v1/hub/search", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        try {
            int limit = 20;
            if (req.has_param("limit")) limit = std::clamp(std::stoi(req.get_param_value("limit")), 1, 100);
            Json list = Json::array();
            for (const auto& m : makeHub().search(req.get_param_value("q"), limit)) {
                list.push_back({{"id", m.id}, {"downloads", m.downloads}, {"likes", m.likes}, {"gated", m.gated}, {"license", m.license},
                                {"library", m.library}, {"lastModified", m.lastModified}});
            }
            sendJson(res, {{"models", list}});
        } catch (const models::HfError& e) {
            sendHfError(res, e);
        } catch (const std::exception& e) {
            sendError(res, 400, "invalid_request_error", e.what(), "invalid_parameters");
        }
    });

    // the files of one repository: /api/v1/hub/files?repo=publisher/name
    server.Get("/api/v1/hub/files", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        const std::string repo = req.get_param_value("repo");
        if (repo.find('/') == std::string::npos) {
            sendError(res, 400, "invalid_request_error", "repo must look like publisher/name", "invalid_parameters");
            return;
        }
        try {
            const models::HfModelInfo info = makeHub().info(repo);
            const auto hardware = budget(ctx.host);
            Json files = Json::array();
            for (const auto& f : models::usableFiles(info)) {
                const std::string kind = models::guessFileKind(f.path);
                const auto hint = models::fitHintFor(f.sizeBytes, f.sizeBytes > 5500ull * 1024 * 1024 ? "SDXL" : "SD 1.x", kind, hardware);
                files.push_back({{"path", f.path}, {"sizeBytes", f.sizeBytes}, {"sha256", f.sha256}, {"kind", kind}, {"fit", fitJson(hint)}});
            }
            sendJson(res, {{"id", info.model.id}, {"gated", info.model.gated}, {"license", info.model.license}, {"commit", info.model.commit}, {"files", files}});
        } catch (const models::HfError& e) {
            sendHfError(res, e);
        }
    });

    // ---- getting models ------------------------------------------------------------------------------------------------
    // {"target": "<catalog key>" | "publisher/repo", "file": "...", "companions": true, "dryRun": false}
    server.Post("/api/v1/models/get", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "manage")) return;
        Json body;
        if (!readJsonObject(req, res, body)) return;
        const std::string target = body.value("target", std::string());
        if (target.empty()) {
            sendError(res, 400, "invalid_request_error", "'target' is required: a catalog key or a repository (publisher/name)", "invalid_parameters");
            return;
        }
        models::ModelService& service = ctx.host.modelService();
        models::DownloadPlan plan;
        try {
            if (service.catalog().find(target)) {
                plan = service.planCatalog(target, body.value("companions", true));
            } else if (target.find('/') != std::string::npos) {
                std::string file = body.value("file", std::string());
                if (file.empty()) {
                    const auto files = models::usableFiles(makeHub().info(target));
                    if (files.size() != 1) {
                        sendError(res, 400, "invalid_request_error", target + " has " + std::to_string(files.size()) + " usable files: name one in 'file'", "file_required");
                        return;
                    }
                    file = files[0].path;
                }
                plan = service.planRepoFile(target, file);
            } else {
                sendError(res, 404, "not_found", "'" + target + "' is neither a catalog key nor a repository (publisher/name)", "unknown_model");
                return;
            }
        } catch (const models::ModelServiceError& e) {
            using C = models::ModelServiceError::Code;
            const int status = e.code() == C::UnknownEntry || e.code() == C::NotFound ? 404 : e.code() == C::NeedsToken || e.code() == C::Gated ? 403 : e.code() == C::Network ? 502 : 500;
            sendError(res, status, "hub_error", e.what(), e.code() == C::NeedsToken ? "hub_token_required" : e.code() == C::Gated ? "hub_gated" : "get_failed");
            return;
        } catch (const models::HfError& e) {
            sendHfError(res, e);
            return;
        }
        if (body.value("dryRun", false)) {
            sendJson(res, {{"plan", models::toJson(plan)}, {"started", Json::array()}});
            return;
        }
        Json started = Json::array();
        for (const auto& d : service.start(plan)) started.push_back(downloadJson(*d));
        sendJson(res, {{"plan", models::toJson(plan)}, {"started", started}}, 202);
    });

    server.Get("/api/v1/downloads", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        Json list = Json::array();
        for (const auto& d : ctx.host.downloads().list()) list.push_back(downloadJson(*d));
        sendJson(res, {{"downloads", list}});
    });
    for (const char* action : {"pause", "resume", "cancel"}) {
        server.Post(std::string(R"(/api/v1/downloads/(\d+)/)") + action, [&ctx, action = std::string(action)](const httplib::Request& req, httplib::Response& res) {
            if (!authorize(ctx, req, res, "manage")) return;
            uint64_t id = 0;
            const auto d = decodeId(req, id) ? ctx.host.downloads().find(id) : nullptr;
            if (!d) {
                sendError(res, 404, "not_found", "no such download", "download_not_found");
                return;
            }
            if (action == "pause") ctx.host.downloads().pause(id);
            else if (action == "resume") ctx.host.downloads().resume(id);
            else ctx.host.downloads().cancel(id);
            sendJson(res, downloadJson(*d));
        });
    }

    server.Get("/api/v1/updates", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        Json list = Json::array();
        for (const auto& u : models::checkForUpdates(ctx.host.library().scan(), makeHub())) {
            list.push_back({{"file", u.file}, {"repo", u.repo}, {"status", models::toString(u.status)}, {"message", u.message}, {"latestSha256", u.latestSha256}});
        }
        sendJson(res, {{"updates", list}});
    });
}

}  // namespace gh::server
