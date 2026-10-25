#include "core/Log.h"
#include "core/SecretStore.h"
#include "core/Version.h"
#include "engine/FitEstimator.h"
#include "host/HostJson.h"
#include "server/HttpCommon.h"
#include "server/RouteContext.h"

#include <algorithm>

namespace gh::server {

using Json = nlohmann::json;

namespace {

int levelRank(const std::string& level) {
    log::Level l = log::Level::Info;
    log::parseLevel(level, l);
    return static_cast<int>(l);
}

}  // namespace

void registerSystemRoutes(httplib::Server& server, RouteContext& ctx) {
    server.Get("/api/v1/status", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        const auto* runtime = ctx.host.selectedRuntime();
        const auto loaded = ctx.host.models().loaded();
        const auto counters = ctx.host.generation().counters();
        Json model = nullptr;
        if (loaded) {
            model = {{"path", loaded->options.modelPath}, {"family", loaded->info.family}, {"runtime", loaded->info.runtimeId},
                     {"device", loaded->info.device}, {"ttlMinutes", loaded->options.ttlMinutes}};
        }
        sendJson(res, {{"name", kAppName},
                       {"version", kAppVersion},
                       {"runtime", runtime ? Json({{"id", runtime->id}, {"backend", runtime->backend}}) : Json(nullptr)},
                       {"model", model},
                       {"queue", {{"pending", ctx.host.models().queue().pending()}, {"active", ctx.host.models().queue().active()}}},
                       {"counters", {{"submitted", counters.submitted}, {"succeeded", counters.succeeded}, {"failed", counters.failed},
                                     {"cancelled", counters.cancelled}, {"images", counters.images}, {"seconds", counters.seconds}}},
                       {"downloads", ctx.host.downloads().list().size()},
                       {"huggingFaceToken", core::huggingFaceToken().has_value()}});
    });

    server.Get("/api/v1/hardware", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        const auto* runtime = ctx.host.selectedRuntime();
        const auto memory = engine::systemMemory();
        sendJson(res, {{"cpu", host::cpuJson(ctx.host.cpu())},
                       {"vulkan", host::gpuJson(ctx.host.gpu())},
                       {"runtimes", host::runtimesJson(ctx.host.runtimes())},
                       {"backendSetting", ctx.host.settings().backend},
                       {"selectedRuntime", runtime ? Json(runtime->id) : Json(nullptr)},
                       {"memory", {{"totalBytes", memory.totalBytes}, {"freeBytes", memory.freeBytes}}}});
    });

    // How busy the computer is right now (poll about once a second; CPU load is measured between two polls).
    server.Get("/api/v1/hardware/usage", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        const host::HardwareUsage u = ctx.host.monitor().sample();
        sendJson(res, {{"cpuPercent", u.cpuPercent},
                       {"ramUsedBytes", u.ramUsedBytes},
                       {"ramTotalBytes", u.ramTotalBytes},
                       {"gpuPercent", u.gpuPercent ? Json(*u.gpuPercent) : Json(nullptr)},
                       {"vramUsedBytes", u.vramUsedBytes ? Json(*u.vramUsedBytes) : Json(nullptr)}});
    });

    // What a client may put into a generation request: the names it can choose from and the limits.
    server.Get("/api/v1/parameters", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        sendJson(res, {{"samplers", engine::knownSamplers()},
                       {"schedulers", engine::knownSchedulers()},
                       {"formats", {"png", "jpeg", "webp"}},
                       {"limits", {{"minSize", 64}, {"maxSize", 4096}, {"sizeStep", 8}, {"maxImages", 16}}},
                       {"defaults", toJson(engine::GenerationRequest{})}});
    });

    server.Get("/api/v1/settings", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        sendJson(res, core::toJson(ctx.host.settings()));
    });

    // A partial document: only the keys present change ({"server": {"port": 4321}} leaves everything else alone).
    server.Put("/api/v1/settings", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "admin")) return;
        Json patch;
        if (!readJsonObject(req, res, patch)) return;
        const core::Settings before = ctx.host.settings();
        try {
            Json merged = core::toJson(before);
            merged.merge_patch(patch);
            const core::Settings after = core::settingsFromJson(merged);
            ctx.host.updateSettings(after);
            const bool restart = after.server.port != before.server.port || after.server.lan != before.server.lan ||
                                 after.server.enabled != before.server.enabled || after.server.https != before.server.https ||
                                 after.server.allowedAddresses != before.server.allowedAddresses || after.server.rateLimitPerMinute != before.server.rateLimitPerMinute ||
                                 after.server.tlsCertFile != before.server.tlsCertFile || after.server.tlsKeyFile != before.server.tlsKeyFile;
            sendJson(res, {{"settings", core::toJson(ctx.host.settings())}, {"restartRequired", restart}});
        } catch (const std::invalid_argument& e) {
            sendError(res, 400, "invalid_request_error", e.what(), "invalid_settings");
        } catch (const Json::exception& e) {
            sendError(res, 400, "invalid_request_error", std::string("invalid settings: ") + e.what(), "invalid_settings");
        }
    });

    // The newest log lines: ?limit=200 (at most 2000) &level=warn (this level and worse)
    server.Get("/api/v1/logs", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        size_t limit = 200;
        try {
            if (req.has_param("limit")) limit = static_cast<size_t>(std::clamp(std::stoi(req.get_param_value("limit")), 1, 2000));
        } catch (const std::exception&) {
            sendError(res, 400, "invalid_request_error", "limit must be a number", "invalid_parameters");
            return;
        }
        const int minimum = req.has_param("level") ? levelRank(req.get_param_value("level")) : 0;
        Json lines = Json::array();
        for (const auto& e : log::tail(2000)) {
            if (levelRank(e.level) >= minimum) lines.push_back({{"time", e.time}, {"level", e.level}, {"message", e.message}});
        }
        while (lines.size() > limit) lines.erase(lines.begin());
        sendJson(res, {{"entries", lines}});
    });
}

}  // namespace gh::server
