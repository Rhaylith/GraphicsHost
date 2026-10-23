#include "server/HttpCommon.h"
#include "server/RouteContext.h"

#include <algorithm>
#include <sstream>

namespace gh::server {

using Json = nlohmann::json;

void registerMetricsRoutes(httplib::Server& server, RouteContext& ctx) {
    // Who asked for what, recently. The request log shows other clients' activity, so it is for administrators.
    server.Get("/api/v1/requests", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "admin")) return;
        size_t limit = 100;
        try {
            if (req.has_param("limit")) limit = static_cast<size_t>(std::clamp(std::stoi(req.get_param_value("limit")), 1, 500));
        } catch (const std::exception&) {
            sendError(res, 400, "invalid_request_error", "limit must be a number", "invalid_parameters");
            return;
        }
        Json list = Json::array();
        for (const auto& r : ctx.requests.recent(limit)) {
            list.push_back({{"time", r.time}, {"method", r.method}, {"path", r.path}, {"status", r.status}, {"ms", r.milliseconds}, {"remote", r.remote}});
        }
        sendJson(res, {{"requests", list}, {"total", ctx.requests.total()}});
    });

    // Prometheus text format
    server.Get("/metrics", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        const auto counters = ctx.host.generation().counters();
        const auto& queue = ctx.host.models().queue();
        std::ostringstream m;
        auto metric = [&](const char* name, const char* type, const char* help, auto value) {
            m << "# HELP " << name << " " << help << "\n# TYPE " << name << " " << type << "\n" << name << " " << value << "\n";
        };
        metric("graphicshost_uptime_seconds", "gauge", "Seconds since the server started.", ctx.requests.uptimeSeconds());
        metric("graphicshost_model_loaded", "gauge", "1 when a model is in memory.", ctx.host.models().loaded() ? 1 : 0);
        metric("graphicshost_queue_pending", "gauge", "Generations waiting for their turn.", queue.pending());
        metric("graphicshost_queue_active", "gauge", "Generations queued or running.", queue.active());
        metric("graphicshost_images_generated_total", "counter", "Images generated.", counters.images);
        metric("graphicshost_generation_seconds_total", "counter", "Seconds spent generating.", counters.seconds);
        m << "# HELP graphicshost_jobs_total Generation jobs by outcome.\n# TYPE graphicshost_jobs_total counter\n";
        m << "graphicshost_jobs_total{outcome=\"succeeded\"} " << counters.succeeded << "\n";
        m << "graphicshost_jobs_total{outcome=\"failed\"} " << counters.failed << "\n";
        m << "graphicshost_jobs_total{outcome=\"cancelled\"} " << counters.cancelled << "\n";
        m << "# HELP graphicshost_http_requests_total HTTP requests by status code.\n# TYPE graphicshost_http_requests_total counter\n";
        for (const auto& [status, count] : ctx.requests.byStatus()) m << "graphicshost_http_requests_total{status=\"" << status << "\"} " << count << "\n";
        res.set_content(m.str(), "text/plain; version=0.0.4");
    });
}

}  // namespace gh::server
