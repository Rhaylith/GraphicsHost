#include "server/GenerateMapping.h"
#include "server/HttpCommon.h"
#include "server/RouteContext.h"

#include <chrono>

namespace gh::server {

using Json = nlohmann::json;
using namespace std::chrono_literals;

namespace {

Json jobJson(RouteContext& ctx, const engine::Job& job) {
    const auto progress = job.progress();
    Json out = {{"id", job.id()},
                {"state", engine::toString(job.state())},
                {"step", progress.step},
                {"steps", progress.steps},
                {"secondsPerStep", progress.secondsPerStep},
                {"request", engine::toJson(job.request())}};
    if (job.state() == engine::JobState::Failed || job.state() == engine::JobState::Cancelled) {
        out["error"] = {{"message", job.error()}, {"code", static_cast<int>(job.errorCode())}};
    }
    if (job.state() == engine::JobState::Succeeded) {
        ctx.host.generation().waitHandled(job.id(), 5s);   // the gallery entries are stored just after the state changes
        const engine::GenerationResult result = job.result();
        Json images = Json::array();
        const auto entries = ctx.host.generation().entries(job.id());
        for (size_t i = 0; i < result.images.size(); ++i) {
            Json image = {{"seed", i < result.seeds.size() ? result.seeds[i] : result.request.seed},
                          {"width", result.images[i].width}, {"height", result.images[i].height}};
            if (i < entries.size()) {
                image["galleryId"] = entries[i].id;
                image["url"] = "/outputs/" + entries[i].path;
            }
            images.push_back(std::move(image));
        }
        out["images"] = images;
        out["seconds"] = result.seconds;
        out["warnings"] = result.warnings;
    }
    return out;
}

bool parseId(const httplib::Request& req, uint64_t& id) {
    try {
        id = std::stoull(req.matches[1]);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace

void registerJobRoutes(httplib::Server& server, RouteContext& ctx) {
    // Queues a generation and returns at once; follow it with GET /jobs/{id} or /jobs/{id}/events.
    server.Post("/api/v1/generate", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "generate")) return;
        Json body;
        if (!readJsonObject(req, res, body)) return;
        host::GenerateCall call;
        try {
            call = callFromBody(body);
            call.addToGallery = body.value("store", true);
        } catch (const std::exception& e) {
            sendError(res, 400, "invalid_request_error", e.what(), "invalid_parameters");
            return;
        }
        try {
            const host::Submission s = ctx.host.generation().submit(call);
            sendJson(res, {{"job", jobJson(ctx, *s.job)}, {"model", s.model}, {"notes", s.notes}}, 202);
        } catch (const engine::EngineError& e) {
            sendEngineError(res, e);
        } catch (const Json::exception& e) {
            sendError(res, 400, "invalid_request_error", e.what(), "invalid_parameters");
        }
    });

    server.Get("/api/v1/jobs", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        Json list = Json::array();
        const auto jobs = ctx.host.models().queue().list();
        for (auto it = jobs.rbegin(); it != jobs.rend(); ++it) list.push_back(jobJson(ctx, **it));   // newest first
        sendJson(res, {{"jobs", list}, {"pending", ctx.host.models().queue().pending()}, {"active", ctx.host.models().queue().active()}});
    });

    server.Get(R"(/api/v1/jobs/(\d+))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        uint64_t id = 0;
        const auto job = parseId(req, id) ? ctx.host.models().queue().find(id) : nullptr;
        if (!job) {
            sendError(res, 404, "not_found", "no such job (finished jobs are only kept for a while)", "job_not_found");
            return;
        }
        sendJson(res, jobJson(ctx, *job));
    });

    server.Post(R"(/api/v1/jobs/(\d+)/cancel)", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "generate")) return;
        uint64_t id = 0;
        const auto job = parseId(req, id) ? ctx.host.models().queue().find(id) : nullptr;
        if (!job) {
            sendError(res, 404, "not_found", "no such job", "job_not_found");
            return;
        }
        job->cancel();
        sendJson(res, jobJson(ctx, *job));
    });

    // Server-Sent Events: "progress" while it runs, then one "done" with the same document as GET /jobs/{id}.
    server.Get(R"(/api/v1/jobs/(\d+)/events)", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        uint64_t id = 0;
        const auto job = parseId(req, id) ? ctx.host.models().queue().find(id) : nullptr;
        if (!job) {
            sendError(res, 404, "not_found", "no such job", "job_not_found");
            return;
        }
        sendEventStream(res, [&ctx, job](SseStream& stream) {
            int lastStep = -1;
            engine::JobState lastState = engine::JobState::Queued;
            bool first = true;
            int quiet = 0;
            while (stream.connected()) {
                const auto p = job->progress();
                const auto state = job->state();
                if (first || p.step != lastStep || state != lastState) {
                    stream.send("progress", {{"state", engine::toString(state)}, {"step", p.step}, {"steps", p.steps}, {"secondsPerStep", p.secondsPerStep}});
                    lastStep = p.step;
                    lastState = state;
                    first = false;
                    quiet = 0;
                }
                if (job->finished()) break;
                if (job->waitFor(200ms)) break;
                if (++quiet % 50 == 0) stream.comment();   // every ~10 s: keeps proxies from dropping the connection
            }
            if (job->finished()) {
                ctx.host.generation().waitHandled(job->id());
                stream.send("done", jobJson(ctx, *job));
            }
        });
    });
}

}  // namespace gh::server
