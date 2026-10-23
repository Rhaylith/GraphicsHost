#include "engine/ImageCodec.h"
#include "server/GenerateMapping.h"
#include "server/HttpCommon.h"
#include "server/RouteContext.h"

#include <chrono>
#include <fstream>
#include <iterator>

namespace gh::server {

using Json = nlohmann::json;
using namespace std::chrono_literals;

namespace {

std::vector<uint8_t> readFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string base64(const std::vector<uint8_t>& bytes) {
    static const char* table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < bytes.size(); i += 3) {
        const uint32_t v = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
        out += table[v >> 18];
        out += table[(v >> 12) & 63];
        out += table[(v >> 6) & 63];
        out += table[v & 63];
    }
    if (i + 1 == bytes.size()) {
        const uint32_t v = bytes[i] << 16;
        out += table[v >> 18];
        out += table[(v >> 12) & 63];
        out += "==";
    } else if (i + 2 == bytes.size()) {
        const uint32_t v = (bytes[i] << 16) | (bytes[i + 1] << 8);
        out += table[v >> 18];
        out += table[(v >> 12) & 63];
        out += table[(v >> 6) & 63];
        out += '=';
    }
    return out;
}

// Waits for the job; gives up on it (cancels) when the client hangs up.
void waitForJob(const httplib::Request& req, engine::Job& job) {
    while (!job.waitFor(200ms)) {
        if (req.is_connection_closed && req.is_connection_closed()) {
            job.cancel();
            job.wait();
            return;
        }
    }
}

}  // namespace

void registerOpenAiRoutes(httplib::Server& server, RouteContext& ctx) {
    server.Get("/v1/models", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "read")) return;
        const auto loaded = ctx.host.models().loaded();
        Json data = Json::array();
        for (const auto& card : ctx.host.library().ofKind(models::Kind::Checkpoint)) {
            if (!card.supported) continue;
            const bool isLoaded = loaded && std::filesystem::path(loaded->options.modelPath) == card.path;
            data.push_back({{"id", card.file}, {"object", "model"}, {"created", 0}, {"owned_by", "local"},
                            {"family", card.familyName}, {"loaded", isLoaded}});
        }
        sendJson(res, {{"object", "list"}, {"data", data}});
    });

    server.Post("/v1/images/generations", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "generate")) return;
        Json body;
        if (!readJsonObject(req, res, body)) return;

        host::GenerateCall call;
        try {
            call = callFromBody(body);
        } catch (const std::exception& e) {  // wrong shape: invalid_argument, or a json type error
            sendError(res, 400, "invalid_request_error", e.what(), "invalid_parameters");
            return;
        }
        if (!body.contains("prompt") || !body["prompt"].is_string() || body["prompt"].get<std::string>().empty()) {
            sendError(res, 400, "invalid_request_error", "'prompt' is required", "missing_prompt");
            return;
        }
        const std::string responseFormat = body.value("response_format", std::string("b64_json"));
        if (responseFormat != "b64_json" && responseFormat != "url") {
            sendError(res, 400, "invalid_request_error", "response_format must be b64_json or url", "invalid_parameters");
            return;
        }
        const bool store = body.value("store", true);
        if (responseFormat == "url" && !store) {
            sendError(res, 400, "invalid_request_error", "response_format url needs the image stored in the gallery (store: true)", "invalid_parameters");
            return;
        }
        call.addToGallery = store;

        host::GenerationService& generation = ctx.host.generation();
        host::Submission submission;
        try {
            submission = generation.submit(call);
        } catch (const engine::EngineError& e) {
            sendEngineError(res, e);
            return;
        } catch (const Json::exception& e) {
            sendError(res, 400, "invalid_request_error", e.what(), "invalid_parameters");
            return;
        }

        const std::shared_ptr<engine::Job> job = submission.job;
        waitForJob(req, *job);
        generation.waitHandled(job->id());
        if (job->state() != engine::JobState::Succeeded) {
            sendEngineError(res, engine::EngineError(job->errorCode(), job->error()));
            return;
        }

        const engine::GenerationResult result = job->result();
        const auto entries = generation.entries(job->id());
        const auto format = engine::parseImageFormat(result.request.format).value_or(engine::ImageFormat::Png);
        const std::string host = req.get_header_value("Host");
        Json data = Json::array();
        for (size_t i = 0; i < result.images.size(); ++i) {
            Json item = {{"seed", i < result.seeds.size() ? result.seeds[i] : result.request.seed}};
            if (i < entries.size()) {
                item["gallery_id"] = entries[i].id;
                if (responseFormat == "url") {
                    item["url"] = "http://" + (host.empty() ? std::string("127.0.0.1") : host) + "/outputs/" + entries[i].path;
                } else {
                    item["b64_json"] = base64(readFile(ctx.host.gallery().absolutePath(entries[i])));
                }
            } else {
                item["b64_json"] = base64(engine::encodeImage(result.images[i], format, std::nullopt));
            }
            data.push_back(std::move(item));
        }
        Json out = {{"created", std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()},
                    {"data", data},
                    {"model", submission.model},
                    {"mime_type", engine::mimeTypeFor(format)}};
        if (!result.warnings.empty() || !submission.notes.empty()) {
            Json notes = Json::array();
            for (const auto& n : submission.notes) notes.push_back(n);
            for (const auto& n : result.warnings) notes.push_back(n);
            out["notes"] = notes;
        }
        sendJson(res, out);
    });

    // Editing needs the image-to-image support of phase 5.
    server.Post("/v1/images/edits", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "generate")) return;
        sendError(res, 501, "not_implemented", "image edits are not available in this version", "not_implemented");
    });
}

}  // namespace gh::server
