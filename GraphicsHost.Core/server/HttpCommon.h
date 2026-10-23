#pragma once

// Helpers shared by the route files. Including this pulls in the whole of cpp-httplib, so only .cpp files of the
// server include it; the rest of the application sees ApiServer.h only.

#include "engine/ImageEngine.h"

#include <httplib.h>

#include <nlohmann/json.hpp>

#include <functional>
#include <string>

namespace gh::server {

// Errors look like OpenAI's: {"error": {"message", "type", "code"}}, for every status, on every route.
inline void sendError(httplib::Response& res, int status, const std::string& type, const std::string& message,
                      const std::string& code = {}) {
    res.status = status;
    res.set_content(nlohmann::json{{"error", {{"message", message}, {"type", type}, {"code", code.empty() ? nlohmann::json(nullptr) : nlohmann::json(code)}}}}.dump(),
                    "application/json");
}

inline void sendJson(httplib::Response& res, const nlohmann::json& body, int status = 200) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

// The HTTP face of an engine failure.
inline void sendEngineError(httplib::Response& res, const engine::EngineError& e) {
    switch (e.code()) {
        case engine::EngineErrorCode::InvalidRequest: sendError(res, 400, "invalid_request_error", e.what(), "invalid_parameters"); break;
        case engine::EngineErrorCode::NotFound: sendError(res, 404, "not_found", e.what(), "model_not_found"); break;
        case engine::EngineErrorCode::LoadFailed: sendError(res, 500, "server_error", e.what(), "model_load_failed"); break;
        case engine::EngineErrorCode::OutOfMemory: sendError(res, 507, "server_error", e.what(), "out_of_memory"); break;
        case engine::EngineErrorCode::Cancelled: sendError(res, 499, "cancelled", e.what(), "cancelled"); break;
        case engine::EngineErrorCode::Failed: sendError(res, 409, "server_error", e.what(), "busy"); break;
    }
}

// The body parsed as a JSON object; on failure sends a 400 and returns false.
inline bool readJsonObject(const httplib::Request& req, httplib::Response& res, nlohmann::json& out) {
    out = nlohmann::json::parse(req.body, nullptr, false);
    if (!out.is_object()) {
        sendError(res, 400, "invalid_request_error", "the request body must be a JSON object", "invalid_json");
        return false;
    }
    return true;
}

// Server-Sent Events: write events from a producer function that runs while the client stays connected.
class SseStream {
public:
    explicit SseStream(httplib::DataSink& sink) : sink_(sink) {}
    // false once the client has gone away
    bool send(const std::string& event, const nlohmann::json& data) {
        const std::string text = "event: " + event + "\ndata: " + data.dump() + "\n\n";
        return sink_.is_writable() && sink_.write(text.data(), text.size());
    }
    bool comment(const std::string& text = "keep-alive") {
        const std::string line = ": " + text + "\n\n";
        return sink_.is_writable() && sink_.write(line.data(), line.size());
    }
    bool connected() const { return sink_.is_writable(); }

private:
    httplib::DataSink& sink_;
};

inline void sendEventStream(httplib::Response& res, std::function<void(SseStream&)> producer) {
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider("text/event-stream", [producer = std::move(producer)](size_t, httplib::DataSink& sink) {
        SseStream stream(sink);
        producer(stream);
        sink.done();
        return true;
    });
}

}  // namespace gh::server
