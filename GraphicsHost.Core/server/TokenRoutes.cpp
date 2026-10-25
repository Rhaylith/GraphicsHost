#include "server/HttpCommon.h"
#include "server/RouteContext.h"

namespace gh::server {

using Json = nlohmann::json;

bool authorize(RouteContext& ctx, const httplib::Request& req, httplib::Response& res, const std::string& permission,
               std::string* principal) {
    const std::string secret = Auth::secretFrom(req.get_header_value("Authorization"), req.get_header_value("x-api-key"));
    const AuthDecision d = ctx.auth.decide(secret, req.remote_addr, permission);
    if (!d.allowed) {
        res.set_header("WWW-Authenticate", "Bearer");
        sendError(res, d.status, d.status == 401 ? "authentication_error" : "permission_error", d.message, d.code);
        return false;
    }
    if (principal) *principal = d.principal;
    return true;
}

namespace {

Json describe(const TokenInfo& t) {
    return {{"id", t.id}, {"name", t.name}, {"permissions", t.permissions}, {"created", t.created}, {"lastUsed", t.lastUsed}};
}

}  // namespace

void registerTokenRoutes(httplib::Server& server, RouteContext& ctx) {
    server.Get("/api/v1/whoami", [&ctx](const httplib::Request& req, httplib::Response& res) {
        std::string who;
        if (!authorize(ctx, req, res, "read", &who)) return;
        const bool local = Auth::isLoopback(req.remote_addr);
        Json permissions = Json::array();
        if (who == "(local)") {
            for (const auto& p : allPermissions()) permissions.push_back(p);
        } else {
            for (const auto& t : ctx.host.tokens().list()) {
                if (t.name == who) permissions = t.permissions;
            }
        }
        sendJson(res, {{"principal", who}, {"permissions", permissions}, {"local", local},
                       {"tokenRequired", ctx.host.settings().server.requireToken}});
    });

    server.Get("/api/v1/tokens", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "admin")) return;
        Json list = Json::array();
        for (const auto& t : ctx.host.tokens().list()) list.push_back(describe(t));
        sendJson(res, {{"tokens", list}});
    });

    server.Post("/api/v1/tokens", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "admin")) return;
        Json body;
        if (!readJsonObject(req, res, body)) return;
        try {
            const std::string name = body.value("name", "");
            std::vector<std::string> permissions;
            if (body.contains("permissions") && body["permissions"].is_array()) permissions = body["permissions"].get<std::vector<std::string>>();
            const auto created = ctx.host.tokens().create(name, permissions);
            sendJson(res, {{"token", describe(created.info)}, {"secret", created.secret},
                           {"note", "this is the only time the secret is shown"}}, 201);
        } catch (const std::invalid_argument& e) {
            sendError(res, 400, "invalid_request_error", e.what(), "invalid_token_request");
        } catch (const nlohmann::json::exception& e) {
            sendError(res, 400, "invalid_request_error", std::string("invalid token request: ") + e.what(), "invalid_token_request");
        }
    });

    server.Delete(R"(/api/v1/tokens/([^/]+))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!authorize(ctx, req, res, "admin")) return;
        if (!ctx.host.tokens().revoke(req.matches[1])) {
            sendError(res, 404, "not_found", "no token with that id or name", "token_not_found");
            return;
        }
        res.status = 204;
    });
}

}  // namespace gh::server
