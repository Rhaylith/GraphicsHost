#pragma once

#include "host/AppHost.h"
#include "server/Auth.h"
#include "server/RequestLog.h"

#include <string>

namespace httplib {
class Server;
struct Request;
struct Response;
}  // namespace httplib

namespace gh::server {

// What every route file needs. Route files are registered by ApiServer::start.
struct RouteContext {
    host::AppHost& host;
    Auth& auth;
    RequestLog& requests;
};

// Checks the request against the permission; sends the 401/403 and returns false when it is not allowed.
// principal (optional) receives who is calling ("(local)" or the token's name).
bool authorize(RouteContext& ctx, const httplib::Request& req, httplib::Response& res, const std::string& permission,
               std::string* principal = nullptr);

void registerTokenRoutes(httplib::Server& server, RouteContext& ctx);
void registerOpenAiRoutes(httplib::Server& server, RouteContext& ctx);
void registerJobRoutes(httplib::Server& server, RouteContext& ctx);
void registerGalleryRoutes(httplib::Server& server, RouteContext& ctx);
void registerModelRoutes(httplib::Server& server, RouteContext& ctx);
void registerSystemRoutes(httplib::Server& server, RouteContext& ctx);
void registerMetricsRoutes(httplib::Server& server, RouteContext& ctx);

}  // namespace gh::server
