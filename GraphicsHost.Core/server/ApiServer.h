#pragma once

#include "host/AppHost.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace gh::server {

struct ServerOptions {
    std::string bindAddress = "127.0.0.1";   // 0.0.0.0 for the LAN
    int port = 1234;                         // 0 = any free port (tests)
    size_t maxBodyBytes = 64ull * 1024 * 1024;   // requests above this are refused with 413 before they are parsed
    int workerThreads = 16;
    std::filesystem::path webRoot;           // static files served at / (the browser UI); empty = <exe folder>/web
    bool https = false;                      // serve https instead of http
    std::filesystem::path tlsCertFile, tlsKeyFile;   // PEM files; both empty = a self-signed certificate in tlsDir
    std::filesystem::path tlsDir;            // where the generated certificate lives (the data folder's tls folder)
    std::vector<std::string> allowedAddresses;   // empty = anyone; this computer is always allowed
    int rateLimitPerMinute = 0;              // per client, 0 = no limit (the health check is exempt)
};

// The options the settings ask for (port, LAN, https, allow-list, rate limit).
ServerOptions serverOptionsFrom(const host::AppHost& host);

// The HTTP server: OpenAI compatible endpoints under /v1, the native API under /api/v1, the browser UI at /.
// One instance runs in the background of the application; routes are added by the other files in this folder.
class ApiServer {
public:
    explicit ApiServer(host::AppHost& host);
    ~ApiServer();
    ApiServer(const ApiServer&) = delete;
    ApiServer& operator=(const ApiServer&) = delete;

    // Starts listening in a background thread. Returns an empty string on success, otherwise why not (port in use, ...).
    std::string start(const ServerOptions& options);
    void stop();
    bool running() const;
    int port() const;                        // the port actually bound
    std::string url() const;                 // http(s)://127.0.0.1:<port>

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gh::server
