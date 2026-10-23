#pragma once

#include "TestUtil.h"

#include "host/AppHost.h"
#include "server/ApiServer.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <memory>

namespace gh::test {

// An application with its own temporary data folder and an HTTP server on a free port.
struct TestApp {
    TempDir home;
    std::unique_ptr<host::AppHost> host;
    std::unique_ptr<server::ApiServer> server;
    std::string error;

    explicit TestApp(server::ServerOptions options = {}, bool start = true) {
        host = std::make_unique<host::AppHost>(home.path());
        server = std::make_unique<server::ApiServer>(*host);
        if (start) {
            options.port = options.port;
            error = server->start(options);
        }
    }

    static server::ServerOptions ephemeral() {
        server::ServerOptions o;
        o.port = 0;
        return o;
    }

    httplib::Client client() const {
        httplib::Client c("127.0.0.1", server->port());
        c.set_read_timeout(120);
        c.set_connection_timeout(10);
        return c;
    }
};

inline nlohmann::json bodyJson(const httplib::Result& r) {
    return r ? nlohmann::json::parse(r->body, nullptr, false) : nlohmann::json();
}

}  // namespace gh::test
