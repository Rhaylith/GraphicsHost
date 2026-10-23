#include "server/ApiServer.h"

#include "core/Log.h"
#include "core/Version.h"
#include "server/HttpCommon.h"

#include <atomic>
#include <thread>

namespace gh::server {

using Json = nlohmann::json;
namespace fs = std::filesystem;

struct ApiServer::Impl {
    host::AppHost& host;
    std::unique_ptr<httplib::Server> server;   // a new one for every start: a stopped or failed httplib server is not reusable
    std::thread thread;
    std::atomic<bool> running{false};
    int port = 0;
    std::string bindAddress;

    explicit Impl(host::AppHost& h) : host(h) {}
};

ApiServer::ApiServer(host::AppHost& host) : impl_(std::make_unique<Impl>(host)) {}

ApiServer::~ApiServer() {
    stop();
}

bool ApiServer::running() const { return impl_->running.load(); }
int ApiServer::port() const { return impl_->port; }
std::string ApiServer::url() const { return "http://127.0.0.1:" + std::to_string(impl_->port); }

std::string ApiServer::start(const ServerOptions& o) {
    if (impl_->running) return "the server is already running";
    impl_->server = std::make_unique<httplib::Server>();
    httplib::Server& s = *impl_->server;

    // a bounded pool: the default would use as many threads as the machine has cores (192 here)
    s.new_task_queue = [n = static_cast<size_t>(std::max(2, o.workerThreads))] { return new httplib::ThreadPool(n); };
    s.set_payload_max_length(o.maxBodyBytes);
    // Windows lets two programs listen on one port with SO_REUSEADDR; an exclusive bind makes "port in use" a real error
    s.set_socket_options([](SOCKET sock) {
        const int on = 1;
        setsockopt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&on), sizeof(on));
        setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&on), sizeof(on));
    });
    s.set_read_timeout(60);
    s.set_write_timeout(600);   // a generated image can take minutes
    s.set_keep_alive_max_count(100);

    // every failure is JSON, never an HTML page
    s.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (!res.body.empty() || res.has_header("Content-Type")) return;  // a route already answered
        const char* type = res.status == 404 ? "not_found" : res.status == 413 ? "payload_too_large" : res.status == 405 ? "method_not_allowed" : "error";
        const std::string message = res.status == 404 ? "no such route" : res.status == 413 ? "the request is larger than the server accepts" :
                                    res.status == 405 ? "method not allowed" : "request failed";
        sendError(res, res.status, type, message);
    });
    s.set_exception_handler([](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
        std::string what = "unknown error";
        try { if (ep) std::rethrow_exception(ep); } catch (const std::exception& e) { what = e.what(); } catch (...) {}
        log::error("unhandled exception in " + req.method + " " + req.path + ": " + what);
        sendError(res, 500, "server_error", "internal error: " + what);
    });

    s.Get("/api/v1/health", [](const httplib::Request&, httplib::Response& res) {
        sendJson(res, {{"status", "ok"}, {"name", kAppName}, {"version", kAppVersion}});
    });

    // the browser UI
    fs::path web = o.webRoot.empty() ? core::AppPaths::executableDir() / "web" : o.webRoot;
    std::error_code ec;
    if (fs::is_directory(web, ec)) {
        s.set_mount_point("/", web.string());
    }

    int port = o.port;
    if (port == 0) {
        port = s.bind_to_any_port(o.bindAddress);
        if (port < 0) return "cannot listen on " + o.bindAddress + ": no free port";
    } else if (!s.bind_to_port(o.bindAddress, port)) {
        return "cannot listen on " + o.bindAddress + ":" + std::to_string(port) + " (is another program using the port?)";
    }
    impl_->port = port;
    impl_->bindAddress = o.bindAddress;
    impl_->running = true;
    impl_->thread = std::thread([this] {
        impl_->server->listen_after_bind();
        impl_->running = false;
    });
    impl_->server->wait_until_ready();
    log::info("API server listening on " + o.bindAddress + ":" + std::to_string(port));
    return {};
}

void ApiServer::stop() {
    if (impl_->thread.joinable()) {
        impl_->server->stop();
        impl_->thread.join();
        log::info("API server stopped");
    }
    impl_->running = false;
}

}  // namespace gh::server
