#pragma once

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gh::test {

// Deterministic content: any byte range of a "blob" can be regenerated and compared.
inline std::string blobBytes(uint64_t begin, uint64_t end) {
    std::string s;
    s.reserve(static_cast<size_t>(end - begin));
    for (uint64_t i = begin; i < end; ++i) s.push_back(static_cast<char>((i * 131 + 17) & 0xFF));
    return s;
}

// A local HTTP server on an ephemeral port for the download and API tests:
//   /blob/<n>         n bytes of blobBytes, with Range support; slowMs per 16 KiB chunk
//   /norange/<n>      the same, but the server pretends not to know Range requests
//   /gated/<n>        needs "Authorization: Bearer secret", else 401
//   anything else     404
class TestHttpServer {
public:
    TestHttpServer() {
        server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response&) {
            std::lock_guard<std::mutex> lock(mutex_);
            requests.push_back(req.path);
            lastRange = req.has_header("Range") ? req.get_header_value("Range") : "";
            lastAuthorization = req.has_header("Authorization") ? req.get_header_value("Authorization") : "";
            if (req.path.rfind("/norange/", 0) == 0) {
                const_cast<httplib::Request&>(req).headers.erase("Range");
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        auto blob = [this](const httplib::Request& req, httplib::Response& res) {
            const uint64_t size = std::stoull(req.matches[1]);
            res.set_content_provider(size, "application/octet-stream",
                                     [this, size](size_t offset, size_t length, httplib::DataSink& sink) {
                                         const size_t chunk = std::min<size_t>(length, 16 * 1024);
                                         if (slowMs > 0) std::this_thread::sleep_for(std::chrono::milliseconds(slowMs));
                                         const std::string data = blobBytes(offset, std::min<uint64_t>(offset + chunk, size));
                                         return sink.write(data.data(), data.size());
                                     });
        };
        server_.Get(R"(/blob/(\d+))", blob);
        server_.Get(R"(/norange/(\d+))", blob);
        server_.Get(R"(/gated/(\d+))", [this, blob](const httplib::Request& req, httplib::Response& res) {
            if (req.get_header_value("Authorization") != "Bearer secret") {
                res.status = 401;
                res.set_content("token needed", "text/plain");
                return;
            }
            blob(req, res);
        });
        port_ = server_.bind_to_any_port("127.0.0.1");
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        server_.wait_until_ready();
    }
    ~TestHttpServer() {
        server_.stop();
        thread_.join();
    }
    TestHttpServer(const TestHttpServer&) = delete;
    TestHttpServer& operator=(const TestHttpServer&) = delete;

    std::string url(const std::string& path) const { return "http://127.0.0.1:" + std::to_string(port_) + path; }
    httplib::Server& server() { return server_; }
    int port() const { return port_; }

    std::atomic<int> slowMs{0};
    std::mutex mutex_;
    std::vector<std::string> requests;
    std::string lastRange, lastAuthorization;

    std::string range() {
        std::lock_guard<std::mutex> lock(mutex_);
        return lastRange;
    }
    std::string authorization() {
        std::lock_guard<std::mutex> lock(mutex_);
        return lastAuthorization;
    }

private:
    httplib::Server server_;
    int port_ = 0;
    std::thread thread_;
};

}  // namespace gh::test
