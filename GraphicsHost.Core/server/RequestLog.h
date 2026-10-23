#pragma once

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>

namespace gh::server {

struct RequestRecord {
    std::string time;       // ISO 8601, UTC
    std::string method;
    std::string path;       // no query string: it can carry search text
    int status = 0;
    double milliseconds = 0;
    std::string remote;
};

// The last requests the server answered, and counters for /metrics. Thread safe.
class RequestLog {
public:
    explicit RequestLog(size_t capacity = 500) : capacity_(capacity), started_(std::chrono::steady_clock::now()) {}

    void add(RequestRecord record);
    std::deque<RequestRecord> recent(size_t limit) const;   // newest last

    uint64_t total() const;
    std::map<int, uint64_t> byStatus() const;
    double uptimeSeconds() const;

private:
    size_t capacity_;
    std::chrono::steady_clock::time_point started_;
    mutable std::mutex mutex_;
    std::deque<RequestRecord> records_;
    uint64_t total_ = 0;
    std::map<int, uint64_t> byStatus_;
};

}  // namespace gh::server
