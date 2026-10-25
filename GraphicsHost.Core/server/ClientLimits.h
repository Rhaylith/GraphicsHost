#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace gh::server {

// Which addresses may connect: "192.168.1.0/24", "10.0.0.5", "::1". Loopback (this computer) is always allowed; an
// empty list means no restriction. Only IPv4 ranges and exact IPv6 addresses are understood.
class IpAllowList {
public:
    // Throws std::invalid_argument naming the entry that cannot be read.
    explicit IpAllowList(const std::vector<std::string>& entries = {});
    bool allows(const std::string& address) const;
    bool restricted() const { return !ranges_.empty() || !exact_.empty(); }

    static bool valid(const std::string& entry);

private:
    struct Range {
        uint32_t base, mask;
    };
    std::vector<Range> ranges_;
    std::vector<std::string> exact_;
};

// At most `perMinute` requests a minute for each client (a sliding window of one minute). 0 = no limit.
class RateLimiter {
public:
    using Clock = std::chrono::steady_clock;
    explicit RateLimiter(int perMinute = 0) : perMinute_(perMinute) {}

    // True when the request may go ahead; false when the client is over its limit, with the seconds until it may try again.
    bool allow(const std::string& client, int* retryAfterSeconds = nullptr, Clock::time_point now = Clock::now());
    int perMinute() const { return perMinute_; }

private:
    int perMinute_;
    std::mutex mutex_;
    std::map<std::string, std::vector<Clock::time_point>> seen_;
};

}  // namespace gh::server
