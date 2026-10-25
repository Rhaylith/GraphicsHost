#include "server/ClientLimits.h"

#include <algorithm>
#include <stdexcept>

namespace gh::server {

namespace {

bool parseIpv4(const std::string& text, uint32_t& out) {
    uint32_t value = 0;
    int parts = 0;
    size_t i = 0;
    while (i <= text.size()) {
        size_t j = text.find('.', i);
        if (j == std::string::npos) j = text.size();
        const std::string part = text.substr(i, j - i);
        if (part.empty() || part.size() > 3 || !std::all_of(part.begin(), part.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) return false;
        const int n = std::stoi(part);
        if (n > 255) return false;
        value = (value << 8) | static_cast<uint32_t>(n);
        ++parts;
        if (j == text.size()) break;
        i = j + 1;
    }
    out = value;
    return parts == 4;
}

bool isLoopback(const std::string& a) {
    return a == "::1" || a == "localhost" || a.rfind("127.", 0) == 0 || a == "::ffff:127.0.0.1";
}

}  // namespace

bool IpAllowList::valid(const std::string& entry) {
    try {
        IpAllowList list({entry});
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

IpAllowList::IpAllowList(const std::vector<std::string>& entries) {
    for (const auto& entry : entries) {
        const auto slash = entry.find('/');
        const std::string addr = entry.substr(0, slash);
        uint32_t base = 0;
        if (parseIpv4(addr, base)) {
            int bits = 32;
            if (slash != std::string::npos) {
                const std::string b = entry.substr(slash + 1);
                if (b.empty() || b.size() > 2 || !std::all_of(b.begin(), b.end(), [](unsigned char c) { return std::isdigit(c) != 0; }) || std::stoi(b) > 32) {
                    throw std::invalid_argument("not an address range: " + entry);
                }
                bits = std::stoi(b);
            }
            const uint32_t mask = bits == 0 ? 0 : 0xffffffffu << (32 - bits);
            ranges_.push_back({base & mask, mask});
        } else if (slash == std::string::npos && addr.find(':') != std::string::npos) {
            exact_.push_back(addr);
        } else {
            throw std::invalid_argument("not an address or range: " + entry);
        }
    }
}

bool IpAllowList::allows(const std::string& address) const {
    if (!restricted() || isLoopback(address)) return true;
    uint32_t v = 0;
    std::string a = address;
    if (a.rfind("::ffff:", 0) == 0) a = a.substr(7);   // an IPv4 client on a dual-stack socket
    if (parseIpv4(a, v)) {
        return std::any_of(ranges_.begin(), ranges_.end(), [&](const Range& r) { return (v & r.mask) == r.base; });
    }
    return std::find(exact_.begin(), exact_.end(), address) != exact_.end();
}

bool RateLimiter::allow(const std::string& client, int* retryAfterSeconds, Clock::time_point now) {
    if (perMinute_ <= 0) return true;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& times = seen_[client];
    const auto cutoff = now - std::chrono::minutes(1);
    times.erase(std::remove_if(times.begin(), times.end(), [&](const Clock::time_point& t) { return t <= cutoff; }), times.end());
    if (static_cast<int>(times.size()) >= perMinute_) {
        if (retryAfterSeconds) {
            const auto wait = std::chrono::duration_cast<std::chrono::seconds>(times.front() + std::chrono::minutes(1) - now).count();
            *retryAfterSeconds = static_cast<int>(std::max<long long>(1, wait + 1));
        }
        return false;
    }
    times.push_back(now);
    if (seen_.size() > 4096) {   // forget clients that have been quiet
        for (auto it = seen_.begin(); it != seen_.end();) it = it->second.empty() || it->second.back() <= cutoff ? seen_.erase(it) : std::next(it);
    }
    return true;
}

}  // namespace gh::server
