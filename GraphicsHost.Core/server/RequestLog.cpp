#include "server/RequestLog.h"

namespace gh::server {

void RequestLog::add(RequestRecord record) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++total_;
    ++byStatus_[record.status];
    records_.push_back(std::move(record));
    while (records_.size() > capacity_) records_.pop_front();
}

std::deque<RequestRecord> RequestLog::recent(size_t limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (records_.size() <= limit) return records_;
    return {records_.end() - static_cast<std::ptrdiff_t>(limit), records_.end()};
}

uint64_t RequestLog::total() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_;
}

std::map<int, uint64_t> RequestLog::byStatus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return byStatus_;
}

double RequestLog::uptimeSeconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
}

}  // namespace gh::server
