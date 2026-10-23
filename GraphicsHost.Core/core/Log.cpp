#include "core/Log.h"

#include <spdlog/details/null_mutex.h>
#include <spdlog/formatter.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>

namespace gh::log {

namespace {

constexpr size_t kRingCapacity = 2000;

std::string isoTime(spdlog::log_clock::time_point tp) {
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()) % 1000;
    const std::time_t seconds = std::chrono::system_clock::to_time_t(tp);
    std::tm utc{};
    gmtime_s(&utc, &seconds);
    char buffer[40];
    std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", utc.tm_year + 1900, utc.tm_mon + 1,
                  utc.tm_mday, utc.tm_hour, utc.tm_min, utc.tm_sec, static_cast<int>(ms.count()));
    return buffer;
}

const char* levelName(spdlog::level::level_enum level) {
    switch (level) {
        case spdlog::level::trace: return "trace";
        case spdlog::level::debug: return "debug";
        case spdlog::level::info: return "info";
        case spdlog::level::warn: return "warn";
        default: return "error";
    }
}

std::string jsonEscape(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char escaped[8];
                    std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
                    out += escaped;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// {"ts":"...","level":"info","msg":"..."}
class JsonLinesFormatter final : public spdlog::formatter {
public:
    void format(const spdlog::details::log_msg& msg, spdlog::memory_buf_t& dest) override {
        std::string line = "{\"ts\":\"" + isoTime(msg.time) + "\",\"level\":\"" + levelName(msg.level) + "\",\"msg\":\"" +
                           jsonEscape(std::string_view(msg.payload.data(), msg.payload.size())) + "\"}\n";
        dest.append(line.data(), line.data() + line.size());
    }
    std::unique_ptr<spdlog::formatter> clone() const override { return std::make_unique<JsonLinesFormatter>(); }
};

// Keeps the last entries in memory for the UI.
class RingSink final : public spdlog::sinks::base_sink<std::mutex> {
public:
    std::vector<Entry> tail(size_t maxEntries) {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t count = std::min(maxEntries, entries_.size());
        return std::vector<Entry>(entries_.end() - static_cast<std::ptrdiff_t>(count), entries_.end());
    }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        entries_.push_back({isoTime(msg.time), levelName(msg.level),
                            std::string(msg.payload.data(), msg.payload.size())});
        if (entries_.size() > kRingCapacity) {
            entries_.pop_front();
        }
    }
    void flush_() override {}

private:
    std::deque<Entry> entries_;
};

// Recent warnings/errors from the native engine, numbered so callers can ask for "the ones since my call".
std::mutex g_nativeMutex;
uint64_t g_nativeSequence = 0;
std::deque<std::pair<uint64_t, std::string>> g_nativeErrors;

std::mutex g_mutex;
std::shared_ptr<spdlog::logger> g_logger;
std::shared_ptr<RingSink> g_ring;

spdlog::level::level_enum toSpdlog(Level level) {
    switch (level) {
        case Level::Trace: return spdlog::level::trace;
        case Level::Debug: return spdlog::level::debug;
        case Level::Info: return spdlog::level::info;
        case Level::Warn: return spdlog::level::warn;
        default: return spdlog::level::err;
    }
}

}  // namespace

bool parseLevel(std::string_view name, Level& out) {
    if (name == "trace") out = Level::Trace;
    else if (name == "debug") out = Level::Debug;
    else if (name == "info") out = Level::Info;
    else if (name == "warn") out = Level::Warn;
    else if (name == "error") out = Level::Error;
    else return false;
    return true;
}

void init(const std::filesystem::path& logsDir, std::string_view level) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_logger) {
        std::filesystem::create_directories(logsDir);
        auto file = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            (logsDir / "graphicshost.log").string(), 5 * 1024 * 1024, 5);
        g_ring = std::make_shared<RingSink>();
        file->set_formatter(std::make_unique<JsonLinesFormatter>());
        g_logger = std::make_shared<spdlog::logger>("graphicshost", spdlog::sinks_init_list{file, g_ring});
        g_logger->flush_on(spdlog::level::warn);
    }
    Level parsed = Level::Info;
    parseLevel(level, parsed);
    g_logger->set_level(toSpdlog(parsed));
}

void shutdown() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_logger) {
        g_logger->flush();
    }
    g_logger.reset();
    g_ring.reset();
}

void setLevel(std::string_view level) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_logger) {
        return;
    }
    Level parsed = Level::Info;
    parseLevel(level, parsed);
    g_logger->set_level(toSpdlog(parsed));
}

void write(Level level, std::string_view message) {
    std::shared_ptr<spdlog::logger> logger;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        logger = g_logger;
    }
    if (logger) {
        logger->log(toSpdlog(level), spdlog::string_view_t(message.data(), message.size()));
    }
}

std::vector<Entry> tail(size_t maxEntries) {
    std::shared_ptr<RingSink> ring;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        ring = g_ring;
    }
    return ring ? ring->tail(maxEntries) : std::vector<Entry>{};
}

void nativeLog(int level, const char* text) {
    if (!text) {
        return;
    }
    std::string message(text);
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
        message.pop_back();
    }
    if (message.empty()) {
        return;
    }
    Level mapped = Level::Info;
    switch (level) {
        case 0: mapped = Level::Debug; break;
        case 1: mapped = Level::Debug; break;
        case 2: mapped = Level::Info; break;
        case 3: mapped = Level::Warn; break;
        default: mapped = Level::Error; break;
    }
    if (level >= 3) {
        std::lock_guard<std::mutex> lock(g_nativeMutex);
        g_nativeErrors.push_back({++g_nativeSequence, message});
        if (g_nativeErrors.size() > 64) {
            g_nativeErrors.pop_front();
        }
    }
    write(mapped, "[engine] " + message);
}

uint64_t nativeMark() {
    std::lock_guard<std::mutex> lock(g_nativeMutex);
    return g_nativeSequence;
}

std::vector<std::string> nativeErrorsSince(uint64_t mark) {
    std::lock_guard<std::mutex> lock(g_nativeMutex);
    std::vector<std::string> out;
    for (const auto& [sequence, message] : g_nativeErrors) {
        if (sequence > mark) {
            out.push_back(message);
        }
    }
    return out;
}

}  // namespace gh::log
