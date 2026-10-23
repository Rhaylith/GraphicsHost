#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace gh::log {

enum class Level { Trace, Debug, Info, Warn, Error };

struct Entry {
    std::string time;     // ISO 8601, UTC, milliseconds
    std::string level;    // trace|debug|info|warn|error
    std::string message;
};

// Starts logging to <logsDir>/graphicshost.log as JSON lines (one object per line, 5 MB x 5 files rotation) and
// into an in-memory ring buffer for the UI's log tab. Safe to call again; later calls only change the level.
// level: trace|debug|info|warn|error (unknown values mean info).
void init(const std::filesystem::path& logsDir, std::string_view level);
void shutdown();

void setLevel(std::string_view level);
bool parseLevel(std::string_view name, Level& out);

void write(Level level, std::string_view message);
inline void debug(std::string_view m) { write(Level::Debug, m); }
inline void info(std::string_view m) { write(Level::Info, m); }
inline void warn(std::string_view m) { write(Level::Warn, m); }
inline void error(std::string_view m) { write(Level::Error, m); }

// Newest entries last, at most maxEntries (the buffer keeps the last 2000).
std::vector<Entry> tail(size_t maxEntries);

// Receives the native engine's log (gh_log_fn): 0 debug, 1 verbose, 2 info, 3 warning, 4 error. Trailing newlines
// are trimmed and the text is prefixed with "[engine] ".
void nativeLog(int level, const char* text);

}  // namespace gh::log
