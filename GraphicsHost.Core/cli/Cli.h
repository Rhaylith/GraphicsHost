#pragma once

#include <iosfwd>
#include <optional>
#include <filesystem>

namespace gh::cli {

// Same table as LLMHost, so scripts can treat both tools alike.
enum ExitCode : int {
    kOk = 0,
    kFailure = 1,
    kUsage = 2,
    kNotFound = 3,
    kNotAuthorised = 4,
    kServerUnreachable = 5,
    kOutOfMemory = 6,
    kCancelled = 7,
    kConflict = 8,
};

// True when the command line asks for the CLI instead of the desktop window: any argument other than
// --smoke-test (or the explicit "gui" verb, which opens the window).
bool wantsCli(int argc, char** argv);
bool wantsGui(int argc, char** argv);

// Runs one command (argv[0] is the program name). Normal output goes to out, errors and diagnostics to err.
//   graphicshost [--json] [--home DIR] <command>
//   commands: version | hardware | runtime info | config show|path|set <key> <value>
// home overrides the data folder (tests); --home overrides that again.
int run(int argc, char** argv, std::ostream& out, std::ostream& err,
        std::optional<std::filesystem::path> home = std::nullopt);

}  // namespace gh::cli
