#include "cli/Cli.h"
#include "cli/Commands.h"
#include "server/ApiServer.h"

#include <windows.h>

#include <atomic>
#include <chrono>
#include <ostream>
#include <thread>

namespace gh::cli {

using Json = nlohmann::json;

namespace {

std::atomic<bool> g_stop{false};

BOOL WINAPI onConsoleControl(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        g_stop = true;
        return TRUE;
    }
    return FALSE;
}

}  // namespace

void requestServeStop() {
    g_stop = true;
}

int runServe(host::AppHost& host, int port, const std::string& bind, bool json, std::ostream& out, std::ostream& err) {
    server::ServerOptions options = server::serverOptionsFrom(host);
    if (port >= 0) options.port = port;
    if (!bind.empty()) options.bindAddress = bind;

    server::ApiServer api(host);
    const std::string problem = api.start(options);
    if (!problem.empty()) {
        err << "error: " << problem << "\n";
        return kFailure;
    }
    if (json) {
        out << Json{{"listening", api.url()}, {"port", api.port()}, {"bind", options.bindAddress}}.dump() << "\n";
    } else {
        out << "listening on " << api.url() << "  (browser UI at the same address; Ctrl+C to stop)\n";
        if (!host.tokens().any() && !host.settings().server.requireToken) {
            out << "note: no access tokens exist; only programs on this computer can connect. Create one with: graphicshost token create <name>\n";
        }
    }
    out.flush();

    g_stop = false;
    SetConsoleCtrlHandler(onConsoleControl, TRUE);
    while (!g_stop && api.running()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    SetConsoleCtrlHandler(onConsoleControl, FALSE);
    api.stop();
    if (!json) out << "stopped\n";
    return kOk;
}

}  // namespace gh::cli
