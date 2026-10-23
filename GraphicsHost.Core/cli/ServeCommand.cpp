#include "cli/Cli.h"
#include "cli/Commands.h"
#include "core/Log.h"
#include "server/ApiServer.h"

#include <windows.h>

#include <atomic>
#include <filesystem>
#include <chrono>
#include <ostream>
#include <thread>

namespace gh::cli {

using Json = nlohmann::json;

namespace {

constexpr wchar_t kServiceName[] = L"GraphicsHost";

std::atomic<bool> g_stop{false};

BOOL WINAPI onConsoleControl(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        g_stop = true;
        return TRUE;
    }
    return FALSE;
}

// ---- running as a Windows service --------------------------------------------------------------------------------
struct ServiceRun {
    host::AppHost* host = nullptr;
    server::ServerOptions options;
    SERVICE_STATUS_HANDLE handle = nullptr;
    DWORD exitCode = 0;
} g_service;

void reportState(DWORD state, DWORD exitCode = 0, DWORD waitHintMs = 0) {
    static DWORD checkpoint = 1;
    SERVICE_STATUS status{};
    status.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    status.dwCurrentState = state;
    status.dwControlsAccepted = state == SERVICE_START_PENDING ? 0 : (SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN);
    status.dwWin32ExitCode = exitCode;
    status.dwCheckPoint = (state == SERVICE_RUNNING || state == SERVICE_STOPPED) ? 0 : checkpoint++;
    status.dwWaitHint = waitHintMs;
    SetServiceStatus(g_service.handle, &status);
}

void WINAPI onServiceControl(DWORD control) {
    if (control == SERVICE_CONTROL_STOP || control == SERVICE_CONTROL_SHUTDOWN) {
        reportState(SERVICE_STOP_PENDING, 0, 10000);
        g_stop = true;
    }
}

void WINAPI serviceMain(DWORD, LPWSTR*) {
    g_service.handle = RegisterServiceCtrlHandlerW(kServiceName, onServiceControl);
    if (!g_service.handle) return;
    reportState(SERVICE_START_PENDING, 0, 15000);
    server::ApiServer api(*g_service.host);
    const std::string problem = api.start(g_service.options);
    if (!problem.empty()) {
        log::error("service: " + problem);
        reportState(SERVICE_STOPPED, ERROR_SERVICE_SPECIFIC_ERROR);
        return;
    }
    reportState(SERVICE_RUNNING);
    g_stop = false;
    while (!g_stop && api.running()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    api.stop();
    reportState(SERVICE_STOPPED);
}

std::string lastErrorText(DWORD code) {
    char buffer[512] = {};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0, buffer, sizeof(buffer), nullptr);
    std::string text = buffer;
    while (!text.empty() && (text.back() == '\r' || text.back() == '\n' || text.back() == ' ')) text.pop_back();
    return text.empty() ? "error " + std::to_string(code) : text;
}

struct ScHandle {
    SC_HANDLE h = nullptr;
    ~ScHandle() { if (h) CloseServiceHandle(h); }
    explicit operator bool() const { return h != nullptr; }
};

const char* stateName(DWORD s) {
    switch (s) {
        case SERVICE_STOPPED: return "stopped";
        case SERVICE_START_PENDING: return "starting";
        case SERVICE_STOP_PENDING: return "stopping";
        case SERVICE_RUNNING: return "running";
        default: return "other";
    }
}

}  // namespace

void requestServeStop() {
    g_stop = true;
}

int runServe(host::AppHost& host, int port, const std::string& bind, bool asService, bool json, std::ostream& out, std::ostream& err) {
    server::ServerOptions options = server::serverOptionsFrom(host);
    if (port >= 0) options.port = port;
    if (!bind.empty()) options.bindAddress = bind;

    if (asService) {
        g_service.host = &host;
        g_service.options = options;
        wchar_t name[] = L"GraphicsHost";
        SERVICE_TABLE_ENTRYW table[] = {{name, serviceMain}, {nullptr, nullptr}};
        if (!StartServiceCtrlDispatcherW(table)) {
            err << "error: --service only works when Windows starts the program as a service (graphicshost service install)\n";
            return kFailure;
        }
        return kOk;
    }

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

int runService(host::AppHost& host, const std::string& action, bool json, std::ostream& out, std::ostream& err) {
    if (action == "status") {
        ScHandle manager{OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT)};
        if (!manager) {
            err << "error: cannot reach the service manager: " << lastErrorText(GetLastError()) << "\n";
            return kFailure;
        }
        ScHandle service{OpenServiceW(manager.h, kServiceName, SERVICE_QUERY_STATUS)};
        if (!service) {
            const DWORD e = GetLastError();
            if (e == ERROR_SERVICE_DOES_NOT_EXIST) {
                if (json) out << Json{{"installed", false}}.dump() << "\n";
                else out << "the GraphicsHost service is not installed (graphicshost service install)\n";
                return kOk;
            }
            err << "error: " << lastErrorText(e) << "\n";
            return kFailure;
        }
        SERVICE_STATUS status{};
        QueryServiceStatus(service.h, &status);
        if (json) out << Json{{"installed", true}, {"state", stateName(status.dwCurrentState)}}.dump() << "\n";
        else out << "the GraphicsHost service is installed and " << stateName(status.dwCurrentState) << "\n";
        return kOk;
    }

    if (action == "install") {
        wchar_t exe[MAX_PATH * 2] = {};
        GetModuleFileNameW(nullptr, exe, static_cast<DWORD>(std::size(exe)));
        // the service runs as LocalSystem, which has another user profile: give it this data folder explicitly
        const std::wstring command = L"\"" + std::wstring(exe) + L"\" --home \"" + host.paths().root().wstring() + L"\" serve --service";
        ScHandle manager{OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CREATE_SERVICE)};
        if (!manager) {
            err << "error: installing a service needs an administrator prompt: " << lastErrorText(GetLastError()) << "\n";
            return kNotAuthorised;
        }
        ScHandle service{CreateServiceW(manager.h, kServiceName, L"GraphicsHost image generation server", SERVICE_CHANGE_CONFIG | SERVICE_START,
                                        SERVICE_WIN32_OWN_PROCESS, SERVICE_AUTO_START, SERVICE_ERROR_NORMAL, command.c_str(), nullptr, nullptr, nullptr,
                                        nullptr, nullptr)};
        if (!service) {
            const DWORD e = GetLastError();
            err << "error: " << (e == ERROR_SERVICE_EXISTS ? "the service is already installed" : lastErrorText(e)) << "\n";
            return e == ERROR_SERVICE_EXISTS ? kConflict : kFailure;
        }
        wchar_t description[] = L"Runs the GraphicsHost API server (OpenAI-compatible image generation) without a window.";
        SERVICE_DESCRIPTIONW d{description};
        ChangeServiceConfig2W(service.h, SERVICE_CONFIG_DESCRIPTION, &d);
        if (json) out << Json{{"installed", true}, {"command", std::filesystem::path(command).string()}}.dump() << "\n";
        else out << "service installed; it starts with Windows. Start it now with: sc start GraphicsHost\nThe server settings (port, tokens, https) come from " << host.paths().root().string() << "\n";
        return kOk;
    }

    if (action == "uninstall") {
        ScHandle manager{OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT)};
        ScHandle service{manager ? OpenServiceW(manager.h, kServiceName, DELETE | SERVICE_STOP | SERVICE_QUERY_STATUS) : nullptr};
        if (!service) {
            const DWORD e = GetLastError();
            if (e == ERROR_SERVICE_DOES_NOT_EXIST) {
                err << "error: the service is not installed\n";
                return kNotFound;
            }
            err << "error: uninstalling needs an administrator prompt: " << lastErrorText(e) << "\n";
            return kNotAuthorised;
        }
        SERVICE_STATUS status{};
        ControlService(service.h, SERVICE_CONTROL_STOP, &status);
        if (!DeleteService(service.h)) {
            err << "error: " << lastErrorText(GetLastError()) << "\n";
            return kFailure;
        }
        if (json) out << Json{{"installed", false}}.dump() << "\n";
        else out << "service removed\n";
        return kOk;
    }
    err << "usage: graphicshost service [status | install | uninstall]\n";
    return kUsage;
}

}  // namespace gh::cli
