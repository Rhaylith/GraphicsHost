#include "cli/Cli.h"
#include "host/AppHost.h"
#include "ui/MainWindow.h"

#include <QApplication>
#include <QCoreApplication>
#include <QStringList>
#include <QTimer>

#include <windows.h>

#include <fcntl.h>
#include <io.h>

#include <cstdio>
#include <iostream>

namespace {

// The exe is a GUI-subsystem program, so a console has to be attached for the command-line mode. Redirected handles
// (pipes, files) are used as they are; otherwise the parent console is attached.
void bindStandardStreams() {
    struct Stream {
        DWORD handleId;
        int fd;
        const char* consoleDevice;
        const char* mode;
        FILE* stream;
    };
    Stream streams[] = {{STD_OUTPUT_HANDLE, 1, "CONOUT$", "w", stdout}, {STD_ERROR_HANDLE, 2, "CONOUT$", "w", stderr}};
    bool attached = false;
    for (const Stream& s : streams) {
        HANDLE handle = GetStdHandle(s.handleId);
        const bool usable = handle != nullptr && handle != INVALID_HANDLE_VALUE && GetFileType(handle) != FILE_TYPE_UNKNOWN;
        if (usable) {
            const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_TEXT);
            if (fd >= 0) {
                _dup2(fd, s.fd);
            }
        } else {
            if (!attached) {
                attached = AttachConsole(ATTACH_PARENT_PROCESS) != 0;
            }
            if (attached) {
                FILE* reopened = nullptr;
                freopen_s(&reopened, s.consoleDevice, s.mode, s.stream);
            }
        }
    }
    std::ios::sync_with_stdio(true);
}

int runGui(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("GraphicsHost"));
    QCoreApplication::setOrganizationName(QStringLiteral("GraphicsHost"));

    gh::host::AppHost host;
    gh::ui::MainWindow window(host);
    window.show();

    // --smoke-test: open the window, run one event-loop pass and exit (used by CI and the build checks).
    if (app.arguments().contains(QStringLiteral("--smoke-test"))) {
        QTimer::singleShot(0, &app, &QCoreApplication::quit);
    }
    return app.exec();
}

}  // namespace

int main(int argc, char* argv[]) {
    if (gh::cli::wantsCli(argc, argv)) {
        bindStandardStreams();
        const int code = gh::cli::run(argc, argv, std::cout, std::cerr);
        std::cout.flush();
        std::cerr.flush();
        return code;
    }
    return runGui(argc, argv);
}
