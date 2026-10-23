#include "cli/Cli.h"
#include "host/AppHost.h"
#include "ui/MainWindow.h"

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QStringList>
#include <QTimer>
#include <QTabWidget>

#include <windows.h>

#include <fcntl.h>
#include <io.h>

#include <cstdio>
#include <iostream>
#include <string>

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
    // Only one GraphicsHost per user session: a second start asks the first to show its window and ends. Not for the
    // automated modes, which must be able to run next to a real instance.
    const bool automated = [&] {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--smoke-test" || a == "--screenshot-dir") return true;
        }
        return false;
    }();
    HANDLE instanceMutex = nullptr;
    if (!automated) {
        instanceMutex = CreateMutexW(nullptr, TRUE, L"Local\\GraphicsHost.SingleInstance");
        if (instanceMutex && GetLastError() == ERROR_ALREADY_EXISTS) {
            PostMessageW(HWND_BROADCAST, gh::ui::MainWindow::raiseMessageId(), 0, 0);
            CloseHandle(instanceMutex);
            return 0;
        }
    }

    QApplication app(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("GraphicsHost"));
    QCoreApplication::setOrganizationName(QStringLiteral("GraphicsHost"));
    QApplication::setQuitOnLastWindowClosed(true);

    gh::host::AppHost host;
    gh::ui::MainWindow window(host);
    window.show();

    const QStringList args = app.arguments();
    // --smoke-test: open the window, run one event-loop pass and exit (used by CI and the build checks).
    if (args.contains(QStringLiteral("--smoke-test"))) {
        QTimer::singleShot(0, &app, &QCoreApplication::quit);
    }
    // --screenshot-dir <dir>: save a picture of every tab and exit (the UI checks in the test plan).
    const int shotArg = static_cast<int>(args.indexOf(QStringLiteral("--screenshot-dir")));
    if (shotArg >= 0 && shotArg + 1 < args.size()) {
        const QString dir = args[shotArg + 1];
        QTimer::singleShot(300, &app, [&window, &app, dir] {
            QDir().mkpath(dir);
            for (int i = 0; i < window.tabs()->count(); ++i) {
                window.tabs()->setCurrentIndex(i);
                QCoreApplication::processEvents();
                const QString name = window.tabs()->tabText(i).remove(QLatin1Char('&')).replace(QLatin1Char(' '), QLatin1Char('-')).toLower();
                window.grab().save(dir + QStringLiteral("/") + QString::number(i + 1) + QStringLiteral("-") + name + QStringLiteral(".png"));
            }
            app.quit();
        });
    }
    const int code = app.exec();
    if (instanceMutex) CloseHandle(instanceMutex);
    return code;
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
