#include "cli/Cli.h"
#include "host/AppHost.h"
#include "ui/GenerateTab.h"
#include "ui/MainWindow.h"

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
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
    // --generate-prompt <text>: first generate a small image on the Generate tab and wait for it, so the picture shows
    // a finished generation (the model must be in the data folder; use GRAPHICSHOST_HOME to point at a prepared one).
    const int shotArg = static_cast<int>(args.indexOf(QStringLiteral("--screenshot-dir")));
    const int promptArg = static_cast<int>(args.indexOf(QStringLiteral("--generate-prompt")));
    if (shotArg >= 0 && shotArg + 1 < args.size()) {
        const QString dir = args[shotArg + 1];
        const QString prompt = promptArg >= 0 && promptArg + 1 < args.size() ? args[promptArg + 1] : QString();
        auto takeShots = [&window, &app, dir] {
            QDir().mkpath(dir);
            for (int i = 0; i < window.tabs()->count(); ++i) {
                window.tabs()->setCurrentIndex(i);
                QEventLoop settle;   // let the tab load what it loads in the background
                QTimer::singleShot(700, &settle, &QEventLoop::quit);
                settle.exec();
                const QString name = window.tabs()->tabText(i).remove(QLatin1Char('&')).replace(QLatin1Char(' '), QLatin1Char('-')).toLower();
                window.grab().save(dir + QStringLiteral("/") + QString::number(i + 1) + QStringLiteral("-") + name + QStringLiteral(".png"));
            }
            app.quit();
        };
        QTimer::singleShot(300, &app, [&window, &app, takeShots, prompt] {
            if (prompt.isEmpty()) {
                takeShots();
                return;
            }
            auto* tab = window.generateTab();
            tab->loadParameters({{"prompt", prompt.toStdString()}, {"width", 128}, {"height", 128}, {"steps", 4}, {"seed", 7}}, QString());
            tab->onGenerate();
            auto* wait = new QTimer(&app);
            auto* ticks = new int(0);
            QObject::connect(wait, &QTimer::timeout, &app, [=, &window] {
                ++*ticks;
                if (!window.generateTab()->busy() || *ticks > 1200) {   // up to 10 minutes
                    wait->stop();
                    std::cout << "generate: " << window.generateTab()->statusText().toStdString() << std::endl;
                    takeShots();
                }
            });
            wait->start(500);
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
