#include "ui/MainWindow.h"

#include <QApplication>
#include <QCoreApplication>
#include <QStringList>
#include <QTimer>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("GraphicsHost"));
    QCoreApplication::setOrganizationName(QStringLiteral("GraphicsHost"));

    gh::ui::MainWindow window;
    window.show();

    // --smoke-test: open the window, run one event-loop pass and exit (used by CI and the build checks).
    if (app.arguments().contains(QStringLiteral("--smoke-test"))) {
        QTimer::singleShot(0, &app, &QCoreApplication::quit);
    }
    return app.exec();
}
