#include "MainWindow.h"

#include "host/AppHost.h"

#include <QLabel>
#include <QStatusBar>
#include <QTabWidget>

namespace gh::ui {

MainWindow::MainWindow(gh::host::AppHost& host, QWidget* parent)
    : QMainWindow(parent), host_(host), tabs_(new QTabWidget(this)) {
    setWindowTitle(tr("GraphicsHost"));
    resize(1100, 720);
    setCentralWidget(tabs_);

    addPlaceholderTab(tr("Generate"), tr("Image generation arrives in phase 4."));
    addPlaceholderTab(tr("Gallery"), tr("The gallery arrives in phase 5."));
    addPlaceholderTab(tr("My Models"), tr("The model library arrives in phase 4."));
    addPlaceholderTab(tr("Discover"), tr("Model discovery and downloads arrive in phase 4."));
    addPlaceholderTab(tr("Settings"), tr("Settings arrive in phase 4."));

    statusBar()->showMessage(runtimeSummary());
}

QString MainWindow::runtimeSummary() const {
    if (const auto* runtime = host_.selectedRuntime()) {
        return tr("Runtime: %1 (backend setting: %2)")
            .arg(QString::fromStdString(runtime->id), QString::fromStdString(host_.settings().backend));
    }
    return tr("No runtime available for backend setting '%1'")
        .arg(QString::fromStdString(host_.settings().backend));
}

void MainWindow::addPlaceholderTab(const QString& title, const QString& text) {
    auto* label = new QLabel(text, tabs_);
    label->setAlignment(Qt::AlignCenter);
    tabs_->addTab(label, title);
}

}  // namespace gh::ui
