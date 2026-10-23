#pragma once

#include <QMainWindow>

class QTabWidget;

namespace gh::host {
class AppHost;
}

namespace gh::ui {

// Main window: one tab per area (Generate, Gallery, My Models, Discover, Settings). The tabs are placeholders until
// the services behind them exist (phases 2 to 4). The status bar shows which runtime is in use.
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(gh::host::AppHost& host, QWidget* parent = nullptr);

private:
    void addPlaceholderTab(const QString& title, const QString& text);
    QString runtimeSummary() const;

    gh::host::AppHost& host_;
    QTabWidget* tabs_ = nullptr;
};

}  // namespace gh::ui
