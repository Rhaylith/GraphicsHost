#pragma once

#include <QMainWindow>

class QTabWidget;

namespace gh::ui {

// Main window: one tab per area (Generate, Gallery, My Models, Discover, Settings). The tabs are placeholders until
// the services behind them exist (phases 2 to 4).
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

private:
    void addPlaceholderTab(const QString& title, const QString& text);

    QTabWidget* tabs_ = nullptr;
};

}  // namespace gh::ui
