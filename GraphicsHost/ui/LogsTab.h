#pragma once

#include <QWidget>

class QCheckBox;
class QComboBox;
class QLabel;
class QTableWidget;
class QTimer;

namespace gh::host {
class AppHost;
}

namespace gh::ui {

// The application log (the last 2000 lines kept in memory), filterable by level, refreshed while visible.
class LogsTab : public QWidget {
    Q_OBJECT

public:
    explicit LogsTab(gh::host::AppHost& host, QWidget* parent = nullptr);

protected:
    void showEvent(QShowEvent* event) override;

private:
    void refresh();
    void copyAll();

    gh::host::AppHost& host_;
    QComboBox* level_ = nullptr;
    QCheckBox* follow_ = nullptr;
    QTableWidget* table_ = nullptr;
    QLabel* count_ = nullptr;
    QTimer* timer_ = nullptr;
    size_t lastSize_ = 0;
    std::string lastTime_;
};

}  // namespace gh::ui
