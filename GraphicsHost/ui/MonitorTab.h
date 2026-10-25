#pragma once

#include <QWidget>

class QLabel;
class QTimer;
class QChartView;
class QLineSeries;
class QValueAxis;


namespace gh::host {
class AppHost;
}

namespace gh::ui {

// Live graphs of CPU, GPU, memory and video memory over the last two minutes, read about once a second while the tab is
// shown. GPU numbers come from the Windows performance counters and are missing on machines that do not have them.
class MonitorTab : public QWidget {
    Q_OBJECT

public:
    explicit MonitorTab(gh::host::AppHost& host, QWidget* parent = nullptr);

protected:
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;

private:
    void tick();

    gh::host::AppHost& host_;
    QTimer* timer_ = nullptr;
    QLabel* summary_ = nullptr;
    QLineSeries* cpu_ = nullptr;
    QLineSeries* gpu_ = nullptr;
    QLineSeries* ram_ = nullptr;
    QLineSeries* vram_ = nullptr;
    QValueAxis* vramAxis_ = nullptr;
    int sample_ = 0;
    double vramTotalGiB_ = 0;
};

}  // namespace gh::ui
