#include "MonitorTab.h"

#include "UiUtil.h"
#include "host/AppHost.h"

#include <QChart>
#include <QChartView>
#include <QLabel>
#include <QLineSeries>
#include <QPainter>
#include <QTimer>
#include <QValueAxis>
#include <QVBoxLayout>

namespace gh::ui {

namespace {

constexpr int kHistory = 120;   // seconds on the chart

QValueAxis* makeXAxis() {
    auto* x = new QValueAxis();
    x->setRange(0, kHistory);
    x->setLabelsVisible(false);
    x->setTickCount(5);
    x->setGridLineVisible(true);
    return x;
}

QChartView* makeChart(const QString& title, QValueAxis* y, std::initializer_list<QLineSeries*> series, QWidget* parent) {
    auto* chart = new QChart();
    chart->setTitle(title);
    chart->legend()->setAlignment(Qt::AlignBottom);
    auto* x = makeXAxis();
    chart->addAxis(x, Qt::AlignBottom);
    chart->addAxis(y, Qt::AlignLeft);
    for (auto* s : series) {
        chart->addSeries(s);
        s->attachAxis(x);
        s->attachAxis(y);
    }
    auto* view = new QChartView(chart, parent);
    view->setRenderHint(QPainter::Antialiasing);
    return view;
}

void push(QLineSeries* series, int at, double value) {
    series->append(at, value);
    if (series->count() > kHistory + 1) series->remove(0);
}

}  // namespace

MonitorTab::MonitorTab(gh::host::AppHost& host, QWidget* parent) : QWidget(parent), host_(host) {
    auto* layout = new QVBoxLayout(this);
    summary_ = new QLabel(this);
    layout->addWidget(summary_);

    cpu_ = new QLineSeries(this);
    cpu_->setName(tr("CPU"));
    gpu_ = new QLineSeries(this);
    gpu_->setName(tr("GPU"));
    ram_ = new QLineSeries(this);
    ram_->setName(tr("Memory"));
    vram_ = new QLineSeries(this);
    vram_->setName(tr("Video memory"));

    auto* percent = new QValueAxis();
    percent->setRange(0, 100);
    percent->setLabelFormat(QStringLiteral("%d %%"));
    layout->addWidget(makeChart(tr("Load"), percent, {cpu_, gpu_, ram_}, this), 1);

    // the video memory axis goes up to what the GPU has
    for (const auto& d : host_.gpu().devices) {
        if (d.supported) {
            vramTotalGiB_ = static_cast<double>(d.vramBytes) / (1024.0 * 1024.0 * 1024.0);
            break;
        }
    }
    vramAxis_ = new QValueAxis();
    vramAxis_->setRange(0, vramTotalGiB_ > 0 ? vramTotalGiB_ : 8);
    vramAxis_->setLabelFormat(QStringLiteral("%.0f GiB"));
    layout->addWidget(makeChart(tr("Video memory"), vramAxis_, {vram_}, this), 1);

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &MonitorTab::tick);
}

void MonitorTab::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    host_.monitor().sample();   // a baseline for the CPU delta
    timer_->start(1000);
}

void MonitorTab::hideEvent(QHideEvent* event) {
    QWidget::hideEvent(event);
    timer_->stop();
}

void MonitorTab::tick() {
    const auto u = host_.monitor().sample();
    const int t = sample_++;
    // the chart scrolls: x is the age of the sample, so shift by moving every point
    push(cpu_, t, u.cpuPercent);
    push(ram_, t, u.ramTotalBytes ? 100.0 * static_cast<double>(u.ramUsedBytes) / static_cast<double>(u.ramTotalBytes) : 0.0);
    if (u.gpuPercent) push(gpu_, t, *u.gpuPercent);
    if (u.vramUsedBytes) push(vram_, t, static_cast<double>(*u.vramUsedBytes) / (1024.0 * 1024.0 * 1024.0));
    const double first = std::max(0, t - kHistory);
    for (auto* axisOwner : {cpu_, gpu_, ram_, vram_}) {
        for (auto* axis : axisOwner->attachedAxes()) {
            if (axis->orientation() == Qt::Horizontal) static_cast<QValueAxis*>(axis)->setRange(first, first + kHistory);
        }
    }
    QString text = tr("CPU %1 %   Memory %2 of %3").arg(u.cpuPercent, 0, 'f', 0).arg(humanSize(u.ramUsedBytes), humanSize(u.ramTotalBytes));
    if (u.gpuPercent) text += tr("   GPU %1 %").arg(*u.gpuPercent, 0, 'f', 0);
    if (u.vramUsedBytes) text += tr("   Video memory %1").arg(humanSize(*u.vramUsedBytes));
    if (!u.gpuPercent && !u.vramUsedBytes) text += tr("   (Windows gives no GPU counters on this computer)");
    summary_->setText(text);
}

}  // namespace gh::ui
