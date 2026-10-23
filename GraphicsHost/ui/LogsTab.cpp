#include "LogsTab.h"

#include "UiUtil.h"
#include "core/Log.h"
#include "host/AppHost.h"

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QComboBox>
#include <QDesktopServices>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QScrollBar>
#include <QTableWidget>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

namespace gh::ui {

namespace {

int rank(const std::string& level) {
    gh::log::Level l = gh::log::Level::Info;
    gh::log::parseLevel(level, l);
    return static_cast<int>(l);
}

}  // namespace

LogsTab::LogsTab(gh::host::AppHost& host, QWidget* parent) : QWidget(parent), host_(host) {
    auto* layout = new QVBoxLayout(this);
    auto* row = new QHBoxLayout();
    level_ = new QComboBox(this);
    for (const char* l : {"trace", "debug", "info", "warn", "error"}) level_->addItem(tr("%1 and worse").arg(QString::fromLatin1(l)), QString::fromLatin1(l));
    level_->setCurrentIndex(2);
    follow_ = new QCheckBox(tr("Follow"), this);
    follow_->setChecked(true);
    count_ = new QLabel(this);
    auto* copy = new QPushButton(tr("Copy"), this);
    auto* folder = new QPushButton(tr("Open log folder"), this);
    row->addWidget(new QLabel(tr("Show"), this));
    row->addWidget(level_);
    row->addWidget(follow_);
    row->addWidget(count_, 1);
    row->addWidget(copy);
    row->addWidget(folder);
    layout->addLayout(row);

    table_ = new QTableWidget(0, 3, this);
    table_->setHorizontalHeaderLabels({tr("Time"), tr("Level"), tr("Message")});
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    table_->verticalHeader()->setVisible(false);
    table_->verticalHeader()->setDefaultSectionSize(20);
    table_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    table_->setColumnWidth(0, 170);
    table_->setColumnWidth(1, 60);
    layout->addWidget(table_, 1);

    connect(level_, &QComboBox::currentIndexChanged, this, [this] { lastSize_ = 0; refresh(); });
    connect(copy, &QPushButton::clicked, this, &LogsTab::copyAll);
    connect(folder, &QPushButton::clicked, this, [this] { QDesktopServices::openUrl(QUrl::fromLocalFile(q(host_.paths().logsDir().string()))); });
    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, [this] { if (isVisible()) refresh(); });
    timer_->start(1000);
    refresh();
}

void LogsTab::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    refresh();
}

void LogsTab::refresh() {
    const auto entries = gh::log::tail(2000);
    if (!entries.empty() && entries.size() == lastSize_ && entries.back().time == lastTime_) return;   // nothing new
    lastSize_ = entries.size();
    lastTime_ = entries.empty() ? std::string() : entries.back().time;

    const int minimum = rank(s(level_->currentData().toString()));
    table_->setUpdatesEnabled(false);
    table_->setRowCount(0);
    for (const auto& e : entries) {
        if (rank(e.level) < minimum) continue;
        const int row = table_->rowCount();
        table_->insertRow(row);
        table_->setItem(row, 0, new QTableWidgetItem(q(e.time)));
        auto* level = new QTableWidgetItem(q(e.level));
        if (e.level == "error") level->setForeground(QColor(0xb0, 0x00, 0x20));
        else if (e.level == "warn") level->setForeground(QColor(0xa0, 0x6a, 0x00));
        table_->setItem(row, 1, level);
        table_->setItem(row, 2, new QTableWidgetItem(q(e.message)));
    }
    table_->setUpdatesEnabled(true);
    count_->setText(tr("%n line(s)", "", table_->rowCount()));
    if (follow_->isChecked()) table_->scrollToBottom();
}

void LogsTab::copyAll() {
    QString text;
    for (int r = 0; r < table_->rowCount(); ++r) {
        text += table_->item(r, 0)->text() + QLatin1Char(' ') + table_->item(r, 1)->text() + QLatin1Char(' ') + table_->item(r, 2)->text() + QLatin1Char('\n');
    }
    QApplication::clipboard()->setText(text);
}

}  // namespace gh::ui
