#include "GalleryTab.h"

#include "ImageView.h"
#include "UiUtil.h"
#include "host/AppHost.h"

#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProcess>
#include <QPushButton>
#include <QSplitter>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <thread>

namespace gh::ui {

namespace {
constexpr int kPage = 60;
constexpr int kThumb = 160;
}  // namespace

GalleryTab::GalleryTab(gh::host::AppHost& host, QWidget* parent) : QWidget(parent), host_(host) {
    auto* layout = new QVBoxLayout(this);
    auto* top = new QHBoxLayout();
    search_ = new QLineEdit(this);
    search_->setPlaceholderText(tr("Search prompts…"));
    search_->setClearButtonEnabled(true);
    count_ = new QLabel(this);
    auto* exportButton = new QPushButton(tr("Export…"), this);
    exportButton->setToolTip(tr("Copy the selected images, or all that match the search, into a folder"));
    top->addWidget(search_, 1);
    top->addWidget(count_);
    top->addWidget(exportButton);
    layout->addLayout(top);

    auto* split = new QSplitter(Qt::Horizontal, this);
    auto* left = new QWidget(split);
    auto* leftLayout = new QVBoxLayout(left);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    list_ = new QListWidget(left);
    list_->setViewMode(QListView::IconMode);
    list_->setIconSize(QSize(kThumb, kThumb));
    list_->setGridSize(QSize(kThumb + 16, kThumb + 28));
    list_->setResizeMode(QListView::Adjust);
    list_->setMovement(QListView::Static);
    list_->setSelectionMode(QAbstractItemView::ExtendedSelection);
    list_->setUniformItemSizes(true);
    list_->setContextMenuPolicy(Qt::CustomContextMenu);
    more_ = new QPushButton(tr("Load more"), left);
    more_->setVisible(false);
    leftLayout->addWidget(list_, 1);
    leftLayout->addWidget(more_);

    auto* right = new QWidget(split);
    auto* rightLayout = new QVBoxLayout(right);
    preview_ = new ImageView(right);
    details_ = new QPlainTextEdit(right);
    details_->setReadOnly(true);
    details_->setMaximumHeight(170);
    auto* buttons = new QHBoxLayout();
    reuse_ = new QPushButton(tr("Reuse parameters"), right);
    save_ = new QPushButton(tr("Save as…"), right);
    folder_ = new QPushButton(tr("Show in folder"), right);
    remove_ = new QPushButton(tr("Delete"), right);
    for (auto* b : {reuse_, save_, folder_, remove_}) buttons->addWidget(b);
    rightLayout->addWidget(preview_, 1);
    rightLayout->addWidget(details_);
    rightLayout->addLayout(buttons);

    split->addWidget(left);
    split->addWidget(right);
    split->setStretchFactor(0, 3);
    split->setStretchFactor(1, 2);
    layout->addWidget(split, 1);

    searchTimer_ = new QTimer(this);
    searchTimer_->setSingleShot(true);
    searchTimer_->setInterval(300);
    connect(search_, &QLineEdit::textChanged, this, [this] { searchTimer_->start(); });
    connect(searchTimer_, &QTimer::timeout, this, &GalleryTab::reload);
    connect(more_, &QPushButton::clicked, this, &GalleryTab::loadPage);
    connect(list_, &QListWidget::currentRowChanged, this, &GalleryTab::select);
    connect(list_, &QListWidget::itemDoubleClicked, this, [this] { reuse_->click(); });
    connect(exportButton, &QPushButton::clicked, this, &GalleryTab::exportSelected);
    connect(save_, &QPushButton::clicked, this, &GalleryTab::saveSelectedAs);
    connect(remove_, &QPushButton::clicked, this, &GalleryTab::deleteSelected);
    connect(folder_, &QPushButton::clicked, this, [this] {
        if (const auto* e = current()) QProcess::startDetached(QStringLiteral("explorer.exe"), {QStringLiteral("/select,"), QDir::toNativeSeparators(q(host_.gallery().absolutePath(*e).string()))});
    });
    connect(reuse_, &QPushButton::clicked, this, [this] {
        if (const auto* e = current()) emit reuseRequested(e->parameters.value("request", nlohmann::json::object()), q(e->model));
    });
    connect(list_, &QWidget::customContextMenuRequested, this, [this](const QPoint& p) {
        if (!current()) return;
        QMenu menu(this);
        menu.addAction(tr("Reuse parameters"), this, [this] { reuse_->click(); });
        menu.addAction(tr("Use as source image"), this, [this] { if (const auto* e = current()) emit sourceRequested(QImage(q(host_.gallery().absolutePath(*e).string()))); });
        menu.addAction(tr("Save as…"), this, &GalleryTab::saveSelectedAs);
        menu.addAction(tr("Show in folder"), this, [this] { folder_->click(); });
        menu.addSeparator();
        menu.addAction(tr("Delete"), this, &GalleryTab::deleteSelected);
        menu.exec(list_->viewport()->mapToGlobal(p));
    });
    select(-1);
}

void GalleryTab::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    if (dirty_) reload();
}

const gh::gallery::GalleryEntry* GalleryTab::current() const {
    const int row = list_->currentRow();
    return row >= 0 && row < static_cast<int>(entries_.size()) ? &entries_[row] : nullptr;
}

void GalleryTab::reload() {
    dirty_ = false;
    ++generation_;
    entries_.clear();
    list_->clear();
    loading_ = false;
    total_ = host_.gallery().count(s(search_->text().trimmed()));
    select(-1);
    loadPage();
}

void GalleryTab::loadPage() {
    if (loading_) return;
    loading_ = true;
    more_->setEnabled(false);
    const int generation = generation_;
    const std::string text = s(search_->text().trimmed());
    const int offset = static_cast<int>(entries_.size());
    QPointer<GalleryTab> self(this);
    std::thread([self, text, offset, generation, this] {
        gh::gallery::GalleryQuery query;
        query.text = text;
        query.limit = kPage;
        query.offset = offset;
        auto entries = host_.gallery().list(query);
        std::vector<QImage> thumbs;
        for (const auto& e : entries) {
            QImage image(q(host_.gallery().absolutePath(e).string()));
            thumbs.push_back(image.isNull() ? QImage() : image.scaled(kThumb, kThumb, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
        if (!self) return;
        QMetaObject::invokeMethod(self.data(), [self, entries = std::move(entries), thumbs = std::move(thumbs), generation] {
            if (self) self->addPage(entries, thumbs, generation);
        }, Qt::QueuedConnection);
    }).detach();
}

void GalleryTab::addPage(const std::vector<gh::gallery::GalleryEntry>& entries, const std::vector<QImage>& thumbnails, int generation) {
    if (generation != generation_) return;   // a newer search replaced this one
    loading_ = false;
    for (size_t i = 0; i < entries.size(); ++i) {
        entries_.push_back(entries[i]);
        QString prompt = q(entries[i].prompt);
        auto* item = new QListWidgetItem(QIcon(QPixmap::fromImage(thumbnails[i])), prompt.left(24), list_);
        item->setToolTip(QStringLiteral("%1\n%2 · %3×%4 · seed %5").arg(prompt, q(entries[i].model)).arg(entries[i].width).arg(entries[i].height).arg(entries[i].seed));
    }
    count_->setText(tr("%1 of %2 image(s)").arg(entries_.size()).arg(total_));
    more_->setVisible(entries_.size() < total_);
    more_->setEnabled(true);
    if (list_->currentRow() < 0 && list_->count() > 0 && entries.size() == entries_.size()) list_->setCurrentRow(0);
}

void GalleryTab::select(int row) {
    const bool has = row >= 0 && row < static_cast<int>(entries_.size());
    for (auto* b : {reuse_, save_, folder_, remove_}) b->setEnabled(has);
    if (!has) {
        preview_->setImage({});
        details_->setPlainText(entries_.empty() && total_ == 0 ? tr("Nothing here yet: generate something, or change the search.") : QString());
        return;
    }
    const auto& e = entries_[row];
    preview_->setImage(QImage(q(host_.gallery().absolutePath(e).string())));
    QString text = tr("%1\n\nModel: %2 (%3)\nSize: %4×%5   Seed: %6\nRuntime: %7\nCreated: %8\n").arg(q(e.prompt), q(e.model), q(e.family)).arg(e.width).arg(e.height).arg(e.seed).arg(q(e.runtime), q(e.created));
    const auto& req = e.parameters.value("request", nlohmann::json::object());
    for (const char* key : {"negativePrompt", "steps", "cfgScale", "sampler", "scheduler", "clipSkip"}) {
        const bool empty = !req.contains(key) || (req[key].is_string() && req[key].get<std::string>().empty()) || (req[key].is_number() && req[key].get<double>() < 0);
        if (!empty) text += QStringLiteral("%1: %2\n").arg(QString::fromLatin1(key), q(req[key].is_string() ? req[key].get<std::string>() : req[key].dump()));
    }
    details_->setPlainText(text);
}

void GalleryTab::deleteSelected() {
    const auto selected = list_->selectedItems();
    if (selected.isEmpty()) return;
    if (QMessageBox::question(this, tr("Delete images"), tr("Delete %n image(s) and their settings files? This cannot be undone.", "", selected.size())) != QMessageBox::Yes) return;
    for (auto* item : selected) {
        const int row = list_->row(item);
        if (row >= 0 && row < static_cast<int>(entries_.size())) host_.gallery().remove(entries_[row].id);
    }
    reload();
}

void GalleryTab::saveSelectedAs() {
    const auto* e = current();
    if (!e) return;
    const QString source = q(host_.gallery().absolutePath(*e).string());
    const QString suffix = QFileInfo(source).suffix();
    const QString target = QFileDialog::getSaveFileName(this, tr("Save image"), QStringLiteral("graphicshost-%1.%2").arg(e->seed).arg(suffix), tr("Images (*.%1)").arg(suffix));
    if (target.isEmpty()) return;
    QFile::remove(target);
    QFile::copy(source, target);   // the original file, with its embedded parameters
}

void GalleryTab::exportSelected() {
    std::vector<gh::gallery::GalleryEntry> chosen;
    for (auto* item : list_->selectedItems()) {
        const int row = list_->row(item);
        if (row >= 0 && row < static_cast<int>(entries_.size())) chosen.push_back(entries_[row]);
    }
    if (chosen.size() <= 1) {   // nothing (or one) selected: everything that matches the search
        gh::gallery::GalleryQuery all;
        all.text = s(search_->text().trimmed());
        all.limit = 100000;
        chosen = host_.gallery().list(all);
        if (QMessageBox::question(this, tr("Export"), tr("Export all %n image(s) that match?", "", static_cast<int>(chosen.size()))) != QMessageBox::Yes) return;
    }
    const QString dir = QFileDialog::getExistingDirectory(this, tr("Export to folder"));
    if (dir.isEmpty()) return;
    int copied = 0;
    for (const auto& e : chosen) {
        const QString source = q(host_.gallery().absolutePath(e).string());
        const QString target = QDir(dir).filePath(QFileInfo(source).fileName());
        if (QFile::exists(target)) QFile::remove(target);
        if (QFile::copy(source, target)) ++copied;
    }
    QMessageBox::information(this, tr("Export"), tr("Copied %1 of %2 image(s) to %3.").arg(copied).arg(chosen.size()).arg(QDir::toNativeSeparators(dir)));
}

}  // namespace gh::ui
