#include "DiscoverTab.h"

#include "UiUtil.h"
#include "host/AppHost.h"
#include "models/FitHints.h"

#include <QComboBox>
#include <QDesktopServices>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPointer>
#include <QProgressBar>
#include <QPushButton>
#include <QSplitter>
#include <QTableWidget>
#include <QTabWidget>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <filesystem>
#include <set>
#include <thread>

namespace gh::ui {

namespace fs = std::filesystem;
using gh::models::DownloadState;

namespace {

QTableWidget* makeTable(const QStringList& headers, QWidget* parent, int stretchColumn = 0) {
    auto* t = new QTableWidget(0, headers.size(), parent);
    t->setHorizontalHeaderLabels(headers);
    t->setSelectionBehavior(QAbstractItemView::SelectRows);
    t->setSelectionMode(QAbstractItemView::SingleSelection);
    t->setEditTriggers(QAbstractItemView::NoEditTriggers);
    t->verticalHeader()->setVisible(false);
    t->horizontalHeader()->setSectionResizeMode(stretchColumn, QHeaderView::Stretch);
    return t;
}

QTableWidgetItem* cell(const QString& text, Qt::Alignment align = Qt::AlignLeft | Qt::AlignVCenter) {
    auto* item = new QTableWidgetItem(text);
    item->setTextAlignment(align);
    return item;
}

QString fitText(const gh::models::FitHint& h) {
    return q(h.label);
}

gh::models::HardwareBudget budget(gh::host::AppHost& host) {
    const auto* runtime = host.selectedRuntime();
    return gh::models::budgetFrom(host.gpu(), runtime && runtime->backend == "vulkan");
}

}  // namespace

DiscoverTab::DiscoverTab(gh::host::AppHost& host, QWidget* parent) : QWidget(parent), host_(host) {
    auto* layout = new QVBoxLayout(this);
    auto* split = new QSplitter(Qt::Vertical, this);
    layout->addWidget(split);

    auto* tabs = new QTabWidget(split);
    auto* catalogPage = new QWidget(tabs);
    auto* hubPage = new QWidget(tabs);
    tabs->addTab(catalogPage, tr("Catalog"));
    tabs->addTab(hubPage, tr("Search Hugging Face"));
    buildCatalogPage(catalogPage);
    buildHubPage(hubPage);

    auto* downloads = new QWidget(split);
    buildDownloads(downloads);
    split->addWidget(tabs);
    split->addWidget(downloads);
    split->setStretchFactor(0, 3);
    split->setStretchFactor(1, 1);

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, [this] { refreshDownloads(); });
    timer_->start(500);
    refreshCatalog();
}

void DiscoverTab::buildCatalogPage(QWidget* page) {
    auto* layout = new QVBoxLayout(page);
    auto* row = new QHBoxLayout();
    filter_ = new QLineEdit(page);
    filter_->setPlaceholderText(tr("Filter by name, tag or note"));
    family_ = new QComboBox(page);
    family_->addItem(tr("Any family"), QString());
    family_->addItem(QStringLiteral("SD 1.x"), QStringLiteral("SD 1.x"));
    family_->addItem(QStringLiteral("SDXL"), QStringLiteral("SDXL"));
    kind_ = new QComboBox(page);
    kind_->addItem(tr("Any kind"), QString());
    for (const char* k : {"checkpoint", "vae", "lora"}) kind_->addItem(QString::fromLatin1(k), QString::fromLatin1(k));
    row->addWidget(filter_, 1);
    row->addWidget(family_);
    row->addWidget(kind_);
    layout->addLayout(row);

    catalog_ = makeTable({tr("Name"), tr("Kind"), tr("Family"), tr("Size"), tr("On this computer"), tr("State"), tr("License")}, page);
    layout->addWidget(catalog_, 1);
    auto* buttons = new QHBoxLayout();
    getCatalog_ = new QPushButton(tr("Get…"), page);
    getCatalog_->setToolTip(tr("Download the selected entry (a VAE it needs comes along)"));
    buttons->addWidget(getCatalog_);
    buttons->addStretch(1);
    layout->addLayout(buttons);

    connect(filter_, &QLineEdit::textChanged, this, [this] { refreshCatalog(); });
    connect(family_, &QComboBox::currentIndexChanged, this, [this] { refreshCatalog(); });
    connect(kind_, &QComboBox::currentIndexChanged, this, [this] { refreshCatalog(); });
    connect(getCatalog_, &QPushButton::clicked, this, [this] {
        const auto rows = catalog_->selectionModel()->selectedRows();
        if (rows.isEmpty()) return;
        const QString key = catalog_->item(rows.first().row(), 0)->data(Qt::UserRole).toString();
        const QString name = catalog_->item(rows.first().row(), 0)->text();
        const std::string family = s(catalog_->item(rows.first().row(), 2)->text());
        getWith(name, [this, key] { return host_.modelService().planCatalog(s(key), true); }, q(family));
    });
    connect(catalog_, &QTableWidget::cellDoubleClicked, this, [this] { getCatalog_->click(); });
}

void DiscoverTab::buildHubPage(QWidget* page) {
    auto* layout = new QVBoxLayout(page);
    auto* row = new QHBoxLayout();
    query_ = new QLineEdit(page);
    query_->setPlaceholderText(tr("Search text-to-image models on Hugging Face, e.g. \"dreamshaper\""));
    auto* search = new QPushButton(tr("Search"), page);
    row->addWidget(query_, 1);
    row->addWidget(search);
    layout->addLayout(row);

    auto* split = new QSplitter(Qt::Horizontal, page);
    repos_ = makeTable({tr("Repository"), tr("Downloads"), tr("Likes"), tr("License"), tr("Access")}, split);
    files_ = makeTable({tr("File"), tr("Kind"), tr("Size"), tr("On this computer")}, split);
    split->addWidget(repos_);
    split->addWidget(files_);
    split->setStretchFactor(0, 1);
    split->setStretchFactor(1, 1);
    layout->addWidget(split, 1);

    auto* buttons = new QHBoxLayout();
    getFile_ = new QPushButton(tr("Get selected file…"), page);
    hubStatus_ = new QLabel(tr("Select a repository to see its files. Gated repositories need a Hugging Face token (graphicshost login)."), page);
    hubStatus_->setWordWrap(true);
    buttons->addWidget(getFile_);
    buttons->addWidget(hubStatus_, 1);
    layout->addLayout(buttons);

    connect(search, &QPushButton::clicked, this, &DiscoverTab::searchHub);
    connect(query_, &QLineEdit::returnPressed, this, &DiscoverTab::searchHub);
    connect(repos_, &QTableWidget::itemSelectionChanged, this, &DiscoverTab::showRepoFiles);
    connect(getFile_, &QPushButton::clicked, this, [this] {
        const auto rows = files_->selectionModel()->selectedRows();
        if (rows.isEmpty() || repoShown_.isEmpty()) return;
        const QString file = files_->item(rows.first().row(), 0)->text();
        const QString repo = repoShown_;
        const QString family = files_->item(rows.first().row(), 2)->data(Qt::UserRole).toULongLong() > 5500ull * 1024 * 1024 ? QStringLiteral("SDXL") : QStringLiteral("SD 1.x");
        getWith(file, [this, repo, file] { return host_.modelService().planRepoFile(s(repo), s(file)); }, family);
    });
}

void DiscoverTab::buildDownloads(QWidget* panel) {
    auto* layout = new QVBoxLayout(panel);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(new QLabel(tr("Downloads"), panel));
    downloads_ = makeTable({tr("File"), tr("State"), tr("Progress"), tr("Speed"), tr("Detail")}, panel);
    downloads_->setColumnWidth(2, 220);
    layout->addWidget(downloads_, 1);
    auto* buttons = new QHBoxLayout();
    for (const auto& [label, action] : {std::pair{tr("Pause"), "pause"}, std::pair{tr("Resume"), "resume"}, std::pair{tr("Cancel"), "cancel"}}) {
        auto* b = new QPushButton(label, panel);
        buttons->addWidget(b);
        connect(b, &QPushButton::clicked, this, [this, action] { downloadAction(action); });
    }
    downloadStatus_ = new QLabel(panel);
    buttons->addWidget(downloadStatus_, 1);
    layout->addLayout(buttons);
}

void DiscoverTab::refreshCatalog() {
    gh::models::CatalogFilter f;
    f.text = s(filter_->text());
    f.family = s(family_->currentData().toString());
    f.kind = s(kind_->currentData().toString());
    const auto found = host_.catalog().search(f);
    const auto hardware = budget(host_);
    catalog_->setRowCount(0);
    for (const auto* e : found) {
        const int row = catalog_->rowCount();
        catalog_->insertRow(row);
        std::error_code ec;
        const fs::path local = host_.modelsDir() / e->relativePath();
        const bool installed = fs::is_regular_file(local, ec) && fs::file_size(local, ec) == e->sizeBytes;
        const auto hint = gh::models::fitHintFor(e->sizeBytes, e->family, e->kind, hardware);
        QString state = installed ? tr("installed") : QString();
        if (e->gated) state += state.isEmpty() ? tr("needs token") : tr(", needs token");
        if (e->recommended && !installed) state += state.isEmpty() ? tr("recommended") : tr(", recommended");
        auto* name = cell(q(e->name));
        name->setData(Qt::UserRole, q(e->key));
        name->setToolTip(q(e->notes));
        catalog_->setItem(row, 0, name);
        catalog_->setItem(row, 1, cell(q(e->kind)));
        catalog_->setItem(row, 2, cell(q(e->family)));
        catalog_->setItem(row, 3, cell(humanSize(e->sizeBytes), Qt::AlignRight | Qt::AlignVCenter));
        auto* fit = cell(fitText(hint));
        fit->setToolTip(q(hint.detail));
        catalog_->setItem(row, 4, fit);
        catalog_->setItem(row, 5, cell(state));
        catalog_->setItem(row, 6, cell(q(e->license)));
    }
    catalog_->resizeColumnsToContents();
    catalog_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
}

void DiscoverTab::searchHub() {
    const QString text = query_->text().trimmed();
    if (text.isEmpty() || busy_) return;
    busy_ = true;
    hubStatus_->setText(tr("Searching…"));
    QPointer<DiscoverTab> self(this);
    std::thread([self, text] {
        std::vector<gh::models::HfModel> found;
        QString error;
        try {
            found = gh::models::makeConfiguredHub().search(s(text), 30);
        } catch (const std::exception& e) {
            error = q(e.what());
        }
        if (!self) return;
        QMetaObject::invokeMethod(self.data(), [self, found, error] {
            if (!self) return;
            self->busy_ = false;
            self->repos_->setRowCount(0);
            self->files_->setRowCount(0);
            self->repoShown_.clear();
            for (const auto& m : found) {
                const int row = self->repos_->rowCount();
                self->repos_->insertRow(row);
                self->repos_->setItem(row, 0, cell(q(m.id)));
                self->repos_->setItem(row, 1, cell(QString::number(m.downloads), Qt::AlignRight | Qt::AlignVCenter));
                self->repos_->setItem(row, 2, cell(QString::number(m.likes), Qt::AlignRight | Qt::AlignVCenter));
                self->repos_->setItem(row, 3, cell(q(m.license)));
                self->repos_->setItem(row, 4, cell(m.isGated() ? tr("gated") : tr("open")));
            }
            self->hubStatus_->setText(!error.isEmpty() ? tr("Search failed: %1").arg(error)
                                      : found.empty() ? tr("Nothing found.") : tr("%n repository(ies). Select one to see its files.", "", static_cast<int>(found.size())));
        }, Qt::QueuedConnection);
    }).detach();
}

void DiscoverTab::showRepoFiles() {
    const auto rows = repos_->selectionModel()->selectedRows();
    if (rows.isEmpty() || busy_) return;
    const QString repo = repos_->item(rows.first().row(), 0)->text();
    busy_ = true;
    hubStatus_->setText(tr("Reading the files of %1…").arg(repo));
    QPointer<DiscoverTab> self(this);
    std::thread([self, repo, this] {
        gh::models::HfModelInfo info;
        QString error;
        try {
            info = gh::models::makeConfiguredHub().info(s(repo));
        } catch (const std::exception& e) {
            error = q(e.what());
        }
        const auto usable = error.isEmpty() ? gh::models::usableFiles(info) : std::vector<gh::models::HfFile>{};
        const auto hardware = budget(host_);
        if (!self) return;
        QMetaObject::invokeMethod(self.data(), [self, repo, info, usable, error, hardware] {
            if (!self) return;
            self->busy_ = false;
            self->files_->setRowCount(0);
            self->repoShown_ = error.isEmpty() ? repo : QString();
            for (const auto& f : usable) {
                const int row = self->files_->rowCount();
                self->files_->insertRow(row);
                const std::string kind = gh::models::guessFileKind(f.path);
                const auto hint = gh::models::fitHintFor(f.sizeBytes, f.sizeBytes > 5500ull * 1024 * 1024 ? "SDXL" : "SD 1.x", kind, hardware);
                self->files_->setItem(row, 0, cell(q(f.path)));
                self->files_->setItem(row, 1, cell(q(kind)));
                auto* size = cell(humanSize(f.sizeBytes), Qt::AlignRight | Qt::AlignVCenter);
                size->setData(Qt::UserRole, static_cast<qulonglong>(f.sizeBytes));
                self->files_->setItem(row, 2, size);
                auto* fit = cell(fitText(hint));
                fit->setToolTip(q(hint.detail));
                self->files_->setItem(row, 3, fit);
            }
            if (!error.isEmpty()) self->hubStatus_->setText(tr("Could not read %1: %2").arg(repo, error));
            else if (usable.empty()) self->hubStatus_->setText(tr("%1 has no checkpoint files GraphicsHost can use (a diffusers-format repository, perhaps).").arg(repo));
            else self->hubStatus_->setText(tr("%1: %n file(s)%2", "", static_cast<int>(usable.size())).arg(repo, info.model.isGated() ? tr("  (gated: needs a token with access)") : QString()));
        }, Qt::QueuedConnection);
    }).detach();
}

void DiscoverTab::getWith(const QString& title, std::function<gh::models::DownloadPlan()> planner, const QString& fitFamily) {
    if (busy_) return;
    busy_ = true;
    downloadStatus_->setText(tr("Preparing %1…").arg(title));
    QPointer<DiscoverTab> self(this);
    std::thread([self, title, planner, fitFamily] {
        gh::models::DownloadPlan plan;
        QString error;
        try {
            plan = planner();
        } catch (const std::exception& e) {
            error = q(e.what());
        }
        if (!self) return;
        QMetaObject::invokeMethod(self.data(), [self, title, plan, error, fitFamily] {
            if (!self) return;
            self->busy_ = false;
            if (!error.isEmpty()) {
                self->downloadStatus_->setText(tr("Cannot get %1: %2").arg(title, error));
                return;
            }
            self->confirmAndStart(title, plan, fitFamily);
        }, Qt::QueuedConnection);
    }).detach();
}

void DiscoverTab::confirmAndStart(const QString& title, const gh::models::DownloadPlan& plan, const QString& fitFamily) {
    QString text;
    for (const auto& item : plan.items) {
        text += QStringLiteral("• %1  (%2, %3)%4\n").arg(q(item.file), q(item.role), humanSize(item.sizeBytes), item.present ? tr("  already here") : QString());
    }
    for (const auto& note : plan.notes) text += QStringLiteral("\n%1").arg(q(note));
    if (!plan.items.empty() && plan.items[0].role == "checkpoint") {
        const auto hint = gh::models::fitHintFor(plan.items[0].sizeBytes, s(fitFamily), "checkpoint", budget(host_));
        text += tr("\n\nOn this computer: %1. %2").arg(q(hint.label), q(hint.detail));
    }
    text += tr("\n\nTo download: %1").arg(humanSize(plan.bytesToDownload));
    if (plan.bytesToDownload == 0) {
        QMessageBox::information(this, title, tr("Everything is already here.\n\n%1").arg(text));
        host_.modelService().start(plan);   // (re)writes the source records
        emit modelsChanged();
        return;
    }
    if (QMessageBox::question(this, tr("Get %1").arg(title), text, QMessageBox::Yes | QMessageBox::No) != QMessageBox::Yes) return;
    host_.modelService().start(plan);
    downloadStatus_->setText(tr("Downloading %1…").arg(title));
    refreshDownloads();
}

void DiscoverTab::refreshDownloads() {
    static std::set<uint64_t> seenCompleted;
    const auto list = host_.downloads().list();
    const uint64_t keep = selectedDownload();
    downloads_->setRowCount(0);
    bool completedNow = false;
    for (const auto& d : list) {
        const int row = downloads_->rowCount();
        downloads_->insertRow(row);
        auto* name = cell(q(d->request().label.empty() ? d->request().destination.filename().string() : d->request().label));
        name->setData(Qt::UserRole, static_cast<qulonglong>(d->id()));
        downloads_->setItem(row, 0, name);
        downloads_->setItem(row, 1, cell(q(gh::models::toString(d->state()))));
        auto* bar = new QProgressBar(downloads_);
        const uint64_t total = d->bytesTotal();
        bar->setRange(0, 1000);
        bar->setValue(total > 0 ? static_cast<int>(1000.0 * static_cast<double>(d->bytesDone()) / static_cast<double>(total)) : (d->state() == DownloadState::Completed ? 1000 : 0));
        bar->setFormat(QStringLiteral("%1 / %2").arg(humanSize(d->bytesDone()), total ? humanSize(total) : QStringLiteral("?")));
        downloads_->setCellWidget(row, 2, bar);
        downloads_->setItem(row, 3, cell(d->state() == DownloadState::Running ? QStringLiteral("%1 MB/s").arg(d->bytesPerSecond() / 1e6, 0, 'f', 1) : QString()));
        downloads_->setItem(row, 4, cell(q(d->message())));
        if (static_cast<uint64_t>(row) < 1000 && d->id() == keep) downloads_->selectRow(row);
        if (d->state() == DownloadState::Completed && seenCompleted.insert(d->id()).second) completedNow = true;
    }
    if (completedNow) {
        refreshCatalog();
        emit modelsChanged();
    }
}

uint64_t DiscoverTab::selectedDownload() const {
    const auto rows = downloads_->selectionModel()->selectedRows();
    return rows.isEmpty() || !downloads_->item(rows.first().row(), 0) ? 0 : downloads_->item(rows.first().row(), 0)->data(Qt::UserRole).toULongLong();
}

void DiscoverTab::downloadAction(const char* action) {
    const uint64_t id = selectedDownload();
    if (id == 0) return;
    const std::string a = action;
    if (a == "pause") host_.downloads().pause(id);
    else if (a == "resume") host_.downloads().resume(id);
    else host_.downloads().cancel(id);
    refreshDownloads();
}

}  // namespace gh::ui
