#include "ModelsTab.h"

#include "UiUtil.h"
#include "host/AppHost.h"

#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMenu>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProcess>
#include <QPushButton>
#include <QTableWidget>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <filesystem>
#include <thread>

namespace gh::ui {

using Json = nlohmann::json;
namespace fs = std::filesystem;

namespace {
enum Column { File, Kind, Family, Size, Vae, State, ColumnCount };
}

ModelsTab::ModelsTab(gh::host::AppHost& host, QWidget* parent) : QWidget(parent), host_(host) {
    auto* layout = new QVBoxLayout(this);
    summary_ = new QLabel(this);
    layout->addWidget(summary_);

    table_ = new QTableWidget(0, ColumnCount, this);
    table_->setHorizontalHeaderLabels({tr("File"), tr("Kind"), tr("Family"), tr("Size"), tr("VAE"), tr("State")});
    table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    table_->setSelectionMode(QAbstractItemView::SingleSelection);
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table_->verticalHeader()->setVisible(false);
    table_->setSortingEnabled(true);
    table_->horizontalHeader()->setStretchLastSection(true);
    table_->horizontalHeader()->setSectionResizeMode(File, QHeaderView::Stretch);
    table_->setContextMenuPolicy(Qt::CustomContextMenu);
    layout->addWidget(table_, 1);

    auto* buttons = new QHBoxLayout();
    load_ = new QPushButton(tr("Load"), this);
    unload_ = new QPushButton(tr("Unload"), this);
    defaults_ = new QPushButton(tr("Defaults…"), this);
    remove_ = new QPushButton(tr("Delete…"), this);
    auto* folder = new QPushButton(tr("Open folder"), this);
    auto* reload = new QPushButton(tr("Rescan"), this);
    load_->setToolTip(tr("Put the model in memory now (generating loads it on demand anyway)"));
    defaults_->setToolTip(tr("Parameters that are used with this model unless you set them yourself"));
    for (auto* b : {load_, unload_, defaults_, remove_}) buttons->addWidget(b);
    buttons->addStretch(1);
    buttons->addWidget(folder);
    buttons->addWidget(reload);
    layout->addLayout(buttons);
    status_ = new QLabel(this);
    status_->setWordWrap(true);
    layout->addWidget(status_);

    connect(load_, &QPushButton::clicked, this, &ModelsTab::loadSelected);
    connect(unload_, &QPushButton::clicked, this, &ModelsTab::unload);
    connect(defaults_, &QPushButton::clicked, this, &ModelsTab::editDefaults);
    connect(remove_, &QPushButton::clicked, this, &ModelsTab::deleteSelected);
    connect(reload, &QPushButton::clicked, this, &ModelsTab::refresh);
    connect(folder, &QPushButton::clicked, this, [this] {
        std::error_code ec;
        fs::create_directories(host_.modelsDir(), ec);
        QDesktopServices::openUrl(QUrl::fromLocalFile(q(host_.modelsDir().string())));
    });
    connect(table_, &QTableWidget::itemSelectionChanged, this, &ModelsTab::updateButtons);
    connect(table_, &QWidget::customContextMenuRequested, this, [this](const QPoint& p) { showMenu(table_->viewport()->mapToGlobal(p)); });
    connect(table_, &QTableWidget::cellDoubleClicked, this, [this] { loadSelected(); });

    // files appearing in the folder show up without a manual rescan
    QPointer<ModelsTab> self(this);
    host_.library().startWatching([self] {
        if (self) QMetaObject::invokeMethod(self.data(), [self] { if (self) self->refresh(); }, Qt::QueuedConnection);
    });
    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, [this] { if (isVisible() && !loading_) refresh(); });
    timer_->start(3000);
    refresh();
}

ModelsTab::~ModelsTab() {
    host_.library().stopWatching();
}

void ModelsTab::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    refresh();
}

QString ModelsTab::selectedFile() const {
    const auto rows = table_->selectionModel()->selectedRows();
    return rows.isEmpty() ? QString() : table_->item(rows.first().row(), File)->data(Qt::UserRole).toString();
}

void ModelsTab::refresh() {
    const QString keep = selectedFile();
    const auto loaded = host_.models().loaded();
    const fs::path loadedPath = loaded ? fs::path(loaded->options.modelPath) : fs::path();
    const auto cards = host_.library().scan();

    table_->setSortingEnabled(false);
    table_->setRowCount(0);
    uint64_t total = 0;
    for (const auto& c : cards) {
        const int row = table_->rowCount();
        table_->insertRow(row);
        QString state;
        if (loaded && loadedPath == c.path) state = tr("loaded");
        else if (!c.supported) state = c.note.empty() ? tr("not supported") : q(c.note);
        else if (!c.problem.empty()) state = q(c.problem);
        if (host_.modelDefaults().get(c.file)) state += state.isEmpty() ? tr("has defaults") : tr(", has defaults");
        auto* file = new QTableWidgetItem(q(c.file));
        file->setData(Qt::UserRole, q(c.file));
        file->setToolTip(q(c.path.string()));
        auto* size = new QTableWidgetItem(humanSize(c.sizeBytes));
        size->setData(Qt::UserRole, static_cast<qulonglong>(c.sizeBytes));
        size->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
        table_->setItem(row, File, file);
        table_->setItem(row, Kind, new QTableWidgetItem(q(gh::models::toString(c.kind))));
        table_->setItem(row, Family, new QTableWidgetItem(c.kind == gh::models::Kind::Checkpoint ? q(c.familyName) : QString()));
        table_->setItem(row, Size, size);
        table_->setItem(row, Vae, new QTableWidgetItem(q(c.vae)));
        auto* stateItem = new QTableWidgetItem(state);
        if (!c.supported) stateItem->setForeground(Qt::darkYellow);
        table_->setItem(row, State, stateItem);
        total += c.sizeBytes;
        if (q(c.file) == keep) table_->selectRow(row);
    }
    table_->setSortingEnabled(true);
    summary_->setText(tr("%1 file(s), %2, in %3").arg(cards.size()).arg(humanSize(total), q(host_.modelsDir().string())));
    if (cards.empty()) status_->setText(tr("Nothing here yet. Get models in Discover, or copy .safetensors / .ckpt / .gguf files into the folder."));
    updateButtons();
}

void ModelsTab::updateButtons() {
    const QString file = selectedFile();
    bool checkpoint = false;
    if (!file.isEmpty()) {
        const auto r = host_.library().resolve(s(file));
        checkpoint = r.card && r.card->kind == gh::models::Kind::Checkpoint && r.card->supported;
    }
    load_->setEnabled(checkpoint && !loading_);
    unload_->setEnabled(host_.models().loaded().has_value() && !loading_);
    defaults_->setEnabled(checkpoint);
    remove_->setEnabled(!file.isEmpty() && !loading_);
}

void ModelsTab::loadSelected() {
    const QString file = selectedFile();
    if (file.isEmpty() || loading_) return;
    loading_ = true;
    updateButtons();
    status_->setText(tr("Loading %1…").arg(file));
    QPointer<ModelsTab> self(this);
    std::thread([self, file, this] {
        QString error;
        QString notes;
        try {
            gh::host::GenerateCall call;
            call.model = s(file);
            const auto submission = host_.generation().loadModel(call);
            for (const auto& n : submission.notes) notes += q(n) + QStringLiteral(" ");
        } catch (const std::exception& e) {
            error = q(e.what());
        }
        if (!self) return;
        QMetaObject::invokeMethod(self.data(), [self, file, error, notes] {
            if (!self) return;
            self->loading_ = false;
            self->status_->setText(error.isEmpty() ? tr("%1 is loaded. %2").arg(file, notes.trimmed()) : tr("Could not load %1: %2").arg(file, error));
            self->refresh();
            emit self->modelsChanged();
        }, Qt::QueuedConnection);
    }).detach();
}

void ModelsTab::unload() {
    if (host_.models().queue().active() > 0) {
        status_->setText(tr("A generation is running or queued: cancel it first."));
        return;
    }
    host_.models().unload();
    status_->setText(tr("Model unloaded."));
    refresh();
    emit modelsChanged();
}

void ModelsTab::editDefaults() {
    const QString file = selectedFile();
    if (file.isEmpty()) return;
    const auto current = host_.modelDefaults().get(s(file));
    const Json doc = {{"request", current ? current->request : Json::object()}, {"load", current ? current->load : Json::object()}};

    QDialog dialog(this);
    dialog.setWindowTitle(tr("Defaults for %1").arg(file));
    auto* layout = new QVBoxLayout(&dialog);
    layout->addWidget(new QLabel(tr("Only the parameters listed here are changed; everything else keeps the model's own default.\n\"request\": steps, cfgScale, sampler, scheduler, width, height, clipSkip, ...   \"load\": device, flashAttention, offloadParamsToCpu, ttlMinutes, ..."), &dialog));
    auto* text = new QPlainTextEdit(q(doc.dump(2)), &dialog);
    text->setMinimumSize(480, 280);
    layout->addWidget(text);
    auto* error = new QLabel(&dialog);
    error->setStyleSheet(QStringLiteral("color: #b00020"));
    layout->addWidget(error);
    auto* box = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, &dialog);
    auto* clear = box->addButton(tr("Remove defaults"), QDialogButtonBox::DestructiveRole);
    layout->addWidget(box);
    connect(box, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    connect(clear, &QPushButton::clicked, &dialog, [&] {
        host_.modelDefaults().remove(s(file));
        dialog.accept();
    });
    connect(box, &QDialogButtonBox::accepted, &dialog, [&] {
        const Json parsed = Json::parse(s(text->toPlainText()), nullptr, false);
        if (!parsed.is_object()) {
            error->setText(tr("This is not a valid JSON object."));
            return;
        }
        try {
            gh::presets::ModelDefaultsStore::Defaults d;
            if (parsed.contains("request")) d.request = parsed["request"];
            if (parsed.contains("load")) d.load = parsed["load"];
            host_.modelDefaults().set(s(file), d);
            dialog.accept();
        } catch (const std::exception& e) {
            error->setText(q(e.what()));
        }
    });
    if (dialog.exec() == QDialog::Accepted) {
        refresh();
        emit modelsChanged();
    }
}

void ModelsTab::deleteSelected() {
    const QString file = selectedFile();
    if (file.isEmpty()) return;
    const auto r = host_.library().resolve(s(file));
    if (!r.card) return;
    const auto loaded = host_.models().loaded();
    if (loaded && fs::path(loaded->options.modelPath) == r.card->path) {
        status_->setText(tr("Unload the model first."));
        return;
    }
    if (QMessageBox::question(this, tr("Delete model"), tr("Delete %1 (%2) from disk? This cannot be undone.").arg(file, humanSize(r.card->sizeBytes))) != QMessageBox::Yes) return;
    std::error_code ec;
    fs::remove(r.card->path, ec);
    fs::remove(gh::models::sourceSidecarFor(r.card->path), ec);   // the download record next to it
    status_->setText(ec ? tr("Could not delete: %1").arg(q(ec.message())) : tr("Deleted %1.").arg(file));
    refresh();
    emit modelsChanged();
}

void ModelsTab::showMenu(const QPoint& where) {
    if (selectedFile().isEmpty()) return;
    QMenu menu(this);
    menu.addAction(tr("Load"), this, &ModelsTab::loadSelected)->setEnabled(load_->isEnabled());
    menu.addAction(tr("Defaults…"), this, &ModelsTab::editDefaults)->setEnabled(defaults_->isEnabled());
    menu.addAction(tr("Show in folder"), this, [this] {
        const auto r = host_.library().resolve(s(selectedFile()));
        if (r.card) QProcess::startDetached(QStringLiteral("explorer.exe"), {QStringLiteral("/select,"), QDir::toNativeSeparators(q(r.card->path.string()))});
    });
    menu.addSeparator();
    menu.addAction(tr("Delete…"), this, &ModelsTab::deleteSelected);
    menu.exec(where);
}

}  // namespace gh::ui
