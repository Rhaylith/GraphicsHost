#include "GenerateTab.h"

#include "UiUtil.h"
#include "engine/ImageCodec.h"
#include "host/AppHost.h"

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QComboBox>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProgressBar>
#include <QPointer>
#include <QPushButton>
#include <QRandomGenerator>
#include <QScrollArea>
#include <QSpinBox>
#include <QSplitter>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <climits>
#include <QDir>
#include <QFileInfo>
#include <thread>

namespace gh::ui {

using Json = nlohmann::json;

// A label that shows a picture as large as fits, keeping the aspect ratio.
class ImageView : public QLabel {
public:
    explicit ImageView(QWidget* parent = nullptr) : QLabel(parent) {
        setAlignment(Qt::AlignCenter);
        setMinimumSize(256, 256);
        setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
        setFrameShape(QFrame::StyledPanel);
        setContextMenuPolicy(Qt::CustomContextMenu);
    }
    void setImage(const QImage& image) {
        image_ = image;
        refit();
    }
    bool hasImage() const { return !image_.isNull(); }
    const QImage& image() const { return image_; }

protected:
    void resizeEvent(QResizeEvent* event) override {
        QLabel::resizeEvent(event);
        refit();
    }

private:
    void refit() {
        if (image_.isNull()) {
            clear();
            return;
        }
        setPixmap(QPixmap::fromImage(image_).scaled(size() - QSize(8, 8), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    QImage image_;
};

namespace {

QSpinBox* autoSpin(int minimum, int maximum, int step, const QString& autoText, QWidget* parent) {
    auto* box = new QSpinBox(parent);
    box->setRange(minimum, maximum);
    box->setSingleStep(step);
    box->setSpecialValueText(autoText);
    box->setValue(minimum);
    return box;
}

QCheckBox* tristate(const QString& text, QWidget* parent) {
    auto* box = new QCheckBox(text, parent);
    box->setTristate(true);
    box->setCheckState(Qt::PartiallyChecked);   // "model default"
    box->setToolTip(QObject::tr("Partly filled: use the model's own default."));
    return box;
}

}  // namespace

GenerateTab::GenerateTab(gh::host::AppHost& host, QWidget* parent) : QWidget(parent), host_(host) {
    buildUi();
    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &GenerateTab::poll);
    timer_->start(150);
    refreshLists();
    applyInterfaceMode();
}

GenerateTab::~GenerateTab() {
    if (job_) job_->cancel();
}

void GenerateTab::buildUi() {
    auto* root = new QHBoxLayout(this);
    auto* split = new QSplitter(Qt::Horizontal, this);
    root->addWidget(split);

    // ---- left: the parameters
    auto* left = new QWidget(split);
    auto* leftLayout = new QVBoxLayout(left);

    auto* form = new QFormLayout();
    model_ = new QComboBox(left);
    model_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    model_->setMinimumContentsLength(24);
    preset_ = new QComboBox(left);
    form->addRow(tr("Model"), model_);
    form->addRow(tr("Preset"), preset_);
    leftLayout->addLayout(form);

    prompt_ = new QPlainTextEdit(left);
    prompt_->setPlaceholderText(tr("Describe the image…"));
    prompt_->setMinimumHeight(90);
    negative_ = new QPlainTextEdit(left);
    negative_->setPlaceholderText(tr("Negative prompt: what to keep out"));
    negative_->setMaximumHeight(60);
    leftLayout->addWidget(prompt_, 3);
    leftLayout->addWidget(negative_, 1);

    auto* basics = new QFormLayout();
    auto* sizeRow = new QHBoxLayout();
    width_ = autoSpin(0, 4096, 64, tr("Auto"), left);
    height_ = autoSpin(0, 4096, 64, tr("Auto"), left);
    sizeRow->addWidget(width_);
    sizeRow->addWidget(new QLabel(QStringLiteral("×"), left));
    sizeRow->addWidget(height_);
    basics->addRow(tr("Size"), sizeRow);
    count_ = new QSpinBox(left);
    count_->setRange(1, 16);
    basics->addRow(tr("Images"), count_);
    auto* seedRow = new QHBoxLayout();
    seed_ = new QSpinBox(left);
    seed_->setRange(-1, INT_MAX);
    seed_->setSpecialValueText(tr("Random"));
    seed_->setValue(-1);
    auto* dice = new QPushButton(tr("New"), left);
    dice->setToolTip(tr("Pick a random seed now"));
    connect(dice, &QPushButton::clicked, this, [this] { seed_->setValue(static_cast<int>(QRandomGenerator::global()->bounded(1, INT_MAX))); });
    seedRow->addWidget(seed_, 1);
    seedRow->addWidget(dice);
    basics->addRow(tr("Seed"), seedRow);
    leftLayout->addLayout(basics);

    // power mode: the sampling parameters
    powerWidgets_ = new QWidget(left);
    auto* power = new QFormLayout(powerWidgets_);
    power->setContentsMargins(0, 0, 0, 0);
    steps_ = autoSpin(0, 150, 1, tr("Auto"), powerWidgets_);
    cfg_ = new QDoubleSpinBox(powerWidgets_);
    cfg_->setRange(-1, 30);
    cfg_->setDecimals(1);
    cfg_->setSingleStep(0.5);
    cfg_->setSpecialValueText(tr("Auto"));
    cfg_->setValue(-1);
    sampler_ = new QComboBox(powerWidgets_);
    scheduler_ = new QComboBox(powerWidgets_);
    sampler_->addItem(tr("Default"), QString());
    for (const auto& name : gh::engine::knownSamplers()) sampler_->addItem(q(name), q(name));
    scheduler_->addItem(tr("Default"), QString());
    for (const auto& name : gh::engine::knownSchedulers()) scheduler_->addItem(q(name), q(name));
    power->addRow(tr("Steps"), steps_);
    power->addRow(tr("CFG scale"), cfg_);
    power->addRow(tr("Sampler"), sampler_);
    power->addRow(tr("Scheduler"), scheduler_);
    advancedRequest_ = powerWidgets_;
    leftLayout->addWidget(powerWidgets_);

    advancedLoad_ = new QWidget(left);
    auto* load = new QFormLayout(advancedLoad_);
    load->setContentsMargins(0, 0, 0, 0);
    clipSkip_ = autoSpin(-1, 12, 1, tr("Default"), advancedLoad_);
    clipSkip_->setMinimum(-1);
    clipSkip_->setValue(-1);
    vaeTiling_ = tristate(tr("VAE tiling (less memory, slower)"), advancedLoad_);
    format_ = new QComboBox(advancedLoad_);
    format_->addItem(tr("Default (PNG)"), QString());
    for (const char* f : {"png", "jpeg", "webp"}) format_->addItem(QString::fromLatin1(f), QString::fromLatin1(f));
    device_ = new QComboBox(advancedLoad_);
    device_->addItem(tr("Automatic"), QString());
    device_->addItem(QStringLiteral("CPU"), QStringLiteral("cpu"));
    for (const auto& d : host_.gpu().devices) {
        if (d.supported) device_->addItem(QStringLiteral("GPU %1: %2").arg(d.index).arg(q(d.name)), QStringLiteral("vulkan%1").arg(d.index));
    }
    flash_ = tristate(tr("Flash attention"), advancedLoad_);
    offload_ = tristate(tr("Keep weights in RAM (low VRAM)"), advancedLoad_);
    load->addRow(tr("CLIP skip"), clipSkip_);
    load->addRow(tr("Output format"), format_);
    load->addRow(tr("Device"), device_);
    load->addRow(QString(), vaeTiling_);
    load->addRow(QString(), flash_);
    load->addRow(QString(), offload_);
    leftLayout->addWidget(advancedLoad_);

    showJson_ = new QPushButton(tr("Show request JSON…"), left);
    connect(showJson_, &QPushButton::clicked, this, &GenerateTab::showRequestJson);
    leftLayout->addWidget(showJson_);
    leftLayout->addStretch(1);

    auto* buttons = new QHBoxLayout();
    generate_ = new QPushButton(tr("Generate"), left);
    generate_->setDefault(true);
    generate_->setMinimumHeight(34);
    cancel_ = new QPushButton(tr("Cancel"), left);
    cancel_->setEnabled(false);
    buttons->addWidget(generate_, 2);
    buttons->addWidget(cancel_, 1);
    leftLayout->addLayout(buttons);
    connect(generate_, &QPushButton::clicked, this, &GenerateTab::onGenerate);
    connect(cancel_, &QPushButton::clicked, this, &GenerateTab::onCancel);

    auto* scroll = new QScrollArea(split);
    scroll->setWidgetResizable(true);
    scroll->setWidget(left);
    scroll->setMinimumWidth(340);
    scroll->setFrameShape(QFrame::NoFrame);

    // ---- right: progress and results
    auto* right = new QWidget(split);
    auto* rightLayout = new QVBoxLayout(right);
    view_ = new ImageView(right);
    connect(view_, &QWidget::customContextMenuRequested, this, [this](const QPoint& p) { resultMenu(view_->mapToGlobal(p)); });
    thumbs_ = new QListWidget(right);
    thumbs_->setViewMode(QListView::IconMode);
    thumbs_->setIconSize(QSize(96, 96));
    thumbs_->setResizeMode(QListView::Adjust);
    thumbs_->setMovement(QListView::Static);
    thumbs_->setFlow(QListView::LeftToRight);
    thumbs_->setWrapping(false);
    thumbs_->setFixedHeight(124);
    thumbs_->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(thumbs_, &QListWidget::currentRowChanged, this, [this](int row) { showResult(row); });
    connect(thumbs_, &QWidget::customContextMenuRequested, this, [this](const QPoint& p) { resultMenu(thumbs_->mapToGlobal(p)); });
    progress_ = new QProgressBar(right);
    progress_->setRange(0, 1);
    progress_->setValue(0);
    progress_->setTextVisible(false);
    status_ = new QLabel(tr("Ready."), right);
    status_->setWordWrap(true);
    rightLayout->addWidget(view_, 1);
    rightLayout->addWidget(thumbs_);
    rightLayout->addWidget(progress_);
    rightLayout->addWidget(status_);

    split->addWidget(scroll);
    split->addWidget(right);
    split->setStretchFactor(0, 0);
    split->setStretchFactor(1, 1);
    split->setSizes({400, 700});

    connect(model_, &QComboBox::currentIndexChanged, this, [this] { updateAutoLabels(); });
}

void GenerateTab::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    refreshLists();
}

void GenerateTab::refreshLists() {
    const QString oldModel = model_->currentData().toString();
    const QString oldPreset = preset_->currentData().toString();
    model_->blockSignals(true);
    model_->clear();
    for (const auto& card : host_.library().ofKind(gh::models::Kind::Checkpoint)) {
        if (!card.supported) continue;
        model_->addItem(QStringLiteral("%1  (%2)").arg(q(card.file), q(card.familyName)), q(card.file));
    }
    if (model_->count() == 0) model_->addItem(tr("No models yet: see My Models or Discover"), QString());
    int at = model_->findData(oldModel);
    if (at < 0) {   // the model that is loaded now is the natural choice
        if (const auto loaded = host_.models().loaded()) at = model_->findData(q(std::filesystem::path(loaded->options.modelPath).filename().string()));
    }
    model_->setCurrentIndex(at >= 0 ? at : 0);
    model_->blockSignals(false);

    preset_->clear();
    preset_->addItem(tr("(none)"), QString());
    for (const auto& name : host_.presets().list()) preset_->addItem(q(name), q(name));
    const int p = preset_->findData(oldPreset);
    preset_->setCurrentIndex(p >= 0 ? p : 0);
    updateAutoLabels();
}

void GenerateTab::updateAutoLabels() {
    gh::engine::FamilyDefaults d;   // SD 1.x defaults when no model is selected
    const std::string file = s(model_->currentData().toString());
    if (!file.empty()) {
        const auto resolved = host_.library().resolve(file);
        if (resolved.card) d = resolved.card->defaults;
    }
    width_->setSpecialValueText(tr("Auto (%1)").arg(d.width));
    height_->setSpecialValueText(tr("Auto (%1)").arg(d.height));
    steps_->setSpecialValueText(tr("Auto (%1)").arg(d.steps));
    cfg_->setSpecialValueText(tr("Auto (%1)").arg(d.cfgScale, 0, 'f', 1));
}

void GenerateTab::applyInterfaceMode() {
    const std::string mode = host_.settings().interfaceMode;
    const bool power = mode == "power" || mode == "developer";
    powerWidgets_->setVisible(power);
    advancedLoad_->setVisible(power);
    showJson_->setVisible(mode == "developer");
}

gh::host::GenerateCall GenerateTab::currentCall() const {
    gh::host::GenerateCall call;
    call.model = s(model_->currentData().toString());
    call.preset = s(preset_->currentData().toString());
    Json& r = call.request;
    r["prompt"] = s(prompt_->toPlainText().trimmed());
    if (!negative_->toPlainText().trimmed().isEmpty()) r["negativePrompt"] = s(negative_->toPlainText().trimmed());
    if (width_->value() > 0) r["width"] = width_->value();
    if (height_->value() > 0) r["height"] = height_->value();
    if (count_->value() > 1) r["batchCount"] = count_->value();
    if (seed_->value() >= 0) r["seed"] = seed_->value();
    if (powerWidgets_->isVisible()) {
        if (steps_->value() > 0) r["steps"] = steps_->value();
        if (cfg_->value() >= 0) r["cfgScale"] = cfg_->value();
        if (!sampler_->currentData().toString().isEmpty()) r["sampler"] = s(sampler_->currentData().toString());
        if (!scheduler_->currentData().toString().isEmpty()) r["scheduler"] = s(scheduler_->currentData().toString());
    }
    Json& l = call.loadOptions;
    if (advancedLoad_->isVisible()) {
        if (clipSkip_->value() >= 0) r["clipSkip"] = clipSkip_->value();
        if (vaeTiling_->checkState() != Qt::PartiallyChecked) r["vaeTiling"] = vaeTiling_->isChecked();
        if (!format_->currentData().toString().isEmpty()) r["format"] = s(format_->currentData().toString());
        if (!device_->currentData().toString().isEmpty()) l["device"] = s(device_->currentData().toString());
        if (flash_->checkState() != Qt::PartiallyChecked) l["flashAttention"] = flash_->isChecked();
        if (offload_->checkState() != Qt::PartiallyChecked) l["offloadParamsToCpu"] = offload_->isChecked();
    }
    return call;
}

void GenerateTab::loadParameters(const Json& request, const QString& model) {
    const int m = model_->findData(model);
    if (m >= 0) model_->setCurrentIndex(m);
    auto number = [&](const char* key, auto apply) {
        if (request.contains(key) && request[key].is_number()) apply(request[key]);
    };
    prompt_->setPlainText(q(request.value("prompt", std::string())));
    negative_->setPlainText(q(request.value("negativePrompt", std::string())));
    number("width", [&](const Json& v) { width_->setValue(v.get<int>()); });
    number("height", [&](const Json& v) { height_->setValue(v.get<int>()); });
    number("steps", [&](const Json& v) { steps_->setValue(v.get<int>()); });
    number("cfgScale", [&](const Json& v) { cfg_->setValue(v.get<double>()); });
    number("seed", [&](const Json& v) { seed_->setValue(static_cast<int>(std::clamp<int64_t>(v.get<int64_t>(), -1, INT_MAX))); });
    number("batchCount", [&](const Json& v) { count_->setValue(v.get<int>()); });
    const int smp = sampler_->findData(q(request.value("sampler", std::string())));
    if (smp >= 0) sampler_->setCurrentIndex(smp);
    const int sch = scheduler_->findData(q(request.value("scheduler", std::string())));
    if (sch >= 0) scheduler_->setCurrentIndex(sch);
}

void GenerateTab::setBusy(bool busy) {
    generate_->setEnabled(!busy);
    cancel_->setEnabled(busy);
    model_->setEnabled(!busy);
}

void GenerateTab::onGenerate() {
    if (job_ || submitting_) return;
    const gh::host::GenerateCall call = currentCall();
    if (call.model.empty()) {
        QMessageBox::information(this, tr("No model"), tr("There is no model to generate with yet. Get one in Discover, or put a checkpoint into the models folder."));
        return;
    }
    if (call.request["prompt"].get<std::string>().empty()) {
        status_->setText(tr("Write a prompt first."));
        prompt_->setFocus();
        return;
    }
    submitting_ = true;
    setBusy(true);
    progress_->setRange(0, 0);   // busy: the model may need loading
    status_->setText(tr("Loading the model and queueing…"));

    // loading a model takes seconds: not on the UI thread
    QPointer<GenerateTab> self(this);
    std::thread([self, call, this] {
        QString error;
        QStringList notes;
        std::shared_ptr<gh::engine::Job> job;
        try {
            auto submission = host_.generation().submit(call);
            job = submission.job;
            for (const auto& n : submission.notes) notes << q(n);
        } catch (const std::exception& e) {
            error = q(e.what());
        }
        if (!self) return;
        QMetaObject::invokeMethod(self.data(), [self, job, error, notes] {
            if (!self) return;
            self->submitting_ = false;
            if (!error.isEmpty() || !job) {
                self->status_->setText(tr("Could not start: %1").arg(error));
                self->progress_->setRange(0, 1);
                self->progress_->setValue(0);
                self->setBusy(false);
                return;
            }
            self->job_ = job;
            if (!notes.isEmpty()) self->status_->setText(notes.join(QStringLiteral("; ")));
        }, Qt::QueuedConnection);
    }).detach();
}

void GenerateTab::onCancel() {
    if (job_) job_->cancel();
    status_->setText(tr("Cancelling…"));
}

void GenerateTab::poll() {
    if (!job_) return;
    const auto p = job_->progress();
    if (!job_->finished()) {
        if (p.steps > 0) {
            progress_->setRange(0, p.steps);
            progress_->setValue(p.step);
            QString text = tr("Step %1 of %2").arg(p.step).arg(p.steps);
            if (p.secondsPerStep > 0) text += tr("  (%1 s/step)").arg(p.secondsPerStep, 0, 'f', 2);
            status_->setText(text);
        } else {
            progress_->setRange(0, 0);
            status_->setText(job_->state() == gh::engine::JobState::Queued ? tr("Waiting in the queue…") : tr("Starting…"));
        }
        return;
    }

    const auto job = job_;
    job_.reset();
    setBusy(false);
    progress_->setRange(0, 1);
    progress_->setValue(job->state() == gh::engine::JobState::Succeeded ? 1 : 0);
    if (job->state() == gh::engine::JobState::Cancelled) {
        status_->setText(tr("Cancelled."));
        return;
    }
    if (job->state() == gh::engine::JobState::Failed) {
        status_->setText(tr("Failed: %1").arg(q(job->error())));
        return;
    }

    host_.generation().waitHandled(job->id());
    const auto entries = host_.generation().entries(job->id());
    const gh::engine::GenerationResult result = job->result();
    results_.clear();
    thumbs_->clear();
    for (size_t i = 0; i < result.images.size(); ++i) {
        Result r;
        if (i < entries.size()) {
            r.path = q((host_.gallery().absolutePath(entries[i])).string());
            r.galleryId = entries[i].id;
            r.image = QImage(r.path);
        }
        if (r.image.isNull()) r.image = toQImage(result.images[i]);
        r.seed = i < result.seeds.size() ? result.seeds[i] : result.request.seed;
        results_.push_back(r);
        auto* item = new QListWidgetItem(QIcon(QPixmap::fromImage(r.image.scaled(96, 96, Qt::KeepAspectRatio, Qt::SmoothTransformation))),
                                         tr("seed %1").arg(r.seed), thumbs_);
        item->setToolTip(tr("Seed %1").arg(r.seed));
    }
    thumbs_->setCurrentRow(0);
    QString done = tr("Done in %1 s: %n image(s).", "", static_cast<int>(result.images.size())).arg(result.seconds, 0, 'f', 1);
    if (!result.warnings.empty()) done += QStringLiteral("  ") + q(result.warnings.front());
    status_->setText(done);
    emit imagesAdded();
}

void GenerateTab::showResult(int index) {
    if (index < 0 || index >= results_.size()) {
        view_->setImage({});
        return;
    }
    view_->setImage(results_[index].image);
}

void GenerateTab::resultMenu(const QPoint& where) {
    const int row = thumbs_->currentRow();
    if (row < 0 || row >= results_.size()) return;
    const Result r = results_[row];
    QMenu menu(this);
    menu.addAction(tr("Save as…"), this, [this, r] {
        const QString target = QFileDialog::getSaveFileName(this, tr("Save image"), QStringLiteral("graphicshost-%1.png").arg(r.seed), tr("Images (*.png *.jpg *.webp)"));
        if (target.isEmpty()) return;
        if (!r.path.isEmpty() && target.endsWith(QFileInfo(r.path).suffix(), Qt::CaseInsensitive)) {
            QFile::remove(target);
            QFile::copy(r.path, target);   // the original file, with its embedded parameters
        } else {
            r.image.save(target);
        }
    });
    menu.addAction(tr("Copy image"), this, [r] { QApplication::clipboard()->setImage(r.image); });
    menu.addAction(tr("Use this seed"), this, [this, r] { seed_->setValue(static_cast<int>(std::clamp<qint64>(r.seed, -1, INT_MAX))); });
    if (!r.path.isEmpty()) {
        menu.addAction(tr("Show in folder"), this, [r] { QProcess::startDetached(QStringLiteral("explorer.exe"), {QStringLiteral("/select,"), QDir::toNativeSeparators(r.path)}); });
        menu.addSeparator();
        menu.addAction(tr("Delete from gallery"), this, [this, r, row] {
            if (QMessageBox::question(this, tr("Delete image"), tr("Delete this image and its settings file?")) != QMessageBox::Yes) return;
            host_.gallery().remove(r.galleryId);
            results_.removeAt(row);
            delete thumbs_->takeItem(row);
            showResult(thumbs_->currentRow());
            emit imagesAdded();   // the gallery changed
        });
    }
    menu.exec(where);
}

void GenerateTab::showRequestJson() {
    const auto call = currentCall();
    const Json doc = {{"model", call.model}, {"preset", call.preset}, {"request", call.request}, {"load", call.loadOptions}};
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Request"));
    auto* layout = new QVBoxLayout(&dialog);
    auto* text = new QPlainTextEdit(q(doc.dump(2)), &dialog);
    text->setReadOnly(true);
    text->setMinimumSize(420, 320);
    layout->addWidget(text);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, &dialog);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::accept);
    layout->addWidget(buttons);
    dialog.exec();
}

}  // namespace gh::ui

namespace gh::ui {

QString GenerateTab::statusText() const {
    return status_->text();
}

}  // namespace gh::ui
