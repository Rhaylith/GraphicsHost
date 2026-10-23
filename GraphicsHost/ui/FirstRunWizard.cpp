#include "FirstRunWizard.h"

#include "UiUtil.h"
#include "host/AppHost.h"
#include "models/FitHints.h"
#include "models/Recommendation.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

#include <filesystem>

namespace gh::ui {

namespace fs = std::filesystem;

namespace {

class FieldPage : public QWizardPage {
public:
    using QWizardPage::QWizardPage;
    using QWizardPage::registerField;   // public here: the pages are built from outside
};

gh::models::HardwareBudget budgetOf(gh::host::AppHost& host) {
    const auto* runtime = host.selectedRuntime();
    return gh::models::budgetFrom(host.gpu(), runtime && runtime->backend == "vulkan");
}

QString freeSpace(const QString& folder) {
    std::error_code ec;
    fs::path p = folder.toStdWString();
    while (!p.empty() && !fs::exists(p, ec)) p = p.parent_path();   // the folder may not exist yet
    if (p.empty()) return {};
    const auto info = fs::space(p, ec);
    return ec ? QString() : humanSize(info.available);
}

}  // namespace

FirstRunWizard::FirstRunWizard(gh::host::AppHost& host, QWidget* parent) : QWizard(parent), host_(host) {
    setWindowTitle(tr("Welcome to GraphicsHost"));
    setWizardStyle(QWizard::ModernStyle);
    setOption(QWizard::NoBackButtonOnStartPage);
    setButtonText(QWizard::CancelButton, tr("Skip"));
    addPage(surveyPage());
    addPage(folderPage());
    addPage(modelPage());
    resize(640, 460);
}

QWizardPage* FirstRunWizard::surveyPage() {
    auto* page = new FieldPage(this);
    page->setTitle(tr("Your computer"));
    page->setSubTitle(tr("This is what GraphicsHost found. Nothing here needs to be set up by hand."));
    auto* layout = new QVBoxLayout(page);
    const auto& cpu = host_.cpu();
    const auto memory = gh::engine::systemMemory();
    QString text = tr("Processor: %1 (%2 logical cores)\n").arg(q(cpu.brand)).arg(cpu.logicalCores);
    text += tr("Memory: %1 free of %2\n").arg(humanSize(memory.freeBytes), humanSize(memory.totalBytes));
    bool anyGpu = false;
    for (const auto& d : host_.gpu().devices) {
        text += tr("Graphics card %1: %2, %3%4\n").arg(d.index).arg(q(d.name), humanSize(d.vramBytes), d.supported ? QString() : tr(" (not supported: %1)").arg(q(d.note)));
        anyGpu = anyGpu || d.supported;
    }
    if (host_.gpu().devices.empty()) text += tr("Graphics card: none found for Vulkan (%1)\n").arg(q(host_.gpu().error));
    const auto* runtime = host_.selectedRuntime();
    QString where = tr("nowhere yet: no inference runtime was found");
    if (runtime) where = runtime->backend == "vulkan" ? tr("on the graphics card (Vulkan)") : tr("on the processor (%1)").arg(q(runtime->id));
    text += tr("\nImages will be made %1.").arg(where);
    if (!anyGpu) text += tr("\nWithout a usable graphics card images take minutes instead of seconds.");
    auto* label = new QLabel(text, page);
    label->setWordWrap(true);
    label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(label);
    layout->addStretch(1);
    return page;
}

QWizardPage* FirstRunWizard::folderPage() {
    auto* page = new FieldPage(this);
    page->setTitle(tr("Where models are kept"));
    page->setSubTitle(tr("Models are large (2 to 7 GiB each). Pick a folder on a drive with room."));
    auto* layout = new QVBoxLayout(page);
    auto* row = new QHBoxLayout();
    auto* folder = new QLineEdit(q(host_.modelsDir().string()), page);
    auto* browse = new QPushButton(tr("Browse…"), page);
    row->addWidget(folder, 1);
    row->addWidget(browse);
    layout->addLayout(row);
    auto* space = new QLabel(page);
    layout->addWidget(space);
    layout->addStretch(1);
    page->registerField(QStringLiteral("modelsDir"), folder);
    auto refresh = [folder, space] {
        const QString f = freeSpace(folder->text());
        space->setText(f.isEmpty() ? QString() : tr("%1 free on that drive").arg(f));
    };
    connect(folder, &QLineEdit::textChanged, page, refresh);
    connect(browse, &QPushButton::clicked, page, [this, folder] {
        const QString dir = QFileDialog::getExistingDirectory(this, tr("Models folder"), folder->text());
        if (!dir.isEmpty()) folder->setText(QDir::toNativeSeparators(dir));
    });
    refresh();
    return page;
}

QWizardPage* FirstRunWizard::modelPage() {
    auto* page = new FieldPage(this);
    page->setTitle(tr("A first model"));
    page->setSubTitle(tr("A good starting point for this computer."));
    auto* layout = new QVBoxLayout(page);
    const auto rec = gh::models::recommendCheckpoint(host_.catalog(), budgetOf(host_));
    auto* download = new QCheckBox(page);
    if (rec.entry) {
        std::error_code ec;
        const bool have = fs::is_regular_file(host_.modelsDir() / rec.entry->relativePath(), ec);
        QString why = q(rec.why);
        why = why.left(1).toUpper() + why.mid(1);
        auto* label = new QLabel(tr("<b>%1</b> (%2, %3)<br>%4. On this computer: %5.")
                                     .arg(q(rec.entry->name).toHtmlEscaped(), humanSize(rec.entry->sizeBytes), q(rec.entry->license).toHtmlEscaped(), why, q(rec.fit.label)),
                                 page);
        label->setWordWrap(true);
        layout->addWidget(label);
        download->setText(have ? tr("Already downloaded") : tr("Download it now (%1)").arg(humanSize(rec.entry->sizeBytes)));
        download->setChecked(!have);
        download->setEnabled(!have);
        page->setProperty("catalogKey", q(rec.entry->key));
    } else {
        layout->addWidget(new QLabel(tr("No model is suggested: the catalog is empty. You can add your own in the models folder."), page));
        download->hide();
    }
    layout->addWidget(download);
    page->registerField(QStringLiteral("download"), download);

    auto* form = new QFormLayout();
    auto* mode = new QComboBox(page);
    mode->addItem(tr("Simple: prompt, size, count, seed"), QStringLiteral("user"));
    mode->addItem(tr("Advanced: sampler, steps, CFG, device"), QStringLiteral("power"));
    form->addRow(tr("Interface"), mode);
    layout->addLayout(form);
    layout->addStretch(1);
    page->registerField(QStringLiteral("mode"), mode, "currentData");
    return page;
}

void FirstRunWizard::markDone() {
    auto settings = host_.settings();
    if (settings.firstRunDone) return;
    settings.firstRunDone = true;
    try {
        host_.updateSettings(settings);
    } catch (const std::exception&) {
    }
}

void FirstRunWizard::reject() {
    markDone();   // skipping counts: the wizard does not come back
    QWizard::reject();
}

void FirstRunWizard::accept() {
    auto settings = host_.settings();
    const std::string folder = s(field(QStringLiteral("modelsDir")).toString().trimmed());
    settings.modelsDir = fs::path(folder) == host_.paths().defaultModelsDir() ? std::string() : folder;
    settings.interfaceMode = s(field(QStringLiteral("mode")).toString());
    settings.firstRunDone = true;
    try {
        host_.updateSettings(settings);
        std::error_code ec;
        fs::create_directories(host_.modelsDir(), ec);
        const QString key = page(2)->property("catalogKey").toString();
        if (field(QStringLiteral("download")).toBool() && !key.isEmpty()) {
            host_.modelService().start(host_.modelService().planCatalog(s(key), true));
        }
    } catch (const std::exception&) {
        // the settings or the download could not be applied; the Settings and Discover tabs can do the same later
    }
    QWizard::accept();
}

}  // namespace gh::ui
