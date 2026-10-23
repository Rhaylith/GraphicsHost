#include "SettingsTab.h"

#include "UiUtil.h"
#include "core/SecretStore.h"
#include "host/AppHost.h"
#include "models/HuggingFace.h"

#include <QApplication>
#include <QCheckBox>
#include <QColorDialog>
#include <QClipboard>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPointer>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QTableWidget>
#include <QVBoxLayout>

#include <thread>

namespace gh::ui {

SettingsTab::SettingsTab(gh::host::AppHost& host, QWidget* parent) : QWidget(parent), host_(host) {
    auto* outer = new QVBoxLayout(this);
    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    auto* body = new QWidget(scroll);
    auto* layout = new QVBoxLayout(body);
    scroll->setWidget(body);
    outer->addWidget(scroll, 1);

    // ---- engine
    auto* engineBox = new QGroupBox(tr("Engine and models"), body);
    auto* engineForm = new QFormLayout(engineBox);
    backend_ = new QComboBox(engineBox);
    backend_->addItem(tr("Automatic (GPU when there is a usable one)"), QStringLiteral("auto"));
    backend_->addItem(tr("Vulkan GPU"), QStringLiteral("vulkan"));
    backend_->addItem(tr("CPU"), QStringLiteral("cpu"));
    gpu_ = new QComboBox(engineBox);
    for (const auto& d : host_.gpu().devices) {
        gpu_->addItem(QStringLiteral("%1: %2 (%3)").arg(d.index).arg(q(d.name), humanSize(d.vramBytes)) + (d.supported ? QString() : tr("  - not supported")), d.index);
    }
    if (gpu_->count() == 0) {
        gpu_->addItem(tr("No Vulkan GPU found"), 0);
        gpu_->setEnabled(false);
    }
    threads_ = new QSpinBox(engineBox);
    threads_->setRange(0, 512);
    threads_->setSpecialValueText(tr("Automatic"));
    modelsDir_ = new QLineEdit(engineBox);
    modelsDir_->setPlaceholderText(q(host_.paths().defaultModelsDir().string()));
    auto* browse = new QPushButton(tr("Browse…"), engineBox);
    auto* dirRow = new QHBoxLayout();
    dirRow->addWidget(modelsDir_, 1);
    dirRow->addWidget(browse);
    mode_ = new QComboBox(engineBox);
    mode_->addItem(tr("Simple: prompt, size, count, seed"), QStringLiteral("user"));
    mode_->addItem(tr("Advanced: sampler, steps, CFG, device"), QStringLiteral("power"));
    mode_->addItem(tr("Developer: also the request JSON"), QStringLiteral("developer"));
    engineForm->addRow(tr("Backend"), backend_);
    engineForm->addRow(tr("GPU"), gpu_);
    engineForm->addRow(tr("CPU threads"), threads_);
    engineForm->addRow(tr("Models folder"), dirRow);
    engineForm->addRow(tr("Interface"), mode_);
    theme_ = new QComboBox(engineBox);
    theme_->addItem(tr("Follow Windows"), QStringLiteral("system"));
    theme_->addItem(tr("Light"), QStringLiteral("light"));
    theme_->addItem(tr("Dark"), QStringLiteral("dark"));
    accent_ = new QPushButton(tr("Default blue"), engineBox);
    accent_->setToolTip(tr("Pick the accent colour (selection, links); click with Shift held to go back to the default"));
    auto* themeRow = new QHBoxLayout();
    themeRow->addWidget(theme_, 1);
    themeRow->addWidget(accent_);
    engineForm->addRow(tr("Theme"), themeRow);
    connect(accent_, &QPushButton::clicked, this, [this] {
        if (QApplication::keyboardModifiers() & Qt::ShiftModifier) {
            accentValue_.clear();
        } else {
            const QColor c = QColorDialog::getColor(accentValue_.isEmpty() ? QColor(0x2b, 0x6c, 0xb0) : QColor(accentValue_), this, tr("Accent colour"));
            if (!c.isValid()) return;
            accentValue_ = c.name();
        }
        accent_->setText(accentValue_.isEmpty() ? tr("Default blue") : accentValue_);
        accent_->setStyleSheet(accentValue_.isEmpty() ? QString() : QStringLiteral("QPushButton { border-left: 12px solid %1; }").arg(accentValue_));
    });
    layout->addWidget(engineBox);
    connect(browse, &QPushButton::clicked, this, [this] {
        const QString dir = QFileDialog::getExistingDirectory(this, tr("Models folder"), modelsDir_->text());
        if (!dir.isEmpty()) modelsDir_->setText(QDir::toNativeSeparators(dir));
    });

    // ---- server
    auto* serverBox = new QGroupBox(tr("API server (browser UI, OpenAI-compatible and native API)"), body);
    auto* serverForm = new QFormLayout(serverBox);
    serverEnabled_ = new QCheckBox(tr("Run the server"), serverBox);
    port_ = new QSpinBox(serverBox);
    port_->setRange(1, 65535);
    lan_ = new QCheckBox(tr("Let other computers connect (needs an access token)"), serverBox);
    requireToken_ = new QCheckBox(tr("Also require a token on this computer"), serverBox);
    serverForm->addRow(QString(), serverEnabled_);
    serverForm->addRow(tr("Port"), port_);
    serverForm->addRow(QString(), lan_);
    serverForm->addRow(QString(), requireToken_);
    https_ = new QCheckBox(tr("Use https (a self-signed certificate is made for you)"), serverBox);
    allowed_ = new QLineEdit(serverBox);
    allowed_->setPlaceholderText(tr("everyone, e.g. 192.168.1.0/24, 10.0.0.5"));
    allowed_->setToolTip(tr("Only these addresses may connect (this computer always may). Separate entries with commas."));
    rate_ = new QSpinBox(serverBox);
    rate_->setRange(0, 100000);
    rate_->setSpecialValueText(tr("No limit"));
    rate_->setSuffix(tr(" requests a minute"));
    serverForm->addRow(QString(), https_);
    serverForm->addRow(tr("Allowed addresses"), allowed_);
    serverForm->addRow(tr("Rate limit"), rate_);

    tokens_ = new QTableWidget(0, 4, serverBox);
    tokens_->setHorizontalHeaderLabels({tr("Name"), tr("Permissions"), tr("Created"), tr("Last used")});
    tokens_->setSelectionBehavior(QAbstractItemView::SelectRows);
    tokens_->setSelectionMode(QAbstractItemView::SingleSelection);
    tokens_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    tokens_->verticalHeader()->setVisible(false);
    tokens_->horizontalHeader()->setStretchLastSection(true);
    tokens_->setMinimumHeight(110);
    auto* tokenButtons = new QHBoxLayout();
    auto* create = new QPushButton(tr("New token…"), serverBox);
    auto* revoke = new QPushButton(tr("Revoke"), serverBox);
    tokenButtons->addWidget(create);
    tokenButtons->addWidget(revoke);
    tokenButtons->addStretch(1);
    serverForm->addRow(tr("Access tokens"), tokens_);
    serverForm->addRow(QString(), tokenButtons);
    layout->addWidget(serverBox);
    connect(create, &QPushButton::clicked, this, &SettingsTab::createToken);
    connect(revoke, &QPushButton::clicked, this, &SettingsTab::revokeToken);

    // ---- logging, safety, Hugging Face
    auto* otherBox = new QGroupBox(tr("Other"), body);
    auto* otherForm = new QFormLayout(otherBox);
    logLevel_ = new QComboBox(otherBox);
    for (const char* level : {"trace", "debug", "info", "warn", "error"}) logLevel_->addItem(QString::fromLatin1(level), QString::fromLatin1(level));
    safety_ = new QCheckBox(tr("Safety filter (off by default)"), otherBox);
    hubState_ = new QLabel(otherBox);
    auto* hubSet = new QPushButton(tr("Set token…"), otherBox);
    auto* hubRemove = new QPushButton(tr("Remove token"), otherBox);
    auto* hubRow = new QHBoxLayout();
    hubRow->addWidget(hubState_, 1);
    hubRow->addWidget(hubSet);
    hubRow->addWidget(hubRemove);
    otherForm->addRow(tr("Log level"), logLevel_);
    otherForm->addRow(QString(), safety_);
    otherForm->addRow(tr("Hugging Face token"), hubRow);
    layout->addWidget(otherBox);
    layout->addStretch(1);
    connect(hubSet, &QPushButton::clicked, this, &SettingsTab::setHubToken);
    connect(hubRemove, &QPushButton::clicked, this, &SettingsTab::removeHubToken);

    // ---- apply
    auto* bottom = new QHBoxLayout();
    message_ = new QLabel(this);
    message_->setWordWrap(true);
    auto* applyButton = new QPushButton(tr("Apply"), this);
    auto* revert = new QPushButton(tr("Revert"), this);
    applyButton->setDefault(true);
    bottom->addWidget(message_, 1);
    bottom->addWidget(revert);
    bottom->addWidget(applyButton);
    outer->addLayout(bottom);
    connect(applyButton, &QPushButton::clicked, this, &SettingsTab::apply);
    connect(revert, &QPushButton::clicked, this, [this] {
        loadFromSettings();
        message_->setText(tr("Reverted to the saved settings."));
    });

    loadFromSettings();
    refreshTokens();
    refreshHubToken();
}

void SettingsTab::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    refreshTokens();
    refreshHubToken();
}

void SettingsTab::loadFromSettings() {
    const auto& st = host_.settings();
    backend_->setCurrentIndex(std::max(0, backend_->findData(q(st.backend))));
    const int g = gpu_->findData(st.gpuDevice);
    if (g >= 0) gpu_->setCurrentIndex(g);
    threads_->setValue(st.threads);
    modelsDir_->setText(q(st.modelsDir));
    mode_->setCurrentIndex(std::max(0, mode_->findData(q(st.interfaceMode))));
    theme_->setCurrentIndex(std::max(0, theme_->findData(q(st.theme))));
    accentValue_ = q(st.accent);
    accent_->setText(accentValue_.isEmpty() ? tr("Default blue") : accentValue_);
    serverEnabled_->setChecked(st.server.enabled);
    port_->setValue(st.server.port);
    lan_->setChecked(st.server.lan);
    requireToken_->setChecked(st.server.requireToken);
    https_->setChecked(st.server.https);
    QStringList addresses;
    for (const auto& a : st.server.allowedAddresses) addresses << q(a);
    allowed_->setText(addresses.join(QStringLiteral(", ")));
    rate_->setValue(st.server.rateLimitPerMinute);
    logLevel_->setCurrentIndex(std::max(0, logLevel_->findData(q(st.logging.level))));
    safety_->setChecked(st.safety.enabled);
}

void SettingsTab::apply() {
    gh::core::Settings st = host_.settings();
    st.backend = s(backend_->currentData().toString());
    if (gpu_->isEnabled()) st.gpuDevice = gpu_->currentData().toInt();
    st.threads = threads_->value();
    st.modelsDir = s(modelsDir_->text().trimmed());
    st.interfaceMode = s(mode_->currentData().toString());
    st.theme = s(theme_->currentData().toString());
    st.accent = s(accentValue_);
    st.server.enabled = serverEnabled_->isChecked();
    st.server.port = port_->value();
    st.server.lan = lan_->isChecked();
    st.server.requireToken = requireToken_->isChecked();
    st.server.https = https_->isChecked();
    st.server.allowedAddresses.clear();
    for (const QString& part : allowed_->text().split(QLatin1Char(','), Qt::SkipEmptyParts)) st.server.allowedAddresses.push_back(s(part.trimmed()));
    st.server.rateLimitPerMinute = rate_->value();
    st.logging.level = s(logLevel_->currentData().toString());
    st.safety.enabled = safety_->isChecked();
    if (st.server.enabled && st.server.lan && !host_.tokens().any()) {
        message_->setText(tr("Letting other computers connect needs at least one access token: create one below first."));
        return;
    }
    try {
        host_.updateSettings(st);
    } catch (const std::exception& e) {
        message_->setText(tr("Not saved: %1").arg(q(e.what())));
        return;
    }
    message_->setText(tr("Saved."));
    emit applied();
}

void SettingsTab::refreshTokens() {
    tokens_->setRowCount(0);
    for (const auto& t : host_.tokens().list()) {
        const int row = tokens_->rowCount();
        tokens_->insertRow(row);
        QStringList perms;
        for (const auto& p : t.permissions) perms << q(p);
        auto* name = new QTableWidgetItem(q(t.name));
        name->setData(Qt::UserRole, q(t.id));
        tokens_->setItem(row, 0, name);
        tokens_->setItem(row, 1, new QTableWidgetItem(perms.join(QStringLiteral(", "))));
        tokens_->setItem(row, 2, new QTableWidgetItem(q(t.created)));
        tokens_->setItem(row, 3, new QTableWidgetItem(t.lastUsed.empty() ? tr("never") : q(t.lastUsed)));
    }
}

void SettingsTab::createToken() {
    QDialog dialog(this);
    dialog.setWindowTitle(tr("New access token"));
    auto* layout = new QFormLayout(&dialog);
    auto* name = new QLineEdit(&dialog);
    name->setPlaceholderText(tr("what it is for, e.g. \"my phone\""));
    layout->addRow(tr("Name"), name);
    QList<QCheckBox*> boxes;
    const QStringList hints = {tr("read: look at status, models, the gallery"), tr("generate: queue and cancel generations"),
                               tr("manage: load models, download, presets, delete images"), tr("admin: settings, tokens, request log")};
    int i = 0;
    for (const auto& p : gh::server::allPermissions()) {
        auto* box = new QCheckBox(hints.value(i++), &dialog);
        box->setProperty("permission", q(p));
        box->setChecked(p == "read" || p == "generate");
        boxes << box;
        layout->addRow(QString(), box);
    }
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    layout->addRow(buttons);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    if (dialog.exec() != QDialog::Accepted) return;

    std::vector<std::string> permissions;
    for (auto* b : boxes) {
        if (b->isChecked()) permissions.push_back(s(b->property("permission").toString()));
    }
    try {
        const auto created = host_.tokens().create(s(name->text().trimmed()), permissions);
        QDialog shown(this);
        shown.setWindowTitle(tr("Token created"));
        auto* v = new QVBoxLayout(&shown);
        v->addWidget(new QLabel(tr("Copy this token now: it is not shown again. Send it as \"Authorization: Bearer <token>\"."), &shown));
        auto* field = new QLineEdit(q(created.secret), &shown);
        field->setReadOnly(true);
        field->setMinimumWidth(520);
        v->addWidget(field);
        auto* ok = new QDialogButtonBox(QDialogButtonBox::Close, &shown);
        auto* copy = ok->addButton(tr("Copy"), QDialogButtonBox::ActionRole);
        connect(copy, &QPushButton::clicked, &shown, [field] { QApplication::clipboard()->setText(field->text()); });
        connect(ok, &QDialogButtonBox::rejected, &shown, &QDialog::accept);
        v->addWidget(ok);
        shown.exec();
    } catch (const std::exception& e) {
        message_->setText(tr("Could not create the token: %1").arg(q(e.what())));
    }
    refreshTokens();
}

void SettingsTab::revokeToken() {
    const auto rows = tokens_->selectionModel()->selectedRows();
    if (rows.isEmpty()) return;
    const QString name = tokens_->item(rows.first().row(), 0)->text();
    if (QMessageBox::question(this, tr("Revoke token"), tr("Revoke the token \"%1\"? Programs using it stop working.").arg(name)) != QMessageBox::Yes) return;
    host_.tokens().revoke(s(tokens_->item(rows.first().row(), 0)->data(Qt::UserRole).toString()));
    refreshTokens();
}

void SettingsTab::refreshHubToken() {
    const bool have = gh::core::huggingFaceToken().has_value();
    hubState_->setText(have ? tr("a token is stored") : tr("none (needed only for gated models)"));
}

void SettingsTab::setHubToken() {
    bool ok = false;
    const QString token = QInputDialog::getText(this, tr("Hugging Face token"), tr("Paste an access token (huggingface.co/settings/tokens):"), QLineEdit::Password, QString(), &ok).trimmed();
    if (!ok || token.isEmpty()) return;
    message_->setText(tr("Checking the token…"));
    QPointer<SettingsTab> self(this);
    std::thread([self, token] {
        QString user, error;
        try {
            const char* url = std::getenv("GRAPHICSHOST_HUB_URL");
            user = q(gh::models::HuggingFace(s(token), url && *url ? url : "https://huggingface.co").whoami());
            if (!gh::core::storeSecret(gh::core::kHuggingFaceSecret, s(token))) error = tr("The token is valid but could not be stored in the Windows Credential Manager.");
        } catch (const std::exception& e) {
            error = q(e.what());
        }
        if (!self) return;
        QMetaObject::invokeMethod(self.data(), [self, user, error] {
            if (!self) return;
            self->message_->setText(error.isEmpty() ? tr("Logged in as %1.").arg(user) : tr("Token not saved: %1").arg(error));
            self->refreshHubToken();
        }, Qt::QueuedConnection);
    }).detach();
}

void SettingsTab::removeHubToken() {
    if (QMessageBox::question(this, tr("Remove token"), tr("Remove the stored Hugging Face token?")) != QMessageBox::Yes) return;
    gh::core::deleteSecret(gh::core::kHuggingFaceSecret);
    refreshHubToken();
}

}  // namespace gh::ui
