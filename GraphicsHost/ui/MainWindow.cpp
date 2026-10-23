#include "MainWindow.h"

#include "DiscoverTab.h"
#include "GalleryTab.h"
#include "GenerateTab.h"
#include "ModelsTab.h"
#include "LogsTab.h"
#include "SettingsTab.h"
#include "UiUtil.h"
#include "core/Log.h"
#include "host/AppHost.h"
#include "server/ApiServer.h"

#include <QAction>
#include <QApplication>
#include <QCloseEvent>
#include <QLabel>
#include <QMenu>
#include <QSettings>
#include <QStatusBar>
#include <QSystemTrayIcon>
#include <QTabWidget>
#include <QTimer>

#include <windows.h>

namespace gh::ui {

namespace {

// The message a second instance broadcasts to make the first one show its window.
UINT raiseMessageImpl() {
    static const UINT id = RegisterWindowMessageW(L"GraphicsHost.RaiseWindow");
    return id;
}

}  // namespace

MainWindow::MainWindow(gh::host::AppHost& host, QWidget* parent)
    : QMainWindow(parent), host_(host), server_(std::make_unique<gh::server::ApiServer>(host)), tabs_(new QTabWidget(this)) {
    setWindowTitle(QStringLiteral("GraphicsHost"));   // second instances look for this title: keep it fixed
    setWindowIcon(appIcon());
    resize(1180, 780);
    setCentralWidget(tabs_);

    serverLabel_ = new QLabel(this);
    modelLabel_ = new QLabel(this);
    statusBar()->addPermanentWidget(modelLabel_);
    statusBar()->addPermanentWidget(serverLabel_);

    buildTabs();
    buildTray();
    restoreWindowState();

    serverError_ = applyServerSettings();

    statusTimer_ = new QTimer(this);
    connect(statusTimer_, &QTimer::timeout, this, [this] { refreshStatus(); });
    statusTimer_->start(1000);
    refreshStatus();
}

unsigned MainWindow::raiseMessageId() { return raiseMessageImpl(); }

MainWindow::~MainWindow() {
    if (server_) server_->stop();
}

void MainWindow::buildTabs() {
    generate_ = new GenerateTab(host_, tabs_);
    tabs_->addTab(generate_, tr("Generate"));
    gallery_ = new GalleryTab(host_, tabs_);
    tabs_->addTab(gallery_, tr("Gallery"));
    connect(generate_, &GenerateTab::imagesAdded, gallery_, &GalleryTab::reload);
    connect(gallery_, &GalleryTab::sourceRequested, this, [this](const QImage& image) {
        generate_->setSourceImage(image);
        tabs_->setCurrentWidget(generate_);
    });
    connect(gallery_, &GalleryTab::reuseRequested, this, [this](const nlohmann::json& request, const QString& model) {
        generate_->loadParameters(request, model);
        tabs_->setCurrentWidget(generate_);
    });
    auto* models = new ModelsTab(host_, tabs_);
    tabs_->addTab(models, tr("My Models"));
    auto* discover = new DiscoverTab(host_, tabs_);
    tabs_->addTab(discover, tr("Discover"));
    connect(models, &ModelsTab::modelsChanged, generate_, &GenerateTab::refreshLists);
    connect(discover, &DiscoverTab::modelsChanged, generate_, &GenerateTab::refreshLists);
    connect(discover, &DiscoverTab::modelsChanged, models, &ModelsTab::refresh);
    auto* settings = new SettingsTab(host_, tabs_);
    tabs_->addTab(settings, tr("Settings"));
    connect(settings, &SettingsTab::applied, this, [this] {
        serverError_ = applyServerSettings();
        generate_->applyInterfaceMode();
        generate_->refreshLists();
        refreshStatus();
    });
    tabs_->addTab(new LogsTab(host_, tabs_), tr("Logs"));
}

void MainWindow::buildTray() {
    if (!QSystemTrayIcon::isSystemTrayAvailable()) return;
    tray_ = new QSystemTrayIcon(appIcon(), this);
    tray_->setToolTip(QStringLiteral("GraphicsHost"));
    auto* menu = new QMenu(this);
    menu->addAction(tr("Show GraphicsHost"), this, &MainWindow::raiseFromOutside);
    menu->addSeparator();
    menu->addAction(tr("Quit"), this, [this] {
        reallyQuit_ = true;
        close();
        QApplication::quit();
    });
    tray_->setContextMenu(menu);
    connect(tray_, &QSystemTrayIcon::activated, this, [this](QSystemTrayIcon::ActivationReason reason) {
        if (reason == QSystemTrayIcon::Trigger || reason == QSystemTrayIcon::DoubleClick) raiseFromOutside();
    });
    tray_->show();
}

QString MainWindow::serverUrl() const {
    return server_ && server_->running() ? q(server_->url()) : QString();
}

QString MainWindow::applyServerSettings() {
    server_->stop();
    const auto& settings = host_.settings();
    if (!settings.server.enabled) return {};
    gh::server::ServerOptions options;
    options.port = settings.server.port;
    options.bindAddress = settings.server.lan ? "0.0.0.0" : "127.0.0.1";
    const std::string error = server_->start(options);
    if (!error.empty()) {
        gh::log::warn("API server not started: " + error);
        return q(error);
    }
    return {};
}

QString MainWindow::runtimeSummary() const {
    if (const auto* runtime = host_.selectedRuntime()) return tr("Runtime: %1").arg(q(runtime->id));
    return tr("No runtime for backend '%1'").arg(q(host_.settings().backend));
}

void MainWindow::refreshStatus() {
    statusBar()->showMessage(runtimeSummary());
    const auto loaded = host_.models().loaded();
    const auto active = host_.models().queue().active();
    QString model = loaded ? tr("Model: %1").arg(QString::fromStdString(std::filesystem::path(loaded->options.modelPath).filename().string())) : tr("No model loaded");
    if (active > 0) model += tr("  |  %n job(s) queued or running", "", static_cast<int>(active));
    modelLabel_->setText(model);
    if (server_->running()) serverLabel_->setText(tr("API: %1").arg(q(server_->url())));
    else if (!host_.settings().server.enabled) serverLabel_->setText(tr("API: off"));
    else serverLabel_->setText(tr("API: not running (%1)").arg(serverError_));
}

void MainWindow::raiseFromOutside() {
    showNormal();
    raise();
    activateWindow();
}

void MainWindow::closeEvent(QCloseEvent* event) {
    saveWindowState();
    // with the API server running, closing the window leaves GraphicsHost in the tray so that clients keep working
    if (!reallyQuit_ && tray_ && tray_->isVisible() && server_->running()) {
        hide();
        if (!toldAboutTray_) {
            tray_->showMessage(tr("GraphicsHost is still running"), tr("The API server keeps running. Quit from the tray icon."), QSystemTrayIcon::Information, 4000);
            toldAboutTray_ = true;
        }
        event->ignore();
        return;
    }
    event->accept();
}

bool MainWindow::nativeEvent(const QByteArray& eventType, void* message, qintptr* result) {
    if (eventType == "windows_generic_MSG") {
        const MSG* msg = static_cast<const MSG*>(message);
        if (msg->message == raiseMessageImpl()) {
            raiseFromOutside();
            *result = 0;
            return true;
        }
    }
    return QMainWindow::nativeEvent(eventType, message, result);
}

void MainWindow::restoreWindowState() {
    QSettings settings(q((host_.paths().root() / "window.ini").string()), QSettings::IniFormat);
    restoreGeometry(settings.value(QStringLiteral("geometry")).toByteArray());
    restoreState(settings.value(QStringLiteral("state")).toByteArray());
    const int tab = settings.value(QStringLiteral("tab"), 0).toInt();
    if (tab >= 0 && tab < tabs_->count()) tabs_->setCurrentIndex(tab);
}

void MainWindow::saveWindowState() {
    QSettings settings(q((host_.paths().root() / "window.ini").string()), QSettings::IniFormat);
    settings.setValue(QStringLiteral("geometry"), saveGeometry());
    settings.setValue(QStringLiteral("state"), saveState());
    settings.setValue(QStringLiteral("tab"), tabs_->currentIndex());
}

}  // namespace gh::ui

