#pragma once

#include <QMainWindow>

#include <memory>

class QLabel;
class QSystemTrayIcon;
class QTabWidget;
class QTimer;

namespace gh::host {
class AppHost;
}
namespace gh::server {
class ApiServer;
}

namespace gh::ui {

// Main window: one tab per area (Generate, Gallery, My Models, Discover, Settings, Logs). It also owns the HTTP API
// server (started from the settings), the tray icon and the remembered window position. The status bar shows the
// runtime, the server address, the loaded model and the queue.
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(gh::host::AppHost& host, QWidget* parent = nullptr);
    ~MainWindow() override;

    // The server address when it is running, empty otherwise.
    QString serverUrl() const;
    // Starts, restarts or stops the server to match the settings. Returns an error message, empty on success.
    QString applyServerSettings();

    QTabWidget* tabs() const { return tabs_; }

    // The window message a second instance broadcasts so that the running one shows itself.
    static unsigned raiseMessageId();

public slots:
    // Brings the window back from the tray / behind other windows.
    void raiseFromOutside();

protected:
    void closeEvent(QCloseEvent* event) override;
    bool nativeEvent(const QByteArray& eventType, void* message, qintptr* result) override;

private:
    void buildTabs();
    void buildTray();
    void refreshStatus();
    void restoreWindowState();
    void saveWindowState();
    QString runtimeSummary() const;

    gh::host::AppHost& host_;
    std::unique_ptr<gh::server::ApiServer> server_;
    QString serverError_;
    QTabWidget* tabs_ = nullptr;
    QSystemTrayIcon* tray_ = nullptr;
    QTimer* statusTimer_ = nullptr;
    QLabel* serverLabel_ = nullptr;
    QLabel* modelLabel_ = nullptr;
    bool reallyQuit_ = false;
    bool toldAboutTray_ = false;
};

}  // namespace gh::ui
