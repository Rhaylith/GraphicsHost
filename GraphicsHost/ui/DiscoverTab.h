#pragma once

#include "models/DownloadManager.h"
#include "models/HuggingFace.h"
#include "models/ModelService.h"

#include <QWidget>

#include <functional>

class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QTableWidget;
class QTimer;

namespace gh::host {
class AppHost;
}

namespace gh::ui {

// Finds models to download: the curated catalog (with a "will it fit" hint for this computer) and a Hugging Face
// search, plus the queue of running downloads with pause / resume / cancel.
class DiscoverTab : public QWidget {
    Q_OBJECT

public:
    explicit DiscoverTab(gh::host::AppHost& host, QWidget* parent = nullptr);

signals:
    // A download finished: the model list changed.
    void modelsChanged();

private:
    void buildCatalogPage(QWidget* page);
    void buildHubPage(QWidget* page);
    void buildDownloads(QWidget* panel);
    void refreshCatalog();
    void searchHub();
    void showRepoFiles();
    void refreshDownloads();
    // Plans on a worker thread (it may talk to Hugging Face), asks, then starts the downloads.
    void getWith(const QString& title, std::function<gh::models::DownloadPlan()> planner, const QString& fitFamily);
    void confirmAndStart(const QString& title, const gh::models::DownloadPlan& plan, const QString& fitFamily);
    void downloadAction(const char* action);
    uint64_t selectedDownload() const;

    gh::host::AppHost& host_;
    QLineEdit* filter_ = nullptr;
    QComboBox* family_ = nullptr;
    QComboBox* kind_ = nullptr;
    QTableWidget* catalog_ = nullptr;
    QPushButton* getCatalog_ = nullptr;
    QLineEdit* query_ = nullptr;
    QTableWidget* repos_ = nullptr;
    QTableWidget* files_ = nullptr;
    QPushButton* getFile_ = nullptr;
    QLabel* hubStatus_ = nullptr;
    QTableWidget* downloads_ = nullptr;
    QLabel* downloadStatus_ = nullptr;
    QTimer* timer_ = nullptr;
    QString repoShown_;
    bool busy_ = false;
};

}  // namespace gh::ui
