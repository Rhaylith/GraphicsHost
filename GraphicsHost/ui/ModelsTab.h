#pragma once

#include <QWidget>

class QLabel;
class QPushButton;
class QTableWidget;
class QTimer;

namespace gh::host {
class AppHost;
}

namespace gh::ui {

// The files in the models folder: checkpoints, VAEs and LoRAs with their family, size and whether this version can
// run them. Load / unload the model in memory, edit the per-model defaults, open or clean the folder.
class ModelsTab : public QWidget {
    Q_OBJECT

public:
    explicit ModelsTab(gh::host::AppHost& host, QWidget* parent = nullptr);
    ~ModelsTab() override;

public slots:
    void refresh();

signals:
    // The loaded model or the list of models changed (the Generate tab refreshes its lists).
    void modelsChanged();

protected:
    void showEvent(QShowEvent* event) override;

private:
    QString selectedFile() const;
    void loadSelected();
    void unload();
    void editDefaults();
    void deleteSelected();
    void showMenu(const QPoint& where);
    void updateButtons();

    gh::host::AppHost& host_;
    QTableWidget* table_ = nullptr;
    QLabel* summary_ = nullptr;
    QLabel* status_ = nullptr;
    QPushButton* load_ = nullptr;
    QPushButton* unload_ = nullptr;
    QPushButton* defaults_ = nullptr;
    QPushButton* remove_ = nullptr;
    QTimer* timer_ = nullptr;
    bool loading_ = false;
};

}  // namespace gh::ui
