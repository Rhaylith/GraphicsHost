#pragma once

#include <QWidget>

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
class QTableWidget;

namespace gh::host {
class AppHost;
}

namespace gh::ui {

// Backend and GPU, the models folder, the API server, access tokens, the Hugging Face token, logging and the
// interface mode. Nothing changes until Apply is pressed.
class SettingsTab : public QWidget {
    Q_OBJECT

public:
    explicit SettingsTab(gh::host::AppHost& host, QWidget* parent = nullptr);

signals:
    // Settings were saved: the server, the interface mode and the lists need to follow.
    void applied();

protected:
    void showEvent(QShowEvent* event) override;

private:
    void loadFromSettings();
    void apply();
    void refreshTokens();
    void createToken();
    void revokeToken();
    void refreshHubToken();
    void setHubToken();
    void removeHubToken();

    gh::host::AppHost& host_;
    QComboBox* backend_ = nullptr;
    QComboBox* gpu_ = nullptr;
    QSpinBox* threads_ = nullptr;
    QLineEdit* modelsDir_ = nullptr;
    QComboBox* mode_ = nullptr;
    QComboBox* theme_ = nullptr;
    QPushButton* accent_ = nullptr;
    QString accentValue_;
    QCheckBox* serverEnabled_ = nullptr;
    QSpinBox* port_ = nullptr;
    QCheckBox* lan_ = nullptr;
    QCheckBox* requireToken_ = nullptr;
    QComboBox* logLevel_ = nullptr;
    QCheckBox* safety_ = nullptr;
    QLabel* message_ = nullptr;
    QLabel* hubState_ = nullptr;
    QTableWidget* tokens_ = nullptr;
};

}  // namespace gh::ui
