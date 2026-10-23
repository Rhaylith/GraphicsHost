#pragma once

#include "engine/JobQueue.h"
#include "host/GenerationService.h"

#include <QImage>
#include <QString>
#include <QVector>
#include <QWidget>

#include <memory>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QLabel;
class QListWidget;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QTimer;

namespace gh::host {
class AppHost;
}

namespace gh::ui {

class ImageView;

// Prompt, parameters, the model to use, a Generate / Cancel button, live progress and the resulting images.
// What the interface mode hides (settings.interfaceMode): user = prompt, size, count, seed; power = also steps, CFG,
// sampler, scheduler, load options; developer = also the request JSON.
class GenerateTab : public QWidget {
    Q_OBJECT

public:
    explicit GenerateTab(gh::host::AppHost& host, QWidget* parent = nullptr);
    ~GenerateTab() override;

    // Re-reads the interface mode: shows or hides the advanced controls.
    void applyInterfaceMode();
    // Reloads the model and preset lists (the library may have changed).
    void refreshLists();
    // The call the current widget values stand for (what Generate would send).
    gh::host::GenerateCall currentCall() const;
    // Fills the widgets from a request document (used by "reuse parameters" in the gallery).
    void loadParameters(const nlohmann::json& request, const QString& model);
    // Image to image: start from this picture (a null image clears it).
    void setSourceImage(const QImage& image);

signals:
    // A generation finished and its images are in the gallery.
    void imagesAdded();

protected:
    void showEvent(QShowEvent* event) override;

public slots:
    // Starts a generation with the current values, as the Generate button does.
    void onGenerate();

public:
    bool busy() const { return job_ || submitting_; }
    QString statusText() const;

private slots:
    void onCancel();
    void poll();

private:
    struct Result {
        QImage image;
        QString path;       // the file in the gallery
        qint64 galleryId = 0;
        qint64 seed = 0;
    };

    void buildUi();
    void setBusy(bool busy);
    void showResult(int index);
    void resultMenu(const QPoint& where);
    void updateAutoLabels();
    void showRequestJson();

    gh::host::AppHost& host_;
    std::shared_ptr<gh::engine::Job> job_;
    bool submitting_ = false;
    QVector<Result> results_;
    QTimer* timer_ = nullptr;

    QComboBox* model_ = nullptr;
    QComboBox* preset_ = nullptr;
    QPlainTextEdit* prompt_ = nullptr;
    QPlainTextEdit* negative_ = nullptr;
    QSpinBox* width_ = nullptr;
    QSpinBox* height_ = nullptr;
    QSpinBox* steps_ = nullptr;
    QDoubleSpinBox* cfg_ = nullptr;
    QSpinBox* seed_ = nullptr;
    QSpinBox* count_ = nullptr;
    QComboBox* sampler_ = nullptr;
    QComboBox* scheduler_ = nullptr;
    QSpinBox* clipSkip_ = nullptr;
    QCheckBox* vaeTiling_ = nullptr;
    QComboBox* format_ = nullptr;
    QComboBox* device_ = nullptr;
    QCheckBox* flash_ = nullptr;
    QCheckBox* offload_ = nullptr;
    QWidget* advancedRequest_ = nullptr;
    QWidget* advancedLoad_ = nullptr;
    QWidget* powerWidgets_ = nullptr;
    QPushButton* showJson_ = nullptr;
    QImage source_;
    QLabel* sourceView_ = nullptr;
    QDoubleSpinBox* strength_ = nullptr;
    QPushButton* clearSource_ = nullptr;
    QPushButton* generate_ = nullptr;
    QPushButton* cancel_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QLabel* status_ = nullptr;
    ImageView* view_ = nullptr;
    QListWidget* thumbs_ = nullptr;
};

}  // namespace gh::ui
