#pragma once

#include "gallery/GalleryStore.h"

#include <QImage>
#include <QWidget>

#include <vector>

class QLabel;
class QLineEdit;
class QListWidget;
class QListWidgetItem;
class QPlainTextEdit;
class QPushButton;
class QTimer;

namespace gh::host {
class AppHost;
}

namespace gh::ui {

class ImageView;

// Everything that was generated, newest first: thumbnails (loaded in pages, off the UI thread), search by prompt, a
// detail pane with the parameters, and reuse / save / show in folder / delete / export.
class GalleryTab : public QWidget {
    Q_OBJECT

public:
    explicit GalleryTab(gh::host::AppHost& host, QWidget* parent = nullptr);

public slots:
    // Starts again from the newest image (after a generation, a delete, a new search).
    void reload();

signals:
    // "Reuse parameters": the Generate tab should take these values (a request document and the model's file name).
    void reuseRequested(const nlohmann::json& request, const QString& model);
    // "Use as source image": start an image to image generation from this picture.
    void sourceRequested(const QImage& image);

protected:
    void showEvent(QShowEvent* event) override;

private:
    void loadPage();
    void addPage(const std::vector<gh::gallery::GalleryEntry>& entries, const std::vector<QImage>& thumbnails, int generation);
    void select(int row);
    void deleteSelected();
    void exportSelected();
    void saveSelectedAs();
    const gh::gallery::GalleryEntry* current() const;

    gh::host::AppHost& host_;
    QLineEdit* search_ = nullptr;
    QLabel* count_ = nullptr;
    QListWidget* list_ = nullptr;
    ImageView* preview_ = nullptr;
    QPlainTextEdit* details_ = nullptr;
    QPushButton* reuse_ = nullptr;
    QPushButton* save_ = nullptr;
    QPushButton* folder_ = nullptr;
    QPushButton* remove_ = nullptr;
    QPushButton* more_ = nullptr;
    QTimer* searchTimer_ = nullptr;

    std::vector<gh::gallery::GalleryEntry> entries_;
    size_t total_ = 0;
    int generation_ = 0;     // answers of an older search are ignored
    bool loading_ = false;
    bool dirty_ = true;
};

}  // namespace gh::ui
