#pragma once

#include <QImage>
#include <QLabel>
#include <QPixmap>

namespace gh::ui {

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

}  // namespace gh::ui
