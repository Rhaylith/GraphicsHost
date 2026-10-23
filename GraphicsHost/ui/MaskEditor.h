#pragma once

#include <QDialog>
#include <QImage>

namespace gh::ui {

// Paint over the part of a picture that should be regenerated (inpainting). The result is a mask of the picture's
// size: white where it was painted, black elsewhere. No Q_OBJECT: nothing here needs signals.
class MaskEditor : public QDialog {
public:
    // existing: an earlier mask of the same picture to continue from (may be null)
    MaskEditor(const QImage& source, const QImage& existing, QWidget* parent = nullptr);
    QImage mask() const;   // Format_Grayscale8, the size of the source

private:
    class Canvas;
    Canvas* canvas_ = nullptr;
};

}  // namespace gh::ui
