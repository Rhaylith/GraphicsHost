#include "MaskEditor.h"

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>

namespace gh::ui {

class MaskEditor::Canvas : public QWidget {
public:
    Canvas(const QImage& source, const QImage& existing, QWidget* parent) : QWidget(parent), source_(source.convertToFormat(QImage::Format_RGB888)) {
        mask_ = existing.size() == source.size() ? existing.convertToFormat(QImage::Format_Grayscale8) : QImage(source.size(), QImage::Format_Grayscale8);
        if (existing.size() != source.size()) mask_.fill(0);
        setMinimumSize(420, 320);
        setCursor(Qt::CrossCursor);
        setMouseTracking(true);
    }

    QImage mask() const { return mask_; }
    int brush = 40;      // radius in picture pixels
    bool erase = false;

    void clear() { mask_.fill(0); update(); }
    void invert() { mask_.invertPixels(); update(); }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.fillRect(rect(), palette().window());
        const QRectF target = imageRect();
        p.drawImage(target, source_);
        // the painted part in translucent red
        QImage overlay(mask_.size(), QImage::Format_ARGB32);
        overlay.fill(Qt::transparent);
        for (int y = 0; y < mask_.height(); ++y) {
            const uchar* in = mask_.constScanLine(y);
            QRgb* out = reinterpret_cast<QRgb*>(overlay.scanLine(y));
            for (int x = 0; x < mask_.width(); ++x) out[x] = in[x] >= 128 ? qRgba(255, 40, 60, 140) : 0;
        }
        p.drawImage(target, overlay);
        if (hover_) {
            p.setPen(QPen(Qt::white, 1));
            p.setBrush(Qt::NoBrush);
            const double r = brush * target.width() / source_.width();
            p.drawEllipse(QPointF(hoverAt_), r, r);
        }
    }
    void mousePressEvent(QMouseEvent* e) override { paintAt(e->position(), e->buttons()); }
    void mouseMoveEvent(QMouseEvent* e) override {
        hover_ = true;
        hoverAt_ = e->position();
        if (e->buttons() & (Qt::LeftButton | Qt::RightButton)) paintAt(e->position(), e->buttons());
        else update();
    }
    void leaveEvent(QEvent*) override { hover_ = false; update(); }

private:
    QRectF imageRect() const {
        const double scale = std::min(static_cast<double>(width()) / source_.width(), static_cast<double>(height()) / source_.height());
        const double w = source_.width() * scale, h = source_.height() * scale;
        return QRectF((width() - w) / 2, (height() - h) / 2, w, h);
    }
    void paintAt(const QPointF& at, Qt::MouseButtons buttons) {
        const QRectF target = imageRect();
        const QPointF pic((at.x() - target.x()) * source_.width() / target.width(), (at.y() - target.y()) * source_.height() / target.height());
        QPainter p(&mask_);
        p.setRenderHint(QPainter::Antialiasing, false);
        p.setPen(Qt::NoPen);
        const bool wipe = erase || (buttons & Qt::RightButton);
        p.setBrush(wipe ? Qt::black : Qt::white);
        p.drawEllipse(pic, static_cast<qreal>(brush), static_cast<qreal>(brush));
        hover_ = true;
        hoverAt_ = at;
        update();
    }

    QImage source_, mask_;
    bool hover_ = false;
    QPointF hoverAt_;
};

MaskEditor::MaskEditor(const QImage& source, const QImage& existing, QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("Mask: paint what should be regenerated"));
    auto* layout = new QVBoxLayout(this);
    canvas_ = new Canvas(source, existing, this);
    layout->addWidget(canvas_, 1);

    auto* row = new QHBoxLayout();
    row->addWidget(new QLabel(tr("Brush"), this));
    auto* size = new QSlider(Qt::Horizontal, this);
    const int longest = std::max(source.width(), source.height());
    size->setRange(4, std::max(8, longest / 4));
    size->setValue(std::clamp(longest / 16, 4, size->maximum()));
    canvas_->brush = size->value();
    row->addWidget(size, 1);
    auto* erase = new QCheckBox(tr("Erase"), this);
    erase->setToolTip(tr("The right mouse button erases too"));
    auto* invert = new QPushButton(tr("Invert"), this);
    auto* clear = new QPushButton(tr("Clear"), this);
    row->addWidget(erase);
    row->addWidget(invert);
    row->addWidget(clear);
    layout->addLayout(row);
    layout->addWidget(new QLabel(tr("Red = regenerated, the rest is kept. Left button paints, right button erases."), this));

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(size, &QSlider::valueChanged, this, [this](int v) { canvas_->brush = v; });
    connect(erase, &QCheckBox::toggled, this, [this](bool on) { canvas_->erase = on; });
    connect(invert, &QPushButton::clicked, this, [this] { canvas_->invert(); });
    connect(clear, &QPushButton::clicked, this, [this] { canvas_->clear(); });
    resize(760, 620);
}

QImage MaskEditor::mask() const {
    return canvas_->mask();
}

}  // namespace gh::ui
