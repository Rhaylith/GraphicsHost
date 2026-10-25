#include "UiUtil.h"

#include <QFont>
#include <QPainter>
#include <QPixmap>

namespace gh::ui {

QString humanSize(uint64_t bytes) {
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    if (gib >= 1.0) return QString::number(gib, 'f', 2) + QStringLiteral(" GiB");
    const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
    if (mib >= 1.0) return QString::number(mib, 'f', 0) + QStringLiteral(" MiB");
    return QString::number(bytes / 1024) + QStringLiteral(" KiB");
}

QImage toQImage(const engine::Image& image) {
    if (image.width == 0 || image.height == 0 || image.data.empty()) return {};
    const QImage::Format format = image.channels == 4 ? QImage::Format_RGBA8888 : QImage::Format_RGB888;
    const QImage view(image.data.data(), static_cast<int>(image.width), static_cast<int>(image.height),
                      static_cast<qsizetype>(image.width) * image.channels, format);
    return view.copy();
}

QIcon appIcon() {
    QIcon icon;
    for (int size : {16, 24, 32, 48, 64, 128, 256}) {
        QPixmap pixmap(size, size);
        pixmap.fill(Qt::transparent);
        QPainter p(&pixmap);
        p.setRenderHint(QPainter::Antialiasing);
        p.setBrush(QColor(0x2b, 0x6c, 0xb0));
        p.setPen(Qt::NoPen);
        p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.22, size * 0.22);
        p.setPen(Qt::white);
        QFont font;
        font.setBold(true);
        font.setPixelSize(static_cast<int>(size * 0.62));
        p.setFont(font);
        p.drawText(QRectF(0, 0, size, size), Qt::AlignCenter, QStringLiteral("G"));
        p.end();
        icon.addPixmap(pixmap);
    }
    return icon;
}

}  // namespace gh::ui
