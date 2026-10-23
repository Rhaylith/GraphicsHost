#include "Theme.h"

#include "core/Paths.h"

#include <QApplication>
#include <QColor>
#include <QDir>
#include <QLocale>
#include <QPalette>
#include <QStyleFactory>
#include <QStyleHints>
#include <QTranslator>

namespace gh::ui {

namespace {

QPalette lightPalette(const QColor& accent) {
    QPalette p;
    p.setColor(QPalette::Window, QColor(0xf0, 0xf0, 0xf0));
    p.setColor(QPalette::WindowText, QColor(0x1a, 0x1a, 0x1a));
    p.setColor(QPalette::Base, Qt::white);
    p.setColor(QPalette::AlternateBase, QColor(0xf5, 0xf6, 0xf8));
    p.setColor(QPalette::Text, QColor(0x1a, 0x1a, 0x1a));
    p.setColor(QPalette::Button, QColor(0xf7, 0xf7, 0xf7));
    p.setColor(QPalette::ButtonText, QColor(0x1a, 0x1a, 0x1a));
    p.setColor(QPalette::ToolTipBase, QColor(0xff, 0xff, 0xdc));
    p.setColor(QPalette::ToolTipText, Qt::black);
    p.setColor(QPalette::PlaceholderText, QColor(0x80, 0x80, 0x80));
    p.setColor(QPalette::Highlight, accent);
    p.setColor(QPalette::HighlightedText, Qt::white);
    p.setColor(QPalette::Link, accent);
    return p;
}

QPalette darkPalette(const QColor& accent) {
    QPalette p;
    p.setColor(QPalette::Window, QColor(0x23, 0x26, 0x2d));
    p.setColor(QPalette::WindowText, QColor(0xe6, 0xe8, 0xec));
    p.setColor(QPalette::Base, QColor(0x1a, 0x1d, 0x23));
    p.setColor(QPalette::AlternateBase, QColor(0x2a, 0x2e, 0x36));
    p.setColor(QPalette::Text, QColor(0xe6, 0xe8, 0xec));
    p.setColor(QPalette::Button, QColor(0x2f, 0x34, 0x3d));
    p.setColor(QPalette::ButtonText, QColor(0xe6, 0xe8, 0xec));
    p.setColor(QPalette::ToolTipBase, QColor(0x2f, 0x34, 0x3d));
    p.setColor(QPalette::ToolTipText, QColor(0xe6, 0xe8, 0xec));
    p.setColor(QPalette::PlaceholderText, QColor(0x8a, 0x93, 0xa3));
    p.setColor(QPalette::Highlight, accent);
    p.setColor(QPalette::HighlightedText, QColor(0x0b, 0x12, 0x20));
    p.setColor(QPalette::Link, accent.lighter(130));
    for (auto role : {QPalette::WindowText, QPalette::Text, QPalette::ButtonText}) {
        p.setColor(QPalette::Disabled, role, QColor(0x6b, 0x72, 0x80));
    }
    return p;
}

}  // namespace

void applyTheme(QApplication& app, const gh::core::Settings& settings) {
    app.setStyle(QStyleFactory::create(QStringLiteral("Fusion")));
    bool dark = settings.theme == "dark";
    if (settings.theme == "system") dark = app.styleHints()->colorScheme() == Qt::ColorScheme::Dark;
    QColor accent = settings.accent.empty() ? QColor(0x2b, 0x6c, 0xb0) : QColor(QString::fromStdString(settings.accent));
    if (!accent.isValid()) accent = QColor(0x2b, 0x6c, 0xb0);
    if (dark && settings.accent.empty()) accent = QColor(0x4a, 0x90, 0xd9);   // the blue that reads on a dark window
    app.setPalette(dark ? darkPalette(accent) : lightPalette(accent));
}

void installTranslations(QApplication& app) {
    const QString dir = QString::fromStdWString((gh::core::AppPaths::executableDir() / "translations").wstring());
    auto* translator = new QTranslator(&app);
    if (translator->load(QLocale(), QStringLiteral("graphicshost"), QStringLiteral("_"), dir)) {
        app.installTranslator(translator);
    } else {
        delete translator;
    }
}

}  // namespace gh::ui
