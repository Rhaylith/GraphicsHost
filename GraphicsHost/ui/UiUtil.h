#pragma once

#include "engine/ImageEngine.h"

#include <QIcon>
#include <QImage>
#include <QString>

#include <cstdint>
#include <string>

namespace gh::ui {

inline QString q(const std::string& s) { return QString::fromStdString(s); }
inline std::string s(const QString& text) { return text.toStdString(); }

// "4.27 GiB", "812 MiB"
QString humanSize(uint64_t bytes);

// A detached copy of the pixels (the engine's buffer is not kept alive by the QImage).
QImage toQImage(const engine::Image& image);

// The application icon, drawn in code so no resource file is needed.
QIcon appIcon();

}  // namespace gh::ui
