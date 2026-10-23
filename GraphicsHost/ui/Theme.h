#pragma once

#include "core/Settings.h"

#include <QString>

class QApplication;

namespace gh::ui {

// The palette for a theme ("light", "dark"; "system" follows Windows) and an accent colour ("" = the default blue).
// Uses the Fusion style, which paints the same everywhere and honours the palette.
void applyTheme(QApplication& app, const gh::core::Settings& settings);

// Loads translations/graphicshost_<language>.qm from next to the executable for the system language, when there is one.
// Strings without a translation stay in English.
void installTranslations(QApplication& app);

}  // namespace gh::ui
