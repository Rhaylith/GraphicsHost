# Translations

The window's text is written in English with `tr("...")`. To translate it:

1. Update the catalogue of source strings (adds new strings, keeps finished translations):
   `lupdate.exe ui\*.cpp main.cpp -source-language en -target-language <code> -ts translations\graphicshost_<code>.ts`
   (`lupdate.exe` is in `<vcpkg>\installed\x64-windows\tools\Qt6\bin`).
2. Translate the `<translation>` entries in the `.ts` file with Qt Linguist (or by hand) and remove `type="unfinished"`.
3. Build: the project compiles every `translations\*.ts` into `translations\*.qm` next to the exe, and the application
   loads `graphicshost_<language>.qm` for the Windows display language at start-up. Strings without a translation stay
   in English.

`graphicshost_de.ts` is the German catalogue; only the main tab and button names are translated so far. The browser UI is English only.
