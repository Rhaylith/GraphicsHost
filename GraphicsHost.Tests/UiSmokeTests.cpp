#include "TestUtil.h"

#include <fstream>

#include "core/Paths.h"

#include <catch2/catch_test_macros.hpp>

#include <windows.h>

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Starts the real desktop executable with the offscreen Qt platform: the window must open, every tab must render, and
// the program must end by itself. (The window is never shown on screen.)
TEST_CASE("ui smoke: the desktop app opens and every tab renders", "[integration][ui]") {
    const fs::path exe = gh::core::AppPaths::executableDir() / "GraphicsHost.exe";
    if (!fs::exists(exe)) SKIP("GraphicsHost.exe is not next to the test executable");

    gh::test::TempDir home, shots;
    SetEnvironmentVariableW(L"QT_QPA_PLATFORM", L"offscreen");
    SetEnvironmentVariableW(L"QT_QPA_FONTDIR", L"C:/Windows/Fonts");
    SetEnvironmentVariableW(L"GRAPHICSHOST_HOME", home.path().c_str());
    // no API server in this run: the port may be taken, and the test must not open one
    fs::create_directories(home.path());
    {
        std::ofstream settings(home.path() / "settings.json");
        settings << R"({"server":{"enabled":false}})";
    }

    std::wstring command = L"\"" + exe.wstring() + L"\" --screenshot-dir \"" + shots.path().wstring() + L"\"";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    REQUIRE(CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process));
    const DWORD waited = WaitForSingleObject(process.hProcess, 60000);
    DWORD exitCode = 1;
    if (waited == WAIT_TIMEOUT) TerminateProcess(process.hProcess, 99);
    GetExitCodeProcess(process.hProcess, &exitCode);
    CloseHandle(process.hProcess);
    CloseHandle(process.hThread);
    SetEnvironmentVariableW(L"QT_QPA_PLATFORM", nullptr);
    SetEnvironmentVariableW(L"QT_QPA_FONTDIR", nullptr);
    SetEnvironmentVariableW(L"GRAPHICSHOST_HOME", nullptr);

    REQUIRE(waited == WAIT_OBJECT_0);
    CHECK(exitCode == 0);
    std::vector<std::string> names;
    for (const auto& e : fs::directory_iterator(shots.path())) {
        names.push_back(e.path().filename().string());
        CHECK(fs::file_size(e.path()) > 5000);   // a blank or failed render is tiny
    }
    CHECK(names.size() == 8);   // Generate, Gallery, My Models, Discover, Settings, Logs, Monitor and the mask editor
}
