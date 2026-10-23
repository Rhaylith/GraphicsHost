#include "ModelFixture.h"
#include "TestUtil.h"

#include <windows.h>
#include <winioctl.h>

#include "cli/Cli.h"
#include "engine/ImageCodec.h"
#include "gallery/GalleryStore.h"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <sstream>

using nlohmann::json;
namespace fs = std::filesystem;

namespace {

struct Result {
    int code = 0;
    std::string out;
    std::string err;
    json asJson() const { return json::parse(out); }
};

Result runCli(const fs::path& home, std::vector<std::string> args) {
    args.insert(args.begin(), "graphicshost");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::ostringstream out, err;
    Result r;
    r.code = gh::cli::run(static_cast<int>(argv.size()), argv.data(), out, err, home);
    r.out = out.str();
    r.err = err.str();
    return r;
}

// A checkpoint-looking file of a given size (sparse: no real data is written).
void fakeCheckpoint(const fs::path& file, uint64_t size) {
    fs::create_directories(file.parent_path());
    HANDLE h = CreateFileW(file.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    REQUIRE(h != INVALID_HANDLE_VALUE);
    DWORD returned = 0;
    DeviceIoControl(h, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr);  // so a huge size costs no disk space
    LARGE_INTEGER end;
    end.QuadPart = static_cast<LONGLONG>(size);
    REQUIRE(SetFilePointerEx(h, end, nullptr, FILE_BEGIN));
    REQUIRE(SetEndOfFile(h));
    CloseHandle(h);
}

constexpr uint64_t kSd15Size = 4265146304ull;
constexpr uint64_t kSdxlSize = 6938078334ull;

}  // namespace

TEST_CASE("cli gen: input problems are reported with the right exit code before anything loads", "[unit][cli][gen]") {
    gh::test::TempDir home;
    fakeCheckpoint(home.path() / "models" / "tiny.safetensors", 1000);

    auto r = runCli(home.path(), {"gen", "a cat"});
    CHECK(r.code == gh::cli::kUsage);
    CHECK(r.err.find("--model") != std::string::npos);

    r = runCli(home.path(), {"gen", "a cat", "-m", "nothere"});
    CHECK(r.code == gh::cli::kNotFound);
    CHECK(r.err.find("models") != std::string::npos);

    r = runCli(home.path(), {"gen", "a cat", "-m", "tiny", "--size", "500x500"});
    CHECK(r.code == gh::cli::kUsage);
    CHECK(r.err.find("width") != std::string::npos);

    CHECK(runCli(home.path(), {"gen", "a cat", "-m", "tiny", "--size", "big"}).code == gh::cli::kUsage);
    CHECK(runCli(home.path(), {"gen", "a cat", "-m", "tiny", "--sampler", "magic"}).code == gh::cli::kUsage);
    CHECK(runCli(home.path(), {"gen", "a cat", "-m", "tiny", "--device", "cuda0"}).code == gh::cli::kUsage);
    CHECK(runCli(home.path(), {"gen", "a cat", "-m", "tiny", "--preset", "nope"}).code == gh::cli::kNotFound);
    CHECK(runCli(home.path(), {"gen", "a cat", "-m", "tiny", "--format", "bmp"}).code == gh::cli::kUsage);
    CHECK(runCli(home.path(), {"gen", "-m", "tiny"}).code == gh::cli::kUsage);  // no prompt
}

TEST_CASE("cli ls: lists checkpoints and finds them by name or unique prefix", "[unit][cli][gen]") {
    gh::test::TempDir home;
    fakeCheckpoint(home.path() / "models" / "sd15.safetensors", kSd15Size);
    fakeCheckpoint(home.path() / "models" / "sub" / "sdxl-base.safetensors", kSdxlSize);
    gh::test::writeText(home.path() / "models" / "notes.txt", "not a checkpoint");

    const auto ls = runCli(home.path(), {"--json", "ls"});
    REQUIRE(ls.code == gh::cli::kOk);
    const auto models = ls.asJson()["models"];
    REQUIRE(models.size() == 2);
    CHECK(models[0]["file"] == "sd15.safetensors");
    CHECK(models[0]["family"] == "SD 1.x");
    CHECK(models[1]["file"] == "sdxl-base.safetensors");
    CHECK(models[1]["family"] == "SDXL");

    // by stem, by file name, by unique prefix; an ambiguous prefix is refused
    for (const char* name : {"sd15", "sd15.safetensors", "sdxl"}) {
        CHECK(runCli(home.path(), {"gen", "x", "-m", name, "--dry-run"}).code == gh::cli::kOk);
    }
    const auto ambiguous = runCli(home.path(), {"gen", "x", "-m", "sd", "--dry-run"});
    CHECK(ambiguous.code == gh::cli::kUsage);
    CHECK(ambiguous.err.find("matches 2") != std::string::npos);

    // a path works anywhere
    CHECK(runCli(home.path(), {"gen", "x", "-m", (home.path() / "models" / "sd15.safetensors").string(), "--dry-run"}).code == gh::cli::kOk);
}

TEST_CASE("cli ls: a custom models folder", "[unit][cli][gen]") {
    gh::test::TempDir home, elsewhere;
    fakeCheckpoint(elsewhere.path() / "m.safetensors", 1000);
    REQUIRE(runCli(home.path(), {"config", "set", "modelsDir", elsewhere.path().string()}).code == gh::cli::kOk);
    const auto ls = runCli(home.path(), {"--json", "ls"});
    CHECK(ls.asJson()["models"].size() == 1);
}

TEST_CASE("cli gen --dry-run: layers, size defaults and the memory estimate", "[unit][cli][gen]") {
    gh::test::TempDir home;
    fakeCheckpoint(home.path() / "models" / "sdxl.safetensors", kSdxlSize);

    auto plan = [&](std::vector<std::string> extra) {
        std::vector<std::string> args = {"--json", "gen", "a lighthouse", "-m", "sdxl", "--dry-run"};
        args.insert(args.end(), extra.begin(), extra.end());
        const auto r = runCli(home.path(), args);
        REQUIRE(r.code == gh::cli::kOk);
        return r.asJson();
    };

    auto p = plan({});
    CHECK(p["request"]["prompt"] == "a lighthouse");
    CHECK(p["request"]["width"] == 1024);       // SDXL size filled in
    CHECK(p["request"]["steps"] == 25);
    CHECK(p["fit"]["status"].is_string());
    CHECK(p["load"]["modelPath"].get<std::string>().find("sdxl.safetensors") != std::string::npos);

    // model defaults < preset < command line
    REQUIRE(runCli(home.path(), {"defaults", "set", "sdxl.safetensors", "steps", "33", "sampler", "euler_a"}).code == gh::cli::kOk);
    REQUIRE(runCli(home.path(), {"preset", "set", "slow", "steps", "44", "scheduler", "karras", "load.flashAttention", "true"}).code == gh::cli::kOk);
    p = plan({});
    CHECK(p["request"]["steps"] == 33);
    CHECK(p["request"]["sampler"] == "euler_a");
    p = plan({"--preset", "slow"});
    CHECK(p["request"]["steps"] == 44);
    CHECK(p["request"]["sampler"] == "euler_a");  // from the model defaults, the preset leaves it alone
    CHECK(p["request"]["scheduler"] == "karras");
    CHECK(p["load"]["flashAttention"] == true);
    p = plan({"--preset", "slow", "--steps", "55", "--sampler", "dpm++2m"});
    CHECK(p["request"]["steps"] == 55);
    CHECK(p["request"]["sampler"] == "dpm++2m");

    // every flag reaches the request or the load options
    p = plan({"--size", "768x512", "--cfg", "5.5", "--seed", "99", "-n", "blurry", "-b", "3", "--clip-skip", "2",
              "--format", "webp", "--vae-tiling", "--no-metadata", "--device", "cpu", "--threads", "8", "--offload",
              "--clip-on-cpu", "--vae-on-cpu"});
    CHECK(p["request"]["width"] == 768);
    CHECK(p["request"]["height"] == 512);
    CHECK(p["request"]["cfgScale"] == 5.5);
    CHECK(p["request"]["seed"] == 99);
    CHECK(p["request"]["negativePrompt"] == "blurry");
    CHECK(p["request"]["batchCount"] == 3);
    CHECK(p["request"]["clipSkip"] == 2);
    CHECK(p["request"]["format"] == "webp");
    CHECK(p["request"]["vaeTiling"] == true);
    CHECK(p["request"]["embedMetadata"] == false);
    CHECK(p["load"]["device"] == "cpu");
    CHECK(p["load"]["threads"] == 8);
    CHECK(p["load"]["offloadParamsToCpu"] == true);
    CHECK(p["load"]["clipOnCpu"] == true);
    CHECK(p["load"]["vaeOnCpu"] == true);
}

TEST_CASE("cli gen --dry-run: a model that cannot fit is refused unless told otherwise", "[unit][cli][gen]") {
    gh::test::TempDir home;
    fakeCheckpoint(home.path() / "models" / "huge.safetensors", 800ull * 1024 * 1024 * 1024);  // 800 GiB
    // not a dry run: the memory check comes first, so nothing is loaded
    const auto refused = runCli(home.path(), {"gen", "x", "-m", "huge", "--device", "cpu"});
    CHECK(refused.code == gh::cli::kOutOfMemory);
    CHECK(refused.err.find("RAM") != std::string::npos);
    CHECK(refused.err.find("--no-fit-check") != std::string::npos);
}

TEST_CASE("cli preset: set, list, show, export, import, delete", "[unit][cli][presets]") {
    gh::test::TempDir home;
    CHECK(runCli(home.path(), {"preset", "set", "portrait", "width", "512", "height", "768", "cfgScale", "6.5",
                               "load.flashAttention", "true"}).code == gh::cli::kOk);
    auto list = runCli(home.path(), {"--json", "preset", "list"});
    CHECK(list.asJson() == json::array({"portrait"}));

    auto shown = runCli(home.path(), {"preset", "show", "portrait"});
    REQUIRE(shown.code == gh::cli::kOk);
    CHECK(shown.asJson()["request"]["height"] == 768);
    CHECK(shown.asJson()["request"]["cfgScale"] == 6.5);
    CHECK(shown.asJson()["load"]["flashAttention"] == true);

    // setting again changes only the given keys
    CHECK(runCli(home.path(), {"preset", "set", "portrait", "steps", "40"}).code == gh::cli::kOk);
    shown = runCli(home.path(), {"preset", "show", "portrait"});
    CHECK(shown.asJson()["request"]["steps"] == 40);
    CHECK(shown.asJson()["request"]["width"] == 512);

    const auto file = (home.path() / "shared.json").string();
    CHECK(runCli(home.path(), {"preset", "export", "portrait", file}).code == gh::cli::kOk);
    CHECK(runCli(home.path(), {"preset", "import", file, "copy"}).code == gh::cli::kOk);
    CHECK(runCli(home.path(), {"--json", "preset", "list"}).asJson() == json::array({"copy", "portrait"}));

    CHECK(runCli(home.path(), {"preset", "delete", "copy"}).code == gh::cli::kOk);
    CHECK(runCli(home.path(), {"preset", "delete", "copy"}).code == gh::cli::kNotFound);
    CHECK(runCli(home.path(), {"preset", "show", "copy"}).code == gh::cli::kNotFound);
}

TEST_CASE("cli preset: bad input is a usage error", "[unit][cli][presets]") {
    gh::test::TempDir home;
    CHECK(runCli(home.path(), {"preset", "set", "x", "sampler", "magic"}).code == gh::cli::kUsage);   // invalid value
    CHECK(runCli(home.path(), {"preset", "set", "x", "nosuchkey", "1"}).code == gh::cli::kUsage);
    CHECK(runCli(home.path(), {"preset", "set", "x", "steps", "many"}).code == gh::cli::kUsage);
    CHECK(runCli(home.path(), {"preset", "set", "x", "steps"}).code == gh::cli::kUsage);              // odd number of args
    CHECK(runCli(home.path(), {"preset", "set", "../evil", "steps", "5"}).code == gh::cli::kUsage);   // unsafe name
    CHECK(runCli(home.path(), {"preset", "set", "x", "load.modelPath", "m.safetensors"}).code == gh::cli::kUsage);
    CHECK(runCli(home.path(), {"preset", "frobnicate"}).code == gh::cli::kUsage);
    CHECK(runCli(home.path(), {"--json", "preset", "list"}).asJson().empty());  // nothing was saved
}

TEST_CASE("cli defaults: per checkpoint", "[unit][cli][presets]") {
    gh::test::TempDir home;
    CHECK(runCli(home.path(), {"defaults", "set", "a.safetensors", "steps", "30", "load.device", "cpu"}).code == gh::cli::kOk);
    CHECK(runCli(home.path(), {"--json", "defaults", "list"}).asJson() == json::array({"a.safetensors"}));
    const auto shown = runCli(home.path(), {"defaults", "show", "a.safetensors"});
    CHECK(shown.asJson()["request"]["steps"] == 30);
    CHECK(shown.asJson()["load"]["device"] == "cpu");
    CHECK(runCli(home.path(), {"defaults", "clear", "a.safetensors"}).code == gh::cli::kOk);
    CHECK(runCli(home.path(), {"defaults", "show", "a.safetensors"}).code == gh::cli::kNotFound);
    CHECK(runCli(home.path(), {"defaults", "set", "a.safetensors", "steps", "9999"}).code == gh::cli::kUsage);
}

TEST_CASE("cli gallery: list, show, rm, rescan", "[unit][cli][gallery]") {
    gh::test::TempDir home;
    {
        gh::gallery::GalleryStore store(home.path() / "outputs");
        gh::engine::Image image{4, 4, 3, std::vector<uint8_t>(48, 90)};
        gh::engine::GenerationRequest r;
        r.prompt = "a lighthouse";
        r.width = r.height = 4;
        gh::engine::ModelInfo model;
        model.modelPath = "m.safetensors";
        store.add(image, r, 1, model, 0.5);
        r.prompt = "a forest";
        store.add(image, r, 2, model, 0.5);
    }
    auto list = runCli(home.path(), {"--json", "gallery", "list"});
    REQUIRE(list.code == gh::cli::kOk);
    CHECK(list.asJson()["total"] == 2);
    CHECK(list.asJson()["images"][0]["prompt"] == "a forest");  // newest first

    list = runCli(home.path(), {"--json", "gallery", "list", "lighthouse"});
    CHECK(list.asJson()["images"].size() == 1);

    const auto shown = runCli(home.path(), {"gallery", "show", "1"});
    REQUIRE(shown.code == gh::cli::kOk);
    CHECK(shown.asJson()["request"]["prompt"] == "a lighthouse");
    CHECK(fs::exists(shown.asJson()["file"].get<std::string>()));

    CHECK(runCli(home.path(), {"gallery", "show", "999"}).code == gh::cli::kNotFound);
    CHECK(runCli(home.path(), {"gallery", "show", "abc"}).code == gh::cli::kUsage);
    CHECK(runCli(home.path(), {"gallery", "rm", "1"}).code == gh::cli::kOk);
    CHECK(runCli(home.path(), {"--json", "gallery", "list"}).asJson()["total"] == 1);
    CHECK(runCli(home.path(), {"gallery", "rescan"}).code == gh::cli::kOk);
}

TEST_CASE("cli gen: generates, writes files and fills the gallery", "[model][cli][gen]") {
    const auto model = gh::test::findTestModel();
    if (!model) SKIP("no SD1.5 checkpoint in models/");
    gh::test::TempDir home;
    // the CPU runtime keeps the test independent of the GPU
    REQUIRE(runCli(home.path(), {"config", "set", "backend", "cpu"}).code == gh::cli::kOk);
    const auto outDir = home.path() / "out";

    auto r = runCli(home.path(), {"--json", "gen", "a red apple", "-m", model->string(), "--steps", "2", "--size", "128x128",
                                  "--seed", "11", "-b", "2", "--format", "jpeg", "-o", outDir.string()});
    if (r.code != gh::cli::kOk) WARN(r.err);
    REQUIRE(r.code == gh::cli::kOk);
    const json result = r.asJson();
    REQUIRE(result["images"].size() == 2);
    CHECK(result["images"][0]["seed"] == 11);
    CHECK(result["images"][1]["seed"] == 12);
    CHECK(result["runtime"].get<std::string>().rfind("cpu", 0) == 0);

    for (const auto& image : result["images"]) {
        const fs::path file = image["file"].get<std::string>();
        REQUIRE(fs::exists(file));
        CHECK(file.extension() == ".jpg");
        std::ifstream in(file, std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(gh::engine::decodeImage(bytes).width == 128);
        // the parameters travel inside the file
        const auto meta = json::parse(gh::engine::readImageMetadata(bytes).value());
        CHECK(meta["request"]["prompt"] == "a red apple");
        CHECK(meta["request"]["seed"] == image["seed"]);
        CHECK(fs::exists(image["galleryPath"].get<std::string>()));
    }
    CHECK(runCli(home.path(), {"--json", "gallery", "list"}).asJson()["total"] == 2);

    // --no-gallery and a single file target
    r = runCli(home.path(), {"--json", "gen", "a red apple", "-m", model->string(), "--steps", "2", "--size", "128x128",
                             "--seed", "11", "--no-gallery", "-o", (home.path() / "one.png").string()});
    REQUIRE(r.code == gh::cli::kOk);
    CHECK(fs::exists(home.path() / "one.png"));
    CHECK(runCli(home.path(), {"--json", "gallery", "list"}).asJson()["total"] == 2);  // unchanged
}

TEST_CASE("cli gen: the same flags give the same picture as the gallery record says", "[model][cli][gen]") {
    const auto model = gh::test::findTestModel();
    if (!model) SKIP("no SD1.5 checkpoint in models/");
    gh::test::TempDir home;
    REQUIRE(runCli(home.path(), {"config", "set", "backend", "cpu"}).code == gh::cli::kOk);
    const auto first = runCli(home.path(), {"--json", "gen", "a red apple", "-m", model->string(), "--steps", "2", "--size", "128x128", "-o", (home.path() / "a.png").string()});
    REQUIRE(first.code == gh::cli::kOk);
    const auto seed = first.asJson()["images"][0]["seed"].get<int64_t>();
    CHECK(seed >= 0);
    const auto second = runCli(home.path(), {"--json", "gen", "a red apple", "-m", model->string(), "--steps", "2", "--size", "128x128", "--seed", std::to_string(seed), "-o", (home.path() / "b.png").string()});
    REQUIRE(second.code == gh::cli::kOk);
    CHECK(gh::test::readText(home.path() / "a.png") != "");
    // identical pixels: compare the decoded images (the files differ in the embedded metadata only by nothing, but compare pixels anyway)
    auto pixels = [&](const fs::path& p) {
        std::ifstream in(p, std::ios::binary);
        return gh::engine::decodeImage(std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>())).data;
    };
    CHECK(pixels(home.path() / "a.png") == pixels(home.path() / "b.png"));
}
