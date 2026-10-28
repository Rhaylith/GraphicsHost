#include "ModelFixture.h"
#include "core/Paths.h"
#include "graph/Executor.h"
#include "host/GraphService.h"
#include "host/RuntimeInference.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>

using namespace gh;
using nlohmann::json;

// The graph nodes on the real Vulkan runtime: checkpoint -> two Text Encode nodes -> Sampler, then the VAE decode of the service.
// Needs a checkpoint (see ModelFixture.h) and runtimes/sd-vulkan.dll next to the test executable.
TEST_CASE("graph on the runtime: SD 1.5 text to latent to picture", "[model][vulkan][graph-run]") {
    const auto checkpoint = test::findTestModel();
    if (!checkpoint) SKIP("no SD 1.5 checkpoint (see ModelFixture.h)");
    const auto dll = core::AppPaths::executableDir() / "runtimes" / "sd-vulkan.dll";
    if (!std::filesystem::exists(dll)) SKIP("sd-vulkan.dll is not next to the test executable");

    models::ModelLibrary library({checkpoint->parent_path()});
    host::LibraryNodeServices services(library);
    host::RuntimeInference inference([&](std::string& error) -> std::shared_ptr<engine::Runtime> {
        std::shared_ptr<engine::Runtime> rt = engine::Runtime::load(dll, error);
        return rt;
    });
    services.setInference(&inference);

    graph::Registry registry;
    for (const graph::NodeDef* d : graph::Registry::builtin().all()) registry.add(*d);
    registry.add(graph::NodeDef("t.emptyLatent").output(graph::Out("latent", graph::ValueType::Latent)).run([](graph::NodeContext&) {
        return graph::NodeResult{{graph::Value::makeTensor(graph::ValueType::Latent, tensor::Tensor({1, 4, 64, 64}, 0.0f))}};
    }));
    const json job = {{"format", "graphicshost.job"},
                      {"version", 1},
                      {"nodes",
                       {{"1", {{"type", "loader.checkpoint"}, {"inputs", {{"checkpoint", checkpoint->filename().string()}}}}},
                        {"2", {{"type", "text.encode"}, {"inputs", {{"text", "a lighthouse on a cliff at sunset"}, {"textEncoder", {"1", 1}}}}}},
                        {"3", {{"type", "text.encode"}, {"inputs", {{"text", ""}, {"textEncoder", {"1", 1}}}}}},
                        {"4", {{"type", "t.emptyLatent"}, {"inputs", json::object()}}},
                        {"5", {{"type", "sampling.sample"},
                               {"inputs", {{"model", {"1", 0}}, {"positive", {"2", 0}}, {"negative", {"3", 0}}, {"latent", {"4", 0}}, {"seed", 42}, {"steps", 4}, {"cfg", 7.5}, {"sampler", "euler"}, {"scheduler", "normal"}}}}}}},
                      {"outputs", json::array({"5"})}};
    graph::OutputCache cache(*graph::CachePolicy::parse("none"));
    graph::RunOptions options;
    options.services = &services;
    const auto result = graph::Executor(registry, cache).run(graph::jobFromJson(job), options);
    INFO(result.errorMessage);
    REQUIRE(result.ok());
    const tensor::Tensor& latent = result.outputs.at("5")[0].asTensor();
    REQUIRE(latent.shape() == std::vector<int64_t>{1, 4, 64, 64});
    bool finite = true, varied = false;
    for (float v : latent.values()) {
        finite = finite && std::isfinite(v);
        varied = varied || std::fabs(v - latent.values()[0]) > 1e-3f;
    }
    CHECK(finite);
    CHECK(varied);

    // the VAE of the checkpoint turns it into a 512 x 512 picture with some contrast
    graph::VaeHandle vae;
    vae.fromCheckpoint = true;
    graph::ModelHandle model;
    model.file.path = checkpoint->string();
    graph::LatentView view{latent.data(), 1, 4, 64, 64};
    const tensor::Tensor image = inference.decode(vae, model, view);
    REQUIRE(image.shape() == std::vector<int64_t>{1, 512, 512, 3});
    float lo = 1e9f, hi = -1e9f;
    for (float v : image.values()) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    CHECK(hi > lo);
    inference.release();
    CHECK_FALSE(inference.loaded());
}
