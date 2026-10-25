#include "RepoRoot.h"
#include "graph/ReferenceDiff.h"
#include "graph/Tips.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>

using namespace gh::graph;

namespace {
bool mentions(const std::vector<std::string>& list, const std::string& text) {
    return std::any_of(list.begin(), list.end(), [&](const std::string& s) { return s.find(text) != std::string::npos; });
}

std::string firstOf(const std::vector<std::string>& list) { return list.empty() ? std::string("none") : list.front(); }

Registry sampleRegistry() {
    Registry r;
    r.add(NodeDef("t.sample")
              .display("Sample")
              .category("Sampling")
              .input(Model("model").tip("Which model to use."))
              .input(Latent("latent").optional().tip("Where to start."))
              .input(Int("seed").range(0, 1e15).def(0).seedControl().tip(tips::seed))
              .input(Int("steps").range(1, 10000).def(20).tip(tips::steps))
              .input(Float("cfg").range(0, 100).step(0.1).def(7.0).tip(tips::common("cfg", "Node note.")))
              .input(tips::withOptions(Combo("sampler").def("euler").tip(tips::sampler), tips::samplerOptions))
              .input(Bool("flag").def(true).tip("Turns it on. Example: off."))
              .input(Combo("lora").modelFolder("loras").tip("A LoRA. Example: style.safetensors.")));
    r.add(NodeDef("t.note").display("Note").category("Utility").description("A comment on the canvas."));
    return r;
}
}  // namespace

TEST_CASE("reference: the node tables of the document are read back", "[unit][reference]") {
    const auto root = repoRoot();
    if (root.empty()) SKIP("the repository is not above the test executable");
    const ReferenceDoc doc = parseReferenceDoc(readTextFile(root / "docs" / "workflow-node-reference.md"));
    CHECK(doc.nodes.size() > 150);
    const auto it = std::find_if(doc.nodes.begin(), doc.nodes.end(), [](const DocNode& n) { return n.id == "loader.checkpoint"; });
    REQUIRE(it != doc.nodes.end());
    CHECK(it->display == "Load Checkpoint");
    CHECK(it->section == "N1");
    REQUIRE(it->params.size() == 5);
    CHECK(it->params[0].names == std::vector<std::string>{"checkpoint"});
    CHECK(it->params[4].names == std::vector<std::string>{"idleUnloadMinutes"});
    CHECK(it->params[4].type.find("**10**") != std::string::npos);
    const auto grouped = std::find_if(doc.nodes.begin(), doc.nodes.end(), [](const DocNode& n) { return n.id == "cond.combine"; });
    REQUIRE(grouped != doc.nodes.end());
    CHECK(grouped->params[0].names == std::vector<std::string>{"a", "b"});
    const auto loose = std::find_if(doc.nodes.begin(), doc.nodes.end(), [](const DocNode& n) { return n.id == "loader.textEncoders"; });
    REQUIRE(loose != doc.nodes.end());
    CHECK(loose->params[0].loose);
}

TEST_CASE("reference: nodes written like the document pass the diff", "[unit][reference]") {
    const auto root = repoRoot();
    if (root.empty()) SKIP("the repository is not above the test executable");
    const ReferenceDoc doc = parseReferenceDoc(readTextFile(root / "docs" / "workflow-node-reference.md"));

    Registry r;
    r.add(NodeDef("cond.average")
              .display("Average Conditioning")
              .category("Conditioning")
              .input(Conditioning("to").tip("The two prompts to blend into one before sampling."))
              .input(Conditioning("from").tip("The two prompts to blend into one before sampling."))
              .input(Float("toStrength").range(0, 1).step(0.01).def(1.0).tip("1.0 keeps only *to*, 0.0 only *from*. Example: 0.5 for a cat that is half fox.")));
    r.add(NodeDef("loader.hypernetwork")
              .display("Load Hypernetwork")
              .category("Loaders")
              .input(Combo("hypernetwork").modelFolder("hypernetworks").tip("An older style add-on that modifies attention layers. Example: `anime_style.pt`."))
              .input(Float("strength").range(-10, 10).step(0.01).def(1.0).tip(tips::strength)));
    const auto diff = diffReference(r, doc);
    INFO(firstOf(diff.problems));
    CHECK(diff.problems.empty());
    CHECK(diff.undocumented.empty());
    CHECK(diff.pending.size() > 100);   // the rest of the document has no code yet

    // the same nodes with a wrong range, default, folder and tooltip are all reported
    Registry bad;
    bad.add(NodeDef("cond.average")
                .display("Average Cond")
                .category("Conditioning")
                .input(Conditioning("to").tip("Something else."))
                .input(Float("toStrength").range(0, 2).step(0.01).def(0.5).tip("1.0 keeps only *to*, 0.0 only *from*. Example: 0.5 for a cat that is half fox."))
                .input(Float("extra").tip("Not in the document. Example: 1.")));
    bad.add(NodeDef("loader.hypernetwork")
                .display("Load Hypernetwork")
                .category("Loaders")
                .input(Combo("hypernetwork").modelFolder("loras").tip("An older style add-on that modifies attention layers. Example: `anime_style.pt`."))
                .input(Float("strength").range(-10, 10).step(0.01).def(1.0).tip("Own text. Example: 1.")));
    const auto d2 = diffReference(bad, doc);
    CHECK(mentions(d2.problems, "cond.average: the display name differs"));
    CHECK(mentions(d2.problems, "cond.average.to: the tooltip differs"));
    CHECK(mentions(d2.problems, "cond.average.toStrength: the maximum differs"));
    CHECK(mentions(d2.problems, "cond.average.toStrength: the default differs"));
    CHECK(mentions(d2.problems, "cond.average.extra: the reference does not list this input"));
    CHECK(mentions(d2.problems, "cond.average.from: the reference lists a parameter that does not exist"));
    CHECK(mentions(d2.problems, "loader.hypernetwork.hypernetwork: the model folder differs"));
    CHECK(mentions(d2.problems, "loader.hypernetwork.strength: the tooltip differs"));
}

TEST_CASE("reference: rendering the registry and diffing it gives no problems", "[unit][reference]") {
    const Registry r = sampleRegistry();
    const std::string text = renderReference(r);
    CHECK(text.find("| Sample (`t.sample`) | model | MODEL |") != std::string::npos);
    CHECK(text.find("latent | LATENT (optional) |") != std::string::npos);
    CHECK(text.find("| seed | int 0 \xE2\x80\xA6 1000000000000000 \xC2\xB7 **0** |") != std::string::npos);
    CHECK(text.find("\xE2\x86\x92 C.steps") != std::string::npos);
    CHECK(text.find("combo euler, euler cfg++") != std::string::npos);
    CHECK(text.find("model list *loras*") != std::string::npos);
    CHECK(text.find("| Note (`t.note`) |") != std::string::npos);

    const ReferenceDoc doc = parseReferenceDoc(text);
    REQUIRE(doc.nodes.size() == 2);
    const auto diff = diffReference(r, doc);
    INFO(firstOf(diff.problems));
    CHECK(diff.problems.empty());
    CHECK(diff.undocumented.empty());
    CHECK(diff.pending.empty());
}

TEST_CASE("reference: the built-in nodes match the reference document", "[unit][reference]") {
    const auto root = repoRoot();
    if (root.empty()) SKIP("the repository is not above the test executable");
    const ReferenceDoc doc = parseReferenceDoc(readTextFile(root / "docs" / "workflow-node-reference.md"));
    const auto diff = diffReference(Registry::builtin(), doc);
    INFO(firstOf(diff.problems));
    CHECK(diff.problems.empty());
    // regenerate the rows of a node with renderReference() and paste them into docs/workflow-node-reference.md
    INFO(firstOf(diff.undocumented));
    CHECK(diff.undocumented.empty());
}
