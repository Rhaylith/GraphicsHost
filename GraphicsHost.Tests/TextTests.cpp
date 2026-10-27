#include "RepoRoot.h"
#include "text/PromptParser.h"

#include <catch2/catch_test_macros.hpp>

#include <set>

using namespace gh::text;

namespace {
const ClipTokenizer& tokenizer() {
    static const ClipTokenizer t = ClipTokenizer::fromFile(repoRoot() / "data" / "text" / "clip_merges.txt");
    return t;
}
std::vector<std::string> texts(const ParsedPrompt& p) {
    std::vector<std::string> out;
    for (const auto& piece : p.pieces) out.push_back(piece.kind == PromptPiece::Kind::Break ? "<BREAK>" : piece.text);
    return out;
}
std::vector<float> weights(const ParsedPrompt& p) {
    std::vector<float> out;
    for (const auto& piece : p.pieces) out.push_back(piece.weight);
    return out;
}
}  // namespace

TEST_CASE("clip tokenizer: known token ids", "[unit][text][tokenizer]") {
    if (repoRoot().empty()) SKIP("the repository is not above the test executable");
    const auto& t = tokenizer();
    CHECK(t.encode("a photo of a cat") == std::vector<int>{320, 1125, 539, 320, 2368});
    CHECK(t.encode("hello world") == std::vector<int>{3306, 1002});
    CHECK(t.encode("cat, dog") == std::vector<int>{2368, 267, 1929});
    CHECK(t.encode("A PHOTO of a CAT") == t.encode("a photo of a cat"));   // lower-cased
    CHECK(t.encode("  a \n\t photo  ") == t.encode("a photo"));             // whitespace collapsed
    CHECK(t.encode("").empty());
    CHECK(t.encode("<|startoftext|>a<|endoftext|>") == std::vector<int>{ClipTokenizer::kStart, 320, ClipTokenizer::kEnd});
    CHECK(t.encode("it's") == t.encode("it 's"));                          // contractions split off
    CHECK(t.encode("123").size() == 3);                                    // digits one by one
}

TEST_CASE("clip tokenizer: unusual input becomes tokens and decodes back", "[unit][text][tokenizer]") {
    if (repoRoot().empty()) SKIP("the repository is not above the test executable");
    const auto& t = tokenizer();
    CHECK(t.decode(t.encode("a photo of a cat")) == "a photo of a cat");
    CHECK(t.decode(t.encode("supercalifragilisticexpialidocious")) == "supercalifragilisticexpialidocious");   // split into many pieces
    CHECK(t.encode("supercalifragilisticexpialidocious").size() > 3);
    CHECK(t.decode(t.encode("caf\xC3\xA9 na\xC3\xAFve")) == "caf\xC3\xA9 na\xC3\xAFve");   // non-ASCII letters keep their bytes
    CHECK_FALSE(t.encode("\xF0\x9F\x98\x80").empty());                                       // an emoji is a few byte tokens
    for (const int id : t.encode("The quick brown fox, 12 jumps!")) CHECK((id >= 0 && id < ClipTokenizer::kVocabSize));
    CHECK_THROWS(ClipTokenizer("#version\na b\n"));   // an incomplete vocabulary is refused
}

TEST_CASE("prompt: comments are removed", "[unit][text][prompt]") {
    CHECK(removeComments("a cat // not this\nsitting") == "a cat \nsitting");
    CHECK(removeComments("a /* hidden */ cat") == "a   cat");
    CHECK(removeComments("a /* open") == "a ");
    CHECK(removeComments("path\\//ok") == "path\\//ok");   // an escaped slash starts no comment
}

TEST_CASE("prompt: choices are resolved by the seed", "[unit][text][prompt]") {
    CHECK(resolveChoices("a {red|green|blue} car", 1) == resolveChoices("a {red|green|blue} car", 1));   // same seed, same pick
    std::set<std::string> seen;
    for (uint64_t seed = 0; seed < 40; ++seed) {
        const std::string r = resolveChoices("a {red|green|blue} car", seed);
        CHECK((r == "a red car" || r == "a green car" || r == "a blue car"));
        seen.insert(r);
    }
    CHECK(seen.size() == 3);   // all of them come up
    for (uint64_t seed = 0; seed < 20; ++seed) {
        const std::string r = resolveChoices("{a {x|y}|b}", seed);
        CHECK((r == "ax" || r == "ay" || r == "b"));
    }
    CHECK(resolveChoices("{only}", 3) == "{only}");                  // no bar: left alone
    CHECK(resolveChoices("{a|b", 3) == "{a|b");                      // unclosed
    CHECK(resolveChoices("\\{a|b\\}", 3) == "\\{a|b\\}");            // escaped braces are text
    CHECK(resolveChoices("{|x}", 0).size() <= 1);                    // an empty option is allowed
}

TEST_CASE("prompt: weights, groups and escapes", "[unit][text][prompt]") {
    auto p = parsePrompt("a (red:1.3) car", 0);
    CHECK(texts(p) == std::vector<std::string>{"a ", "red", " car"});
    CHECK(weights(p) == std::vector<float>{1.0f, 1.3f, 1.0f});

    p = parsePrompt("(red) car", 0);
    CHECK(weights(p)[0] == 1.1f);

    p = parsePrompt("a [dull] car", 0);
    CHECK(weights(p)[1] == 1.0f / 1.1f);

    p = parsePrompt("((very red:1.5) car:1.2)", 0);   // nested weights multiply
    REQUIRE(p.pieces.size() == 2);
    CHECK(texts(p) == std::vector<std::string>{"very red", " car"});
    CHECK(p.pieces[0].weight == 1.5f * 1.2f);
    CHECK(p.pieces[1].weight == 1.2f);

    p = parsePrompt("\\(not weighted\\) and \\[this\\]", 0);
    CHECK(texts(p) == std::vector<std::string>{"(not weighted) and [this]"});
    CHECK(p.pieces[0].weight == 1.0f);

    p = parsePrompt("unclosed (paren", 0);   // no match: plain text
    CHECK(texts(p) == std::vector<std::string>{"unclosed (paren"});

    p = parsePrompt("a (b:0.5) c (d:0.5)", 0);
    CHECK(texts(p) == std::vector<std::string>{"a ", "b", " c ", "d"});

    p = parsePrompt("cat // comment\n{x|x}", 0);   // all three stages together
    CHECK(texts(p) == std::vector<std::string>{"cat \nx"});
}

TEST_CASE("prompt: embeddings and BREAK", "[unit][text][prompt]") {
    auto p = parsePrompt("a cat embedding:bad-hands, (embedding:style:0.8) embedding:other:0.5 x", 0);
    std::vector<std::pair<std::string, float>> found;
    for (const auto& piece : p.pieces)
        if (piece.kind == PromptPiece::Kind::Embedding) found.push_back({piece.text, piece.weight});
    REQUIRE(found.size() == 3);
    CHECK(found[0] == std::make_pair(std::string("bad-hands"), 1.0f));
    CHECK(found[1] == std::make_pair(std::string("style"), 0.8f));
    CHECK(found[2] == std::make_pair(std::string("other"), 0.5f));
    CHECK(p.embeddings == std::vector<std::string>{"bad-hands", "style", "other"});

    p = parsePrompt("one BREAK two\nBREAK three NOBREAK BREAKS", 0);
    CHECK(texts(p) == std::vector<std::string>{"one ", "<BREAK>", " two\n", "<BREAK>", " three NOBREAK BREAKS"});
}

TEST_CASE("chunks: 75 tokens each, wrapped in start and end, padded to 77", "[unit][text][chunks]") {
    if (repoRoot().empty()) SKIP("the repository is not above the test executable");
    const auto& t = tokenizer();
    auto chunks = buildChunks(t, parsePrompt("a photo of a cat", 0));
    REQUIRE(chunks.chunks.size() == 1);
    const Chunk& c = chunks.chunks[0];
    REQUIRE(c.ids.size() == 77);
    CHECK(c.ids[0] == ClipTokenizer::kStart);
    CHECK(std::vector<int>(c.ids.begin() + 1, c.ids.begin() + 6) == std::vector<int>{320, 1125, 539, 320, 2368});
    CHECK(c.ids[6] == ClipTokenizer::kEnd);
    CHECK(c.ids[7] == ClipTokenizer::kEnd);   // padding with the end token
    CHECK(c.weights.size() == 77);

    chunks = buildChunks(t, parsePrompt("cat", 0), 0);   // the second SDXL encoder pads with 0
    CHECK(chunks.chunks[0].ids[3] == 0);

    // weights follow the words
    chunks = buildChunks(t, parsePrompt("cat (dog:1.4)", 0));
    CHECK(chunks.chunks[0].weights[1] == 1.0f);
    CHECK(chunks.chunks[0].weights[2] == 1.4f);

    // an empty prompt is one chunk of nothing
    chunks = buildChunks(t, parsePrompt("", 0));
    REQUIRE(chunks.chunks.size() == 1);
    CHECK(chunks.chunks[0].ids[1] == ClipTokenizer::kEnd);

    // 100 words are 75 + 25 tokens in two chunks; BREAK forces a new chunk early
    std::string many;
    for (int i = 0; i < 100; ++i) many += "cat ";
    chunks = buildChunks(t, parsePrompt(many, 0));
    REQUIRE(chunks.chunks.size() == 2);
    CHECK(chunks.chunks[0].ids[75] == 2368);
    CHECK(chunks.chunks[0].ids[76] == ClipTokenizer::kEnd);
    CHECK(chunks.chunks[1].ids[25] == 2368);
    CHECK(chunks.chunks[1].ids[26] == ClipTokenizer::kEnd);
    chunks = buildChunks(t, parsePrompt("cat BREAK dog", 0));
    REQUIRE(chunks.chunks.size() == 2);
    CHECK(chunks.chunks[1].ids[1] == 1929);
}

TEST_CASE("chunks: embeddings take placeholder ids and keep together", "[unit][text][chunks]") {
    if (repoRoot().empty()) SKIP("the repository is not above the test executable");
    const auto& t = tokenizer();
    const EmbeddingLookup lookup = [](const std::string& name) { return name == "known" ? 4 : 0; };
    auto chunks = buildChunks(t, parsePrompt("cat embedding:known:0.7 dog embedding:unknown", 0), ClipTokenizer::kEnd, lookup);
    REQUIRE(chunks.embeddings.size() == 1);
    const EmbeddingUse& e = chunks.embeddings[0];
    CHECK(e.name == "known");
    CHECK(e.vectors == 4);
    CHECK(e.weight == 0.7f);
    CHECK(e.chunk == 0);
    CHECK(e.position == 2);   // after the start token and "cat"
    const Chunk& c = chunks.chunks[0];
    for (int v = 0; v < 4; ++v) {
        CHECK(c.ids[static_cast<size_t>(e.position + v)] == e.firstId + v);
        CHECK(c.weights[static_cast<size_t>(e.position + v)] == 0.7f);
    }
    CHECK(e.firstId >= kEmbeddingBase);
    CHECK(c.ids[static_cast<size_t>(e.position + 4)] == 1929);   // dog
    // the unknown one was typed as words: "embedding" ":" "unknown" appear as text tokens
    CHECK(c.ids.size() == 77);

    // an embedding that does not fit the room left starts a new chunk
    std::string text;
    for (int i = 0; i < 73; ++i) text += "cat ";
    chunks = buildChunks(t, parsePrompt(text + "embedding:known", 0), ClipTokenizer::kEnd, lookup);
    REQUIRE(chunks.chunks.size() == 2);
    CHECK(chunks.embeddings[0].chunk == 1);
    CHECK(chunks.embeddings[0].position == 1);
}
