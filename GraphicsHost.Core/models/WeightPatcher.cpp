#include "models/WeightPatcher.h"

#include "models/SafetensorsReader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <execution>
#include <numeric>
#include <regex>
#include <set>
#include <stdexcept>

namespace gh::models {

using tensor::Tensor;
namespace fs = std::filesystem;

namespace {

// ---- small linear algebra ------------------------------------------------------------------------------------------------

template <typename Fn>
void parallelRows(int64_t rows, Fn&& fn) {
    if (rows < 8) {
        for (int64_t i = 0; i < rows; ++i) fn(i);
        return;
    }
    std::vector<int64_t> idx(static_cast<size_t>(rows));
    std::iota(idx.begin(), idx.end(), int64_t{0});
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](int64_t i) { fn(i); });
}

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (const int64_t d : shape) n *= d;
    return n;
}

// [m,k] x [k,n] -> [m,n], row-major
Tensor matmul(const float* a, int64_t m, int64_t k, const float* b, int64_t n) {
    Tensor out({m, n});
    float* o = out.data();
    parallelRows(m, [&](int64_t i) {
        float* row = o + i * n;
        for (int64_t p = 0; p < k; ++p) {
            const float av = a[i * k + p];
            const float* brow = b + p * n;
            for (int64_t j = 0; j < n; ++j) row[j] += av * brow[j];
        }
    });
    return out;
}

Tensor reshaped(const Tensor& t, std::vector<int64_t> shape) {
    if (numel(shape) != t.numel()) throw std::runtime_error("the adapter tensor has the wrong number of elements");
    return Tensor(std::move(shape), t.values());
}

// A 2-D view of a tensor: first dimension as rows, the rest flattened.
Tensor rows2d(const Tensor& t) { return reshaped(t, {t.dim(0), t.numel() / t.dim(0)}); }

Tensor scaled(Tensor t, double factor) {
    float* p = t.data();
    for (int64_t i = 0; i < t.numel(); ++i) p[i] = static_cast<float>(p[i] * factor);
    return t;
}

const Tensor* find(const AdapterModule& m, const char* role) {
    const auto it = m.tensors.find(role);
    return it == m.tensors.end() ? nullptr : &it->second;
}

// einsum("i j k l, i p, j r -> p r k l", t, a, b): a Tucker core t [I,J,K,L] with factors a [I,P] and b [J,R]
Tensor tucker(const Tensor& t, const Tensor& a, const Tensor& b) {
    const int64_t I = t.dim(0), J = t.dim(1), K = t.dim(2), L = t.dim(3), P = a.dim(1), R = b.dim(1);
    if (a.dim(0) != I || b.dim(0) != J) throw std::runtime_error("the Tucker factors do not fit the core");
    Tensor out({P, R, K, L});
    parallelRows(P, [&](int64_t p) {
        for (int64_t r = 0; r < R; ++r)
            for (int64_t kl = 0; kl < K * L; ++kl) {
                double sum = 0;
                for (int64_t i = 0; i < I; ++i)
                    for (int64_t j = 0; j < J; ++j) sum += static_cast<double>(t.data()[((i * J + j) * K * L) + kl]) * a.data()[i * P + p] * b.data()[j * R + r];
                out.data()[((p * R + r) * K * L) + kl] = static_cast<float>(sum);
            }
    });
    return out;
}

// kron over the first two dimensions; trailing dimensions (a convolution kernel) are kept from `b`.
Tensor kron(const Tensor& a, const Tensor& b) {
    const int64_t A = a.dim(0), B = a.dim(1);
    const int64_t C = b.dim(0), D = b.dim(1), tail = b.numel() / (C * D);
    std::vector<int64_t> shape = {A * C, B * D};
    for (int i = 2; i < b.ndim(); ++i) shape.push_back(b.dim(i));
    Tensor out(shape);
    parallelRows(A * C, [&](int64_t row) {
        const int64_t i = row / C, p = row % C;
        for (int64_t j = 0; j < B; ++j) {
            const float w = a.data()[i * B + j];
            for (int64_t q = 0; q < D; ++q)
                for (int64_t t = 0; t < tail; ++t) out.data()[((row * (B * D)) + (j * D + q)) * tail + t] = w * b.data()[((p * D) + q) * tail + t];
        }
    });
    return out;
}

// ---- the deltas ---------------------------------------------------------------------------------------------------------

Tensor loraDelta(const AdapterModule& m, const Tensor& down, const Tensor& up) {
    const int64_t rank = down.dim(0);
    const double scale = m.alpha ? *m.alpha / static_cast<double>(rank) : 1.0;
    const Tensor down2 = rows2d(down);   // [rank, in * k * k]
    const Tensor up2 = rows2d(up);       // [out, rank]
    if (up2.dim(1) != rank) throw std::runtime_error("lora_up and lora_down disagree on the rank");
    if (const Tensor* mid = find(m, "mid")) {
        // LoCon with a Tucker core: W[o,i,k,l] = sum_ab up[o,b] * mid[b,a,k,l] * down[a,i]
        const int64_t in = down2.dim(1), kl = mid->numel() / (mid->dim(0) * mid->dim(1));
        const int64_t out = up2.dim(0);
        Tensor w({out, in, mid->dim(2), mid->dim(3)});
        parallelRows(out, [&](int64_t o) {
            for (int64_t i = 0; i < in; ++i)
                for (int64_t k = 0; k < kl; ++k) {
                    double sum = 0;
                    for (int64_t a = 0; a < rank; ++a)
                        for (int64_t b = 0; b < rank; ++b) sum += static_cast<double>(up2.data()[o * rank + b]) * mid->data()[(b * rank + a) * kl + k] * down2.data()[a * in + i];
                    w.data()[(o * in + i) * kl + k] = static_cast<float>(sum);
                }
        });
        return scaled(std::move(w), scale);
    }
    return scaled(matmul(up2.data(), up2.dim(0), rank, down2.data(), down2.dim(1)), scale);
}

Tensor hohaHalf(const AdapterModule& m, const char* a, const char* b, const char* t) {
    const Tensor& wa = m.tensors.at(a);
    const Tensor& wb = m.tensors.at(b);
    if (const Tensor* core = find(m, t)) return tucker(*core, wa, wb);
    return matmul(wa.data(), wa.dim(0), wa.dim(1), wb.data(), wb.numel() / wb.dim(0));
}

Tensor lohaDelta(const AdapterModule& m) {
    const Tensor first = hohaHalf(m, "hada_w1_a", "hada_w1_b", "hada_t1");
    const Tensor second = hohaHalf(m, "hada_w2_a", "hada_w2_b", "hada_t2");
    if (first.numel() != second.numel()) throw std::runtime_error("the two halves of the LoHa have different sizes");
    const int64_t rank = m.tensors.at("hada_w1_b").dim(0);
    const double scale = m.alpha ? *m.alpha / static_cast<double>(rank) : 1.0;
    Tensor out = first;
    for (int64_t i = 0; i < out.numel(); ++i) out.data()[i] = static_cast<float>(first.data()[i] * second.data()[i] * scale);
    return out;
}

Tensor lokrDelta(const AdapterModule& m) {
    const bool decomposed = find(m, "lokr_w1_a") || find(m, "lokr_w2_a");
    const Tensor w1 = find(m, "lokr_w1") ? *find(m, "lokr_w1") : matmul(m.tensors.at("lokr_w1_a").data(), m.tensors.at("lokr_w1_a").dim(0), m.tensors.at("lokr_w1_a").dim(1), m.tensors.at("lokr_w1_b").data(), m.tensors.at("lokr_w1_b").dim(1));
    Tensor w2;
    if (const Tensor* full = find(m, "lokr_w2")) w2 = *full;
    else if (const Tensor* core = find(m, "lokr_t2")) w2 = tucker(*core, m.tensors.at("lokr_w2_a"), m.tensors.at("lokr_w2_b"));
    else w2 = matmul(m.tensors.at("lokr_w2_a").data(), m.tensors.at("lokr_w2_a").dim(0), m.tensors.at("lokr_w2_a").dim(1), m.tensors.at("lokr_w2_b").data(), m.tensors.at("lokr_w2_b").dim(1));
    double scale = 1.0;
    if (decomposed && m.alpha) {
        const Tensor* b = find(m, "lokr_w2_b") ? find(m, "lokr_w2_b") : find(m, "lokr_w1_b");
        scale = *m.alpha / static_cast<double>(b->dim(0));
    }
    return scaled(kron(w1, w2), scale);
}

}  // namespace

Tensor adapterDelta(const AdapterModule& m, const std::vector<int64_t>& weightShape) {
    Tensor delta;
    if (find(m, "down") && find(m, "up")) delta = loraDelta(m, *find(m, "down"), *find(m, "up"));
    else if (find(m, "hada_w1_a")) delta = lohaDelta(m);
    else if (find(m, "lokr_w1") || find(m, "lokr_w1_a")) delta = lokrDelta(m);
    else if (find(m, "diff")) delta = *find(m, "diff");
    else throw std::runtime_error("the module " + m.name + " has no tensors that change a weight");
    if (delta.numel() != numel(weightShape)) {
        throw std::runtime_error("the adapter changes " + std::to_string(delta.numel()) + " values, the weight " + m.name + " has " + std::to_string(numel(weightShape)));
    }
    return delta.shape() == weightShape ? delta : Tensor(weightShape, delta.values());
}

// ---- reading adapter files ------------------------------------------------------------------------------------------------

namespace {

struct RoleSuffix {
    const char* suffix;
    const char* role;
};
// longer suffixes before the shorter ones they end with
constexpr RoleSuffix kRoles[] = {
    {".lora_down.weight", "down"},   {".lora_up.weight", "up"},   {".lora_mid.weight", "mid"},
    {".lora_A.default.weight", "down"}, {".lora_B.default.weight", "up"}, {".lora_A.weight", "down"}, {".lora_B.weight", "up"},
    {".alpha", "alpha"},
    {".hada_w1_a", "hada_w1_a"}, {".hada_w1_b", "hada_w1_b"}, {".hada_w2_a", "hada_w2_a"}, {".hada_w2_b", "hada_w2_b"}, {".hada_t1", "hada_t1"}, {".hada_t2", "hada_t2"},
    {".lokr_w1_a", "lokr_w1_a"}, {".lokr_w1_b", "lokr_w1_b"}, {".lokr_w1", "lokr_w1"}, {".lokr_w2_a", "lokr_w2_a"}, {".lokr_w2_b", "lokr_w2_b"}, {".lokr_w2", "lokr_w2"}, {".lokr_t2", "lokr_t2"},
    {".dora_scale", "dora_scale"}, {".diff_b", "diff_b"}, {".diff", "diff"},
};

bool endsWith(const std::string& s, const char* suffix) {
    const size_t n = std::char_traits<char>::length(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

}  // namespace

std::shared_ptr<const AdapterFile> AdapterFile::fromTensors(const std::map<std::string, Tensor>& tensors, std::string identity) {
    auto file = std::make_shared<AdapterFile>();
    file->identity = std::move(identity);
    std::map<std::string, AdapterModule> byName;
    for (const auto& [name, t] : tensors) {
        for (const RoleSuffix& r : kRoles) {
            if (!endsWith(name, r.suffix)) continue;
            AdapterModule& m = byName[name.substr(0, name.size() - std::char_traits<char>::length(r.suffix))];
            if (std::string(r.role) == "alpha") m.alpha = t.numel() > 0 ? t.data()[0] : 0.0f;
            else m.tensors[r.role] = t;
            break;
        }
    }
    for (auto& [name, m] : byName) {
        m.name = name;
        if (!m.tensors.empty()) file->modules.push_back(std::move(m));
    }
    return file;
}

std::shared_ptr<const AdapterFile> AdapterFile::load(const fs::path& path, std::string* error) {
    std::string problem;
    const auto tensors = readAllSafetensors(path, &problem);
    if (tensors.empty()) {
        if (error) *error = problem.empty() ? "the file holds no tensors" : problem;
        return nullptr;
    }
    std::error_code ec;
    auto file = std::const_pointer_cast<AdapterFile>(fromTensors(tensors));
    file->identity = path.generic_string() + "|" + std::to_string(fs::file_size(path, ec)) + "|" + std::to_string(fs::last_write_time(path, ec).time_since_epoch().count());
    if (file->modules.empty()) {
        if (error) *error = "the file holds no LoRA, LoHa, LoKr or weight-difference tensors";
        return nullptr;
    }
    return file;
}

std::optional<TargetKind> AdapterFile::targetOf(const std::string& name) {
    const auto starts = [&](const char* p) { return name.rfind(p, 0) == 0; };
    if (starts("lora_unet_") || starts("diffusion_model.") || starts("unet.") || starts("lycoris_unet") || starts("model.diffusion_model.")) return TargetKind::Diffusion;
    if (starts("lora_te") || starts("text_encoder") || starts("lycoris_te") || starts("clip_l.") || starts("clip_g.")) return TargetKind::TextEncoder;
    if (starts("base_model.model.unet.")) return TargetKind::Diffusion;
    if (starts("base_model.model.text_encoder")) return TargetKind::TextEncoder;
    return std::nullopt;
}

// ---- key mapping ----------------------------------------------------------------------------------------------------------

std::string diffusersUnetPath(const std::string& p) {
    static const std::regex input(R"(^input_blocks\.(\d+)\.(\d+)\.(.*)$)");
    static const std::regex middle(R"(^middle_block\.(\d+)\.(.*)$)");
    static const std::regex output(R"(^output_blocks\.(\d+)\.(\d+)\.(.*)$)");
    const auto resnet = [](std::string rest) {
        static const std::pair<const char*, const char*> map[] = {{"in_layers.0", "norm1"}, {"in_layers.2", "conv1"}, {"emb_layers.1", "time_emb_proj"},
                                                                   {"out_layers.0", "norm2"}, {"out_layers.3", "conv2"}, {"skip_connection", "conv_shortcut"}};
        for (const auto& [from, to] : map)
            if (rest.rfind(from, 0) == 0) return std::string(to) + rest.substr(std::char_traits<char>::length(from));
        return rest;
    };
    std::smatch m;
    if (p == "time_embed.0") return "time_embedding.linear_1";
    if (p == "time_embed.2") return "time_embedding.linear_2";
    if (p == "label_emb.0.0") return "add_embedding.linear_1";
    if (p == "label_emb.0.2") return "add_embedding.linear_2";
    if (p == "input_blocks.0.0") return "conv_in";
    if (p == "out.0") return "conv_norm_out";
    if (p == "out.2") return "conv_out";
    if (std::regex_match(p, m, input)) {
        const int n = std::stoi(m[1]), part = std::stoi(m[2]);
        const std::string rest = m[3];
        if (n == 0) return p;
        const int level = (n - 1) / 3, index = (n - 1) % 3;
        if (n % 3 == 0 && part == 0 && rest.rfind("op", 0) == 0) return "down_blocks." + std::to_string(n / 3 - 1) + ".downsamplers.0.conv" + rest.substr(2);
        if (part == 0) return "down_blocks." + std::to_string(level) + ".resnets." + std::to_string(index) + "." + resnet(rest);
        return "down_blocks." + std::to_string(level) + ".attentions." + std::to_string(index) + "." + rest;
    }
    if (std::regex_match(p, m, middle)) {
        const int part = std::stoi(m[1]);
        const std::string rest = m[2];
        if (part == 1) return "mid_block.attentions.0." + rest;
        return "mid_block.resnets." + std::to_string(part == 0 ? 0 : 1) + "." + resnet(rest);
    }
    if (std::regex_match(p, m, output)) {
        const int n = std::stoi(m[1]), part = std::stoi(m[2]);
        const std::string rest = m[3];
        const std::string level = std::to_string(n / 3), index = std::to_string(n % 3);
        if (part == 0) return "up_blocks." + level + ".resnets." + index + "." + resnet(rest);
        if (rest.rfind("conv", 0) == 0) return "up_blocks." + level + ".upsamplers.0." + rest;   // the upsampler follows the attention (or the resnet alone)
        return "up_blocks." + level + ".attentions." + index + "." + rest;
    }
    return p;
}

namespace {

std::string underscored(std::string s) {
    std::replace(s.begin(), s.end(), '.', '_');
    return s;
}

bool startsWith(const std::string& s, const std::string& p) { return s.rfind(p, 0) == 0; }

}  // namespace

KeyResolver::KeyResolver(const std::vector<std::string>& modelKeys) : keys_(modelKeys) {
    const auto add = [&](const std::string& alias, const Hit& hit) { aliases_.emplace(alias, hit); };
    for (const std::string& key : modelKeys) {
        // the module a weight belongs to is its key without ".weight"
        const bool isWeight = endsWith(key, ".weight");
        const bool isPacked = endsWith(key, ".in_proj_weight");
        if (!isWeight && !isPacked) continue;
        const std::string path = isWeight ? key.substr(0, key.size() - 7) : key.substr(0, key.size() - 15);
        const Hit whole{key, 0, 1};

        if (startsWith(path, "model.diffusion_model.")) {
            const std::string rel = path.substr(22);
            const std::string dia = diffusersUnetPath(rel);
            add("lora_unet_" + underscored(rel), whole);
            add("lora_unet_" + underscored(dia), whole);
            add("diffusion_model." + rel, whole);
            add("unet." + dia, whole);
            add("lycoris_unet_" + underscored(rel), whole);
        } else if (startsWith(path, "down_blocks.") || startsWith(path, "up_blocks.") || startsWith(path, "mid_block.") || startsWith(path, "conv_in") ||
                   startsWith(path, "conv_out") || startsWith(path, "time_embedding.") || startsWith(path, "add_embedding.") || startsWith(path, "conv_norm_out")) {
            add("lora_unet_" + underscored(path), whole);
            add("unet." + path, whole);
        } else if (startsWith(path, "cond_stage_model.transformer.")) {
            const std::string rel = path.substr(29);   // text_model.encoder.layers.0.self_attn.q_proj
            add("lora_te_" + underscored(rel), whole);
            add("lora_te1_" + underscored(rel), whole);
            add("text_encoder." + rel, whole);
        } else if (startsWith(path, "conditioner.embedders.0.transformer.")) {
            const std::string rel = path.substr(36);
            add("lora_te1_" + underscored(rel), whole);
            add("lora_te_" + underscored(rel), whole);
            add("text_encoder." + rel, whole);
            add("clip_l." + rel, whole);
        } else {
            // OpenCLIP text encoders (SDXL's second encoder, SD 2.x): resblocks with packed attention
            static const std::regex openclip(R"(^(cond_stage_model|conditioner\.embedders\.1)\.model\.transformer\.resblocks\.(\d+)\.(.*)$)");
            std::smatch m;
            if (std::regex_match(path, m, openclip)) {
                const bool sdxl = std::string(m[1]).find("conditioner") == 0;
                const std::string layer = "text_model.encoder.layers." + std::string(m[2]);
                const std::string rest = m[3];
                const std::vector<std::string> prefixes = sdxl ? std::vector<std::string>{"lora_te2_"} : std::vector<std::string>{"lora_te_"};
                const std::vector<std::string> dotted = sdxl ? std::vector<std::string>{"text_encoder_2.", "clip_g."} : std::vector<std::string>{"text_encoder."};
                const auto addBoth = [&](const std::string& hf, const Hit& hit) {
                    for (const auto& p : prefixes) add(p + underscored(hf), hit);
                    for (const auto& d : dotted) add(d + hf, hit);
                };
                if (rest == "attn" && isPacked) {
                    addBoth(layer + ".self_attn.q_proj", Hit{key, 0, 3});
                    addBoth(layer + ".self_attn.k_proj", Hit{key, 1, 3});
                    addBoth(layer + ".self_attn.v_proj", Hit{key, 2, 3});
                } else if (rest == "attn.out_proj") {
                    addBoth(layer + ".self_attn.out_proj", whole);
                } else if (rest == "mlp.c_fc") {
                    addBoth(layer + ".mlp.fc1", whole);
                } else if (rest == "mlp.c_proj") {
                    addBoth(layer + ".mlp.fc2", whole);
                }
            }
        }
    }
}

std::optional<KeyResolver::Hit> KeyResolver::resolve(const std::string& name) const {
    std::string n = name;
    if (startsWith(n, "base_model.model.")) n = n.substr(17);
    if (const auto it = aliases_.find(n); it != aliases_.end()) return it->second;
    // "diff" files name the weight itself: "<key>.weight" with or without the "model." prefix
    for (const std::string& candidate : {n, "model." + n, "model.diffusion_model." + (startsWith(n, "diffusion_model.") ? n.substr(16) : n)}) {
        if (std::find(keys_.begin(), keys_.end(), candidate) != keys_.end()) return Hit{candidate, 0, 1};
    }
    return std::nullopt;
}

// ---- applying -------------------------------------------------------------------------------------------------------------

std::string WeightPatcher::signatureOf(TargetKind kind, const std::vector<PatchSpec>& patches) {
    if (patches.empty()) return "";
    std::string s = kind == TargetKind::Diffusion ? "d" : "t";
    for (const PatchSpec& p : patches) {
        char buf[48];
        std::snprintf(buf, sizeof buf, "|%.6f|%.6f", p.strengthModel, p.strengthText);
        s += "#" + (p.file ? p.file->identity : std::string("none")) + buf;
    }
    return s;
}

void WeightPatcher::undo(WeightTarget& target) {
    for (const auto& [key, original] : originals_) target.write(key, original);
    originals_.clear();
    signature_.clear();
}

WeightPatcher::Report WeightPatcher::ensure(WeightTarget& target, TargetKind kind, const std::vector<PatchSpec>& patches) {
    const std::string wanted = signatureOf(kind, patches);
    if (wanted == signature_) return last_;
    return apply(target, kind, patches);
}

WeightPatcher::Report WeightPatcher::apply(WeightTarget& target, TargetKind kind, const std::vector<PatchSpec>& patches) {
    undo(target);
    Report report;
    if (patches.empty()) {
        last_ = report;
        return report;
    }
    const KeyResolver resolver(target.keys());
    std::map<std::string, Tensor> work;   // the weights as they are while the adapters are added one by one

    const auto current = [&](const std::string& key) -> Tensor* {
        auto it = work.find(key);
        if (it != work.end()) return &it->second;
        Tensor value;
        if (!target.read(key, value)) return nullptr;
        originals_.emplace(key, value);
        return &work.emplace(key, std::move(value)).first->second;
    };

    std::set<std::string> unmatched;
    for (const PatchSpec& spec : patches) {
        if (!spec.file) continue;
        for (const AdapterModule& module : spec.file->modules) {
            const auto owner = AdapterFile::targetOf(module.name);
            if (owner && *owner != kind) continue;   // this one is for the other component
            ++report.modules;
            const double strength = (owner ? *owner : kind) == TargetKind::Diffusion ? spec.strengthModel : spec.strengthText;
            if (strength == 0.0) continue;

            // a bias difference names the module; every other kind of tensor changes its weight
            const bool biasOnly = module.tensors.size() == 1 && module.tensors.count("diff_b");
            std::optional<KeyResolver::Hit> hit = resolver.resolve(module.name);
            if (!hit) {
                unmatched.insert(module.name);
                continue;
            }
            if (biasOnly) {
                const std::string biasKey = hit->key.size() > 7 && endsWith(hit->key, ".weight") ? hit->key.substr(0, hit->key.size() - 7) + ".bias" : hit->key;
                Tensor* bias = current(biasKey);
                if (!bias) {
                    unmatched.insert(module.name + " (bias)");
                    continue;
                }
                const Tensor& diff = module.tensors.at("diff_b");
                if (diff.numel() != bias->numel()) throw std::runtime_error("the bias difference of " + module.name + " has the wrong size");
                for (int64_t i = 0; i < bias->numel(); ++i) bias->data()[i] += static_cast<float>(strength) * diff.data()[i];
                ++report.patched;
                continue;
            }
            Tensor* weight = current(hit->key);
            if (!weight) {
                unmatched.insert(module.name);
                continue;
            }
            // a packed weight (OpenCLIP q, k, v) is changed one slice of rows at a time
            const int64_t rows = weight->dim(0) / hit->parts;
            std::vector<int64_t> sliceShape = weight->shape();
            sliceShape[0] = rows;
            const int64_t stride = weight->numel() / hit->parts;
            float* base = weight->data() + stride * hit->part;
            const Tensor delta = adapterDelta(module, sliceShape);

            if (const Tensor* magnitude = find(module, "dora_scale")) {
                // DoRA: W' = m * (W + dW) / ||W + dW|| per output row; the strength blends between W and W'
                const int64_t width = stride / rows;
                for (int64_t r = 0; r < rows; ++r) {
                    double norm = 0;
                    for (int64_t c = 0; c < width; ++c) {
                        const double v = static_cast<double>(base[r * width + c]) + delta.data()[r * width + c];
                        norm += v * v;
                    }
                    norm = std::sqrt(norm);
                    const double m = magnitude->data()[std::min<int64_t>(r, magnitude->numel() - 1)];
                    const double factor = norm > 0 ? m / norm : 1.0;
                    for (int64_t c = 0; c < width; ++c) {
                        const double original = base[r * width + c];
                        const double dora = (original + delta.data()[r * width + c]) * factor;
                        base[r * width + c] = static_cast<float>(original + strength * (dora - original));
                    }
                }
            } else if (module.tensors.count("set")) {
                // not produced by the readers today; kept for completeness of the format
            } else {
                for (int64_t i = 0; i < delta.numel(); ++i) base[i] += static_cast<float>(strength) * delta.data()[i];
            }
            ++report.patched;
            if (const Tensor* bias = find(module, "diff_b")) {   // a full difference may carry a bias difference too
                const std::string biasKey = endsWith(hit->key, ".weight") ? hit->key.substr(0, hit->key.size() - 7) + ".bias" : hit->key;
                if (Tensor* b = current(biasKey))
                    if (b->numel() == bias->numel())
                        for (int64_t i = 0; i < b->numel(); ++i) b->data()[i] += static_cast<float>(strength) * bias->data()[i];
            }
        }
    }
    for (const auto& [key, value] : work) target.write(key, value);
    report.unmatched.assign(unmatched.begin(), unmatched.end());
    signature_ = signatureOf(kind, patches);
    last_ = report;
    return report;
}

}  // namespace gh::models
