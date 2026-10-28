#include "host/RuntimeInference.h"

#include "core/Log.h"
#include "graph/NodeApi.h"

#include <algorithm>
#include <cstring>

namespace gh::host {

using graph::NodeError;

namespace {

// A prompt encoded by the runtime: the hidden states and the pooled vector (SDXL) with their ggml shapes.
struct Conditioning {
    std::vector<float> hidden, pooled;
    int64_t hiddenShape[5] = {0, 0, 0, 0, 0};
    int hiddenDims = 0;
    int64_t pooledShape[5] = {0, 0, 0, 0, 0};
    int pooledDims = 0;
};

gh_tensor view(const std::vector<float>& data, const int64_t* shape, int dims) {
    gh_tensor t{};
    t.struct_size = sizeof(t);
    t.ndim = dims;
    for (int i = 0; i < dims; ++i) t.shape[i] = shape[i];
    t.data = const_cast<float*>(data.data());
    return t;
}

struct Owned {
    gh_tensor t{};
    const engine::Runtime::Api* api;
    explicit Owned(const engine::Runtime::Api& a) : api(&a) { t.struct_size = sizeof(t); }
    ~Owned() { api->tensorFree(&t); }
    int64_t count() const {
        int64_t n = 1;
        for (int i = 0; i < t.ndim; ++i) n *= t.shape[i];
        return n;
    }
};

}  // namespace

struct RuntimeInference::Open {
    std::shared_ptr<engine::Runtime> runtime;
    gh_model* model = nullptr;
    std::string key;
    std::string path;
    ~Open() {
        if (model && runtime) {
            if (runtime->api().sessionEnd) runtime->api().sessionEnd(model);
            runtime->api().free(model);
        }
    }
};

RuntimeInference::RuntimeInference(RuntimeProvider provider) : provider_(std::move(provider)) {}
RuntimeInference::~RuntimeInference() = default;

void RuntimeInference::release() {
    std::lock_guard<std::mutex> lock(mutex_);
    open_.reset();
}

bool RuntimeInference::loaded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return open_ != nullptr;
}

RuntimeInference::Open& RuntimeInference::open(const graph::ModelFileRef& file, const graph::LoadSettings& load) {
    if (file.path.empty()) throw NodeError("the model has no file");
    const std::string key = file.path + "|" + load.device + "|" + load.flashAttention + "|" + load.offloadToRam;
    if (open_ && open_->key == key) return *open_;
    open_.reset();   // another checkpoint or other options: the open one goes first (one at a time keeps video memory free)

    std::string error;
    std::shared_ptr<engine::Runtime> runtime = provider_ ? provider_(error) : nullptr;
    if (!runtime) throw NodeError("no inference runtime is available" + (error.empty() ? std::string() : ": " + error));
    if (!runtime->hasComponents()) throw NodeError("the " + runtime->id() + " runtime cannot run node graphs: use the Vulkan runtime (backend setting)");

    gh_load_params params{};
    params.struct_size = sizeof(params);
    params.model_path = file.path.c_str();
    params.backend = load.device == "auto" ? nullptr : load.device.c_str();
    params.flash_attention = load.flashAttention == "off" ? 0 : 1;   // on Vulkan it was measured 5x faster for SD 1.x above 512 pixels
    params.offload_params_to_cpu = load.offloadToRam == "on" ? 1 : 0;
    gh_model* model = runtime->api().load(&params);
    if (!model) throw NodeError("cannot load " + file.path + ": " + runtime->api().lastError());
    if (runtime->api().sessionBegin(model) != GH_OK) {
        const std::string why = runtime->api().lastError();
        runtime->api().free(model);
        throw NodeError("cannot start a session on " + file.path + ": " + why);
    }
    auto o = std::make_unique<Open>();
    o->runtime = std::move(runtime);
    o->model = model;
    o->key = key;
    o->path = file.path;
    log::info("graph model loaded: " + file.path);
    open_ = std::move(o);
    return *open_;
}

graph::Value RuntimeInference::encodeText(const graph::TextEncoderHandle& encoder, const text::ParsedPrompt& prompt, int clipSkip, int width, int height, bool zeroOut) {
    if (!encoder.fromCheckpoint || encoder.files.size() != 1) throw NodeError("separate text encoder files are not supported by the runtime yet: use the encoders of a checkpoint", "textEncoder");
    if (!encoder.patches.empty()) throw NodeError("LoRAs cannot be applied through the runtime yet", "textEncoder");
    std::lock_guard<std::mutex> lock(mutex_);
    Open& o = open(encoder.files.front(), encoder.load);
    const auto& api = o.runtime->api();
    Owned hidden(api), pooled(api);
    const std::string text = text::plainText(prompt);
    if (api.textEncode(o.model, text.c_str(), clipSkip > 1 ? clipSkip : -1, width, height, zeroOut ? 1 : 0, &hidden.t, &pooled.t) != GH_OK) throw NodeError("text encoding failed: " + std::string(api.lastError()), "text");
    auto c = std::make_shared<Conditioning>();
    c->hidden.assign(hidden.t.data, hidden.t.data + hidden.count());
    c->hiddenDims = hidden.t.ndim;
    for (int i = 0; i < hidden.t.ndim; ++i) c->hiddenShape[i] = hidden.t.shape[i];
    if (pooled.t.ndim > 0 && pooled.t.data) {
        c->pooled.assign(pooled.t.data, pooled.t.data + pooled.count());
        c->pooledDims = pooled.t.ndim;
        for (int i = 0; i < pooled.t.ndim; ++i) c->pooledShape[i] = pooled.t.shape[i];
    }
    return graph::Value::makeOpaque(graph::ValueType::Conditioning, std::move(c));
}

namespace {
const Conditioning& conditioningOf(const graph::Value& v) {
    const auto p = std::static_pointer_cast<const Conditioning>(std::get<std::shared_ptr<const void>>(v.data));
    if (!p) throw NodeError("the conditioning is empty");
    return *p;
}

gh_tensor latentTensor(const graph::LatentView& x, std::vector<int64_t>& shapeStorage) {
    (void)shapeStorage;
    gh_tensor t{};
    t.struct_size = sizeof(t);
    t.ndim = 4;
    t.shape[0] = x.width;   // ggml order: width varies fastest
    t.shape[1] = x.height;
    t.shape[2] = x.channels;
    t.shape[3] = x.batch;
    t.data = const_cast<float*>(x.data);
    return t;
}
}  // namespace

void RuntimeInference::denoise(const graph::ModelHandle& model, const graph::Value& cond, const graph::LatentView& x, float sigma, std::vector<float>& out) {
    denoiseGuided(model, cond, graph::Value{}, 1.0f, x, sigma, out);
}

void RuntimeInference::denoiseGuided(const graph::ModelHandle& model, const graph::Value& cond, const graph::Value& uncond, float cfg, const graph::LatentView& x, float sigma, std::vector<float>& out) {
    if (!model.patches.empty()) throw NodeError("LoRAs cannot be applied through the runtime yet", "model");
    std::lock_guard<std::mutex> lock(mutex_);
    Open& o = open(model.file, model.load);
    const auto& api = o.runtime->api();
    std::vector<int64_t> unused;
    const gh_tensor xt = latentTensor(x, unused);
    const Conditioning& c = conditioningOf(cond);
    const gh_tensor ch = view(c.hidden, c.hiddenShape, c.hiddenDims);
    gh_tensor cp{};
    if (!c.pooled.empty()) cp = view(c.pooled, c.pooledShape, c.pooledDims);
    gh_denoise_args args{};
    args.struct_size = sizeof(args);
    args.x = &xt;
    args.sigma = sigma;
    args.cond_hidden = &ch;
    args.cond_pooled = c.pooled.empty() ? nullptr : &cp;
    gh_tensor uh{}, up{};
    if (!uncond.empty()) {
        const Conditioning& u = conditioningOf(uncond);
        uh = view(u.hidden, u.hiddenShape, u.hiddenDims);
        args.uncond_hidden = &uh;
        if (!u.pooled.empty()) {
            up = view(u.pooled, u.pooledShape, u.pooledDims);
            args.uncond_pooled = &up;
        }
    }
    args.cfg_scale = cfg;
    Owned result(api);
    if (api.denoise(o.model, &args, &result.t) != GH_OK) throw NodeError("the model call failed: " + std::string(api.lastError()));
    if (static_cast<size_t>(result.count()) != x.count()) throw NodeError("the model returned a latent of the wrong size");
    out.assign(result.t.data, result.t.data + result.count());
}

tensor::Tensor RuntimeInference::decode(const graph::VaeHandle& vae, const graph::ModelHandle& model, const graph::LatentView& latent) {
    if (vae.pixelSpace) throw NodeError("pixel space is not supported by the runtime yet", "vae");
    if (!vae.fromCheckpoint) throw NodeError("separate VAE files are not supported by the runtime yet: use the VAE of the checkpoint", "vae");
    std::lock_guard<std::mutex> lock(mutex_);
    Open& o = open(model.file, model.load);
    const auto& api = o.runtime->api();
    std::vector<int64_t> unused;
    const gh_tensor lt = latentTensor(latent, unused);
    Owned image(api);
    if (api.vaeDecode(o.model, &lt, &image.t) != GH_OK) throw NodeError("decoding failed: " + std::string(api.lastError()), "vae");
    // {W, H, 3[, B]} with width fastest -> [B, H, W, 3]
    const int64_t W = image.t.shape[0], H = image.t.shape[1], C = image.t.shape[2], B = image.t.ndim > 3 ? image.t.shape[3] : 1;
    std::vector<float> data(static_cast<size_t>(B * H * W * C));
    for (int64_t b = 0; b < B; ++b)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t y = 0; y < H; ++y)
                for (int64_t x = 0; x < W; ++x) data[static_cast<size_t>(((b * H + y) * W + x) * C + c)] = image.t.data[((b * C + c) * H + y) * W + x];
    return tensor::Tensor({B, H, W, C}, std::move(data));
}

}  // namespace gh::host
