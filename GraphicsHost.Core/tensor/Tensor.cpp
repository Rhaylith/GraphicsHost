#include "tensor/Tensor.h"

#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>
#include <stdexcept>

namespace gh::tensor {

namespace {

int64_t product(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (const int64_t d : shape) {
        if (d < 0) throw std::invalid_argument("negative dimension");
        n *= d;
    }
    return n;
}

// Runs fn(i) for i in [0, count) on the available cores; small loops stay on the calling thread.
template <typename Fn>
void parallelFor(int64_t count, Fn&& fn) {
    if (count < 64) {
        for (int64_t i = 0; i < count; ++i) fn(i);
        return;
    }
    std::vector<int64_t> idx(static_cast<size_t>(count));
    std::iota(idx.begin(), idx.end(), int64_t{0});
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](int64_t i) { fn(i); });
}

void requireSame(const Tensor& a, const Tensor& b, const char* op) {
    if (!a.sameShape(b)) throw std::invalid_argument(std::string(op) + ": shapes differ");
}

template <typename Op>
Tensor zip(const Tensor& a, const Tensor& b, const char* name, Op op) {
    if (b.numel() == 1 && a.numel() != 1) {
        Tensor out(a.shape());
        const float s = b.data()[0];
        parallelFor(a.numel() / 1024 + 1, [&](int64_t block) {
            const int64_t from = block * 1024, to = std::min(a.numel(), from + 1024);
            for (int64_t i = from; i < to; ++i) out.data()[i] = op(a.data()[i], s);
        });
        return out;
    }
    if (a.numel() == 1 && b.numel() != 1) {
        Tensor out(b.shape());
        const float s = a.data()[0];
        for (int64_t i = 0; i < b.numel(); ++i) out.data()[i] = op(s, b.data()[i]);
        return out;
    }
    requireSame(a, b, name);
    Tensor out(a.shape());
    parallelFor(a.numel() / 1024 + 1, [&](int64_t block) {
        const int64_t from = block * 1024, to = std::min(a.numel(), from + 1024);
        const float* pa = a.data();
        const float* pb = b.data();
        float* po = out.data();
        for (int64_t i = from; i < to; ++i) po[i] = op(pa[i], pb[i]);
    });
    return out;
}

// [.., H, W, C] or [.., H, W] -> (batch, H, W, C)
struct ImageDims {
    int64_t b, h, w, c;
};
ImageDims imageDims(const Tensor& a) {
    if (a.ndim() == 4) return {a.dim(0), a.dim(1), a.dim(2), a.dim(3)};
    if (a.ndim() == 3) return {a.dim(0), a.dim(1), a.dim(2), 1};
    throw std::invalid_argument("expected [B, H, W, C] or [B, H, W]");
}
Tensor makeImage(const Tensor& like, int64_t h, int64_t w) {
    if (like.ndim() == 4) return Tensor({like.dim(0), h, w, like.dim(3)});
    return Tensor({like.dim(0), h, w});
}

}  // namespace

Tensor::Tensor(std::vector<int64_t> shape, float fill) : shape_(std::move(shape)) {
    if (shape_.size() > kMaxDims) throw std::invalid_argument("at most 5 dimensions");
    data_.assign(static_cast<size_t>(product(shape_)), fill);
}

Tensor::Tensor(std::vector<int64_t> shape, std::vector<float> data) : shape_(std::move(shape)), data_(std::move(data)) {
    if (shape_.size() > kMaxDims) throw std::invalid_argument("at most 5 dimensions");
    if (static_cast<int64_t>(data_.size()) != product(shape_)) throw std::invalid_argument("data does not fit the shape");
}

int64_t Tensor::offset(std::initializer_list<int64_t> index) const {
    if (index.size() != shape_.size()) throw std::invalid_argument("index has the wrong number of dimensions");
    int64_t off = 0;
    size_t d = 0;
    for (const int64_t i : index) {
        if (i < 0 || i >= shape_[d]) throw std::out_of_range("tensor index out of range");
        off = off * shape_[d] + i;
        ++d;
    }
    return off;
}

// ---- element-wise ----------------------------------------------------------------------------------------------------------
Tensor add(const Tensor& a, const Tensor& b) { return zip(a, b, "add", [](float x, float y) { return x + y; }); }
Tensor sub(const Tensor& a, const Tensor& b) { return zip(a, b, "sub", [](float x, float y) { return x - y; }); }
Tensor mul(const Tensor& a, const Tensor& b) { return zip(a, b, "mul", [](float x, float y) { return x * y; }); }
Tensor scale(const Tensor& a, float f) { return zip(a, Tensor({1}, f), "scale", [](float x, float y) { return x * y; }); }
Tensor addScalar(const Tensor& a, float v) { return zip(a, Tensor({1}, v), "addScalar", [](float x, float y) { return x + y; }); }
Tensor invert(const Tensor& a) { return zip(Tensor({1}, 1.0f), a, "invert", [](float x, float y) { return x - y; }); }

Tensor clamp(const Tensor& a, float low, float high) {
    Tensor out(a.shape());
    for (int64_t i = 0; i < a.numel(); ++i) out.data()[i] = std::min(high, std::max(low, a.data()[i]));
    return out;
}

Tensor lerp(const Tensor& a, const Tensor& b, float t) {
    requireSame(a, b, "lerp");
    Tensor out(a.shape());
    parallelFor(a.numel() / 1024 + 1, [&](int64_t block) {
        const int64_t from = block * 1024, to = std::min(a.numel(), from + 1024);
        for (int64_t i = from; i < to; ++i) out.data()[i] = a.data()[i] + (b.data()[i] - a.data()[i]) * t;
    });
    return out;
}

Tensor slerp(const Tensor& a, const Tensor& b, float t) {
    requireSame(a, b, "slerp");
    if (a.ndim() == 0 || a.dim(0) == 0) return a;
    const int64_t batch = a.dim(0), per = a.numel() / batch;
    Tensor out(a.shape());
    for (int64_t k = 0; k < batch; ++k) {
        const float* pa = a.data() + k * per;
        const float* pb = b.data() + k * per;
        double dot = 0, na = 0, nb = 0;
        for (int64_t i = 0; i < per; ++i) {
            dot += double(pa[i]) * pb[i];
            na += double(pa[i]) * pa[i];
            nb += double(pb[i]) * pb[i];
        }
        const double denom = std::sqrt(na * nb);
        const double cosine = denom > 0 ? std::min(1.0, std::max(-1.0, dot / denom)) : 1.0;
        const double omega = std::acos(cosine), so = std::sin(omega);
        float* po = out.data() + k * per;
        if (so < 1e-6) {   // (nearly) parallel: the straight line is the same curve
            for (int64_t i = 0; i < per; ++i) po[i] = pa[i] + (pb[i] - pa[i]) * t;
        } else {
            const double wa = std::sin((1.0 - t) * omega) / so, wb = std::sin(t * omega) / so;
            for (int64_t i = 0; i < per; ++i) po[i] = static_cast<float>(wa * pa[i] + wb * pb[i]);
        }
    }
    return out;
}

// ---- statistics ----------------------------------------------------------------------------------------------------------
std::vector<float> channelMean(const Tensor& a) {
    if (a.ndim() == 0) return {};
    const int64_t c = a.dim(a.ndim() - 1), n = a.numel() / std::max<int64_t>(c, 1);
    std::vector<double> sum(static_cast<size_t>(c), 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t k = 0; k < c; ++k) sum[static_cast<size_t>(k)] += a.data()[i * c + k];
    std::vector<float> out(static_cast<size_t>(c));
    for (int64_t k = 0; k < c; ++k) out[static_cast<size_t>(k)] = static_cast<float>(sum[static_cast<size_t>(k)] / std::max<int64_t>(n, 1));
    return out;
}

std::vector<float> channelStd(const Tensor& a) {
    const auto mean = channelMean(a);
    if (mean.empty()) return {};
    const int64_t c = static_cast<int64_t>(mean.size()), n = a.numel() / std::max<int64_t>(c, 1);
    std::vector<double> sum(static_cast<size_t>(c), 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t k = 0; k < c; ++k) {
            const double d = a.data()[i * c + k] - mean[static_cast<size_t>(k)];
            sum[static_cast<size_t>(k)] += d * d;
        }
    std::vector<float> out(static_cast<size_t>(c));
    for (int64_t k = 0; k < c; ++k) out[static_cast<size_t>(k)] = static_cast<float>(std::sqrt(sum[static_cast<size_t>(k)] / std::max<int64_t>(n, 1)));
    return out;
}

// ---- structure ----------------------------------------------------------------------------------------------------------
Tensor concat(const std::vector<Tensor>& parts, int axis) {
    if (parts.empty()) throw std::invalid_argument("concat of nothing");
    const int nd = parts[0].ndim();
    if (axis < 0) axis += nd;
    if (axis < 0 || axis >= nd) throw std::invalid_argument("concat axis out of range");
    std::vector<int64_t> shape = parts[0].shape();
    int64_t total = 0;
    for (const auto& p : parts) {
        if (p.ndim() != nd) throw std::invalid_argument("concat: ranks differ");
        for (int d = 0; d < nd; ++d)
            if (d != axis && p.dim(d) != shape[static_cast<size_t>(d)]) throw std::invalid_argument("concat: shapes differ");
        total += p.dim(axis);
    }
    shape[static_cast<size_t>(axis)] = total;
    Tensor out(shape);
    int64_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d) outer *= shape[static_cast<size_t>(d)];
    for (int d = axis + 1; d < nd; ++d) inner *= shape[static_cast<size_t>(d)];
    int64_t at = 0;
    for (const auto& p : parts) {
        const int64_t span = p.dim(axis) * inner;
        for (int64_t o = 0; o < outer; ++o)
            std::copy_n(p.data() + o * span, span, out.data() + (o * total * inner) + at * inner);
        at += p.dim(axis);
    }
    return out;
}

Tensor slice(const Tensor& a, int axis, int64_t begin, int64_t end) {
    const int nd = a.ndim();
    if (axis < 0) axis += nd;
    if (axis < 0 || axis >= nd || begin < 0 || end > a.dim(axis) || begin > end) throw std::invalid_argument("slice out of range");
    std::vector<int64_t> shape = a.shape();
    shape[static_cast<size_t>(axis)] = end - begin;
    Tensor out(shape);
    int64_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d) outer *= shape[static_cast<size_t>(d)];
    for (int d = axis + 1; d < nd; ++d) inner *= shape[static_cast<size_t>(d)];
    const int64_t full = a.dim(axis);
    for (int64_t o = 0; o < outer; ++o) std::copy_n(a.data() + (o * full + begin) * inner, (end - begin) * inner, out.data() + o * (end - begin) * inner);
    return out;
}

Tensor crop(const Tensor& a, int64_t x, int64_t y, int64_t width, int64_t height) {
    const auto d = imageDims(a);
    if (x < 0 || y < 0 || width < 0 || height < 0 || x + width > d.w || y + height > d.h) throw std::invalid_argument("crop outside the image");
    Tensor out = makeImage(a, height, width);
    for (int64_t b = 0; b < d.b; ++b)
        for (int64_t row = 0; row < height; ++row)
            std::copy_n(a.data() + ((b * d.h + y + row) * d.w + x) * d.c, width * d.c, out.data() + ((b * height + row) * width) * d.c);
    return out;
}

Tensor pad(const Tensor& a, int64_t left, int64_t top, int64_t right, int64_t bottom, PadMode mode, float value) {
    const auto d = imageDims(a);
    if (left < 0 || top < 0 || right < 0 || bottom < 0) throw std::invalid_argument("negative padding");
    const int64_t nh = d.h + top + bottom, nw = d.w + left + right;
    Tensor out = makeImage(a, nh, nw);
    auto source = [&](int64_t v, int64_t size) -> int64_t {   // -1: outside (constant mode)
        if (v >= 0 && v < size) return v;
        if (mode == PadMode::Constant) return -1;
        if (mode == PadMode::Replicate) return std::min(size - 1, std::max<int64_t>(0, v));
        if (size == 1) return 0;
        const int64_t period = 2 * (size - 1);   // reflect without repeating the edge
        int64_t m = ((v % period) + period) % period;
        return m < size ? m : period - m;
    };
    for (int64_t b = 0; b < d.b; ++b)
        for (int64_t y = 0; y < nh; ++y) {
            const int64_t sy = source(y - top, d.h);
            for (int64_t x = 0; x < nw; ++x) {
                const int64_t sx = source(x - left, d.w);
                float* po = out.data() + ((b * nh + y) * nw + x) * d.c;
                if (sy < 0 || sx < 0) {
                    std::fill_n(po, d.c, value);
                } else {
                    std::copy_n(a.data() + ((b * d.h + sy) * d.w + sx) * d.c, d.c, po);
                }
            }
        }
    return out;
}

Tensor flip(const Tensor& a, bool horizontal) {
    const auto d = imageDims(a);
    Tensor out(a.shape());
    for (int64_t b = 0; b < d.b; ++b)
        for (int64_t y = 0; y < d.h; ++y)
            for (int64_t x = 0; x < d.w; ++x) {
                const int64_t sx = horizontal ? d.w - 1 - x : x, sy = horizontal ? y : d.h - 1 - y;
                std::copy_n(a.data() + ((b * d.h + sy) * d.w + sx) * d.c, d.c, out.data() + ((b * d.h + y) * d.w + x) * d.c);
            }
    return out;
}

Tensor rotate90(const Tensor& a, int quarterTurns) {
    const auto d = imageDims(a);
    const int turns = ((quarterTurns % 4) + 4) % 4;
    if (turns == 0) return a;
    const bool swap = turns % 2 == 1;
    const int64_t nh = swap ? d.w : d.h, nw = swap ? d.h : d.w;
    Tensor out = makeImage(a, nh, nw);
    for (int64_t b = 0; b < d.b; ++b)
        for (int64_t y = 0; y < d.h; ++y)
            for (int64_t x = 0; x < d.w; ++x) {
                int64_t ny, nx;
                if (turns == 1) { ny = x; nx = d.h - 1 - y; }          // clockwise
                else if (turns == 2) { ny = d.h - 1 - y; nx = d.w - 1 - x; }
                else { ny = d.w - 1 - x; nx = y; }
                std::copy_n(a.data() + ((b * d.h + y) * d.w + x) * d.c, d.c, out.data() + ((b * nh + ny) * nw + nx) * d.c);
            }
    return out;
}

// ---- resampling ------------------------------------------------------------------------------------------------------------
namespace {

struct Taps {
    std::vector<int64_t> first;     // first source index per output position
    std::vector<int> count;         // taps per output position
    std::vector<float> weights;     // count * max taps, flattened per position
    int stride = 0;
};

double lanczos3(double x) {
    x = std::fabs(x);
    if (x >= 3.0) return 0.0;
    if (x < 1e-8) return 1.0;
    const double px = 3.14159265358979323846 * x;
    return 3.0 * std::sin(px) * std::sin(px / 3.0) / (px * px);
}
double cubic(double x) {   // Catmull-Rom
    x = std::fabs(x);
    if (x < 1.0) return 1.5 * x * x * x - 2.5 * x * x + 1.0;
    if (x < 2.0) return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
    return 0.0;
}
double triangle(double x) {
    x = std::fabs(x);
    return x < 1.0 ? 1.0 - x : 0.0;
}

// Separable resampling weights from `in` samples to `out` samples; shrinking widens the kernel (antialiasing).
Taps makeTaps(int64_t in, int64_t out, Resample method) {
    Taps t;
    const double ratio = static_cast<double>(in) / static_cast<double>(out);
    double support = 0.5;
    double (*kernel)(double) = nullptr;
    switch (method) {
        case Resample::Bilinear: kernel = triangle; support = 1.0; break;
        case Resample::Bicubic: kernel = cubic; support = 2.0; break;
        case Resample::Lanczos: kernel = lanczos3; support = 3.0; break;
        case Resample::Area: kernel = nullptr; support = 0.5; break;
        case Resample::Nearest: break;
    }
    const double filterScale = (method == Resample::Nearest || method == Resample::Area) ? 1.0 : std::max(1.0, ratio);
    const int maxTaps = method == Resample::Nearest ? 1 : static_cast<int>(std::ceil(2.0 * support * filterScale)) + (method == Resample::Area ? static_cast<int>(std::ceil(ratio)) + 1 : 1);
    t.stride = maxTaps;
    t.first.assign(static_cast<size_t>(out), 0);
    t.count.assign(static_cast<size_t>(out), 0);
    t.weights.assign(static_cast<size_t>(out * maxTaps), 0.0f);
    for (int64_t o = 0; o < out; ++o) {
        float* w = &t.weights[static_cast<size_t>(o * maxTaps)];
        if (method == Resample::Nearest) {
            t.first[static_cast<size_t>(o)] = std::min(in - 1, static_cast<int64_t>(std::floor(o * ratio)));
            t.count[static_cast<size_t>(o)] = 1;
            w[0] = 1.0f;
            continue;
        }
        if (method == Resample::Area) {   // box average over [o*ratio, (o+1)*ratio), with fractional edge weights
            const double lo = o * ratio, hi = (o + 1) * ratio;
            const int64_t i0 = static_cast<int64_t>(std::floor(lo)), i1 = std::min(in, static_cast<int64_t>(std::ceil(hi)));
            double total = 0;
            int n = 0;
            for (int64_t i = i0; i < i1 && n < maxTaps; ++i, ++n) {
                const double cover = std::min(static_cast<double>(i + 1), hi) - std::max(static_cast<double>(i), lo);
                w[n] = static_cast<float>(cover);
                total += cover;
            }
            for (int k = 0; k < n; ++k) w[k] = static_cast<float>(w[k] / total);
            t.first[static_cast<size_t>(o)] = i0;
            t.count[static_cast<size_t>(o)] = n;
            continue;
        }
        const double center = (o + 0.5) * ratio;   // in source coordinates, pixel centres at +0.5
        const int64_t lo = static_cast<int64_t>(std::floor(center - support * filterScale + 0.5));
        const int64_t hi = static_cast<int64_t>(std::floor(center + support * filterScale + 0.5));
        double total = 0;
        int n = 0;
        std::vector<double> raw;
        for (int64_t i = lo; i < hi && n < maxTaps; ++i, ++n) {
            const double v = kernel((i + 0.5 - center) / filterScale);
            raw.push_back(v);
            total += v;
        }
        // positions outside the image replicate the edge sample: fold their weight onto the edge
        std::vector<double> folded(static_cast<size_t>(in), 0.0);
        for (int k = 0; k < n; ++k) {
            const int64_t idx = std::min(in - 1, std::max<int64_t>(0, lo + k));
            folded[static_cast<size_t>(idx)] += raw[static_cast<size_t>(k)];
        }
        const int64_t f0 = std::min(in - 1, std::max<int64_t>(0, lo)), f1 = std::min(in - 1, std::max<int64_t>(0, lo + n - 1));
        int m = 0;
        for (int64_t i = f0; i <= f1 && m < maxTaps; ++i, ++m) w[m] = static_cast<float>(folded[static_cast<size_t>(i)] / total);
        t.first[static_cast<size_t>(o)] = f0;
        t.count[static_cast<size_t>(o)] = m;
    }
    return t;
}

}  // namespace

Tensor resize(const Tensor& a, int64_t height, int64_t width, Resample method) {
    const auto d = imageDims(a);
    if (height <= 0 || width <= 0) throw std::invalid_argument("resize to nothing");
    if (height == d.h && width == d.w) return a;
    const Taps tx = makeTaps(d.w, width, method), ty = makeTaps(d.h, height, method);
    // horizontal pass: [B, H, W, C] -> [B, H, width, C]
    Tensor mid({d.b, d.h, width, d.c});
    parallelFor(d.b * d.h, [&](int64_t row) {
        const float* src = a.data() + row * d.w * d.c;
        float* dst = mid.data() + row * width * d.c;
        for (int64_t x = 0; x < width; ++x) {
            const float* w = &tx.weights[static_cast<size_t>(x * tx.stride)];
            const int n = tx.count[static_cast<size_t>(x)];
            const int64_t f = tx.first[static_cast<size_t>(x)];
            for (int64_t c = 0; c < d.c; ++c) {
                float acc = 0;
                for (int k = 0; k < n; ++k) acc += w[k] * src[(f + k) * d.c + c];
                dst[x * d.c + c] = acc;
            }
        }
    });
    // vertical pass
    Tensor out = makeImage(a, height, width);
    parallelFor(d.b * height, [&](int64_t row) {
        const int64_t b = row / height, y = row % height;
        const float* w = &ty.weights[static_cast<size_t>(y * ty.stride)];
        const int n = ty.count[static_cast<size_t>(y)];
        const int64_t f = ty.first[static_cast<size_t>(y)];
        float* dst = out.data() + row * width * d.c;
        for (int64_t i = 0; i < width * d.c; ++i) {
            float acc = 0;
            for (int k = 0; k < n; ++k) acc += w[k] * mid.data()[((b * d.h + f + k) * width) * d.c + i];
            dst[i] = acc;
        }
    });
    return out;
}

Tensor gaussianBlur(const Tensor& a, int radius, float sigma) {
    const auto d = imageDims(a);
    if (radius <= 0 || sigma <= 0) return a;
    std::vector<float> k(static_cast<size_t>(2 * radius + 1));
    float total = 0;
    for (int i = -radius; i <= radius; ++i) {
        k[static_cast<size_t>(i + radius)] = std::exp(-0.5f * (i * i) / (sigma * sigma));
        total += k[static_cast<size_t>(i + radius)];
    }
    for (auto& v : k) v /= total;
    Tensor mid(a.shape()), out(a.shape());
    parallelFor(d.b * d.h, [&](int64_t row) {
        const float* src = a.data() + row * d.w * d.c;
        float* dst = mid.data() + row * d.w * d.c;
        for (int64_t x = 0; x < d.w; ++x)
            for (int64_t c = 0; c < d.c; ++c) {
                float acc = 0;
                for (int i = -radius; i <= radius; ++i) acc += k[static_cast<size_t>(i + radius)] * src[std::min(d.w - 1, std::max<int64_t>(0, x + i)) * d.c + c];
                dst[x * d.c + c] = acc;
            }
    });
    parallelFor(d.b * d.h, [&](int64_t row) {
        const int64_t b = row / d.h, y = row % d.h;
        float* dst = out.data() + row * d.w * d.c;
        for (int64_t i = 0; i < d.w * d.c; ++i) {
            float acc = 0;
            for (int j = -radius; j <= radius; ++j) acc += k[static_cast<size_t>(j + radius)] * mid.data()[(b * d.h + std::min(d.h - 1, std::max<int64_t>(0, y + j))) * d.w * d.c + i];
            dst[i] = acc;
        }
    });
    return out;
}

Tensor sharpen(const Tensor& a, int radius, float sigma, float amount) {
    const Tensor blurred = gaussianBlur(a, radius, sigma);
    return clamp(add(a, scale(sub(a, blurred), amount)), 0.0f, 1.0f);
}

namespace {
Tensor morphology(const Tensor& a, int radius, bool maximum) {
    const auto d = imageDims(a);
    if (radius <= 0) return a;
    Tensor mid(a.shape()), out(a.shape());
    auto pick = [maximum](float x, float y) { return maximum ? std::max(x, y) : std::min(x, y); };
    parallelFor(d.b * d.h, [&](int64_t row) {
        for (int64_t x = 0; x < d.w; ++x)
            for (int64_t c = 0; c < d.c; ++c) {
                float v = a.data()[(row * d.w + x) * d.c + c];
                for (int i = -radius; i <= radius; ++i) v = pick(v, a.data()[(row * d.w + std::min(d.w - 1, std::max<int64_t>(0, x + i))) * d.c + c]);
                mid.data()[(row * d.w + x) * d.c + c] = v;
            }
    });
    parallelFor(d.b * d.h, [&](int64_t row) {
        const int64_t b = row / d.h, y = row % d.h;
        for (int64_t i = 0; i < d.w * d.c; ++i) {
            float v = mid.data()[row * d.w * d.c + i];
            for (int j = -radius; j <= radius; ++j) v = pick(v, mid.data()[(b * d.h + std::min(d.h - 1, std::max<int64_t>(0, y + j))) * d.w * d.c + i]);
            out.data()[row * d.w * d.c + i] = v;
        }
    });
    return out;
}
}  // namespace

Tensor dilate(const Tensor& a, int radius) { return morphology(a, radius, true); }
Tensor erode(const Tensor& a, int radius) { return morphology(a, radius, false); }

// ---- conversion ---------------------------------------------------------------------------------------------------------
Tensor fromImage(const engine::Image& image) {
    if (image.width == 0 || image.height == 0 || (image.channels != 3 && image.channels != 4) ||
        image.data.size() < static_cast<size_t>(image.width) * image.height * image.channels) {
        throw std::invalid_argument("the image is empty or inconsistent");
    }
    const int64_t h = image.height, w = image.width;
    Tensor out({1, h, w, 3});
    parallelFor(h, [&](int64_t y) {
        for (int64_t x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) out.data()[(y * w + x) * 3 + c] = image.data[static_cast<size_t>((y * w + x) * image.channels + c)] / 255.0f;
    });
    return out;
}

Tensor alphaMask(const engine::Image& image) {
    const int64_t h = image.height, w = image.width;
    Tensor out({1, h, w}, 1.0f);
    if (image.channels == 4) {
        for (int64_t i = 0; i < h * w; ++i) out.data()[i] = image.data[static_cast<size_t>(i * 4 + 3)] / 255.0f;
    }
    return out;
}

engine::Image toImage(const Tensor& a, int64_t index) {
    const auto d = imageDims(a);
    if (index < 0 || index >= d.b) throw std::out_of_range("batch index");
    if (d.c != 3 && d.c != 1) throw std::invalid_argument("expected 1 or 3 channels");
    engine::Image out;
    out.width = static_cast<uint32_t>(d.w);
    out.height = static_cast<uint32_t>(d.h);
    out.channels = 3;
    out.data.resize(static_cast<size_t>(d.w * d.h * 3));
    const float* src = a.data() + index * d.h * d.w * d.c;
    for (int64_t i = 0; i < d.h * d.w; ++i)
        for (int c = 0; c < 3; ++c) {
            const float v = src[i * d.c + (d.c == 1 ? 0 : c)];
            out.data[static_cast<size_t>(i * 3 + c)] = static_cast<uint8_t>(std::lround(std::min(1.0f, std::max(0.0f, v)) * 255.0f));
        }
    return out;
}

}  // namespace gh::tensor
