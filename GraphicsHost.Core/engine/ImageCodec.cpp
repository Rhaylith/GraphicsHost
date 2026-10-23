#include "engine/ImageCodec.h"

#include <png.h>

#include <cstdio>  // jpeglib.h needs FILE
#include <jpeglib.h>

#include <webp/decode.h>
#include <webp/encode.h>
#include <webp/mux.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace gh::engine {

namespace {

constexpr const char* kPngKeyword = "graphicshost";

void requireConsistent(const Image& image) {
    if (image.width == 0 || image.height == 0 || (image.channels != 3 && image.channels != 4) ||
        image.data.size() != static_cast<size_t>(image.width) * image.height * image.channels) {
        throw std::runtime_error("image has inconsistent dimensions or pixel data");
    }
}

// RGB rows (alpha dropped), for the encoders that take RGB only.
std::vector<uint8_t> toRgb(const Image& image) {
    if (image.channels == 3) {
        return image.data;
    }
    std::vector<uint8_t> rgb(static_cast<size_t>(image.width) * image.height * 3);
    for (size_t i = 0, n = static_cast<size_t>(image.width) * image.height; i < n; ++i) {
        std::memcpy(&rgb[i * 3], &image.data[i * 4], 3);
    }
    return rgb;
}

// ---- PNG -------------------------------------------------------------------------------------------------------

struct PngWriteState {
    std::vector<uint8_t>* out;
};

[[noreturn]] void pngError(png_structp, png_const_charp message) {
    throw std::runtime_error(std::string("PNG error: ") + (message ? message : "unknown"));
}
void pngWarning(png_structp, png_const_charp) {}

std::vector<uint8_t> encodePng(const Image& image, const std::optional<std::string>& metadata) {
    std::vector<uint8_t> out;
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, pngError, pngWarning);
    png_infop info = png ? png_create_info_struct(png) : nullptr;
    if (!png || !info) {
        if (png) png_destroy_write_struct(&png, nullptr);
        throw std::runtime_error("cannot create a PNG writer");
    }
    try {
        png_set_write_fn(
            png, &out,
            [](png_structp p, png_bytep data, size_t length) {
                auto* v = static_cast<std::vector<uint8_t>*>(png_get_io_ptr(p));
                v->insert(v->end(), data, data + length);
            },
            [](png_structp) {});
        png_set_IHDR(png, info, image.width, image.height, 8,
                     image.channels == 4 ? PNG_COLOR_TYPE_RGB_ALPHA : PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_text text{};
        std::string keyword = kPngKeyword;
        std::string value;
        if (metadata) {
            value = *metadata;
            text.compression = value.size() > 1024 ? PNG_TEXT_COMPRESSION_zTXt : PNG_TEXT_COMPRESSION_NONE;
            text.key = keyword.data();
            text.text = value.data();
            text.text_length = value.size();
            png_set_text(png, info, &text, 1);
        }
        png_write_info(png, info);
        for (uint32_t y = 0; y < image.height; ++y) {
            png_write_row(png, const_cast<png_bytep>(&image.data[static_cast<size_t>(y) * image.width * image.channels]));
        }
        png_write_end(png, info);
    } catch (...) {
        png_destroy_write_struct(&png, &info);
        throw;
    }
    png_destroy_write_struct(&png, &info);
    return out;
}

struct PngReadState {
    const std::vector<uint8_t>* bytes;
    size_t pos = 0;
};

// Reads the PNG; fills image (as RGB) when wanted and the metadata when present.
void readPng(const std::vector<uint8_t>& bytes, Image* image, std::optional<std::string>* metadata) {
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, pngError, pngWarning);
    png_infop info = png ? png_create_info_struct(png) : nullptr;
    if (!png || !info) {
        if (png) png_destroy_read_struct(&png, nullptr, nullptr);
        throw std::runtime_error("cannot create a PNG reader");
    }
    PngReadState state{&bytes, 0};
    try {
        png_set_read_fn(png, &state, [](png_structp p, png_bytep out, size_t length) {
            auto* s = static_cast<PngReadState*>(png_get_io_ptr(p));
            if (s->pos + length > s->bytes->size()) {
                png_error(p, "unexpected end of data");
            }
            std::memcpy(out, s->bytes->data() + s->pos, length);
            s->pos += length;
        });
        png_read_info(png, info);
        if (metadata) {
            png_textp texts = nullptr;
            int count = 0;
            if (png_get_text(png, info, &texts, &count) > 0) {
                for (int i = 0; i < count; ++i) {
                    if (std::strcmp(texts[i].key, kPngKeyword) == 0) {
                        *metadata = std::string(texts[i].text, texts[i].text_length);
                    }
                }
            }
        }
        if (image) {
            const uint32_t width = png_get_image_width(png, info);
            const uint32_t height = png_get_image_height(png, info);
            png_set_strip_16(png);
            png_set_palette_to_rgb(png);
            png_set_expand_gray_1_2_4_to_8(png);
            png_set_gray_to_rgb(png);
            png_set_strip_alpha(png);
            png_read_update_info(png, info);
            if (png_get_channels(png, info) != 3) {
                png_error(png, "unsupported PNG layout");
            }
            image->width = width;
            image->height = height;
            image->channels = 3;
            image->data.resize(static_cast<size_t>(width) * height * 3);
            for (uint32_t y = 0; y < height; ++y) {
                png_read_row(png, &image->data[static_cast<size_t>(y) * width * 3], nullptr);
            }
        }
    } catch (...) {
        png_destroy_read_struct(&png, &info, nullptr);
        throw;
    }
    png_destroy_read_struct(&png, &info, nullptr);
}

// ---- JPEG ------------------------------------------------------------------------------------------------------

struct JpegErrorManager {
    jpeg_error_mgr base;
};

[[noreturn]] void jpegErrorExit(j_common_ptr cinfo) {
    char buffer[JMSG_LENGTH_MAX];
    (*cinfo->err->format_message)(cinfo, buffer);
    throw std::runtime_error(std::string("JPEG error: ") + buffer);
}

std::vector<uint8_t> encodeJpeg(const Image& image, const std::optional<std::string>& metadata, int quality) {
    const std::vector<uint8_t> rgb = toRgb(image);
    jpeg_compress_struct cinfo{};
    JpegErrorManager errors{};
    cinfo.err = jpeg_std_error(&errors.base);
    errors.base.error_exit = jpegErrorExit;
    unsigned char* buffer = nullptr;
    unsigned long size = 0;
    jpeg_create_compress(&cinfo);
    try {
        jpeg_mem_dest(&cinfo, &buffer, &size);
        cinfo.image_width = image.width;
        cinfo.image_height = image.height;
        cinfo.input_components = 3;
        cinfo.in_color_space = JCS_RGB;
        jpeg_set_defaults(&cinfo);
        jpeg_set_quality(&cinfo, std::clamp(quality, 1, 100), TRUE);
        jpeg_start_compress(&cinfo, TRUE);
        if (metadata) {
            jpeg_write_marker(&cinfo, JPEG_COM, reinterpret_cast<const JOCTET*>(metadata->data()),
                              static_cast<unsigned>(std::min<size_t>(metadata->size(), 65000)));
        }
        while (cinfo.next_scanline < cinfo.image_height) {
            JSAMPROW row = const_cast<JSAMPROW>(&rgb[static_cast<size_t>(cinfo.next_scanline) * image.width * 3]);
            jpeg_write_scanlines(&cinfo, &row, 1);
        }
        jpeg_finish_compress(&cinfo);
    } catch (...) {
        jpeg_destroy_compress(&cinfo);
        std::free(buffer);
        throw;
    }
    jpeg_destroy_compress(&cinfo);
    std::vector<uint8_t> out(buffer, buffer + size);
    std::free(buffer);
    return out;
}

void readJpeg(const std::vector<uint8_t>& bytes, Image* image, std::optional<std::string>* metadata) {
    jpeg_decompress_struct cinfo{};
    JpegErrorManager errors{};
    cinfo.err = jpeg_std_error(&errors.base);
    errors.base.error_exit = jpegErrorExit;
    jpeg_create_decompress(&cinfo);
    try {
        jpeg_mem_src(&cinfo, bytes.data(), static_cast<unsigned long>(bytes.size()));
        if (metadata) {
            jpeg_save_markers(&cinfo, JPEG_COM, 0xFFFF);
        }
        jpeg_read_header(&cinfo, TRUE);
        if (metadata) {
            for (jpeg_saved_marker_ptr m = cinfo.marker_list; m; m = m->next) {
                if (m->marker == JPEG_COM) {
                    *metadata = std::string(reinterpret_cast<const char*>(m->data), m->data_length);
                    break;
                }
            }
        }
        if (image) {
            cinfo.out_color_space = JCS_RGB;
            jpeg_start_decompress(&cinfo);
            image->width = cinfo.output_width;
            image->height = cinfo.output_height;
            image->channels = 3;
            image->data.resize(static_cast<size_t>(image->width) * image->height * 3);
            while (cinfo.output_scanline < cinfo.output_height) {
                JSAMPROW row = &image->data[static_cast<size_t>(cinfo.output_scanline) * image->width * 3];
                jpeg_read_scanlines(&cinfo, &row, 1);
            }
            jpeg_finish_decompress(&cinfo);
        }
    } catch (...) {
        jpeg_destroy_decompress(&cinfo);
        throw;
    }
    jpeg_destroy_decompress(&cinfo);
}

// ---- WebP ------------------------------------------------------------------------------------------------------

struct WebpMemory {
    uint8_t* data = nullptr;
    size_t size = 0;
    ~WebpMemory() { WebPFree(data); }
};

std::vector<uint8_t> encodeWebp(const Image& image, const std::optional<std::string>& metadata, int quality) {
    const std::vector<uint8_t> rgb = toRgb(image);
    WebpMemory encoded;
    const int stride = static_cast<int>(image.width) * 3;
    if (quality >= 100) {
        encoded.size = WebPEncodeLosslessRGB(rgb.data(), static_cast<int>(image.width), static_cast<int>(image.height),
                                             stride, &encoded.data);
    } else {
        encoded.size = WebPEncodeRGB(rgb.data(), static_cast<int>(image.width), static_cast<int>(image.height), stride,
                                     static_cast<float>(std::max(1, quality)), &encoded.data);
    }
    if (encoded.size == 0) {
        throw std::runtime_error("WebP encoder failed");
    }
    if (!metadata) {
        return std::vector<uint8_t>(encoded.data, encoded.data + encoded.size);
    }
    WebPData source{encoded.data, encoded.size};
    WebPMux* mux = WebPMuxCreate(&source, 1);
    if (!mux) {
        throw std::runtime_error("cannot open the WebP for metadata");
    }
    WebPData xmp{reinterpret_cast<const uint8_t*>(metadata->data()), metadata->size()};
    const WebPMuxError setError = WebPMuxSetChunk(mux, "XMP ", &xmp, 1);
    WebPData assembled{};
    const WebPMuxError assembleError = setError == WEBP_MUX_OK ? WebPMuxAssemble(mux, &assembled) : setError;
    WebPMuxDelete(mux);
    if (assembleError != WEBP_MUX_OK) {
        WebPDataClear(&assembled);
        throw std::runtime_error("cannot add metadata to the WebP");
    }
    std::vector<uint8_t> out(assembled.bytes, assembled.bytes + assembled.size);
    WebPDataClear(&assembled);
    return out;
}

std::optional<std::string> readWebpMetadata(const std::vector<uint8_t>& bytes) {
    WebPData data{bytes.data(), bytes.size()};
    WebPMux* mux = WebPMuxCreate(&data, 0);
    if (!mux) {
        return std::nullopt;
    }
    WebPData chunk{};
    std::optional<std::string> result;
    if (WebPMuxGetChunk(mux, "XMP ", &chunk) == WEBP_MUX_OK && chunk.size > 0) {
        result = std::string(reinterpret_cast<const char*>(chunk.bytes), chunk.size);
    }
    WebPMuxDelete(mux);
    return result;
}

Image decodeWebp(const std::vector<uint8_t>& bytes) {
    int width = 0, height = 0;
    if (!WebPGetInfo(bytes.data(), bytes.size(), &width, &height)) {
        throw std::runtime_error("not a valid WebP");
    }
    Image image;
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.channels = 3;
    image.data.resize(static_cast<size_t>(width) * height * 3);
    if (!WebPDecodeRGBInto(bytes.data(), bytes.size(), image.data.data(), image.data.size(), width * 3)) {
        throw std::runtime_error("WebP decoder failed");
    }
    return image;
}

}  // namespace

std::optional<ImageFormat> parseImageFormat(std::string_view name) {
    if (name == "png") return ImageFormat::Png;
    if (name == "jpeg" || name == "jpg") return ImageFormat::Jpeg;
    if (name == "webp") return ImageFormat::Webp;
    return std::nullopt;
}

const char* extensionFor(ImageFormat format) {
    switch (format) {
        case ImageFormat::Png: return ".png";
        case ImageFormat::Jpeg: return ".jpg";
        case ImageFormat::Webp: return ".webp";
    }
    return "";
}

const char* mimeTypeFor(ImageFormat format) {
    switch (format) {
        case ImageFormat::Png: return "image/png";
        case ImageFormat::Jpeg: return "image/jpeg";
        case ImageFormat::Webp: return "image/webp";
    }
    return "application/octet-stream";
}

std::vector<uint8_t> encodeImage(const Image& image, ImageFormat format, const std::optional<std::string>& metadata,
                                 int quality) {
    requireConsistent(image);
    switch (format) {
        case ImageFormat::Png: return encodePng(image, metadata);
        case ImageFormat::Jpeg: return encodeJpeg(image, metadata, quality);
        case ImageFormat::Webp: return encodeWebp(image, metadata, quality);
    }
    throw std::runtime_error("unknown image format");
}

std::optional<ImageFormat> detectImageFormat(const std::vector<uint8_t>& b) {
    static const uint8_t png[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (b.size() >= 8 && std::memcmp(b.data(), png, 8) == 0) return ImageFormat::Png;
    if (b.size() >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF) return ImageFormat::Jpeg;
    if (b.size() >= 12 && std::memcmp(b.data(), "RIFF", 4) == 0 && std::memcmp(b.data() + 8, "WEBP", 4) == 0) {
        return ImageFormat::Webp;
    }
    return std::nullopt;
}

std::optional<std::string> readImageMetadata(const std::vector<uint8_t>& bytes) {
    const auto format = detectImageFormat(bytes);
    if (!format) {
        return std::nullopt;
    }
    std::optional<std::string> metadata;
    switch (*format) {
        case ImageFormat::Png: readPng(bytes, nullptr, &metadata); break;
        case ImageFormat::Jpeg: readJpeg(bytes, nullptr, &metadata); break;
        case ImageFormat::Webp: metadata = readWebpMetadata(bytes); break;
    }
    return metadata;
}

Image decodeImage(const std::vector<uint8_t>& bytes) {
    const auto format = detectImageFormat(bytes);
    if (!format) {
        throw std::runtime_error("unknown image format");
    }
    Image image;
    switch (*format) {
        case ImageFormat::Png: readPng(bytes, &image, nullptr); break;
        case ImageFormat::Jpeg: readJpeg(bytes, &image, nullptr); break;
        case ImageFormat::Webp: return decodeWebp(bytes);
    }
    return image;
}

}  // namespace gh::engine

namespace gh::engine {

Image resizeImage(const Image& src, uint32_t width, uint32_t height) {
    if (src.width == 0 || src.height == 0 || width == 0 || height == 0 || (src.channels != 3 && src.channels != 4) ||
        src.data.size() < static_cast<size_t>(src.width) * src.height * src.channels) {
        throw std::runtime_error("resizeImage: the picture is empty or inconsistent");
    }
    if (src.width == width && src.height == height) return src;
    Image out;
    out.width = width;
    out.height = height;
    out.channels = src.channels;
    out.data.resize(static_cast<size_t>(width) * height * src.channels);
    const double sx = static_cast<double>(src.width) / width;
    const double sy = static_cast<double>(src.height) / height;
    for (uint32_t y = 0; y < height; ++y) {
        const double fy = std::max(0.0, (y + 0.5) * sy - 0.5);
        const uint32_t y0 = std::min<uint32_t>(static_cast<uint32_t>(fy), src.height - 1);
        const uint32_t y1 = std::min<uint32_t>(y0 + 1, src.height - 1);
        const double wy = fy - y0;
        for (uint32_t x = 0; x < width; ++x) {
            const double fx = std::max(0.0, (x + 0.5) * sx - 0.5);
            const uint32_t x0 = std::min<uint32_t>(static_cast<uint32_t>(fx), src.width - 1);
            const uint32_t x1 = std::min<uint32_t>(x0 + 1, src.width - 1);
            const double wx = fx - x0;
            for (uint32_t c = 0; c < src.channels; ++c) {
                auto at = [&](uint32_t xx, uint32_t yy) { return static_cast<double>(src.data[(static_cast<size_t>(yy) * src.width + xx) * src.channels + c]); };
                const double top = at(x0, y0) * (1 - wx) + at(x1, y0) * wx;
                const double bottom = at(x0, y1) * (1 - wx) + at(x1, y1) * wx;
                out.data[(static_cast<size_t>(y) * width + x) * src.channels + c] = static_cast<uint8_t>(std::lround(top * (1 - wy) + bottom * wy));
            }
        }
    }
    return out;
}

}  // namespace gh::engine
