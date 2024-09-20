#ifndef BVH_V2_STREAM_H
#define BVH_V2_STREAM_H

#include <istream>
#include <ostream>

namespace bvh::v2 {

/// Stream of data that can be used to deserialize data structures.
class InputStream {
public:
    template <typename T>
    T read(T&& default_val = {}) {
        T data;
        if (read_raw(&data, sizeof(T)) != sizeof(T))
            data = std::move(default_val);
        return data;
    }

protected:
    virtual size_t read_raw(void*, size_t) = 0;
};

/// Stream of data that can be used to serialize data structures.
class OutputStream {
public:
    template <typename T>
    bool write(const T& data) { return write_raw(&data, sizeof(T)); }

protected:
    virtual bool write_raw(const void*, size_t) = 0;
};

/// Stream adapter for standard library input streams.
class StdInputStream : public InputStream {
public:
    StdInputStream(std::istream& stream)
        : stream_(stream)
    {}

    using InputStream::read;

protected:
    std::istream& stream_;

    size_t read_raw(void* data, size_t size) override {
        stream_.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
        return static_cast<size_t>(stream_.gcount());
    }
};

/// Stream adapter for standard library output streams.
class StdOutputStream : public OutputStream {
public:
    StdOutputStream(std::ostream& stream)
        : stream_(stream)
    {}

    using OutputStream::write;

protected:
    std::ostream& stream_;

    bool write_raw(const void* data, size_t size) override {
        stream_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        return stream_.good();
    }
};

} // namespace bvh::v2

#endif
