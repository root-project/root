#ifndef BVH_V2_BLOB_PERSISTENCE_H
#define BVH_V2_BLOB_PERSISTENCE_H

#include "bvh/v2/bvh.h"
#include "bvh/v2/stream.h"

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace bvh::v2 {

// BVH blob persistence helpers.
//
// Rationale:
// - "Blob" means a compact byte buffer storing serialized BVH data.
// - The blob begins with a small self-describing header (magic + version + traits).
// - If the blob is missing, corrupted, or incompatible with the current Node traits, the caller
//   should rebuild the BVH from source geometry (the blob is a cache, not the source of truth).
// - If the BVH serialization layout changes, bump kBvhBlobVersion and update deserialization to
//   invalidate older blobs.
// - Legacy version 1 blobs did not encode Node traits. If you change Node template parameters and
//   want to force a rebuild of old files, either drop v1 compatibility or bump kBvhBlobVersion.
//
// Usage (example):
//   using Node = bvh::v2::Node<float, 3>;
//   using Bvh  = bvh::v2::Bvh<Node>;
//   std::vector<unsigned char> blob;
//   bvh::v2::SerializeBvhToBlob(bvh, blob);
//   Bvh restored;
//   if (!bvh::v2::DeserializeBvhFromBlob(blob, restored)) {
//      // rebuild BVH from primitives
//   }
//
// Added by andrei.gheata@cern.ch on 03.02.2026

constexpr uint32_t kBvhBlobMagic = 0x54474248U; // "TGBH" marker to detect valid BVH blobs
constexpr uint32_t kBvhBlobEndianTag = 0x01020304U;
constexpr uint16_t kBvhBlobVersion = 2; // serialized BVH blob format version
constexpr uint16_t kBvhBlobHeaderSizeV2 = 14;
constexpr uint16_t kBvhBlobScalarFlagFloat = 1u << 0;
constexpr uint16_t kBvhBlobScalarFlagSigned = 1u << 1;

namespace detail {

inline void AppendU16(std::vector<unsigned char> &out, uint16_t value)
{
   out.push_back(static_cast<unsigned char>(value & 0xFFu));
   out.push_back(static_cast<unsigned char>((value >> 8) & 0xFFu));
}

inline void AppendU32(std::vector<unsigned char> &out, uint32_t value)
{
   out.push_back(static_cast<unsigned char>(value & 0xFFu));
   out.push_back(static_cast<unsigned char>((value >> 8) & 0xFFu));
   out.push_back(static_cast<unsigned char>((value >> 16) & 0xFFu));
   out.push_back(static_cast<unsigned char>((value >> 24) & 0xFFu));
}

inline bool ReadU16(const std::vector<unsigned char> &data, size_t &offset, uint16_t &value)
{
   if (offset + 2 > data.size())
      return false;
   value = static_cast<uint16_t>(data[offset]) | (static_cast<uint16_t>(data[offset + 1]) << 8);
   offset += 2;
   return true;
}

inline bool ReadU32(const std::vector<unsigned char> &data, size_t &offset, uint32_t &value)
{
   if (offset + 4 > data.size())
      return false;
   value = static_cast<uint32_t>(data[offset]) | (static_cast<uint32_t>(data[offset + 1]) << 8) |
           (static_cast<uint32_t>(data[offset + 2]) << 16) | (static_cast<uint32_t>(data[offset + 3]) << 24);
   offset += 4;
   return true;
}

class BvhVectorOutputStream final : public bvh::v2::OutputStream {
public:
   explicit BvhVectorOutputStream(std::vector<unsigned char> &data) : data_(data) {}

protected:
   std::vector<unsigned char> &data_;

   bool write_raw(const void *data, size_t size) override
   {
      const auto *bytes = static_cast<const unsigned char *>(data);
      data_.insert(data_.end(), bytes, bytes + size);
      return true;
   }
};

class BvhVectorInputStream final : public bvh::v2::InputStream {
public:
   explicit BvhVectorInputStream(const std::vector<unsigned char> &data, size_t offset)
      : data_(data), offset_(offset), ok_(true)
   {
   }

   bool ok() const { return ok_; }

protected:
   const std::vector<unsigned char> &data_;
   size_t offset_;
   bool ok_;

   size_t read_raw(void *data, size_t size) override
   {
      if (!ok_ || offset_ + size > data_.size()) {
         ok_ = false;
         return 0;
      }
      std::memcpy(data, data_.data() + offset_, size);
      offset_ += size;
      return size;
   }
};

} // namespace detail

// Serialize the BVH into a blob with a small header (magic + version + Node traits).
template <typename Node>
inline bool SerializeBvhToBlob(const bvh::v2::Bvh<Node> &bvh, std::vector<unsigned char> &out)
{
   out.clear();
   detail::AppendU32(out, kBvhBlobMagic);
   detail::AppendU16(out, kBvhBlobVersion);
   detail::AppendU16(out, kBvhBlobHeaderSizeV2);
   detail::AppendU16(out, static_cast<uint16_t>(Node::dimension));
   detail::AppendU16(out, static_cast<uint16_t>(Node::index_bits));
   detail::AppendU16(out, static_cast<uint16_t>(Node::prim_count_bits));
   detail::AppendU16(out, static_cast<uint16_t>(sizeof(typename Node::Scalar)));
   uint16_t scalar_flags = 0;
   if constexpr (std::is_floating_point_v<typename Node::Scalar>) {
      scalar_flags |= kBvhBlobScalarFlagFloat;
   }
   if constexpr (std::is_signed_v<typename Node::Scalar>) {
      scalar_flags |= kBvhBlobScalarFlagSigned;
   }
   detail::AppendU16(out, scalar_flags);
   detail::AppendU32(out, kBvhBlobEndianTag);
   detail::BvhVectorOutputStream stream(out);
   bvh.serialize(stream);
   return true;
}

// Deserialize the BVH from a blob; returns false if the header or payload is invalid.
template <typename Node>
inline bool DeserializeBvhFromBlob(const std::vector<unsigned char> &data, bvh::v2::Bvh<Node> &out)
{
   if (data.size() < 8)
      return false;
   size_t offset = 0;
   uint32_t magic = 0;
   uint16_t version = 0;
   uint16_t header_size = 0;
   if (!detail::ReadU32(data, offset, magic) || magic != kBvhBlobMagic)
      return false;
   if (!detail::ReadU16(data, offset, version) || version == 0)
      return false;
   if (!detail::ReadU16(data, offset, header_size))
      return false;
   if (version == 1) {
      detail::BvhVectorInputStream stream(data, offset);
      out = bvh::v2::Bvh<Node>::deserialize(stream);
      if (!stream.ok())
         return false;
      return !out.nodes.empty();
   }
   if (version != kBvhBlobVersion)
      return false;
   if (header_size < kBvhBlobHeaderSizeV2)
      return false;
   if (offset + header_size > data.size())
      return false;
   uint16_t dim = 0;
   uint16_t index_bits = 0;
   uint16_t prim_count_bits = 0;
   uint16_t scalar_size = 0;
   uint16_t scalar_flags = 0;
   uint32_t endian_tag = 0;
   if (!detail::ReadU16(data, offset, dim) || !detail::ReadU16(data, offset, index_bits) ||
       !detail::ReadU16(data, offset, prim_count_bits) || !detail::ReadU16(data, offset, scalar_size) ||
       !detail::ReadU16(data, offset, scalar_flags) || !detail::ReadU32(data, offset, endian_tag)) {
      return false;
   }
   if (endian_tag != kBvhBlobEndianTag)
      return false;
   const uint16_t expected_flags = (std::is_floating_point_v<typename Node::Scalar> ? kBvhBlobScalarFlagFloat : 0) |
                                   (std::is_signed_v<typename Node::Scalar> ? kBvhBlobScalarFlagSigned : 0);
   if (dim != Node::dimension || index_bits != Node::index_bits || prim_count_bits != Node::prim_count_bits ||
       scalar_size != sizeof(typename Node::Scalar) || scalar_flags != expected_flags) {
      return false;
   }
   const size_t extra = static_cast<size_t>(header_size - kBvhBlobHeaderSizeV2);
   if (extra > 0) {
      if (offset + extra > data.size())
         return false;
      offset += extra;
   }
   detail::BvhVectorInputStream stream(data, offset);
   out = bvh::v2::Bvh<Node>::deserialize(stream);
   if (!stream.ok())
      return false;
   return !out.nodes.empty();
}

} // namespace bvh::v2

#endif
