// RColumnElement concrete implementations
//
// Note that this file is in the src directory and not in the inc directory because we need the ability
// to override R__LITTLE_ENDIAN for testing purposes.
// This is not a particularly clean or correct solution, as the tests that do this will end up with two different
// definitions of some RColumnElements, so we might want to change this mechanism in the future. In any case, these
// definitions are implementation details and should not be exposed to a public interface.

#include <ROOT/RColumnElementBase.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RConfig.hxx>
#include <ROOT/RError.hxx>
#include <Byteswap.h>

#include <bitset>
#include <cassert>
#include <limits>
#include <type_traits>
#include <cmath>

// NOTE: some tests might define R__LITTLE_ENDIAN to simulate a different-endianness machine
#ifndef R__LITTLE_ENDIAN
#ifdef R__BYTESWAP
// `R__BYTESWAP` is defined in RConfig.hxx for little-endian architectures; undefined otherwise
#define R__LITTLE_ENDIAN 1
#else
#define R__LITTLE_ENDIAN 0
#endif
#endif /* R__LITTLE_ENDIAN */

namespace ROOT::Experimental::Internal::BitPacking {

using Word_t = std::uintmax_t;
inline constexpr std::size_t kBitsPerWord = sizeof(Word_t) * 8;

/// Returns the minimum safe size (in bytes) of a buffer that is intended to be used as a destination for PackBits
/// or a source for UnpackBits.
/// Passing a buffer that's less than this size will cause invalid memory reads and writes.
constexpr std::size_t MinBufSize(std::size_t count, std::size_t nDstBits)
{
   return (count * nDstBits + 7) / 8;
}

/// Tightly packs `count` items of size `sizeofSrc` contained in `src` into `dst` using `nDstBits` per item.
/// It must be  `0 < sizeofSrc <= 8`  and  `0 < nDstBits <= sizeofSrc * 8`.
/// The extra least significant bits are dropped (assuming LE ordering of the items in `src`).
/// Note that this function doesn't do any byte reordering for you.
/// IMPORTANT: the size of `dst` must be at least `MinBufSize(count, nBitBits)`
void PackBits(void *dst, const void *src, std::size_t count, std::size_t sizeofSrc, std::size_t nDstBits);

/// Undoes the effect of `PackBits`. The bits that were truncated in the packed representation
/// are filled with zeroes.
/// `src` must be at least `MinBufSize(count, nDstBits)` bytes long.
/// `dst` must be at least `count * sizeofDst` bytes long.
void UnpackBits(void *dst, const void *src, std::size_t count, std::size_t sizeofDst, std::size_t nSrcBits);

} // namespace ROOT::Experimental::Internal::BitPacking

namespace {

// In this namespace, common routines are defined for element packing and unpacking of ints and floats.
// The following conversions and encodings exist:
//
//   - Byteswap:  on big endian machines, ints and floats are byte-swapped to the little endian on-disk format
//   - Cast:      in-memory values can be stored in narrower on-disk columns.  Currently without bounds checks.
//                For instance, for Double32_t, an in-memory double value is stored as a float on disk.
//   - Split:     rearranges the bytes of an array of elements such that all the first bytes are stored first,
//                followed by all the second bytes, etc. This often clusters similar values, e.g. all the zero bytes
//                for arrays of small integers.
//   - Delta:     Delta encoding stores on disk the delta to the previous element.  This is useful for offsets,
//                because it transforms potentially large offset values into small deltas, which are then better
//                suited for split encoding.
//   - Zigzag:    Zigzag encoding is used on signed integers only. It maps x to 2x if x is positive and to -(2x+1) if
//                x is negative. For series of positive and negative values of small absolute value, it will produce
//                a bit pattern that is favorable for split encoding.
//
// Encodings/conversions can be fused:
//
//  - Delta/Zigzag + Splitting (there is no only-delta/zigzag encoding)
//  - (Delta/Zigzag + ) Splitting + Casting
//  - Everything + Byteswap

/// \brief Copy and byteswap `count` elements of size `N` from `source` to `destination`.
///
/// Used on big-endian architectures for packing/unpacking elements whose column type requires
/// a little-endian on-disk representation.
template <std::size_t N>
inline void CopyBswap(void *destination, const void *source, std::size_t count)
{
   auto dst = reinterpret_cast<typename RByteSwap<N>::value_type *>(destination);
   auto src = reinterpret_cast<const typename RByteSwap<N>::value_type *>(source);
   for (std::size_t i = 0; i < count; ++i) {
      dst[i] = RByteSwap<N>::bswap(src[i]);
   }
}

template <std::size_t N>
void InPlaceBswap(void *array, std::size_t count)
{
   auto arr = reinterpret_cast<typename RByteSwap<N>::value_type *>(array);
   for (std::size_t i = 0; i < count; ++i) {
      arr[i] = RByteSwap<N>::bswap(arr[i]);
   }
}

/// Casts T to one of the ints used in RByteSwap and back to its original type, which may be float or double
#if R__LITTLE_ENDIAN == 0
template <typename T>
inline void ByteSwapIfNecessary(T &value)
{
   constexpr auto N = sizeof(T);
   using bswap_value_type = typename RByteSwap<N>::value_type;
   void *valuePtr = &value;
   auto swapped = RByteSwap<N>::bswap(*reinterpret_cast<bswap_value_type *>(valuePtr));
   *reinterpret_cast<bswap_value_type *>(valuePtr) = swapped;
}
template <>
inline void ByteSwapIfNecessary<char>(char &)
{
}
template <>
inline void ByteSwapIfNecessary<signed char>(signed char &)
{
}
template <>
inline void ByteSwapIfNecessary<unsigned char>(unsigned char &)
{
}
#else
#define ByteSwapIfNecessary(x) ((void)0)
#endif

/// For integral types, ensures that the value of type SourceT is representable as DestT
template <typename DestT, typename SourceT>
inline void EnsureValidRange(SourceT val [[maybe_unused]])
{
   using ROOT::RException;

   if constexpr (!std::is_integral_v<DestT> || !std::is_integral_v<SourceT>)
      return;

   if constexpr (static_cast<double>(std::numeric_limits<SourceT>::min()) <
                 static_cast<double>(std::numeric_limits<DestT>::min())) {
      if constexpr (!std::is_signed_v<DestT>) {
         if (val < 0) {
            throw RException(R__FAIL(std::string("value out of range: ") + std::to_string(val) + " for type " +
                                     typeid(DestT).name()));
         }
      } else if (val < std::numeric_limits<DestT>::min()) {
         throw RException(
            R__FAIL(std::string("value out of range: ") + std::to_string(val) + " for type " + typeid(DestT).name()));
      }
   }

   if constexpr (static_cast<double>(std::numeric_limits<SourceT>::max()) >
                 static_cast<double>(std::numeric_limits<DestT>::max())) {
      if (val > std::numeric_limits<DestT>::max()) {
         throw RException(
            R__FAIL(std::string("value out of range: ") + std::to_string(val) + " for type " + typeid(DestT).name()));
      }
   }
}

/// \brief Pack `count` elements into narrower (or wider) type
///
/// Used to convert in-memory elements to smaller column types of comatible types
/// (e.g., double to float, int64 to int32). Takes care of byte swap if necessary.
template <typename DestT, typename SourceT>
inline void CastPack(void *destination, const void *source, std::size_t count)
{
   static_assert(std::is_convertible_v<SourceT, DestT>);
   auto dst = reinterpret_cast<DestT *>(destination);
   auto src = reinterpret_cast<const SourceT *>(source);
   for (std::size_t i = 0; i < count; ++i) {
      dst[i] = src[i];
      ByteSwapIfNecessary(dst[i]);
   }
}

/// \brief Unpack `count` on-disk elements into wider (or narrower) in-memory array
///
/// Used to convert on-disk elements to larger C++ types of comatible types
/// (e.g., float to double, int32 to int64). Takes care of byte swap if necessary.
template <typename DestT, typename SourceT>
inline void CastUnpack(void *destination, const void *source, std::size_t count)
{
   auto dst = reinterpret_cast<DestT *>(destination);
   auto src = reinterpret_cast<const SourceT *>(source);
   for (std::size_t i = 0; i < count; ++i) {
      SourceT val = src[i];
      ByteSwapIfNecessary(val);
      EnsureValidRange<DestT, SourceT>(val);
      dst[i] = val;
   }
}

/// \brief Split encoding of elements, possibly into narrower column
///
/// Used to first cast and then split-encode in-memory values to the on-disk column. Swap bytes if necessary.
template <typename DestT, typename SourceT>
inline void CastSplitPack(void *destination, const void *source, std::size_t count)
{
   constexpr std::size_t N = sizeof(DestT);
   auto splitArray = reinterpret_cast<char *>(destination);
   auto src = reinterpret_cast<const SourceT *>(source);
   for (std::size_t i = 0; i < count; ++i) {
      DestT val = src[i];
      ByteSwapIfNecessary(val);
      for (std::size_t b = 0; b < N; ++b) {
         splitArray[b * count + i] = reinterpret_cast<const char *>(&val)[b];
      }
   }
}

/// \brief Reverse split encoding of elements
///
/// Used to first unsplit a column, possibly storing elements in wider C++ types. Swaps bytes if necessary
template <typename DestT, typename SourceT>
inline void CastSplitUnpack(void *destination, const void *source, std::size_t count)
{
   constexpr std::size_t N = sizeof(SourceT);
   auto dst = reinterpret_cast<DestT *>(destination);
   auto splitArray = reinterpret_cast<const char *>(source);
   for (std::size_t i = 0; i < count; ++i) {
      SourceT val = 0;
      for (std::size_t b = 0; b < N; ++b) {
         reinterpret_cast<char *>(&val)[b] = splitArray[b * count + i];
      }
      ByteSwapIfNecessary(val);
      EnsureValidRange<DestT, SourceT>(val);
      dst[i] = val;
   }
}

/// \brief Packing of columns with delta + split encoding
///
/// Apply split encoding to delta-encoded values, currently used only for index columns
template <typename DestT, typename SourceT>
inline void CastDeltaSplitPack(void *destination, const void *source, std::size_t count)
{
   constexpr std::size_t N = sizeof(DestT);
   auto src = reinterpret_cast<const SourceT *>(source);
   auto splitArray = reinterpret_cast<char *>(destination);
   for (std::size_t i = 0; i < count; ++i) {
      DestT val = (i == 0) ? src[0] : src[i] - src[i - 1];
      ByteSwapIfNecessary(val);
      for (std::size_t b = 0; b < N; ++b) {
         splitArray[b * count + i] = reinterpret_cast<char *>(&val)[b];
      }
   }
}

/// \brief Unsplit and unwind delta encoding
///
/// Unsplit a column and reverse the delta encoding, currently used only for index columns
template <typename DestT, typename SourceT>
inline void CastDeltaSplitUnpack(void *destination, const void *source, std::size_t count)
{
   constexpr std::size_t N = sizeof(SourceT);
   auto splitArray = reinterpret_cast<const char *>(source);
   auto dst = reinterpret_cast<DestT *>(destination);
   for (std::size_t i = 0; i < count; ++i) {
      SourceT val = 0;
      for (std::size_t b = 0; b < N; ++b) {
         reinterpret_cast<char *>(&val)[b] = splitArray[b * count + i];
      }
      ByteSwapIfNecessary(val);
      val = (i == 0) ? val : val + dst[i - 1];
      EnsureValidRange<DestT, SourceT>(val);
      dst[i] = val;
   }
}

/// \brief Packing of columns with zigzag + split encoding
///
/// Apply split encoding to zigzag-encoded values, used for signed integers
template <typename DestT, typename SourceT>
inline void CastZigzagSplitPack(void *destination, const void *source, std::size_t count)
{
   using UDestT = std::make_unsigned_t<DestT>;
   constexpr std::size_t kNBitsDestT = sizeof(DestT) * 8;
   constexpr std::size_t N = sizeof(DestT);
   auto src = reinterpret_cast<const SourceT *>(source);
   auto splitArray = reinterpret_cast<char *>(destination);
   for (std::size_t i = 0; i < count; ++i) {
      UDestT val = (static_cast<DestT>(src[i]) << 1) ^ (static_cast<DestT>(src[i]) >> (kNBitsDestT - 1));
      ByteSwapIfNecessary(val);
      for (std::size_t b = 0; b < N; ++b) {
         splitArray[b * count + i] = reinterpret_cast<char *>(&val)[b];
      }
   }
}

/// \brief Unsplit and unwind zigzag encoding
///
/// Unsplit a column and reverse the zigzag encoding, used for signed integer columns
template <typename DestT, typename SourceT>
inline void CastZigzagSplitUnpack(void *destination, const void *source, std::size_t count)
{
   using USourceT = std::make_unsigned_t<SourceT>;
   constexpr std::size_t N = sizeof(SourceT);
   auto splitArray = reinterpret_cast<const char *>(source);
   auto dst = reinterpret_cast<DestT *>(destination);
   for (std::size_t i = 0; i < count; ++i) {
      USourceT val = 0;
      for (std::size_t b = 0; b < N; ++b) {
         reinterpret_cast<char *>(&val)[b] = splitArray[b * count + i];
      }
      ByteSwapIfNecessary(val);
      SourceT sval = static_cast<SourceT>((val >> 1) ^ -(static_cast<SourceT>(val) & 1));
      EnsureValidRange<DestT, SourceT>(sval);
      dst[i] = sval;
   }
}
} // namespace

// anonymous namespace because these definitions are not meant to be exported.
namespace {

using ROOT::Experimental::ENTupleColumnType;
using ROOT::Experimental::Internal::kTestFutureType;
using ROOT::Experimental::Internal::MakeUninitArray;
using ROOT::Experimental::Internal::RColumnElementBase;

template <typename CppT, ENTupleColumnType>
class RColumnElement;

template <typename CppT>
std::unique_ptr<RColumnElementBase> GenerateColumnElementInternal(ENTupleColumnType onDiskType)
{
   switch (onDiskType) {
   case ENTupleColumnType::kIndex64: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kIndex64>>();
   case ENTupleColumnType::kIndex32: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kIndex32>>();
   case ENTupleColumnType::kSwitch: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kSwitch>>();
   case ENTupleColumnType::kByte: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kByte>>();
   case ENTupleColumnType::kChar: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kChar>>();
   case ENTupleColumnType::kBit: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kBit>>();
   case ENTupleColumnType::kReal64: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kReal64>>();
   case ENTupleColumnType::kReal32: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kReal32>>();
   case ENTupleColumnType::kReal16: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kReal16>>();
   case ENTupleColumnType::kInt64: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kInt64>>();
   case ENTupleColumnType::kUInt64: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kUInt64>>();
   case ENTupleColumnType::kInt32: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kInt32>>();
   case ENTupleColumnType::kUInt32: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kUInt32>>();
   case ENTupleColumnType::kInt16: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kInt16>>();
   case ENTupleColumnType::kUInt16: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kUInt16>>();
   case ENTupleColumnType::kInt8: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kInt8>>();
   case ENTupleColumnType::kUInt8: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kUInt8>>();
   case ENTupleColumnType::kSplitIndex64:
      return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kSplitIndex64>>();
   case ENTupleColumnType::kSplitIndex32:
      return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kSplitIndex32>>();
   case ENTupleColumnType::kSplitReal64:
      return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kSplitReal64>>();
   case ENTupleColumnType::kSplitReal32:
      return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kSplitReal32>>();
   case ENTupleColumnType::kSplitInt64: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kSplitInt64>>();
   case ENTupleColumnType::kSplitUInt64:
      return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kSplitUInt64>>();
   case ENTupleColumnType::kSplitInt32: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kSplitInt32>>();
   case ENTupleColumnType::kSplitUInt32:
      return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kSplitUInt32>>();
   case ENTupleColumnType::kSplitInt16: return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kSplitInt16>>();
   case ENTupleColumnType::kSplitUInt16:
      return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kSplitUInt16>>();
   case ENTupleColumnType::kReal32Trunc:
      return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kReal32Trunc>>();
   case ENTupleColumnType::kReal32Quant:
      return std::make_unique<RColumnElement<CppT, ENTupleColumnType::kReal32Quant>>();
   default:
      if (onDiskType == kTestFutureType)
         return std::make_unique<RColumnElement<CppT, kTestFutureType>>();
      R__ASSERT(false);
   }
   // never here
   return nullptr;
}

/**
 * Base class for columns whose on-storage representation is little-endian.
 * The implementation of `Pack` and `Unpack` takes care of byteswap if the memory page is big-endian.
 */
template <typename CppT>
class RColumnElementLE : public RColumnElementBase {
protected:
   explicit RColumnElementLE(std::size_t size, std::size_t bitsOnStorage) : RColumnElementBase(size, bitsOnStorage) {}

public:
   static constexpr bool kIsMappable = (R__LITTLE_ENDIAN == 1);

   void Pack(void *dst, const void *src, std::size_t count) const final
   {
#if R__LITTLE_ENDIAN == 1
      RColumnElementBase::Pack(dst, src, count);
#else
      CopyBswap<sizeof(CppT)>(dst, src, count);
#endif
   }
   void Unpack(void *dst, const void *src, std::size_t count) const final
   {
#if R__LITTLE_ENDIAN == 1
      RColumnElementBase::Unpack(dst, src, count);
#else
      CopyBswap<sizeof(CppT)>(dst, src, count);
#endif
   }
}; // class RColumnElementLE

/**
 * Base class for columns storing elements of wider in-memory types,
 * such as 64bit in-memory offsets to Index32 columns.
 */
template <typename CppT, typename NarrowT>
class RColumnElementCastLE : public RColumnElementBase {
protected:
   explicit RColumnElementCastLE(std::size_t size, std::size_t bitsOnStorage) : RColumnElementBase(size, bitsOnStorage)
   {
   }

public:
   static constexpr bool kIsMappable = false;

   void Pack(void *dst, const void *src, std::size_t count) const final { CastPack<NarrowT, CppT>(dst, src, count); }
   void Unpack(void *dst, const void *src, std::size_t count) const final
   {
      CastUnpack<CppT, NarrowT>(dst, src, count);
   }
}; // class RColumnElementCastLE

/**
 * Base class for split columns whose on-storage representation is little-endian.
 * The implementation of `Pack` and `Unpack` takes care of splitting and, if necessary, byteswap.
 * As part of the splitting, can also narrow down the type to NarrowT.
 */
template <typename CppT, typename NarrowT>
class RColumnElementSplitLE : public RColumnElementBase {
protected:
   explicit RColumnElementSplitLE(std::size_t size, std::size_t bitsOnStorage) : RColumnElementBase(size, bitsOnStorage)
   {
   }

public:
   static constexpr bool kIsMappable = false;

   void Pack(void *dst, const void *src, std::size_t count) const final
   {
      CastSplitPack<NarrowT, CppT>(dst, src, count);
   }
   void Unpack(void *dst, const void *src, std::size_t count) const final
   {
      CastSplitUnpack<CppT, NarrowT>(dst, src, count);
   }
}; // class RColumnElementSplitLE

/**
 * Base class for delta + split columns (index columns) whose on-storage representation is little-endian.
 * The implementation of `Pack` and `Unpack` takes care of splitting and, if necessary, byteswap.
 * As part of the encoding, can also narrow down the type to NarrowT.
 */
template <typename CppT, typename NarrowT>
class RColumnElementDeltaSplitLE : public RColumnElementBase {
protected:
   explicit RColumnElementDeltaSplitLE(std::size_t size, std::size_t bitsOnStorage)
      : RColumnElementBase(size, bitsOnStorage)
   {
   }

public:
   static constexpr bool kIsMappable = false;

   void Pack(void *dst, const void *src, std::size_t count) const final
   {
      CastDeltaSplitPack<NarrowT, CppT>(dst, src, count);
   }
   void Unpack(void *dst, const void *src, std::size_t count) const final
   {
      CastDeltaSplitUnpack<CppT, NarrowT>(dst, src, count);
   }
}; // class RColumnElementDeltaSplitLE

/// Reading of unsplit integer columns to boolean
template <typename CppIntT>
class RColumnElementBoolAsUnsplitInt : public RColumnElementBase {
protected:
   explicit RColumnElementBoolAsUnsplitInt(std::size_t size, std::size_t bitsOnStorage)
      : RColumnElementBase(size, bitsOnStorage)
   {
   }

public:
   static constexpr bool kIsMappable = false;

   // We don't implement Pack() because integers must not be written to disk as booleans
   void Pack(void *, const void *, std::size_t) const final { R__ASSERT(false); }

   void Unpack(void *dst, const void *src, std::size_t count) const final
   {
      auto *boolArray = reinterpret_cast<bool *>(dst);
      auto *intArray = reinterpret_cast<const CppIntT *>(src);
      for (std::size_t i = 0; i < count; ++i) {
         boolArray[i] = intArray[i] != 0;
      }
   }
}; // class RColumnElementBoolAsUnsplitInt

/// Reading of split integer columns to boolean
template <typename CppIntT>
class RColumnElementBoolAsSplitInt : public RColumnElementBase {
protected:
   explicit RColumnElementBoolAsSplitInt(std::size_t size, std::size_t bitsOnStorage)
      : RColumnElementBase(size, bitsOnStorage)
   {
   }

public:
   static constexpr bool kIsMappable = false;

   // We don't implement Pack() because integers must not be written to disk as booleans
   void Pack(void *, const void *, std::size_t) const final { R__ASSERT(false); }

   void Unpack(void *dst, const void *src, std::size_t count) const final
   {
      constexpr std::size_t N = sizeof(CppIntT);
      auto *boolArray = reinterpret_cast<bool *>(dst);
      auto *splitArray = reinterpret_cast<const char *>(src);
      for (std::size_t i = 0; i < count; ++i) {
         boolArray[i] = false;
         for (std::size_t b = 0; b < N; ++b) {
            if (splitArray[b * count + i]) {
               boolArray[i] = true;
               break;
            }
         }
      }
   }
}; // RColumnElementBoolAsSplitInt

/// Reading of bit columns as integer
template <typename CppIntT>
class RColumnElementIntAsBool : public RColumnElementBase {
protected:
   explicit RColumnElementIntAsBool(std::size_t size, std::size_t bitsOnStorage)
      : RColumnElementBase(size, bitsOnStorage)
   {
   }

public:
   static constexpr bool kIsMappable = false;

   // We don't implement Pack() because booleans must not be written as integers to disk
   void Pack(void *, const void *, std::size_t) const final { R__ASSERT(false); }

   void Unpack(void *dst, const void *src, std::size_t count) const final
   {
      auto *intArray = reinterpret_cast<CppIntT *>(dst);
      const char *charArray = reinterpret_cast<const char *>(src);
      std::bitset<8> bitSet;
      for (std::size_t i = 0; i < count; i += 8) {
         bitSet = charArray[i / 8];
         for (std::size_t j = i; j < std::min(count, i + 8); ++j) {
            intArray[j] = bitSet[j % 8];
         }
      }
   }
}; // RColumnElementIntAsBool

/**
 * Base class for zigzag + split columns (signed integer columns) whose on-storage representation is little-endian.
 * The implementation of `Pack` and `Unpack` takes care of splitting and, if necessary, byteswap.
 * The NarrowT target type should be an signed integer, which can be smaller than the CppT source type.
 */
template <typename CppT, typename NarrowT>
class RColumnElementZigzagSplitLE : public RColumnElementBase {
protected:
   explicit RColumnElementZigzagSplitLE(std::size_t size, std::size_t bitsOnStorage)
      : RColumnElementBase(size, bitsOnStorage)
   {
   }

public:
   static constexpr bool kIsMappable = false;

   void Pack(void *dst, const void *src, std::size_t count) const final
   {
      CastZigzagSplitPack<NarrowT, CppT>(dst, src, count);
   }
   void Unpack(void *dst, const void *src, std::size_t count) const final
   {
      CastZigzagSplitUnpack<CppT, NarrowT>(dst, src, count);
   }
}; // class RColumnElementZigzagSplitLE

////////////////////////////////////////////////////////////////////////////////
// Pairs of C++ type and column type, like float and ENTupleColumnType::kReal32
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Part 1: C++ type --> unknown column type
////////////////////////////////////////////////////////////////////////////////

template <typename CppT, ENTupleColumnType ColumnT = ENTupleColumnType::kUnknown>
class RColumnElement : public RColumnElementBase {
public:
   RColumnElement() : RColumnElementBase(sizeof(CppT))
   {
      throw ROOT::RException(R__FAIL(std::string("internal error: no column mapping for this C++ type: ") +
                                     typeid(CppT).name() + " --> " + GetColumnTypeName(ColumnT)));
   }

   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(CppT), ENTupleColumnType::kUnknown}; }
};

template <>
class RColumnElement<bool, ENTupleColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(bool);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(bool), ENTupleColumnType::kUnknown}; }
};

template <>
class RColumnElement<std::byte, ENTupleColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::byte);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(std::byte), ENTupleColumnType::kUnknown}; }
};

template <>
class RColumnElement<char, ENTupleColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(char);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(char), ENTupleColumnType::kUnknown}; }
};

template <>
class RColumnElement<std::int8_t, ENTupleColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::int8_t);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(std::int8_t), ENTupleColumnType::kUnknown}; }
};

template <>
class RColumnElement<std::uint8_t, ENTupleColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::uint8_t);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(std::uint8_t), ENTupleColumnType::kUnknown}; }
};

template <>
class RColumnElement<std::int16_t, ENTupleColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::int16_t);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(std::int16_t), ENTupleColumnType::kUnknown}; }
};

template <>
class RColumnElement<std::uint16_t, ENTupleColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::uint16_t);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(std::uint16_t), ENTupleColumnType::kUnknown}; }
};

template <>
class RColumnElement<std::int32_t, ENTupleColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::int32_t);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(std::int32_t), ENTupleColumnType::kUnknown}; }
};

template <>
class RColumnElement<std::uint32_t, ENTupleColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::uint32_t);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(std::uint32_t), ENTupleColumnType::kUnknown}; }
};

template <>
class RColumnElement<std::int64_t, ENTupleColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::int64_t);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(std::int64_t), ENTupleColumnType::kUnknown}; }
};

template <>
class RColumnElement<std::uint64_t, ENTupleColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::uint64_t);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(std::uint64_t), ENTupleColumnType::kUnknown}; }
};

template <>
class RColumnElement<float, ENTupleColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(float);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(float), ENTupleColumnType::kUnknown}; }
};

template <>
class RColumnElement<double, ENTupleColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(double);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(double), ENTupleColumnType::kUnknown}; }
};

template <>
class RColumnElement<ROOT::Experimental::Internal::RColumnIndex, ENTupleColumnType::kUnknown>
   : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(ROOT::Experimental::Internal::RColumnIndex);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final
   {
      return RIdentifier{typeid(ROOT::Experimental::Internal::RColumnIndex), ENTupleColumnType::kUnknown};
   }
};

template <>
class RColumnElement<ROOT::Experimental::Internal::RColumnSwitch, ENTupleColumnType::kUnknown>
   : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(ROOT::Experimental::Internal::RColumnSwitch);
   RColumnElement() : RColumnElementBase(kSize) {}
   RIdentifier GetIdentifier() const final
   {
      return RIdentifier{typeid(ROOT::Experimental::Internal::RColumnSwitch), ENTupleColumnType::kUnknown};
   }
};

////////////////////////////////////////////////////////////////////////////////
// Part 2: C++ type --> supported column representations,
//         ordered by C++ type
////////////////////////////////////////////////////////////////////////////////

template <>
class RColumnElement<ROOT::Experimental::Internal::RColumnSwitch, ENTupleColumnType::kSwitch>
   : public RColumnElementBase {
private:
   struct RSwitchElement {
      std::uint64_t fIndex;
      std::uint32_t fTag;
   };

public:
   static constexpr bool kIsMappable = false;
   static constexpr std::size_t kSize = sizeof(ROOT::Experimental::Internal::RColumnSwitch);
   static constexpr std::size_t kBitsOnStorage = 96;
   RColumnElement() : RColumnElementBase(kSize, kBitsOnStorage) {}
   bool IsMappable() const final { return kIsMappable; }

   void Pack(void *dst, const void *src, std::size_t count) const final
   {
      auto srcArray = reinterpret_cast<const ROOT::Experimental::Internal::RColumnSwitch *>(src);
      auto dstArray = reinterpret_cast<unsigned char *>(dst);
      for (std::size_t i = 0; i < count; ++i) {
         RSwitchElement element{srcArray[i].GetIndex(), srcArray[i].GetTag()};
#if R__LITTLE_ENDIAN == 0
         element.fIndex = RByteSwap<8>::bswap(element.fIndex);
         element.fTag = RByteSwap<4>::bswap(element.fTag);
#endif
         memcpy(dstArray + i * 12, &element, 12);
      }
   }

   void Unpack(void *dst, const void *src, std::size_t count) const final
   {
      auto srcArray = reinterpret_cast<const unsigned char *>(src);
      auto dstArray = reinterpret_cast<ROOT::Experimental::Internal::RColumnSwitch *>(dst);
      for (std::size_t i = 0; i < count; ++i) {
         RSwitchElement element;
         memcpy(&element, srcArray + i * 12, 12);
#if R__LITTLE_ENDIAN == 0
         element.fIndex = RByteSwap<8>::bswap(element.fIndex);
         element.fTag = RByteSwap<4>::bswap(element.fTag);
#endif
         dstArray[i] = ROOT::Experimental::Internal::RColumnSwitch(element.fIndex, element.fTag);
      }
   }

   RIdentifier GetIdentifier() const final
   {
      return RIdentifier{typeid(ROOT::Experimental::Internal::RColumnSwitch), ENTupleColumnType::kSwitch};
   }
};

template <>
class RColumnElement<bool, ENTupleColumnType::kBit> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = false;
   static constexpr std::size_t kSize = sizeof(bool);
   static constexpr std::size_t kBitsOnStorage = 1;
   RColumnElement() : RColumnElementBase(kSize, kBitsOnStorage) {}
   bool IsMappable() const final { return kIsMappable; }

   void Pack(void *dst, const void *src, std::size_t count) const final;
   void Unpack(void *dst, const void *src, std::size_t count) const final;

   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(bool), ENTupleColumnType::kBit}; }
};

template <>
class RColumnElement<float, ENTupleColumnType::kReal16> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = false;
   static constexpr std::size_t kSize = sizeof(float);
   static constexpr std::size_t kBitsOnStorage = 16;
   RColumnElement() : RColumnElementBase(kSize, kBitsOnStorage) {}
   bool IsMappable() const final { return kIsMappable; }

   void Pack(void *dst, const void *src, std::size_t count) const final
   {
      const float *floatArray = reinterpret_cast<const float *>(src);
      std::uint16_t *uint16Array = reinterpret_cast<std::uint16_t *>(dst);

      for (std::size_t i = 0; i < count; ++i) {
         uint16Array[i] = ROOT::Experimental::Internal::FloatToHalf(floatArray[i]);
         ByteSwapIfNecessary(uint16Array[i]);
      }
   }

   void Unpack(void *dst, const void *src, std::size_t count) const final
   {
      float *floatArray = reinterpret_cast<float *>(dst);
      const std::uint16_t *uint16Array = reinterpret_cast<const std::uint16_t *>(src);

      for (std::size_t i = 0; i < count; ++i) {
         std::uint16_t val = uint16Array[i];
         ByteSwapIfNecessary(val);
         floatArray[i] = ROOT::Experimental::Internal::HalfToFloat(val);
      }
   }

   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(float), ENTupleColumnType::kReal16}; }
};

template <>
class RColumnElement<double, ENTupleColumnType::kReal16> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = false;
   static constexpr std::size_t kSize = sizeof(double);
   static constexpr std::size_t kBitsOnStorage = 16;
   RColumnElement() : RColumnElementBase(kSize, kBitsOnStorage) {}
   bool IsMappable() const final { return kIsMappable; }

   void Pack(void *dst, const void *src, std::size_t count) const final
   {
      const double *doubleArray = reinterpret_cast<const double *>(src);
      std::uint16_t *uint16Array = reinterpret_cast<std::uint16_t *>(dst);

      for (std::size_t i = 0; i < count; ++i) {
         uint16Array[i] = ROOT::Experimental::Internal::FloatToHalf(static_cast<float>(doubleArray[i]));
         ByteSwapIfNecessary(uint16Array[i]);
      }
   }

   void Unpack(void *dst, const void *src, std::size_t count) const final
   {
      double *doubleArray = reinterpret_cast<double *>(dst);
      const std::uint16_t *uint16Array = reinterpret_cast<const std::uint16_t *>(src);

      for (std::size_t i = 0; i < count; ++i) {
         std::uint16_t val = uint16Array[i];
         ByteSwapIfNecessary(val);
         doubleArray[i] = static_cast<double>(ROOT::Experimental::Internal::HalfToFloat(val));
      }
   }

   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(double), ENTupleColumnType::kReal16}; }
};

template <typename T>
class RColumnElementTrunc : public RColumnElementBase {
public:
   static_assert(std::is_floating_point_v<T>);
   static constexpr bool kIsMappable = false;
   static constexpr std::size_t kSize = sizeof(T);

   // NOTE: setting bitsOnStorage == 0 by default. This is an invalid value that helps us
   // catch misuses where RColumnElement is used without having explicitly set its bit width
   // (which should never happen).
   RColumnElementTrunc() : RColumnElementBase(kSize, 0) {}

   void SetBitsOnStorage(std::size_t bitsOnStorage) final
   {
      const auto &[minBits, maxBits] = GetValidBitRange(ENTupleColumnType::kReal32Trunc);
      R__ASSERT(bitsOnStorage >= minBits && bitsOnStorage <= maxBits);
      fBitsOnStorage = bitsOnStorage;
   }

   bool IsMappable() const final { return kIsMappable; }

   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(T), ENTupleColumnType::kReal32Trunc}; }
};

template <>
class RColumnElement<float, ENTupleColumnType::kReal32Trunc> : public RColumnElementTrunc<float> {
public:
   void Pack(void *dst, const void *src, std::size_t count) const final
   {
      using namespace ROOT::Experimental::Internal::BitPacking;

      R__ASSERT(GetPackedSize(count) == MinBufSize(count, fBitsOnStorage));

#if R__LITTLE_ENDIAN == 0
      // TODO(gparolini): to avoid this extra allocation we might want to perform byte swapping
      // directly in the Pack/UnpackBits functions.
      auto bswapped = MakeUninitArray<float>(count);
      CopyBswap<sizeof(float)>(bswapped.get(), src, count);
      const auto *srcLe = bswapped.get();
#else
      const auto *srcLe = reinterpret_cast<const float *>(src);
#endif
      PackBits(dst, srcLe, count, sizeof(float), fBitsOnStorage);
   }

   void Unpack(void *dst, const void *src, std::size_t count) const final
   {
      using namespace ROOT::Experimental::Internal::BitPacking;

      R__ASSERT(GetPackedSize(count) == MinBufSize(count, fBitsOnStorage));

      UnpackBits(dst, src, count, sizeof(float), fBitsOnStorage);
#if R__LITTLE_ENDIAN == 0
      InPlaceBswap<sizeof(float)>(dst, count);
#endif
   }
};

template <>
class RColumnElement<double, ENTupleColumnType::kReal32Trunc> : public RColumnElementTrunc<double> {
public:
   void Pack(void *dst, const void *src, std::size_t count) const final
   {
      using namespace ROOT::Experimental::Internal::BitPacking;

      R__ASSERT(GetPackedSize(count) == MinBufSize(count, fBitsOnStorage));

      // Cast doubles to float before packing them
      // TODO(gparolini): avoid this allocation
      auto srcFloat = MakeUninitArray<float>(count);
      const double *srcDouble = reinterpret_cast<const double *>(src);
      for (std::size_t i = 0; i < count; ++i)
         srcFloat[i] = static_cast<float>(srcDouble[i]);

#if R__LITTLE_ENDIAN == 0
      // TODO(gparolini): to avoid this extra allocation we might want to perform byte swapping
      // directly in the Pack/UnpackBits functions.
      auto bswapped = MakeUninitArray<float>(count);
      CopyBswap<sizeof(float)>(bswapped.get(), srcFloat.get(), count);
      const float *srcLe = bswapped.get();
#else
      const float *srcLe = reinterpret_cast<const float *>(srcFloat.get());
#endif
      PackBits(dst, srcLe, count, sizeof(float), fBitsOnStorage);
   }

   void Unpack(void *dst, const void *src, std::size_t count) const final
   {
      using namespace ROOT::Experimental::Internal::BitPacking;

      R__ASSERT(GetPackedSize(count) == MinBufSize(count, fBitsOnStorage));

      // TODO(gparolini): avoid this allocation
      auto dstFloat = MakeUninitArray<float>(count);
      UnpackBits(dstFloat.get(), src, count, sizeof(float), fBitsOnStorage);
#if R__LITTLE_ENDIAN == 0
      InPlaceBswap<sizeof(float)>(dstFloat.get(), count);
#endif

      double *dstDouble = reinterpret_cast<double *>(dst);
      for (std::size_t i = 0; i < count; ++i)
         dstDouble[i] = static_cast<double>(dstFloat[i]);
   }
};

namespace Quantize {

using Quantized_t = std::uint32_t;

[[maybe_unused]] inline std::size_t LeadingZeroes(std::uint32_t x)
{
   if (x == 0)
      return 32;

#ifdef _MSC_VER
   unsigned long idx = 0;
   if (_BitScanReverse(&idx, x))
      return static_cast<std::size_t>(31 - idx);
   return 32;
#else
   return static_cast<std::size_t>(__builtin_clzl(x));
#endif
}

[[maybe_unused]] inline std::size_t TrailingZeroes(std::uint32_t x)
{
   if (x == 0)
      return 32;

#ifdef _MSC_VER
   unsigned long idx = 0;
   if (_BitScanForward(&idx, x))
      return static_cast<std::size_t>(idx);
   return 32;
#else
   return static_cast<std::size_t>(__builtin_ctzl(x));
#endif
}

/// Converts the array `src` of `count` floating point numbers into an array of their quantized representations.
/// Each element of `src` is assumed to be in the inclusive range [min, max].
/// The quantized representation will consist of unsigned integers of at most `nQuantBits` (with `nQuantBits <= 8 *
/// sizeof(Quantized_t)`). The unused bits are kept in the LSB of the quantized integers, to allow for easy bit packing
/// of those integers via BitPacking::PackBits().
/// \return The number of values in `src` that were found to be out of range (0 means all values were in range).
template <typename T>
int QuantizeReals(Quantized_t *dst, const T *src, std::size_t count, double min, double max, std::size_t nQuantBits)
{
   static_assert(std::is_floating_point_v<T>);
   static_assert(sizeof(T) <= sizeof(double));
   assert(1 <= nQuantBits && nQuantBits <= 8 * sizeof(Quantized_t));

   const std::size_t quantMax = (1ull << nQuantBits) - 1;
   const double scale = quantMax / (max - min);
   const std::size_t unusedBits = sizeof(Quantized_t) * 8 - nQuantBits;

   int nOutOfRange = 0;

   for (std::size_t i = 0; i < count; ++i) {
      const T elem = src[i];

      nOutOfRange += !(min <= elem && elem <= max);

      const double e = 0.5 + (elem - min) * scale;
      Quantized_t q = static_cast<Quantized_t>(e);
      ByteSwapIfNecessary(q);

      // double-check we actually used at most `nQuantBits`
      assert(LeadingZeroes(q) >= unusedBits);

      // we want to leave zeroes in the LSB, not the MSB, because we'll then drop the LSB
      // when bit packing.
      dst[i] = q << unusedBits;
   }

   return nOutOfRange;
}

/// Undoes the transformation performed by QuantizeReals() (assuming the same `count`, `min`, `max` and `nQuantBits`).
/// \return The number of unpacked values that were found to be out of range (0 means all values were in range).
template <typename T>
int UnquantizeReals(T *dst, const Quantized_t *src, std::size_t count, double min, double max, std::size_t nQuantBits)
{
   static_assert(std::is_floating_point_v<T>);
   static_assert(sizeof(T) <= sizeof(double));
   assert(1 <= nQuantBits && nQuantBits <= 8 * sizeof(Quantized_t));

   const std::size_t quantMax = (1ull << nQuantBits) - 1;
   const double scale = (max - min) / quantMax;
   const std::size_t unusedBits = sizeof(Quantized_t) * 8 - nQuantBits;
   const double eps = std::numeric_limits<double>::epsilon();
   const double emin = min - std::abs(min) * eps;
   const double emax = max + std::abs(max) * eps;

   int nOutOfRange = 0;

   for (std::size_t i = 0; i < count; ++i) {
      Quantized_t elem = src[i];
      // Undo the LSB-preserving shift performed by QuantizeReals
      assert(TrailingZeroes(elem) >= unusedBits);
      elem >>= unusedBits;
      ByteSwapIfNecessary(elem);

      const double fq = static_cast<double>(elem);
      const double e = fq * scale + min;
      dst[i] = static_cast<T>(e);

      nOutOfRange += !(emin <= dst[i] && dst[i] <= emax);
   }

   return nOutOfRange;
}
} // namespace Quantize

template <typename T>
class RColumnElementQuantized : public RColumnElementBase {
   static_assert(std::is_floating_point_v<T>);

public:
   static constexpr bool kIsMappable = false;
   static constexpr std::size_t kSize = sizeof(T);

   RColumnElementQuantized() : RColumnElementBase(kSize, 0) {}

   void SetBitsOnStorage(std::size_t bitsOnStorage) final
   {
      const auto [minBits, maxBits] = GetValidBitRange(ENTupleColumnType::kReal32Quant);
      R__ASSERT(bitsOnStorage >= minBits && bitsOnStorage <= maxBits);
      fBitsOnStorage = bitsOnStorage;
   }

   void SetValueRange(double min, double max) final
   {
      R__ASSERT(min >= std::numeric_limits<T>::lowest());
      R__ASSERT(max <= std::numeric_limits<T>::max());
      // Disallow denormal, NaN and infinity
      R__ASSERT(std::isnormal(min) || min == 0.0);
      R__ASSERT(std::isnormal(max) || max == 0.0);
      fValueRange = {min, max};
   }

   bool IsMappable() const final { return kIsMappable; }

   void Pack(void *dst, const void *src, std::size_t count) const final
   {
      using namespace ROOT::Experimental;

      // TODO(gparolini): see if we can avoid this allocation
      auto quantized = MakeUninitArray<Quantize::Quantized_t>(count);
      assert(fValueRange);
      const auto [min, max] = *fValueRange;
      const int nOutOfRange =
         Quantize::QuantizeReals(quantized.get(), reinterpret_cast<const T *>(src), count, min, max, fBitsOnStorage);
      if (nOutOfRange) {
         throw ROOT::RException(R__FAIL(std::to_string(nOutOfRange) +
                                        " values were found of of range for quantization while packing (range is [" +
                                        std::to_string(min) + ", " + std::to_string(max) + "])"));
      }
      Internal::BitPacking::PackBits(dst, quantized.get(), count, sizeof(Quantize::Quantized_t), fBitsOnStorage);
   }

   void Unpack(void *dst, const void *src, std::size_t count) const final
   {
      using namespace ROOT::Experimental;

      // TODO(gparolini): see if we can avoid this allocation
      auto quantized = MakeUninitArray<Quantize::Quantized_t>(count);
      assert(fValueRange);
      const auto [min, max] = *fValueRange;
      Internal::BitPacking::UnpackBits(quantized.get(), src, count, sizeof(Quantize::Quantized_t), fBitsOnStorage);
      [[maybe_unused]] const int nOutOfRange =
         Quantize::UnquantizeReals(reinterpret_cast<T *>(dst), quantized.get(), count, min, max, fBitsOnStorage);
      // NOTE: here, differently from Pack(), we don't ever expect to have values out of range, since the quantized
      // integers we pass to UnquantizeReals are by construction limited in value to the proper range. In Pack()
      // this is not the case, as the user may give us float values that are out of range.
      assert(nOutOfRange == 0);
   }

   RIdentifier GetIdentifier() const final { return RIdentifier{typeid(T), ENTupleColumnType::kReal32Quant}; }
};

template <>
class RColumnElement<float, ENTupleColumnType::kReal32Quant> : public RColumnElementQuantized<float> {};

template <>
class RColumnElement<double, ENTupleColumnType::kReal32Quant> : public RColumnElementQuantized<double> {};

#define __RCOLUMNELEMENT_SPEC_BODY(CppT, ColumnT, BaseT, BitsOnStorage) \
   static constexpr std::size_t kSize = sizeof(CppT);                   \
   static constexpr std::size_t kBitsOnStorage = BitsOnStorage;         \
   RColumnElement() : BaseT(kSize, kBitsOnStorage) {}                   \
   bool IsMappable() const final                                        \
   {                                                                    \
      return kIsMappable;                                               \
   }                                                                    \
   RIdentifier GetIdentifier() const final                              \
   {                                                                    \
      return RIdentifier{typeid(CppT), ColumnT};                        \
   }
/// These macros are used to declare `RColumnElement` template specializations below.  Additional arguments can be used
/// to forward template parameters to the base class, e.g.
/// ```
/// DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kInt32, 32,
///                             RColumnElementCastLE, <std::int64_t, std::int32_t>);
/// ```
#define DECLARE_RCOLUMNELEMENT_SPEC(CppT, ColumnT, BitsOnStorage, BaseT, ...) \
   template <>                                                                \
   class RColumnElement<CppT, ColumnT> : public BaseT __VA_ARGS__ {           \
   public:                                                                    \
      __RCOLUMNELEMENT_SPEC_BODY(CppT, ColumnT, BaseT, BitsOnStorage)         \
   }
#define DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(CppT, ColumnT, BitsOnStorage)           \
   template <>                                                                     \
   class RColumnElement<CppT, ColumnT> : public RColumnElementBase {               \
   public:                                                                         \
      static constexpr bool kIsMappable = true;                                    \
      __RCOLUMNELEMENT_SPEC_BODY(CppT, ColumnT, RColumnElementBase, BitsOnStorage) \
   }

DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kChar, 8, RColumnElementBoolAsUnsplitInt, <char>);
DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kInt8, 8, RColumnElementBoolAsUnsplitInt, <std::int8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kUInt8, 8, RColumnElementBoolAsUnsplitInt, <std::uint8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kInt16, 16, RColumnElementBoolAsUnsplitInt, <std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kUInt16, 16, RColumnElementBoolAsUnsplitInt, <std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kInt32, 32, RColumnElementBoolAsUnsplitInt, <std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kUInt32, 32, RColumnElementBoolAsUnsplitInt, <std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kInt64, 64, RColumnElementBoolAsUnsplitInt, <std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kUInt64, 64, RColumnElementBoolAsUnsplitInt, <std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kSplitInt16, 16, RColumnElementBoolAsSplitInt, <std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kSplitUInt16, 16, RColumnElementBoolAsSplitInt, <std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kSplitInt32, 32, RColumnElementBoolAsSplitInt, <std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kSplitUInt32, 32, RColumnElementBoolAsSplitInt, <std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kSplitInt64, 64, RColumnElementBoolAsSplitInt, <std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(bool, ENTupleColumnType::kSplitUInt64, 64, RColumnElementBoolAsSplitInt, <std::uint64_t>);

DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(std::byte, ENTupleColumnType::kByte, 8);

DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(char, ENTupleColumnType::kChar, 8);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kInt8, 8, RColumnElementCastLE, <char, std::int8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kUInt8, 8, RColumnElementCastLE, <char, std::uint8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kInt16, 16, RColumnElementCastLE, <char, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kUInt16, 16, RColumnElementCastLE, <char, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kInt32, 32, RColumnElementCastLE, <char, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kUInt32, 32, RColumnElementCastLE, <char, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kInt64, 64, RColumnElementCastLE, <char, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kUInt64, 64, RColumnElementCastLE, <char, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kSplitInt16, 16, RColumnElementZigzagSplitLE,
                            <char, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kSplitUInt16, 16, RColumnElementSplitLE, <char, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kSplitInt32, 32, RColumnElementZigzagSplitLE,
                            <char, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kSplitUInt32, 32, RColumnElementSplitLE, <char, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kSplitInt64, 64, RColumnElementZigzagSplitLE,
                            <char, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kSplitUInt64, 64, RColumnElementSplitLE, <char, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(char, ENTupleColumnType::kBit, 1, RColumnElementIntAsBool, <char>);

DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(std::int8_t, ENTupleColumnType::kInt8, 8);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kChar, 8, RColumnElementCastLE, <std::int8_t, char>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kUInt8, 8, RColumnElementCastLE,
                            <std::int8_t, std::uint8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kInt16, 16, RColumnElementCastLE,
                            <std::int8_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kUInt16, 16, RColumnElementCastLE,
                            <std::int8_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kInt32, 32, RColumnElementCastLE,
                            <std::int8_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kUInt32, 32, RColumnElementCastLE,
                            <std::int8_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kInt64, 64, RColumnElementCastLE,
                            <std::int8_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kUInt64, 64, RColumnElementCastLE,
                            <std::int8_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kSplitInt16, 16, RColumnElementZigzagSplitLE,
                            <std::int8_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kSplitUInt16, 16, RColumnElementSplitLE,
                            <std::int8_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kSplitInt32, 32, RColumnElementZigzagSplitLE,
                            <std::int8_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kSplitUInt32, 32, RColumnElementSplitLE,
                            <std::int8_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kSplitInt64, 64, RColumnElementZigzagSplitLE,
                            <std::int8_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kSplitUInt64, 64, RColumnElementSplitLE,
                            <std::int8_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int8_t, ENTupleColumnType::kBit, 1, RColumnElementIntAsBool, <std::int8_t>);

DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(std::uint8_t, ENTupleColumnType::kUInt8, 8);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kChar, 8, RColumnElementCastLE, <std::uint8_t, char>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kInt8, 8, RColumnElementCastLE,
                            <std::uint8_t, std::int8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kInt16, 16, RColumnElementCastLE,
                            <std::uint8_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kUInt16, 16, RColumnElementCastLE,
                            <std::uint8_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kInt32, 32, RColumnElementCastLE,
                            <std::uint8_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kUInt32, 32, RColumnElementCastLE,
                            <std::uint8_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kInt64, 64, RColumnElementCastLE,
                            <std::uint8_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kUInt64, 64, RColumnElementCastLE,
                            <std::uint8_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kSplitInt16, 16, RColumnElementZigzagSplitLE,
                            <std::uint8_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kSplitUInt16, 16, RColumnElementSplitLE,
                            <std::uint8_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kSplitInt32, 32, RColumnElementZigzagSplitLE,
                            <std::uint8_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kSplitUInt32, 32, RColumnElementSplitLE,
                            <std::uint8_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kSplitInt64, 64, RColumnElementZigzagSplitLE,
                            <std::uint8_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kSplitUInt64, 64, RColumnElementSplitLE,
                            <std::uint8_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint8_t, ENTupleColumnType::kBit, 1, RColumnElementIntAsBool, <std::uint8_t>);

DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kInt16, 16, RColumnElementLE, <std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kSplitInt16, 16, RColumnElementZigzagSplitLE,
                            <std::int16_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kChar, 8, RColumnElementCastLE, <std::int16_t, char>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kInt8, 8, RColumnElementCastLE,
                            <std::int16_t, std::int8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kUInt8, 8, RColumnElementCastLE,
                            <std::int16_t, std::uint8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kUInt16, 16, RColumnElementCastLE,
                            <std::int16_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kInt32, 32, RColumnElementCastLE,
                            <std::int16_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kUInt32, 32, RColumnElementCastLE,
                            <std::int16_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kInt64, 64, RColumnElementCastLE,
                            <std::int16_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kUInt64, 64, RColumnElementCastLE,
                            <std::int16_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kSplitUInt16, 16, RColumnElementSplitLE,
                            <std::int16_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kSplitInt32, 32, RColumnElementZigzagSplitLE,
                            <std::int16_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kSplitUInt32, 32, RColumnElementSplitLE,
                            <std::int16_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kSplitInt64, 64, RColumnElementZigzagSplitLE,
                            <std::int16_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kSplitUInt64, 64, RColumnElementSplitLE,
                            <std::int16_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, ENTupleColumnType::kBit, 1, RColumnElementIntAsBool, <std::int16_t>);

DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kUInt16, 16, RColumnElementLE, <std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kSplitUInt16, 16, RColumnElementSplitLE,
                            <std::uint16_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kChar, 8, RColumnElementCastLE, <std::uint16_t, char>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kInt8, 8, RColumnElementCastLE,
                            <std::uint16_t, std::int8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kUInt8, 8, RColumnElementCastLE,
                            <std::uint16_t, std::uint8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kInt16, 16, RColumnElementCastLE,
                            <std::uint16_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kInt32, 32, RColumnElementCastLE,
                            <std::uint16_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kUInt32, 32, RColumnElementCastLE,
                            <std::uint16_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kInt64, 64, RColumnElementCastLE,
                            <std::uint16_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kUInt64, 64, RColumnElementCastLE,
                            <std::uint16_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kSplitInt16, 16, RColumnElementZigzagSplitLE,
                            <std::uint16_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kSplitInt32, 32, RColumnElementZigzagSplitLE,
                            <std::uint16_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kSplitUInt32, 32, RColumnElementSplitLE,
                            <std::uint16_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kSplitInt64, 64, RColumnElementZigzagSplitLE,
                            <std::uint16_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kSplitUInt64, 64, RColumnElementSplitLE,
                            <std::uint16_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, ENTupleColumnType::kBit, 1, RColumnElementIntAsBool, <std::uint16_t>);

DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kInt32, 32, RColumnElementLE, <std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kSplitInt32, 32, RColumnElementZigzagSplitLE,
                            <std::int32_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kChar, 8, RColumnElementCastLE, <std::int32_t, char>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kInt8, 8, RColumnElementCastLE,
                            <std::int32_t, std::int8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kUInt8, 8, RColumnElementCastLE,
                            <std::int32_t, std::uint8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kInt16, 16, RColumnElementCastLE,
                            <std::int32_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kUInt16, 16, RColumnElementCastLE,
                            <std::int32_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kUInt32, 32, RColumnElementCastLE,
                            <std::int32_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kInt64, 64, RColumnElementCastLE,
                            <std::int32_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kUInt64, 64, RColumnElementCastLE,
                            <std::int32_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kSplitInt16, 16, RColumnElementZigzagSplitLE,
                            <std::int32_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kSplitUInt16, 16, RColumnElementSplitLE,
                            <std::int32_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kSplitUInt32, 32, RColumnElementSplitLE,
                            <std::int32_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kSplitInt64, 64, RColumnElementZigzagSplitLE,
                            <std::int32_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kSplitUInt64, 64, RColumnElementSplitLE,
                            <std::int32_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, ENTupleColumnType::kBit, 1, RColumnElementIntAsBool, <std::int32_t>);

DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kUInt32, 32, RColumnElementLE, <std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kSplitUInt32, 32, RColumnElementSplitLE,
                            <std::uint32_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kChar, 8, RColumnElementCastLE, <std::uint32_t, char>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kInt8, 8, RColumnElementCastLE,
                            <std::uint32_t, std::int8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kUInt8, 8, RColumnElementCastLE,
                            <std::uint32_t, std::uint8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kInt16, 16, RColumnElementCastLE,
                            <std::uint32_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kUInt16, 16, RColumnElementCastLE,
                            <std::uint32_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kInt32, 32, RColumnElementCastLE,
                            <std::uint32_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kInt64, 64, RColumnElementCastLE,
                            <std::uint32_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kUInt64, 64, RColumnElementCastLE,
                            <std::uint32_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kSplitInt16, 16, RColumnElementZigzagSplitLE,
                            <std::uint32_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kSplitUInt16, 16, RColumnElementSplitLE,
                            <std::uint32_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kSplitInt32, 32, RColumnElementZigzagSplitLE,
                            <std::uint32_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kSplitInt64, 64, RColumnElementZigzagSplitLE,
                            <std::uint32_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kSplitUInt64, 64, RColumnElementSplitLE,
                            <std::uint32_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, ENTupleColumnType::kBit, 1, RColumnElementIntAsBool, <std::uint32_t>);

DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kInt64, 64, RColumnElementLE, <std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kSplitInt64, 64, RColumnElementZigzagSplitLE,
                            <std::int64_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kChar, 8, RColumnElementCastLE, <std::int64_t, char>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kInt8, 8, RColumnElementCastLE,
                            <std::int64_t, std::int8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kUInt8, 8, RColumnElementCastLE,
                            <std::int64_t, std::uint8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kInt16, 16, RColumnElementCastLE,
                            <std::int64_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kUInt16, 16, RColumnElementCastLE,
                            <std::int64_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kInt32, 32, RColumnElementCastLE,
                            <std::int64_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kUInt32, 32, RColumnElementCastLE,
                            <std::int64_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kUInt64, 64, RColumnElementCastLE,
                            <std::int64_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kSplitInt16, 16, RColumnElementZigzagSplitLE,
                            <std::int64_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kSplitUInt16, 16, RColumnElementSplitLE,
                            <std::int64_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kSplitInt32, 32, RColumnElementZigzagSplitLE,
                            <std::int64_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kSplitUInt32, 32, RColumnElementSplitLE,
                            <std::int64_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kSplitUInt64, 64, RColumnElementSplitLE,
                            <std::int64_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, ENTupleColumnType::kBit, 1, RColumnElementIntAsBool, <std::int64_t>);

DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kUInt64, 64, RColumnElementLE, <std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kSplitUInt64, 64, RColumnElementSplitLE,
                            <std::uint64_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kChar, 8, RColumnElementCastLE, <std::uint64_t, char>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kInt8, 8, RColumnElementCastLE,
                            <std::uint64_t, std::int8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kUInt8, 8, RColumnElementCastLE,
                            <std::uint64_t, std::uint8_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kInt16, 16, RColumnElementCastLE,
                            <std::uint64_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kUInt16, 16, RColumnElementCastLE,
                            <std::uint64_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kInt32, 32, RColumnElementCastLE,
                            <std::uint64_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kUInt32, 32, RColumnElementCastLE,
                            <std::uint64_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kInt64, 64, RColumnElementCastLE,
                            <std::uint64_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kSplitInt16, 16, RColumnElementZigzagSplitLE,
                            <std::uint64_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kSplitUInt16, 16, RColumnElementSplitLE,
                            <std::uint64_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kSplitInt32, 32, RColumnElementZigzagSplitLE,
                            <std::uint64_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kSplitUInt32, 32, RColumnElementSplitLE,
                            <std::uint64_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kSplitInt64, 64, RColumnElementZigzagSplitLE,
                            <std::uint64_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, ENTupleColumnType::kBit, 1, RColumnElementIntAsBool, <std::uint64_t>);

DECLARE_RCOLUMNELEMENT_SPEC(float, ENTupleColumnType::kReal32, 32, RColumnElementLE, <float>);
DECLARE_RCOLUMNELEMENT_SPEC(float, ENTupleColumnType::kSplitReal32, 32, RColumnElementSplitLE, <float, float>);
DECLARE_RCOLUMNELEMENT_SPEC(float, ENTupleColumnType::kReal64, 64, RColumnElementCastLE, <float, double>);
DECLARE_RCOLUMNELEMENT_SPEC(float, ENTupleColumnType::kSplitReal64, 64, RColumnElementSplitLE, <float, double>);

DECLARE_RCOLUMNELEMENT_SPEC(double, ENTupleColumnType::kReal64, 64, RColumnElementLE, <double>);
DECLARE_RCOLUMNELEMENT_SPEC(double, ENTupleColumnType::kSplitReal64, 64, RColumnElementSplitLE, <double, double>);
DECLARE_RCOLUMNELEMENT_SPEC(double, ENTupleColumnType::kReal32, 32, RColumnElementCastLE, <double, float>);
DECLARE_RCOLUMNELEMENT_SPEC(double, ENTupleColumnType::kSplitReal32, 32, RColumnElementSplitLE, <double, float>);

DECLARE_RCOLUMNELEMENT_SPEC(ROOT::Experimental::Internal::RColumnIndex, ENTupleColumnType::kIndex64, 64,
                            RColumnElementLE, <std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(ROOT::Experimental::Internal::RColumnIndex, ENTupleColumnType::kIndex32, 32,
                            RColumnElementCastLE, <std::uint64_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(ROOT::Experimental::Internal::RColumnIndex, ENTupleColumnType::kSplitIndex64, 64,
                            RColumnElementDeltaSplitLE, <std::uint64_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(ROOT::Experimental::Internal::RColumnIndex, ENTupleColumnType::kSplitIndex32, 32,
                            RColumnElementDeltaSplitLE, <std::uint64_t, std::uint32_t>);

template <>
class RColumnElement<ROOT::Experimental::Internal::RTestFutureColumn, kTestFutureType> final
   : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = false;
   static constexpr std::size_t kSize = sizeof(ROOT::Experimental::Internal::RTestFutureColumn);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   RColumnElement() : RColumnElementBase(kSize, kBitsOnStorage) {}

   bool IsMappable() const { return kIsMappable; }
   void Pack(void *, const void *, std::size_t) const {}
   void Unpack(void *, const void *, std::size_t) const {}

   RIdentifier GetIdentifier() const final
   {
      return RIdentifier{typeid(ROOT::Experimental::Internal::RTestFutureColumn), kTestFutureType};
   }
};

inline void RColumnElement<bool, ROOT::Experimental::ENTupleColumnType::kBit>::Pack(void *dst, const void *src,
                                                                                    std::size_t count) const
{
   const bool *boolArray = reinterpret_cast<const bool *>(src);
   char *charArray = reinterpret_cast<char *>(dst);
   std::bitset<8> bitSet;
   std::size_t i = 0;
   for (; i < count; ++i) {
      bitSet.set(i % 8, boolArray[i]);
      if (i % 8 == 7) {
         char packed = bitSet.to_ulong();
         charArray[i / 8] = packed;
      }
   }
   if (i % 8 != 0) {
      char packed = bitSet.to_ulong();
      charArray[i / 8] = packed;
   }
}

inline void RColumnElement<bool, ROOT::Experimental::ENTupleColumnType::kBit>::Unpack(void *dst, const void *src,
                                                                                      std::size_t count) const
{
   bool *boolArray = reinterpret_cast<bool *>(dst);
   const char *charArray = reinterpret_cast<const char *>(src);
   std::bitset<8> bitSet;
   for (std::size_t i = 0; i < count; i += 8) {
      bitSet = charArray[i / 8];
      for (std::size_t j = i; j < std::min(count, i + 8); ++j) {
         boolArray[j] = bitSet[j % 8];
      }
   }
}

} // namespace
