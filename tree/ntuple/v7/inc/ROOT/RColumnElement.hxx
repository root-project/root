/// \file ROOT/RColumnElement.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RColumnElement
#define ROOT7_RColumnElement

#include <ROOT/RColumnModel.hxx>
#include <ROOT/RConfig.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RFloat16.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <Byteswap.h>
#include <TError.h>

#include <cstring> // for memcpy
#include <cstddef> // for std::byte
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

#ifndef R__LITTLE_ENDIAN
#ifdef R__BYTESWAP
// `R__BYTESWAP` is defined in RConfig.hxx for little-endian architectures; undefined otherwise
#define R__LITTLE_ENDIAN 1
#else
#define R__LITTLE_ENDIAN 0
#endif
#endif /* R__LITTLE_ENDIAN */

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
void CopyBswap(void *destination, const void *source, std::size_t count)
{
   auto dst = reinterpret_cast<typename RByteSwap<N>::value_type *>(destination);
   auto src = reinterpret_cast<const typename RByteSwap<N>::value_type *>(source);
   for (std::size_t i = 0; i < count; ++i) {
      dst[i] = RByteSwap<N>::bswap(src[i]);
   }
}

/// Casts T to one of the ints used in RByteSwap and back to its original type, which may be float or double
#if R__LITTLE_ENDIAN == 0
template <typename T>
void ByteSwapIfNecessary(T &value)
{
   constexpr auto N = sizeof(T);
   using bswap_value_type = typename RByteSwap<N>::value_type;
   void *valuePtr = &value;
   auto swapped = RByteSwap<N>::bswap(*reinterpret_cast<bswap_value_type *>(valuePtr));
   *reinterpret_cast<bswap_value_type *>(valuePtr) = swapped;
}
#else
#define ByteSwapIfNecessary(x) ((void)0)
#endif

/// \brief Pack `count` elements into narrower (or wider) type
///
/// Used to convert in-memory elements to smaller column types of comatible types
/// (e.g., double to float, int64 to int32). Takes care of byte swap if necessary.
template <typename DestT, typename SourceT>
static void CastPack(void *destination, const void *source, std::size_t count)
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
static void CastUnpack(void *destination, const void *source, std::size_t count)
{
   auto dst = reinterpret_cast<DestT *>(destination);
   auto src = reinterpret_cast<const SourceT *>(source);
   for (std::size_t i = 0; i < count; ++i) {
      SourceT val = src[i];
      ByteSwapIfNecessary(val);
      dst[i] = val;
   }
}

/// \brief Split encoding of elements, possibly into narrower column
///
/// Used to first cast and then split-encode in-memory values to the on-disk column. Swap bytes if necessary.
template <typename DestT, typename SourceT>
static void CastSplitPack(void *destination, const void *source, std::size_t count)
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
static void CastSplitUnpack(void *destination, const void *source, std::size_t count)
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
      dst[i] = val;
   }
}

/// \brief Packing of columns with delta + split encoding
///
/// Apply split encoding to delta-encoded values, currently used only for index columns
template <typename DestT, typename SourceT>
static void CastDeltaSplitPack(void *destination, const void *source, std::size_t count)
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
static void CastDeltaSplitUnpack(void *destination, const void *source, std::size_t count)
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
      dst[i] = (i == 0) ? val : dst[i - 1] + val;
   }
}

/// \brief Packing of columns with zigzag + split encoding
///
/// Apply split encoding to zigzag-encoded values, used for signed integers
template <typename DestT, typename SourceT>
static void CastZigzagSplitPack(void *destination, const void *source, std::size_t count)
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
static void CastZigzagSplitUnpack(void *destination, const void *source, std::size_t count)
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
      dst[i] = static_cast<SourceT>((val >> 1) ^ -(static_cast<SourceT>(val) & 1));
   }
}

} // anonymous namespace

namespace ROOT {
namespace Experimental {

namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RColumnElementBase
\ingroup NTuple
\brief A column element encapsulates the translation between basic C++ types and their column representation.

Usually the on-disk element should map bitwise to the in-memory element. Sometimes that's not the case
though, for instance on big endian platforms or for bools.

There is a template specialization for every valid pair of C++ type and column representation.
These specialized child classes are responsible for overriding `Pack()` / `Unpack()` for packing / unpacking elements
as appropriate.
*/
// clang-format on
class RColumnElementBase {
protected:
   /// Size of the C++ value that corresponds to the on-disk element
   std::size_t fSize;

   explicit RColumnElementBase(std::size_t size) : fSize(size) {}

public:
   RColumnElementBase(const RColumnElementBase& other) = default;
   RColumnElementBase(RColumnElementBase&& other) = default;
   RColumnElementBase& operator =(const RColumnElementBase& other) = delete;
   RColumnElementBase& operator =(RColumnElementBase&& other) = default;
   virtual ~RColumnElementBase() = default;

   /// If CppT == void, use the default C++ type for the given column type
   template <typename CppT = void>
   static std::unique_ptr<RColumnElementBase> Generate(EColumnType type);
   static std::size_t GetBitsOnStorage(EColumnType type);
   static std::string GetTypeName(EColumnType type);

   /// Derived, typed classes tell whether the on-storage layout is bitwise identical to the memory layout
   virtual bool IsMappable() const { R__ASSERT(false); return false; }
   virtual std::size_t GetBitsOnStorage() const { R__ASSERT(false); return 0; }

   /// If the on-storage layout and the in-memory layout differ, packing creates an on-disk page from an in-memory page
   virtual void Pack(void *destination, void *source, std::size_t count) const
   {
      std::memcpy(destination, source, count);
   }

   /// If the on-storage layout and the in-memory layout differ, unpacking creates a memory page from an on-storage page
   virtual void Unpack(void *destination, void *source, std::size_t count) const
   {
      std::memcpy(destination, source, count);
   }

   std::size_t GetSize() const { return fSize; }
   std::size_t GetPackedSize(std::size_t nElements = 1U) const { return (nElements * GetBitsOnStorage() + 7) / 8; }
};

/**
 * Base class for columns whose on-storage representation is little-endian.
 * The implementation of `Pack` and `Unpack` takes care of byteswap if the memory page is big-endian.
 */
template <typename CppT>
class RColumnElementLE : public RColumnElementBase {
protected:
   explicit RColumnElementLE(std::size_t size) : RColumnElementBase(size) {}

public:
   static constexpr bool kIsMappable = (R__LITTLE_ENDIAN == 1);

   void Pack(void *dst, void *src, std::size_t count) const final
   {
#if R__LITTLE_ENDIAN == 1
      RColumnElementBase::Pack(dst, src, count);
#else
      CopyBswap<sizeof(CppT)>(dst, src, count);
#endif
   }
   void Unpack(void *dst, void *src, std::size_t count) const final
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
   explicit RColumnElementCastLE(std::size_t size) : RColumnElementBase(size) {}

public:
   static constexpr bool kIsMappable = false;

   void Pack(void *dst, void *src, std::size_t count) const final { CastPack<NarrowT, CppT>(dst, src, count); }
   void Unpack(void *dst, void *src, std::size_t count) const final { CastUnpack<CppT, NarrowT>(dst, src, count); }
}; // class RColumnElementCastLE

/**
 * Base class for split columns whose on-storage representation is little-endian.
 * The implementation of `Pack` and `Unpack` takes care of splitting and, if necessary, byteswap.
 * As part of the splitting, can also narrow down the type to NarrowT.
 */
template <typename CppT, typename NarrowT>
class RColumnElementSplitLE : public RColumnElementBase {
protected:
   explicit RColumnElementSplitLE(std::size_t size) : RColumnElementBase(size) {}

public:
   static constexpr bool kIsMappable = false;

   void Pack(void *dst, void *src, std::size_t count) const final { CastSplitPack<NarrowT, CppT>(dst, src, count); }
   void Unpack(void *dst, void *src, std::size_t count) const final { CastSplitUnpack<CppT, NarrowT>(dst, src, count); }
}; // class RColumnElementSplitLE

/**
 * Base class for delta + split columns (index columns) whose on-storage representation is little-endian.
 * The implementation of `Pack` and `Unpack` takes care of splitting and, if necessary, byteswap.
 * As part of the encoding, can also narrow down the type to NarrowT.
 */
template <typename CppT, typename NarrowT>
class RColumnElementDeltaSplitLE : public RColumnElementBase {
protected:
   explicit RColumnElementDeltaSplitLE(std::size_t size) : RColumnElementBase(size) {}

public:
   static constexpr bool kIsMappable = false;

   void Pack(void *dst, void *src, std::size_t count) const final
   {
      CastDeltaSplitPack<NarrowT, CppT>(dst, src, count);
   }
   void Unpack(void *dst, void *src, std::size_t count) const final
   {
      CastDeltaSplitUnpack<CppT, NarrowT>(dst, src, count);
   }
}; // class RColumnElementDeltaSplitLE

/**
 * Base class for zigzag + split columns (signed integer columns) whose on-storage representation is little-endian.
 * The implementation of `Pack` and `Unpack` takes care of splitting and, if necessary, byteswap.
 * The NarrowT target type should be an signed integer, which can be smaller than the CppT source type.
 */
template <typename CppT, typename NarrowT>
class RColumnElementZigzagSplitLE : public RColumnElementBase {
protected:
   explicit RColumnElementZigzagSplitLE(std::size_t size) : RColumnElementBase(size) {}

public:
   static constexpr bool kIsMappable = false;

   void Pack(void *dst, void *src, std::size_t count) const final
   {
      CastZigzagSplitPack<NarrowT, CppT>(dst, src, count);
   }
   void Unpack(void *dst, void *src, std::size_t count) const final
   {
      CastZigzagSplitUnpack<CppT, NarrowT>(dst, src, count);
   }
}; // class RColumnElementZigzagSplitLE

////////////////////////////////////////////////////////////////////////////////
// Pairs of C++ type and column type, like float and EColumnType::kReal32
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Part 1: C++ type --> unknown column type
////////////////////////////////////////////////////////////////////////////////

template <typename CppT, EColumnType ColumnT = EColumnType::kUnknown>
class RColumnElement : public RColumnElementBase {
public:
   RColumnElement() : RColumnElementBase(sizeof(CppT))
   {
      throw RException(R__FAIL(std::string("internal error: no column mapping for this C++ type: ") +
                               typeid(CppT).name() + " --> " + GetTypeName(ColumnT)));
   }
};

template <>
class RColumnElement<bool, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(bool);
   RColumnElement() : RColumnElementBase(kSize) {}
};

template <>
class RColumnElement<std::byte, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::byte);
   RColumnElement() : RColumnElementBase(kSize) {}
};

template <>
class RColumnElement<char, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(char);
   RColumnElement() : RColumnElementBase(kSize) {}
};

template <>
class RColumnElement<std::int8_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::int8_t);
   RColumnElement() : RColumnElementBase(kSize) {}
};

template <>
class RColumnElement<std::uint8_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::uint8_t);
   RColumnElement() : RColumnElementBase(kSize) {}
};

template <>
class RColumnElement<std::int16_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::int16_t);
   RColumnElement() : RColumnElementBase(kSize) {}
};

template <>
class RColumnElement<std::uint16_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::uint16_t);
   RColumnElement() : RColumnElementBase(kSize) {}
};

template <>
class RColumnElement<std::int32_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::int32_t);
   RColumnElement() : RColumnElementBase(kSize) {}
};

template <>
class RColumnElement<std::uint32_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::uint32_t);
   RColumnElement() : RColumnElementBase(kSize) {}
};

template <>
class RColumnElement<std::int64_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::int64_t);
   RColumnElement() : RColumnElementBase(kSize) {}
};

template <>
class RColumnElement<std::uint64_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::uint64_t);
   RColumnElement() : RColumnElementBase(kSize) {}
};

template <>
class RColumnElement<float, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(float);
   RColumnElement() : RColumnElementBase(kSize) {}
};

template <>
class RColumnElement<double, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(double);
   RColumnElement() : RColumnElementBase(kSize) {}
};

template <>
class RColumnElement<ClusterSize_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(ClusterSize_t);
   RColumnElement() : RColumnElementBase(kSize) {}
};

template <>
class RColumnElement<RColumnSwitch, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(RColumnSwitch);
   RColumnElement() : RColumnElementBase(kSize) {}
};

////////////////////////////////////////////////////////////////////////////////
// Part 2: C++ type --> supported column representations,
//         ordered by C++ type
////////////////////////////////////////////////////////////////////////////////

template <>
class RColumnElement<RColumnSwitch, EColumnType::kSwitch> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = false;
   static constexpr std::size_t kSize = sizeof(ROOT::Experimental::RColumnSwitch);
   static constexpr std::size_t kBitsOnStorage = 64;
   RColumnElement() : RColumnElementBase(kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }

   void Pack(void *dst, void *src, std::size_t count) const final
   {
      auto srcArray = reinterpret_cast<ROOT::Experimental::RColumnSwitch *>(src);
      auto uint64Array = reinterpret_cast<std::uint64_t *>(dst);
      for (std::size_t i = 0; i < count; ++i) {
         uint64Array[i] =
            (static_cast<std::uint64_t>(srcArray[i].GetTag()) << 44) | (srcArray[i].GetIndex() & 0x0fffffffffff);
#if R__LITTLE_ENDIAN == 0
         uint64Array[i] = RByteSwap<8>::bswap(uint64Array[i]);
#endif
      }
   }

   void Unpack(void *dst, void *src, std::size_t count) const final
   {
      auto uint64Array = reinterpret_cast<std::uint64_t *>(src);
      auto dstArray = reinterpret_cast<ROOT::Experimental::RColumnSwitch *>(dst);
      for (std::size_t i = 0; i < count; ++i) {
#if R__LITTLE_ENDIAN == 1
         const auto value = uint64Array[i];
#else
         const auto value = RByteSwap<8>::bswap(uint64Array[i]);
#endif
         dstArray[i] = ROOT::Experimental::RColumnSwitch(
            ClusterSize_t{static_cast<RClusterSize::ValueType>(value & 0x0fffffffffff)}, (value >> 44));
      }
   }
};

template <>
class RColumnElement<bool, EColumnType::kBit> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = false;
   static constexpr std::size_t kSize = sizeof(bool);
   static constexpr std::size_t kBitsOnStorage = 1;
   RColumnElement() : RColumnElementBase(kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }

   void Pack(void *dst, void *src, std::size_t count) const final;
   void Unpack(void *dst, void *src, std::size_t count) const final;
};

template <>
class RColumnElement<float, EColumnType::kReal16> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = false;
   static constexpr std::size_t kSize = sizeof(float);
   static constexpr std::size_t kBitsOnStorage = 16;
   RColumnElement() : RColumnElementBase(kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }

   void Pack(void *dst, void *src, std::size_t count) const final
   {
      float *floatArray = reinterpret_cast<float *>(src);
      std::uint16_t *uint16Array = reinterpret_cast<std::uint16_t *>(dst);

      for (std::size_t i = 0; i < count; ++i) {
         uint16Array[i] = Internal::FloatToHalf(floatArray[i]);
         ByteSwapIfNecessary(uint16Array[i]);
      }
   }

   void Unpack(void *dst, void *src, std::size_t count) const final
   {
      float *floatArray = reinterpret_cast<float *>(dst);
      std::uint16_t *uint16Array = reinterpret_cast<std::uint16_t *>(src);

      for (std::size_t i = 0; i < count; ++i) {
         ByteSwapIfNecessary(floatArray[i]);
         floatArray[i] = Internal::HalfToFloat(uint16Array[i]);
      }
   }
};

#define __RCOLUMNELEMENT_SPEC_BODY(CppT, BaseT, BitsOnStorage)  \
   static constexpr std::size_t kSize = sizeof(CppT);           \
   static constexpr std::size_t kBitsOnStorage = BitsOnStorage; \
   RColumnElement() : BaseT(kSize) {}                           \
   bool IsMappable() const final                                \
   {                                                            \
      return kIsMappable;                                       \
   }                                                            \
   std::size_t GetBitsOnStorage() const final                   \
   {                                                            \
      return kBitsOnStorage;                                    \
   }
/// These macros are used to declare `RColumnElement` template specializations below.  Additional arguments can be used
/// to forward template parameters to the base class, e.g.
/// ```
/// DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, EColumnType::kInt32, 32,
///                             RColumnElementCastLE, <std::int64_t, std::int32_t>);
/// ```
#define DECLARE_RCOLUMNELEMENT_SPEC(CppT, ColumnT, BitsOnStorage, BaseT, ...) \
   template <>                                                                \
   class RColumnElement<CppT, ColumnT> : public BaseT __VA_ARGS__ {           \
   public:                                                                    \
      __RCOLUMNELEMENT_SPEC_BODY(CppT, BaseT, BitsOnStorage)                  \
   }
#define DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(CppT, ColumnT, BitsOnStorage)  \
   template <>                                                            \
   class RColumnElement<CppT, ColumnT> : public RColumnElementBase {      \
   public:                                                                \
      static constexpr bool kIsMappable = true;                           \
      __RCOLUMNELEMENT_SPEC_BODY(CppT, RColumnElementBase, BitsOnStorage) \
   }

DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(std::byte, EColumnType::kByte, 8);

DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(char, EColumnType::kByte, 8);
DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(char, EColumnType::kChar, 8);

DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(std::int8_t, EColumnType::kInt8, 8);
DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(std::int8_t, EColumnType::kUInt8, 8);
DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(std::int8_t, EColumnType::kByte, 8);

DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(std::uint8_t, EColumnType::kUInt8, 8);
DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(std::uint8_t, EColumnType::kInt8, 8);
DECLARE_RCOLUMNELEMENT_SPEC_SIMPLE(std::uint8_t, EColumnType::kByte, 8);

DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, EColumnType::kInt16, 16, RColumnElementLE, <std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, EColumnType::kUInt16, 16, RColumnElementLE, <std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, EColumnType::kSplitInt16, 16, RColumnElementZigzagSplitLE,
                            <std::int16_t, std::int16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int16_t, EColumnType::kSplitUInt16, 16, RColumnElementSplitLE,
                            <std::int16_t, std::uint16_t>);

DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, EColumnType::kUInt16, 16, RColumnElementLE, <std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, EColumnType::kInt16, 16, RColumnElementLE, <std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, EColumnType::kSplitUInt16, 16, RColumnElementSplitLE,
                            <std::uint16_t, std::uint16_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint16_t, EColumnType::kSplitInt16, 16, RColumnElementZigzagSplitLE,
                            <std::uint16_t, std::int16_t>);

DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, EColumnType::kInt32, 32, RColumnElementLE, <std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, EColumnType::kUInt32, 32, RColumnElementLE, <std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, EColumnType::kSplitInt32, 32, RColumnElementZigzagSplitLE,
                            <std::int32_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int32_t, EColumnType::kSplitUInt32, 32, RColumnElementSplitLE,
                            <std::int32_t, std::uint32_t>);

DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, EColumnType::kUInt32, 32, RColumnElementLE, <std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, EColumnType::kInt32, 32, RColumnElementLE, <std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, EColumnType::kSplitUInt32, 32, RColumnElementSplitLE,
                            <std::uint32_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint32_t, EColumnType::kSplitInt32, 32, RColumnElementZigzagSplitLE,
                            <std::uint32_t, std::int32_t>);

DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, EColumnType::kInt64, 64, RColumnElementLE, <std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, EColumnType::kUInt64, 64, RColumnElementLE, <std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, EColumnType::kSplitInt64, 64, RColumnElementZigzagSplitLE,
                            <std::int64_t, std::int64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, EColumnType::kSplitUInt64, 64, RColumnElementSplitLE,
                            <std::int64_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, EColumnType::kInt32, 32, RColumnElementCastLE, <std::int64_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, EColumnType::kUInt32, 32, RColumnElementCastLE,
                            <std::int64_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, EColumnType::kSplitInt32, 32, RColumnElementZigzagSplitLE,
                            <std::int64_t, std::int32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::int64_t, EColumnType::kSplitUInt32, 32, RColumnElementSplitLE,
                            <std::int64_t, std::uint32_t>);

DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, EColumnType::kUInt64, 64, RColumnElementLE, <std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, EColumnType::kInt64, 64, RColumnElementLE, <std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, EColumnType::kSplitUInt64, 64, RColumnElementSplitLE,
                            <std::uint64_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(std::uint64_t, EColumnType::kSplitInt64, 64, RColumnElementZigzagSplitLE,
                            <std::uint64_t, std::int64_t>);

DECLARE_RCOLUMNELEMENT_SPEC(float, EColumnType::kReal32, 32, RColumnElementLE, <float>);
DECLARE_RCOLUMNELEMENT_SPEC(float, EColumnType::kSplitReal32, 32, RColumnElementSplitLE, <float, float>);

DECLARE_RCOLUMNELEMENT_SPEC(double, EColumnType::kReal64, 64, RColumnElementLE, <double>);
DECLARE_RCOLUMNELEMENT_SPEC(double, EColumnType::kSplitReal64, 64, RColumnElementSplitLE, <double, double>);
DECLARE_RCOLUMNELEMENT_SPEC(double, EColumnType::kReal32, 32, RColumnElementCastLE, <double, float>);
DECLARE_RCOLUMNELEMENT_SPEC(double, EColumnType::kSplitReal32, 32, RColumnElementSplitLE, <double, float>);

DECLARE_RCOLUMNELEMENT_SPEC(ClusterSize_t, EColumnType::kIndex64, 64, RColumnElementLE, <std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(ClusterSize_t, EColumnType::kIndex32, 32, RColumnElementCastLE,
                            <std::uint64_t, std::uint32_t>);
DECLARE_RCOLUMNELEMENT_SPEC(ClusterSize_t, EColumnType::kSplitIndex64, 64, RColumnElementDeltaSplitLE,
                            <std::uint64_t, std::uint64_t>);
DECLARE_RCOLUMNELEMENT_SPEC(ClusterSize_t, EColumnType::kSplitIndex32, 32, RColumnElementDeltaSplitLE,
                            <std::uint64_t, std::uint32_t>);

template <typename CppT>
std::unique_ptr<RColumnElementBase> RColumnElementBase::Generate(EColumnType type)
{
   switch (type) {
   case EColumnType::kIndex64: return std::make_unique<RColumnElement<CppT, EColumnType::kIndex64>>();
   case EColumnType::kIndex32: return std::make_unique<RColumnElement<CppT, EColumnType::kIndex32>>();
   case EColumnType::kSwitch: return std::make_unique<RColumnElement<CppT, EColumnType::kSwitch>>();
   case EColumnType::kByte: return std::make_unique<RColumnElement<CppT, EColumnType::kByte>>();
   case EColumnType::kChar: return std::make_unique<RColumnElement<CppT, EColumnType::kChar>>();
   case EColumnType::kBit: return std::make_unique<RColumnElement<CppT, EColumnType::kBit>>();
   case EColumnType::kReal64: return std::make_unique<RColumnElement<CppT, EColumnType::kReal64>>();
   case EColumnType::kReal32: return std::make_unique<RColumnElement<CppT, EColumnType::kReal32>>();
   case EColumnType::kReal16: return std::make_unique<RColumnElement<CppT, EColumnType::kReal16>>();
   case EColumnType::kInt64: return std::make_unique<RColumnElement<CppT, EColumnType::kInt64>>();
   case EColumnType::kUInt64: return std::make_unique<RColumnElement<CppT, EColumnType::kUInt64>>();
   case EColumnType::kInt32: return std::make_unique<RColumnElement<CppT, EColumnType::kInt32>>();
   case EColumnType::kUInt32: return std::make_unique<RColumnElement<CppT, EColumnType::kUInt32>>();
   case EColumnType::kInt16: return std::make_unique<RColumnElement<CppT, EColumnType::kInt16>>();
   case EColumnType::kUInt16: return std::make_unique<RColumnElement<CppT, EColumnType::kUInt16>>();
   case EColumnType::kInt8: return std::make_unique<RColumnElement<CppT, EColumnType::kInt8>>();
   case EColumnType::kUInt8: return std::make_unique<RColumnElement<CppT, EColumnType::kUInt8>>();
   case EColumnType::kSplitIndex64: return std::make_unique<RColumnElement<CppT, EColumnType::kSplitIndex64>>();
   case EColumnType::kSplitIndex32: return std::make_unique<RColumnElement<CppT, EColumnType::kSplitIndex32>>();
   case EColumnType::kSplitReal64: return std::make_unique<RColumnElement<CppT, EColumnType::kSplitReal64>>();
   case EColumnType::kSplitReal32: return std::make_unique<RColumnElement<CppT, EColumnType::kSplitReal32>>();
   case EColumnType::kSplitInt64: return std::make_unique<RColumnElement<CppT, EColumnType::kSplitInt64>>();
   case EColumnType::kSplitUInt64: return std::make_unique<RColumnElement<CppT, EColumnType::kSplitUInt64>>();
   case EColumnType::kSplitInt32: return std::make_unique<RColumnElement<CppT, EColumnType::kSplitInt32>>();
   case EColumnType::kSplitUInt32: return std::make_unique<RColumnElement<CppT, EColumnType::kSplitUInt32>>();
   case EColumnType::kSplitInt16: return std::make_unique<RColumnElement<CppT, EColumnType::kSplitInt16>>();
   case EColumnType::kSplitUInt16: return std::make_unique<RColumnElement<CppT, EColumnType::kSplitUInt16>>();
   default: R__ASSERT(false);
   }
   // never here
   return nullptr;
}

template <>
std::unique_ptr<RColumnElementBase> RColumnElementBase::Generate<void>(EColumnType type);

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
