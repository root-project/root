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
#include <ROOT/RNTupleUtil.hxx>

#include <Byteswap.h>
#include <TError.h>

#include <cstring> // for memcpy
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
/// \brief Copy and byteswap `count` elements of size `N` from `source` to `destination`.
///
/// Used on big-endian architectures for packing/unpacking elements whose column type requires
/// a little-endian on-disk representation.
template <std::size_t N>
static void CopyElementsBswap(void *destination, const void *source, std::size_t count)
{
   auto dst = reinterpret_cast<typename RByteSwap<N>::value_type *>(destination);
   auto src = reinterpret_cast<const typename RByteSwap<N>::value_type *>(source);
   for (std::size_t i = 0; i < count; ++i) {
      dst[i] = RByteSwap<N>::bswap(src[i]);
   }
}
} // anonymous namespace

namespace ROOT {
namespace Experimental {

namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RColumnElement
\ingroup NTuple
\brief A column element points either to the content of an RFieldValue or into a memory mapped page.

The content pointed to by fRawContent can be a single element or the first element of an array.
Usually the on-disk element should map bitwise to the in-memory element. Sometimes that's not the case
though, for instance on big endian platforms and for exotic physical columns like 8 bit float.

This class does not provide protection around the raw pointer, fRawContent has to be managed correctly
by the user of this class.
*/
// clang-format on
class RColumnElementBase {
protected:
   /// Points to valid C++ data, either a single value or an array of values
   void* fRawContent;
   /// Size of the C++ value pointed to by fRawContent (not necessarily equal to the on-disk element size)
   std::size_t fSize;

public:
   RColumnElementBase()
     : fRawContent(nullptr)
     , fSize(0)
   {}
   RColumnElementBase(void *rawContent, std::size_t size) : fRawContent(rawContent), fSize(size)
   {}
   RColumnElementBase(const RColumnElementBase &elemArray, std::size_t at)
     : fRawContent(static_cast<unsigned char *>(elemArray.fRawContent) + elemArray.fSize * at)
     , fSize(elemArray.fSize)
   {}
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

   /// Write one or multiple column elements into destination
   void WriteTo(void *destination, std::size_t count) const {
      std::memcpy(destination, fRawContent, fSize * count);
   }

   /// Set the column element or an array of elements from the memory location source
   void ReadFrom(void *source, std::size_t count) {
      std::memcpy(fRawContent, source, fSize * count);
   }

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

   void *GetRawContent() const { return fRawContent; }
   std::size_t GetSize() const { return fSize; }
   std::size_t GetPackedSize(std::size_t nElements) const { return (nElements * GetBitsOnStorage() + 7) / 8; }
};

/**
 * Base class for columns whose on-storage representation is little-endian.
 * The implementation of `Pack` and `Unpack` takes care of byteswap if the memory page is big-endian.
 */
template <typename CppT>
class RColumnElementLE : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = (R__LITTLE_ENDIAN == 1);
   RColumnElementLE(void *rawContent, std::size_t size) : RColumnElementBase(rawContent, size) {}

   void Pack(void *dst, void *src, std::size_t count) const final
   {
#if R__LITTLE_ENDIAN == 1
      RColumnElementBase::Pack(dst, src, count);
#else
      CopyElementsBswap<sizeof(CppT)>(dst, src, count);
#endif
   }
   void Unpack(void *dst, void *src, std::size_t count) const final
   {
#if R__LITTLE_ENDIAN == 1
      RColumnElementBase::Unpack(dst, src, count);
#else
      CopyElementsBswap<sizeof(CppT)>(dst, src, count);
#endif
   }
};

/**
 * Pairs of C++ type and column type, like float and EColumnType::kReal32
 */
template <typename CppT, EColumnType ColumnT = EColumnType::kUnknown>
class RColumnElement : public RColumnElementBase {
public:
   explicit RColumnElement(CppT* value) : RColumnElementBase(value, sizeof(CppT))
   {
      throw RException(R__FAIL(std::string("internal error: no column mapping for this C++ type: ") +
                               typeid(CppT).name() + " --> " + GetTypeName(ColumnT)));
   }
};

template <>
class RColumnElement<bool, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(bool);
   explicit RColumnElement(bool *value) : RColumnElementBase(value, kSize) {}
};

template <>
class RColumnElement<char, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(char);
   explicit RColumnElement(char *value) : RColumnElementBase(value, kSize) {}
};

template <>
class RColumnElement<std::int8_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::int8_t);
   explicit RColumnElement(std::int8_t *value) : RColumnElementBase(value, kSize) {}
};

template <>
class RColumnElement<std::uint8_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::uint8_t);
   explicit RColumnElement(std::uint8_t *value) : RColumnElementBase(value, kSize) {}
};

template <>
class RColumnElement<std::int16_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::int16_t);
   explicit RColumnElement(std::int16_t *value) : RColumnElementBase(value, kSize) {}
};

template <>
class RColumnElement<std::uint16_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::uint16_t);
   explicit RColumnElement(std::uint16_t *value) : RColumnElementBase(value, kSize) {}
};

template <>
class RColumnElement<std::int32_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::int32_t);
   explicit RColumnElement(std::int32_t *value) : RColumnElementBase(value, kSize) {}
};

template <>
class RColumnElement<std::uint32_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::uint32_t);
   explicit RColumnElement(std::uint32_t *value) : RColumnElementBase(value, kSize) {}
};

template <>
class RColumnElement<std::int64_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::int64_t);
   explicit RColumnElement(std::int64_t *value) : RColumnElementBase(value, kSize) {}
};

template <>
class RColumnElement<std::uint64_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(std::uint64_t);
   explicit RColumnElement(std::uint64_t *value) : RColumnElementBase(value, kSize) {}
};

template <>
class RColumnElement<float, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(float);
   explicit RColumnElement(float *value) : RColumnElementBase(value, kSize) {}
};

template <>
class RColumnElement<double, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(double);
   explicit RColumnElement(double *value) : RColumnElementBase(value, kSize) {}
};

template <>
class RColumnElement<ClusterSize_t, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(ClusterSize_t);
   explicit RColumnElement(ClusterSize_t *value) : RColumnElementBase(value, kSize) {}
};

template <>
class RColumnElement<RColumnSwitch, EColumnType::kUnknown> : public RColumnElementBase {
public:
   static constexpr std::size_t kSize = sizeof(RColumnSwitch);
   explicit RColumnElement(RColumnSwitch *value) : RColumnElementBase(value, kSize) {}
};

template <>
class RColumnElement<float, EColumnType::kReal32> : public RColumnElementLE<float> {
public:
   static constexpr std::size_t kSize = sizeof(float);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(float *value) : RColumnElementLE(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<double, EColumnType::kReal64> : public RColumnElementLE<double> {
public:
   static constexpr std::size_t kSize = sizeof(double);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(double *value) : RColumnElementLE(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<double, EColumnType::kSplitReal64> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = false;
   static constexpr std::size_t kSize = sizeof(double);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(double *value) : RColumnElementBase(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }

   void Pack(void *dst, void *src, std::size_t count) const final;
   void Unpack(void *dst, void *src, std::size_t count) const final;
};

template <>
class RColumnElement<std::int8_t, EColumnType::kInt8> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = true;
   static constexpr std::size_t kSize = sizeof(std::int8_t);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(std::int8_t *value) : RColumnElementBase(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<std::uint8_t, EColumnType::kInt8> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = true;
   static constexpr std::size_t kSize = sizeof(std::uint8_t);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(std::uint8_t *value) : RColumnElementBase(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<std::int8_t, EColumnType::kByte> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = true;
   static constexpr std::size_t kSize = sizeof(std::int8_t);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(std::int8_t *value) : RColumnElementBase(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<std::uint8_t, EColumnType::kByte> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = true;
   static constexpr std::size_t kSize = sizeof(std::uint8_t);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(std::uint8_t *value) : RColumnElementBase(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<std::int16_t, EColumnType::kInt16> : public RColumnElementLE<std::int16_t> {
public:
   static constexpr std::size_t kSize = sizeof(std::int16_t);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(std::int16_t *value) : RColumnElementLE(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<std::uint16_t, EColumnType::kInt16> : public RColumnElementLE<std::uint16_t> {
public:
   static constexpr std::size_t kSize = sizeof(std::uint16_t);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(std::uint16_t *value) : RColumnElementLE(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<std::int32_t, EColumnType::kInt32> : public RColumnElementLE<std::int32_t> {
public:
   static constexpr std::size_t kSize = sizeof(std::int32_t);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(std::int32_t *value) : RColumnElementLE(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<std::uint32_t, EColumnType::kInt32> : public RColumnElementLE<std::uint32_t> {
public:
   static constexpr std::size_t kSize = sizeof(std::uint32_t);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(std::uint32_t *value) : RColumnElementLE(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<std::int64_t, EColumnType::kInt64> : public RColumnElementLE<std::int64_t> {
public:
   static constexpr std::size_t kSize = sizeof(std::int64_t);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(std::int64_t *value) : RColumnElementLE(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<std::uint64_t, EColumnType::kInt64> : public RColumnElementLE<std::uint64_t> {
public:
   static constexpr std::size_t kSize = sizeof(std::uint64_t);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(std::uint64_t *value) : RColumnElementLE(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<ClusterSize_t, EColumnType::kIndex> : public RColumnElementLE<ClusterSize_t::ValueType> {
public:
   static constexpr std::size_t kSize = sizeof(ROOT::Experimental::ClusterSize_t);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(ClusterSize_t *value) : RColumnElementLE(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<RColumnSwitch, EColumnType::kSwitch> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = false;
   static constexpr std::size_t kSize = sizeof(ROOT::Experimental::RColumnSwitch);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(RColumnSwitch *value) : RColumnElementBase(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }

   void Pack(void *dst, void *src, std::size_t count) const final;
   void Unpack(void *dst, void *src, std::size_t count) const final;
};

template <>
class RColumnElement<char, EColumnType::kByte> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = true;
   static constexpr std::size_t kSize = sizeof(char);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(char *value) : RColumnElementBase(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<char, EColumnType::kChar> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = true;
   static constexpr std::size_t kSize = sizeof(char);
   static constexpr std::size_t kBitsOnStorage = kSize * 8;
   explicit RColumnElement(char *value) : RColumnElementBase(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }
};

template <>
class RColumnElement<bool, EColumnType::kBit> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = false;
   static constexpr std::size_t kSize = sizeof(bool);
   static constexpr std::size_t kBitsOnStorage = 1;
   explicit RColumnElement(bool *value) : RColumnElementBase(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }

   void Pack(void *dst, void *src, std::size_t count) const final;
   void Unpack(void *dst, void *src, std::size_t count) const final;
};

template <>
class RColumnElement<std::int64_t, EColumnType::kInt32> : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = false;
   static constexpr std::size_t kSize = sizeof(std::int64_t);
   static constexpr std::size_t kBitsOnStorage = 32;
   explicit RColumnElement(std::int64_t *value) : RColumnElementBase(value, kSize) {}
   bool IsMappable() const final { return kIsMappable; }
   std::size_t GetBitsOnStorage() const final { return kBitsOnStorage; }

   void Pack(void *dst, void *src, std::size_t count) const final;
   void Unpack(void *dst, void *src, std::size_t count) const final;
};

template <typename CppT>
std::unique_ptr<RColumnElementBase> RColumnElementBase::Generate(EColumnType type)
{
   switch (type) {
   case EColumnType::kReal32: return std::make_unique<RColumnElement<CppT, EColumnType::kReal32>>(nullptr);
   case EColumnType::kReal64: return std::make_unique<RColumnElement<CppT, EColumnType::kReal64>>(nullptr);
   case EColumnType::kSplitReal64: return std::make_unique<RColumnElement<CppT, EColumnType::kSplitReal64>>(nullptr);
   case EColumnType::kChar: return std::make_unique<RColumnElement<CppT, EColumnType::kChar>>(nullptr);
   case EColumnType::kByte: return std::make_unique<RColumnElement<CppT, EColumnType::kByte>>(nullptr);
   case EColumnType::kInt8: return std::make_unique<RColumnElement<CppT, EColumnType::kInt8>>(nullptr);
   case EColumnType::kInt16: return std::make_unique<RColumnElement<CppT, EColumnType::kInt16>>(nullptr);
   case EColumnType::kInt32: return std::make_unique<RColumnElement<CppT, EColumnType::kInt32>>(nullptr);
   case EColumnType::kInt64: return std::make_unique<RColumnElement<CppT, EColumnType::kInt64>>(nullptr);
   case EColumnType::kBit: return std::make_unique<RColumnElement<CppT, EColumnType::kBit>>(nullptr);
   case EColumnType::kIndex: return std::make_unique<RColumnElement<CppT, EColumnType::kIndex>>(nullptr);
   case EColumnType::kSwitch: return std::make_unique<RColumnElement<CppT, EColumnType::kSwitch>>(nullptr);
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
