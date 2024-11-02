/// \file ROOT/RColumnElementBase.hxx
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

#ifndef ROOT7_RColumnElementBase
#define ROOT7_RColumnElementBase

#include "RtypesCore.h"
#include <ROOT/RError.hxx>
#include <ROOT/RFloat16.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <Byteswap.h>
#include <TError.h>

#include <cstring> // for memcpy
#include <cstddef> // for std::byte
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>

namespace ROOT::Experimental::Internal {

// clang-format off
/**
\class ROOT::Experimental::Internal::RColumnElementBase
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
   std::size_t fBitsOnStorage;
   /// This is only meaningful for column elements that support it (e.g. Real32Quant)
   std::optional<std::pair<double, double>> fValueRange = std::nullopt;

   explicit RColumnElementBase(std::size_t size, std::size_t bitsOnStorage = 0)
      : fSize(size), fBitsOnStorage(bitsOnStorage ? bitsOnStorage : 8 * size)
   {
   }

public:
   /// Every concrete RColumnElement type is identified by its on-disk type (column type) and the
   /// in-memory C++ type, given by a type index.
   struct RIdentifier {
      std::type_index fInMemoryType = std::type_index(typeid(void));
      EColumnType fOnDiskType = EColumnType::kUnknown;
   };

   RColumnElementBase(const RColumnElementBase &other) = default;
   RColumnElementBase(RColumnElementBase &&other) = default;
   RColumnElementBase &operator=(const RColumnElementBase &other) = delete;
   RColumnElementBase &operator=(RColumnElementBase &&other) = default;
   virtual ~RColumnElementBase() = default;

   /// If CppT == void, use the default C++ type for the given column type
   template <typename CppT = void>
   static std::unique_ptr<RColumnElementBase> Generate(EColumnType type);
   static const char *GetColumnTypeName(EColumnType type);
   /// Most types have a fixed on-disk bit width. Some low-precision column types
   /// have a range of possible bit widths. Return the minimum and maximum allowed
   /// bit size per type.
   static std::pair<std::uint16_t, std::uint16_t> GetValidBitRange(EColumnType type);

   /// Derived, typed classes tell whether the on-storage layout is bitwise identical to the memory layout
   virtual bool IsMappable() const
   {
      R__ASSERT(false);
      return false;
   }

   virtual void SetBitsOnStorage(std::size_t bitsOnStorage)
   {
      if (bitsOnStorage != fBitsOnStorage)
         throw RException(R__FAIL(std::string("internal error: cannot change bit width of this column type")));
   }

   virtual void SetValueRange(double, double)
   {
      throw RException(R__FAIL(std::string("internal error: cannot change value range of this column type")));
   }

   /// If the on-storage layout and the in-memory layout differ, packing creates an on-disk page from an in-memory page
   virtual void Pack(void *destination, const void *source, std::size_t count) const
   {
      std::memcpy(destination, source, count);
   }

   /// If the on-storage layout and the in-memory layout differ, unpacking creates a memory page from an on-storage page
   virtual void Unpack(void *destination, const void *source, std::size_t count) const
   {
      std::memcpy(destination, source, count);
   }

   std::size_t GetSize() const { return fSize; }
   std::size_t GetBitsOnStorage() const { return fBitsOnStorage; }
   std::optional<std::pair<double, double>> GetValueRange() const { return fValueRange; }
   std::size_t GetPackedSize(std::size_t nElements = 1U) const { return (nElements * fBitsOnStorage + 7) / 8; }

   virtual RIdentifier GetIdentifier() const = 0;
}; // class RColumnElementBase

// All supported C++ in-memory types
enum class EColumnCppType {
   kChar,
   kBool,
   kByte,
   kUint8,
   kUint16,
   kUint32,
   kUint64,
   kInt8,
   kInt16,
   kInt32,
   kInt64,
   kFloat,
   kDouble,
   kClusterSize,
   kColumnSwitch,
   kMax
};

inline constexpr EColumnCppType kTestFutureColumn =
   static_cast<EColumnCppType>(std::numeric_limits<std::underlying_type_t<EColumnCppType>>::max() - 1);

struct RTestFutureColumn {
   std::uint32_t dummy;
};

std::unique_ptr<RColumnElementBase> GenerateColumnElement(EColumnCppType cppType, EColumnType colType);

template <typename CppT>
std::unique_ptr<RColumnElementBase> RColumnElementBase::Generate(EColumnType type)
{
   if constexpr (std::is_same_v<CppT, char>)
      return GenerateColumnElement(EColumnCppType::kChar, type);
   else if constexpr (std::is_same_v<CppT, bool>)
      return GenerateColumnElement(EColumnCppType::kBool, type);
   else if constexpr (std::is_same_v<CppT, std::byte>)
      return GenerateColumnElement(EColumnCppType::kByte, type);
   else if constexpr (std::is_same_v<CppT, std::uint8_t>)
      return GenerateColumnElement(EColumnCppType::kUint8, type);
   else if constexpr (std::is_same_v<CppT, std::uint16_t>)
      return GenerateColumnElement(EColumnCppType::kUint16, type);
   else if constexpr (std::is_same_v<CppT, std::uint32_t>)
      return GenerateColumnElement(EColumnCppType::kUint32, type);
   else if constexpr (std::is_same_v<CppT, std::uint64_t>)
      return GenerateColumnElement(EColumnCppType::kUint64, type);
   else if constexpr (std::is_same_v<CppT, std::int8_t>)
      return GenerateColumnElement(EColumnCppType::kInt8, type);
   else if constexpr (std::is_same_v<CppT, std::int16_t>)
      return GenerateColumnElement(EColumnCppType::kInt16, type);
   else if constexpr (std::is_same_v<CppT, std::int32_t>)
      return GenerateColumnElement(EColumnCppType::kInt32, type);
   else if constexpr (std::is_same_v<CppT, std::int64_t>)
      return GenerateColumnElement(EColumnCppType::kInt64, type);
   else if constexpr (std::is_same_v<CppT, float>)
      return GenerateColumnElement(EColumnCppType::kFloat, type);
   else if constexpr (std::is_same_v<CppT, double>)
      return GenerateColumnElement(EColumnCppType::kDouble, type);
   else if constexpr (std::is_same_v<CppT, ClusterSize_t>)
      return GenerateColumnElement(EColumnCppType::kClusterSize, type);
   else if constexpr (std::is_same_v<CppT, RColumnSwitch>)
      return GenerateColumnElement(EColumnCppType::kColumnSwitch, type);
   else if constexpr (std::is_same_v<CppT, RTestFutureColumn>)
      return GenerateColumnElement(kTestFutureColumn, type);
   else
      static_assert(!sizeof(CppT), "Unsupported Cpp type");
}

template <>
std::unique_ptr<RColumnElementBase> RColumnElementBase::Generate<void>(EColumnType type);

} // namespace ROOT::Experimental::Internal

#endif
