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
      ROOT::ENTupleColumnType fOnDiskType = ROOT::ENTupleColumnType::kUnknown;

      bool operator==(const RIdentifier &other) const
      {
         return this->fInMemoryType == other.fInMemoryType && this->fOnDiskType == other.fOnDiskType;
      }

      bool operator!=(const RIdentifier &other) const { return !(*this == other); }
   };

   RColumnElementBase(const RColumnElementBase &other) = default;
   RColumnElementBase(RColumnElementBase &&other) = default;
   RColumnElementBase &operator=(const RColumnElementBase &other) = delete;
   RColumnElementBase &operator=(RColumnElementBase &&other) = default;
   virtual ~RColumnElementBase() = default;

   /// If CppT == void, use the default C++ type for the given column type
   template <typename CppT = void>
   static std::unique_ptr<RColumnElementBase> Generate(ROOT::ENTupleColumnType type);
   static const char *GetColumnTypeName(ROOT::ENTupleColumnType type);
   /// Most types have a fixed on-disk bit width. Some low-precision column types
   /// have a range of possible bit widths. Return the minimum and maximum allowed
   /// bit size per type.
   static std::pair<std::uint16_t, std::uint16_t> GetValidBitRange(ROOT::ENTupleColumnType type);

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

struct RTestFutureColumn {
   std::uint32_t dummy;
};

std::unique_ptr<RColumnElementBase>
GenerateColumnElement(std::type_index inMemoryType, ROOT::ENTupleColumnType onDiskType);

std::unique_ptr<RColumnElementBase> GenerateColumnElement(const RColumnElementBase::RIdentifier &elementId);

template <typename CppT>
std::unique_ptr<RColumnElementBase> RColumnElementBase::Generate(ROOT::ENTupleColumnType onDiskType)
{
   return GenerateColumnElement(std::type_index(typeid(CppT)), onDiskType);
}

template <>
std::unique_ptr<RColumnElementBase> RColumnElementBase::Generate<void>(ROOT::ENTupleColumnType onDiskType);

} // namespace ROOT::Experimental::Internal

#endif
