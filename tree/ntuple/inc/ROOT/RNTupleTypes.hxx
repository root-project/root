/// \file ROOT/RNTupleTypes.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleTypes
#define ROOT_RNTupleTypes

#include <ROOT/RConfig.hxx>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>
#include <variant>

namespace ROOT {

/// Helper types to present an offset column as array of collection sizes.
/// See RField<RNTupleCardinality<SizeT>> for details.
template <typename SizeT>
struct RNTupleCardinality {
   static_assert(std::is_same_v<SizeT, std::uint32_t> || std::is_same_v<SizeT, std::uint64_t>,
                 "RNTupleCardinality is only supported with std::uint32_t or std::uint64_t template parameters");

   using ValueType = SizeT;

   RNTupleCardinality() : fValue(0) {}
   explicit constexpr RNTupleCardinality(ValueType value) : fValue(value) {}
   RNTupleCardinality &operator=(const ValueType value)
   {
      fValue = value;
      return *this;
   }
   operator ValueType() const { return fValue; }

   ValueType fValue;
};

// clang-format off
/**
\class ROOT::ENTupleColumnType
\ingroup NTuple
\brief The available trivial, native content types of a column

More complex types, such as classes, get translated into columns of such simple types by the RField.
When changed, remember to update
  - RColumnElement::Generate()
  - RColumnElement::GetTypeName()
  - RColumnElement::GetValidBitRange()
  - RColumnElement template specializations / packing & unpacking
  - If necessary, endianess handling for the packing + unit test in ntuple_endian
  - RNTupleSerializer::[Des|S]erializeColumnType
*/
// clang-format on
enum class ENTupleColumnType {
   kUnknown = 0,
   // type for root columns of (nested) collections; offsets are relative to the current cluster
   kIndex64,
   kIndex32,
   // 96 bit column that is a pair of a kIndex64 and a 32bit dispatch tag to a column ID;
   // used to serialize std::variant.
   kSwitch,
   kByte,
   kChar,
   kBit,
   kReal64,
   kReal32,
   kReal16,
   kInt64,
   kUInt64,
   kInt32,
   kUInt32,
   kInt16,
   kUInt16,
   kInt8,
   kUInt8,
   kSplitIndex64,
   kSplitIndex32,
   kSplitReal64,
   kSplitReal32,
   kSplitInt64,
   kSplitUInt64,
   kSplitInt32,
   kSplitUInt32,
   kSplitInt16,
   kSplitUInt16,
   kReal32Trunc,
   kReal32Quant,
   kMax,
};

/// The fields in the ntuple model tree can carry different structural information about the type system.
/// Leaf fields contain just data, collection fields resolve to offset columns, record fields have no
/// materialization on the primitive column layer.
// IMPORTANT: if you add members, remember to change the related `operator<<` below.
enum class ENTupleStructure : std::uint16_t {
   kInvalid,
   kLeaf,
   kCollection,
   kRecord,
   kVariant,
   kStreamer,
   kUnknown
};

inline std::ostream &operator<<(std::ostream &os, ENTupleStructure structure)
{
   static const char *const names[] = {"Invalid", "Leaf", "Collection", "Record", "Variant", "Streamer", "Unknown"};
   static_assert((std::size_t)ENTupleStructure::kUnknown + 1 == std::size(names));

   if (R__likely(static_cast<std::size_t>(structure) <= std::size(names)))
      os << names[static_cast<std::uint16_t>(structure)];
   else
      os << "(invalid)";
   return os;
}

/// Integer type long enough to hold the maximum number of entries in a column
using NTupleSize_t = std::uint64_t;
constexpr NTupleSize_t kInvalidNTupleIndex = std::uint64_t(-1);

/// Distriniguishes elements of the same type within a descriptor, e.g. different fields
using DescriptorId_t = std::uint64_t;
constexpr DescriptorId_t kInvalidDescriptorId = std::uint64_t(-1);

/// Addresses a column element or field item relative to a particular cluster, instead of a global NTupleSize_t index
class RNTupleLocalIndex {
private:
   ROOT::DescriptorId_t fClusterId = ROOT::kInvalidDescriptorId;
   ROOT::NTupleSize_t fIndexInCluster = ROOT::kInvalidNTupleIndex;

public:
   RNTupleLocalIndex() = default;
   RNTupleLocalIndex(const RNTupleLocalIndex &other) = default;
   RNTupleLocalIndex &operator=(const RNTupleLocalIndex &other) = default;
   constexpr RNTupleLocalIndex(ROOT::DescriptorId_t clusterId, ROOT::NTupleSize_t indexInCluster)
      : fClusterId(clusterId), fIndexInCluster(indexInCluster)
   {
   }

   RNTupleLocalIndex operator+(ROOT::NTupleSize_t off) const
   {
      return RNTupleLocalIndex(fClusterId, fIndexInCluster + off);
   }

   RNTupleLocalIndex operator-(ROOT::NTupleSize_t off) const
   {
      return RNTupleLocalIndex(fClusterId, fIndexInCluster - off);
   }

   RNTupleLocalIndex operator++(int) /* postfix */
   {
      auto r = *this;
      fIndexInCluster++;
      return r;
   }

   RNTupleLocalIndex &operator++() /* prefix */
   {
      ++fIndexInCluster;
      return *this;
   }

   bool operator==(RNTupleLocalIndex other) const
   {
      return fClusterId == other.fClusterId && fIndexInCluster == other.fIndexInCluster;
   }

   bool operator!=(RNTupleLocalIndex other) const { return !(*this == other); }

   ROOT::DescriptorId_t GetClusterId() const { return fClusterId; }
   ROOT::NTupleSize_t GetIndexInCluster() const { return fIndexInCluster; }
};

/// RNTupleLocator payload that is common for object stores using 64bit location information.
/// This might not contain the full location of the content. In particular, for page locators this information may be
/// used in conjunction with the cluster and column ID.
class RNTupleLocatorObject64 {
private:
   std::uint64_t fLocation = 0;

public:
   RNTupleLocatorObject64() = default;
   explicit RNTupleLocatorObject64(std::uint64_t location) : fLocation(location) {}
   bool operator==(const RNTupleLocatorObject64 &other) const { return fLocation == other.fLocation; }
   std::uint64_t GetLocation() const { return fLocation; }
};

/// Generic information about the physical location of data. Values depend on the concrete storage type.  E.g.,
/// for a local file `fPosition` might be a 64bit file offset. Referenced objects on storage can be compressed
/// and therefore we need to store their actual size.
class RNTupleLocator {
public:
   /// Values for the _Type_ field in non-disk locators.  Serializable types must have the MSb == 0; see
   /// `doc/BinaryFormatSpecification.md` for details
   enum ELocatorType : std::uint8_t {
      // The kTypeFile locator may translate to an on-disk standard locator (type 0x00) or a large locator (type 0x01),
      // if the size of the referenced data block is >2GB
      kTypeFile = 0x00,
      kTypeDAOS = 0x02,

      kLastSerializableType = 0x7f,
      kTypePageZero = kLastSerializableType + 1,
      kTypeUnknown,
   };

private:
   std::uint64_t fNBytesOnStorage = 0;
   /// Simple on-disk locators consisting of a 64-bit offset use variant type `uint64_t`; extended locators have
   /// `fPosition.index()` > 0
   std::variant<std::uint64_t, RNTupleLocatorObject64> fPosition{};
   /// For non-disk locators, the value for the _Type_ field. This makes it possible to have different type values even
   /// if the payload structure is identical.
   ELocatorType fType = kTypeFile;
   /// Reserved for use by concrete storage backends
   std::uint8_t fReserved = 0;

public:
   RNTupleLocator() = default;

   bool operator==(const RNTupleLocator &other) const
   {
      return fPosition == other.fPosition && fNBytesOnStorage == other.fNBytesOnStorage && fType == other.fType;
   }

   std::uint64_t GetNBytesOnStorage() const { return fNBytesOnStorage; }
   ELocatorType GetType() const { return fType; }
   std::uint8_t GetReserved() const { return fReserved; }

   void SetNBytesOnStorage(std::uint64_t nBytesOnStorage) { fNBytesOnStorage = nBytesOnStorage; }
   void SetType(ELocatorType type) { fType = type; }
   void SetReserved(std::uint8_t reserved) { fReserved = reserved; }

   template <typename T>
   T GetPosition() const
   {
      return std::get<T>(fPosition);
   }

   template <typename T>
   void SetPosition(T position)
   {
      fPosition = position;
   }
};

namespace Internal {

/// The in-memory representation of a 32bit or 64bit on-disk index column. Wraps the integer in a
/// named type so that templates can distinguish between integer data columns and index columns.
class RColumnIndex {
public:
   using ValueType = std::uint64_t;

private:
   ValueType fValue = 0;

public:
   RColumnIndex() = default;
   explicit constexpr RColumnIndex(ValueType value) : fValue(value) {}
   RColumnIndex &operator=(const ValueType value)
   {
      fValue = value;
      return *this;
   }
   RColumnIndex &operator+=(const ValueType value)
   {
      fValue += value;
      return *this;
   }
   RColumnIndex operator++(int)
   {
      auto result = *this;
      fValue++;
      return result;
   }
   operator ValueType() const { return fValue; }
};

/// Holds the index and the tag of a kSwitch column
class RColumnSwitch {
private:
   ROOT::NTupleSize_t fIndex;
   std::uint32_t fTag = 0;

public:
   RColumnSwitch() = default;
   RColumnSwitch(ROOT::NTupleSize_t index, std::uint32_t tag) : fIndex(index), fTag(tag) {}
   ROOT::NTupleSize_t GetIndex() const { return fIndex; }
   std::uint32_t GetTag() const { return fTag; }
};

inline constexpr ENTupleColumnType kTestFutureColumnType =
   static_cast<ENTupleColumnType>(std::numeric_limits<std::underlying_type_t<ENTupleColumnType>>::max() - 1);

inline constexpr ROOT::ENTupleStructure kTestFutureFieldStructure =
   static_cast<ROOT::ENTupleStructure>(std::numeric_limits<std::underlying_type_t<ROOT::ENTupleStructure>>::max() - 1);

inline constexpr RNTupleLocator::ELocatorType kTestLocatorType = static_cast<RNTupleLocator::ELocatorType>(0x7e);
static_assert(kTestLocatorType < RNTupleLocator::ELocatorType::kLastSerializableType);

} // namespace Internal
} // namespace ROOT

#endif
