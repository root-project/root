/// \file ROOT/RNTupleUtil.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleUtil
#define ROOT7_RNTupleUtil

#include <cstdint>

#include <string>
#include <variant>

#include <ROOT/RLogger.hxx>

namespace ROOT {
namespace Experimental {

class RLogChannel;
/// Log channel for RNTuple diagnostics.
RLogChannel &NTupleLog();

/**
 * The fields in the ntuple model tree can carry different structural information about the type system.
 * Leaf fields contain just data, collection fields resolve to offset columns, record fields have no
 * materialization on the primitive column layer.
 */
enum ENTupleStructure {
   kLeaf,
   kCollection,
   kRecord,
   kVariant,
   kReference, // unimplemented so far
   kInvalid,
};

/// Integer type long enough to hold the maximum number of entries in a column
using NTupleSize_t = std::uint64_t;
constexpr NTupleSize_t kInvalidNTupleIndex = std::uint64_t(-1);
/// Wrap the integer in a struct in order to avoid template specialization clash with std::uint32_t
struct RClusterSize {
   using ValueType = std::uint64_t;

   RClusterSize() : fValue(0) {}
   explicit constexpr RClusterSize(ValueType value) : fValue(value) {}
   RClusterSize& operator =(const ValueType value) { fValue = value; return *this; }
   RClusterSize& operator +=(const ValueType value) { fValue += value; return *this; }
   RClusterSize operator++(int) { auto result = *this; fValue++; return result; }
   operator ValueType() const { return fValue; }

   ValueType fValue;
};
using ClusterSize_t = RClusterSize;
constexpr ClusterSize_t kInvalidClusterIndex(std::uint64_t(-1));

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

/// Holds the index and the tag of a kSwitch column
class RColumnSwitch {
private:
   ClusterSize_t fIndex;
   std::uint32_t fTag = 0;

public:
   RColumnSwitch() = default;
   RColumnSwitch(ClusterSize_t index, std::uint32_t tag) : fIndex(index), fTag(tag) { }
   ClusterSize_t GetIndex() const { return fIndex; }
   std::uint32_t GetTag() const { return fTag; }
};

/// Uniquely identifies a physical column within the scope of the current process, used to tag pages
using ColumnId_t = std::int64_t;
constexpr ColumnId_t kInvalidColumnId = -1;

/// Distriniguishes elements of the same type within a descriptor, e.g. different fields
using DescriptorId_t = std::uint64_t;
constexpr DescriptorId_t kInvalidDescriptorId = std::uint64_t(-1);

/// Addresses a column element or field item relative to a particular cluster, instead of a global NTupleSize_t index
class RClusterIndex {
private:
   DescriptorId_t fClusterId = kInvalidDescriptorId;
   ClusterSize_t::ValueType fIndex = kInvalidClusterIndex;
public:
   RClusterIndex() = default;
   RClusterIndex(const RClusterIndex &other) = default;
   RClusterIndex &operator =(const RClusterIndex &other) = default;
   constexpr RClusterIndex(DescriptorId_t clusterId, ClusterSize_t::ValueType index)
      : fClusterId(clusterId), fIndex(index) {}

   RClusterIndex  operator+(ClusterSize_t::ValueType off) const { return RClusterIndex(fClusterId, fIndex + off); }
   RClusterIndex  operator-(ClusterSize_t::ValueType off) const { return RClusterIndex(fClusterId, fIndex - off); }
   RClusterIndex  operator++(int) /* postfix */        { auto r = *this; fIndex++; return r; }
   RClusterIndex& operator++()    /* prefix */         { ++fIndex; return *this; }
   bool operator==(const RClusterIndex &other) const {
      return fClusterId == other.fClusterId && fIndex == other.fIndex;
   }
   bool operator!=(const RClusterIndex &other) const { return !(*this == other); }

   DescriptorId_t GetClusterId() const { return fClusterId; }
   ClusterSize_t::ValueType GetIndex() const { return fIndex; }
};

/// RNTupleLocator payload that is common for object stores using 64bit location information.
/// This might not contain the full location of the content. In particular, for page locators this information may be
/// used in conjunction with the cluster and column ID.
struct RNTupleLocatorObject64 {
   std::uint64_t fLocation = 0;
   bool operator==(const RNTupleLocatorObject64 &other) const { return fLocation == other.fLocation; }
};

/// Generic information about the physical location of data. Values depend on the concrete storage type.  E.g.,
/// for a local file `fPosition` might be a 64bit file offset. Referenced objects on storage can be compressed
/// and therefore we need to store their actual size.
/// TODO(jblomer): consider moving this to `RNTupleDescriptor`
struct RNTupleLocator {
   /// Values for the _Type_ field in non-disk locators.  Serializable types must have the MSb == 0; see
   /// `doc/specifications.md` for details
   enum ELocatorType : std::uint8_t {
      kTypeFile = 0x00,
      kTypeURI = 0x01,
      kTypeDAOS = 0x02,

      kLastSerializableType = 0x7f,
      kTypePageZero = kLastSerializableType + 1,
   };

   /// Simple on-disk locators consisting of a 64-bit offset use variant type `uint64_t`; extended locators have
   /// `fPosition.index()` > 0
   std::variant<std::uint64_t, std::string, RNTupleLocatorObject64> fPosition;
   std::uint32_t fBytesOnStorage = 0;
   /// For non-disk locators, the value for the _Type_ field. This makes it possible to have different type values even
   /// if the payload structure is identical.
   ELocatorType fType = kTypeFile;
   /// Reserved for use by concrete storage backends
   std::uint8_t fReserved = 0;

   bool operator==(const RNTupleLocator &other) const {
      return fPosition == other.fPosition && fBytesOnStorage == other.fBytesOnStorage && fType == other.fType;
   }
   template <typename T>
   const T &GetPosition() const
   {
      return std::get<T>(fPosition);
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
