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

#include <ROOT/RLogger.hxx>

namespace ROOT {
namespace Experimental {

class RLogChannel;
/// Log channel for RNTuple diagnostics.
RLogChannel &NTupleLog();

struct RNTuple;

namespace Internal {

void PrintRNTuple(const RNTuple& ntuple, std::ostream& output);

} // namespace Internal

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
/// Wrap the 32bit integer in a struct in order to avoid template specialization clash with std::uint32_t
struct RClusterSize {
   using ValueType = std::uint32_t;

   RClusterSize() : fValue(0) {}
   explicit constexpr RClusterSize(ValueType value) : fValue(value) {}
   RClusterSize& operator =(const ValueType value) { fValue = value; return *this; }
   RClusterSize& operator +=(const ValueType value) { fValue += value; return *this; }
   RClusterSize operator++(int) { auto result = *this; fValue++; return result; }
   operator ValueType() const { return fValue; }

   ValueType fValue;
};
using ClusterSize_t = RClusterSize;
constexpr ClusterSize_t kInvalidClusterIndex(std::uint32_t(-1));

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

/// Generic information about the physical location of data. Values depend on the concrete storage type.  E.g.,
/// for a local file fUrl might be unsused and fPosition might be a file offset. Objects on storage can be compressed
/// and therefore we need to store their actual size.
/// TODO(jblomer): should move the RNTUpleDescriptor and should be an std::variant
struct RNTupleLocator {
   std::int64_t fPosition = 0;
   std::uint32_t fBytesOnStorage = 0;
   std::string fUrl;

   bool operator==(const RNTupleLocator &other) const {
      return fPosition == other.fPosition && fBytesOnStorage == other.fBytesOnStorage && fUrl == other.fUrl;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
