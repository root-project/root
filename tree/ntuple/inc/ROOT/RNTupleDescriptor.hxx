/// \file ROOT/RNTupleDescriptor.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Javier Lopez-Gomez <javier.lopez.gomez@cern.ch>
/// \date 2018-07-19

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleDescriptor
#define ROOT_RNTupleDescriptor

#include <ROOT/RCreateFieldOptions.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RSpan.hxx>

#include <TError.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <vector>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace ROOT {

class RFieldBase;
class RNTupleModel;

namespace Internal {
class RColumnElementBase;
}

class RNTupleDescriptor;

namespace Internal {
class RColumnDescriptorBuilder;
class RClusterDescriptorBuilder;
class RClusterGroupDescriptorBuilder;
class RExtraTypeInfoDescriptorBuilder;
class RFieldDescriptorBuilder;
class RNTupleDescriptorBuilder;

RNTupleDescriptor CloneDescriptorSchema(const RNTupleDescriptor &desc);
} // namespace Internal

// clang-format off
/**
\class ROOT::RFieldDescriptor
\ingroup NTuple
\brief Metadata stored for every field of an RNTuple
*/
// clang-format on
class RFieldDescriptor final {
   friend class Internal::RNTupleDescriptorBuilder;
   friend class Internal::RFieldDescriptorBuilder;

private:
   ROOT::DescriptorId_t fFieldId = ROOT::kInvalidDescriptorId;
   /// The version of the C++-type-to-column translation mechanics
   std::uint32_t fFieldVersion = 0;
   /// The version of the C++ type itself
   std::uint32_t fTypeVersion = 0;
   /// The leaf name, not including parent fields
   std::string fFieldName;
   /// Free text set by the user
   std::string fFieldDescription;
   /// The C++ type that was used when writing the field
   std::string fTypeName;
   /// A typedef or using directive that resolved to the type name during field creation
   std::string fTypeAlias;
   /// The number of elements per entry for fixed-size arrays
   std::uint64_t fNRepetitions = 0;
   /// The structural information carried by this field in the data model tree
   ROOT::ENTupleStructure fStructure = ROOT::ENTupleStructure::kInvalid;
   /// Establishes sub field relationships, such as classes and collections
   ROOT::DescriptorId_t fParentId = ROOT::kInvalidDescriptorId;
   /// For projected fields, the source field ID
   ROOT::DescriptorId_t fProjectionSourceId = ROOT::kInvalidDescriptorId;
   /// The pointers in the other direction from parent to children. They are serialized, too, to keep the
   /// order of sub fields.
   std::vector<ROOT::DescriptorId_t> fLinkIds;
   /// The number of columns in the column representations of the field. The column cardinality helps to navigate the
   /// list of logical column ids. For example, the second column of the third column representation is
   /// fLogicalColumnIds[2 * fColumnCardinality + 1]
   std::uint32_t fColumnCardinality = 0;
   /// The ordered list of columns attached to this field: first by representation index then by column index.
   std::vector<ROOT::DescriptorId_t> fLogicalColumnIds;
   /// For custom classes, we store the ROOT TClass reported checksum to facilitate the use of I/O rules that
   /// identify types by their checksum
   std::optional<std::uint32_t> fTypeChecksum;

public:
   RFieldDescriptor() = default;
   RFieldDescriptor(const RFieldDescriptor &other) = delete;
   RFieldDescriptor &operator=(const RFieldDescriptor &other) = delete;
   RFieldDescriptor(RFieldDescriptor &&other) = default;
   RFieldDescriptor &operator=(RFieldDescriptor &&other) = default;

   bool operator==(const RFieldDescriptor &other) const;
   /// Get a copy of the descriptor
   RFieldDescriptor Clone() const;

   /// In general, we create a field simply from the C++ type name. For untyped fields, however, we potentially need
   /// access to sub fields, which is provided by the RNTupleDescriptor argument.
   std::unique_ptr<ROOT::RFieldBase>
   CreateField(const RNTupleDescriptor &ntplDesc, const ROOT::RCreateFieldOptions &options = {}) const;

   ROOT::DescriptorId_t GetId() const { return fFieldId; }
   std::uint32_t GetFieldVersion() const { return fFieldVersion; }
   std::uint32_t GetTypeVersion() const { return fTypeVersion; }
   const std::string &GetFieldName() const { return fFieldName; }
   const std::string &GetFieldDescription() const { return fFieldDescription; }
   const std::string &GetTypeName() const { return fTypeName; }
   const std::string &GetTypeAlias() const { return fTypeAlias; }
   std::uint64_t GetNRepetitions() const { return fNRepetitions; }
   ROOT::ENTupleStructure GetStructure() const { return fStructure; }
   ROOT::DescriptorId_t GetParentId() const { return fParentId; }
   ROOT::DescriptorId_t GetProjectionSourceId() const { return fProjectionSourceId; }
   const std::vector<ROOT::DescriptorId_t> &GetLinkIds() const { return fLinkIds; }
   const std::vector<ROOT::DescriptorId_t> &GetLogicalColumnIds() const { return fLogicalColumnIds; }
   std::uint32_t GetColumnCardinality() const { return fColumnCardinality; }
   std::optional<std::uint32_t> GetTypeChecksum() const { return fTypeChecksum; }
   bool IsProjectedField() const { return fProjectionSourceId != ROOT::kInvalidDescriptorId; }
   /// Tells if the field describes a user-defined class rather than a fundamental type, a collection, or one of the
   /// natively supported stdlib classes.
   /// The dictionary does not need to be available for this method.
   bool IsCustomClass() const;
};

// clang-format off
/**
\class ROOT::RColumnDescriptor
\ingroup NTuple
\brief Metadata stored for every column of an RNTuple
*/
// clang-format on
class RColumnDescriptor final {
   friend class Internal::RColumnDescriptorBuilder;
   friend class Internal::RNTupleDescriptorBuilder;

public:
   struct RValueRange {
      double fMin = 0, fMax = 0;

      RValueRange() = default;
      RValueRange(double min, double max) : fMin(min), fMax(max) {}
      RValueRange(std::pair<double, double> range) : fMin(range.first), fMax(range.second) {}

      bool operator==(RValueRange other) const { return fMin == other.fMin && fMax == other.fMax; }
      bool operator!=(RValueRange other) const { return !(*this == other); }
   };

private:
   /// The actual column identifier, which is the link to the corresponding field
   ROOT::DescriptorId_t fLogicalColumnId = ROOT::kInvalidDescriptorId;
   /// Usually identical to the logical column ID, except for alias columns where it references the shadowed column
   ROOT::DescriptorId_t fPhysicalColumnId = ROOT::kInvalidDescriptorId;
   /// Every column belongs to one and only one field
   ROOT::DescriptorId_t fFieldId = ROOT::kInvalidDescriptorId;
   /// The absolute value specifies the index for the first stored element for this column.
   /// For deferred columns the absolute value is larger than zero.
   /// Negative values specify a suppressed and deferred column.
   std::int64_t fFirstElementIndex = 0U;
   /// A field can be serialized into several columns, which are numbered from zero to $n$
   std::uint32_t fIndex = 0;
   /// A field may use multiple column representations, which are numbered from zero to $m$.
   /// Every representation has the same number of columns.
   std::uint16_t fRepresentationIndex = 0;
   /// The size in bits of elements of this column. Most columns have the size fixed by their type
   /// but low-precision float columns have variable bit widths.
   std::uint16_t fBitsOnStorage = 0;
   /// The on-disk column type
   ROOT::ENTupleColumnType fType = ROOT::ENTupleColumnType::kUnknown;
   /// Optional value range (used e.g. by quantized real fields)
   std::optional<RValueRange> fValueRange;

public:
   RColumnDescriptor() = default;
   RColumnDescriptor(const RColumnDescriptor &other) = delete;
   RColumnDescriptor &operator=(const RColumnDescriptor &other) = delete;
   RColumnDescriptor(RColumnDescriptor &&other) = default;
   RColumnDescriptor &operator=(RColumnDescriptor &&other) = default;

   bool operator==(const RColumnDescriptor &other) const;
   /// Get a copy of the descriptor
   RColumnDescriptor Clone() const;

   ROOT::DescriptorId_t GetLogicalId() const { return fLogicalColumnId; }
   ROOT::DescriptorId_t GetPhysicalId() const { return fPhysicalColumnId; }
   ROOT::DescriptorId_t GetFieldId() const { return fFieldId; }
   std::uint32_t GetIndex() const { return fIndex; }
   std::uint16_t GetRepresentationIndex() const { return fRepresentationIndex; }
   std::uint64_t GetFirstElementIndex() const { return std::abs(fFirstElementIndex); }
   std::uint16_t GetBitsOnStorage() const { return fBitsOnStorage; }
   ROOT::ENTupleColumnType GetType() const { return fType; }
   std::optional<RValueRange> GetValueRange() const { return fValueRange; }
   bool IsAliasColumn() const { return fPhysicalColumnId != fLogicalColumnId; }
   bool IsDeferredColumn() const { return fFirstElementIndex != 0; }
   bool IsSuppressedDeferredColumn() const { return fFirstElementIndex < 0; }
};

// clang-format off
/**
\class ROOT::RClusterDescriptor
\ingroup NTuple
\brief Metadata for RNTuple clusters

The cluster descriptor is built in two phases.  In a first phase, the descriptor has only an ID.
In a second phase, the event range, column group, page locations and column ranges are added.
Both phases are populated by the RClusterDescriptorBuilder.
Clusters span across all available columns in the RNTuple.
*/
// clang-format on
class RClusterDescriptor final {
   friend class Internal::RClusterDescriptorBuilder;

public:
   // clang-format off
   /**
   \class ROOT::RClusterDescriptor::RColumnRange
   \ingroup NTuple
   \brief The window of element indexes of a particular column in a particular cluster
   */
   // clang-format on
   class RColumnRange final {
      ROOT::DescriptorId_t fPhysicalColumnId = ROOT::kInvalidDescriptorId;
      /// The global index of the first column element in the cluster
      ROOT::NTupleSize_t fFirstElementIndex = ROOT::kInvalidNTupleIndex;
      /// The number of column elements in the cluster
      ROOT::NTupleSize_t fNElements = ROOT::kInvalidNTupleIndex;
      /// The usual format for ROOT compression settings (see Compression.h).
      /// The pages of a particular column in a particular cluster are all compressed with the same settings.
      /// If unset, the compression settings are undefined (deferred columns, suppressed columns).
      std::optional<std::uint32_t> fCompressionSettings;
      /// Suppressed columns have an empty page range and unknown compression settings.
      /// Their element index range, however, is aligned with the corresponding column of the
      /// primary column representation (see Section "Suppressed Columns" in the specification)
      bool fIsSuppressed = false;

      // TODO(jblomer): we perhaps want to store summary information, such as average, min/max, etc.
      // Should this be done on the field level?

   public:
      RColumnRange() = default;

      RColumnRange(ROOT::DescriptorId_t physicalColumnId, ROOT::NTupleSize_t firstElementIndex,
                   ROOT::NTupleSize_t nElements, std::optional<std::uint32_t> compressionSettings,
                   bool suppressed = false)
         : fPhysicalColumnId(physicalColumnId),
           fFirstElementIndex(firstElementIndex),
           fNElements(nElements),
           fCompressionSettings(compressionSettings),
           fIsSuppressed(suppressed)
      {
      }

      ROOT::DescriptorId_t GetPhysicalColumnId() const { return fPhysicalColumnId; }
      void SetPhysicalColumnId(ROOT::DescriptorId_t id) { fPhysicalColumnId = id; }

      ROOT::NTupleSize_t GetFirstElementIndex() const { return fFirstElementIndex; }
      void SetFirstElementIndex(ROOT::NTupleSize_t idx) { fFirstElementIndex = idx; }
      void IncrementFirstElementIndex(ROOT::NTupleSize_t by) { fFirstElementIndex += by; }

      ROOT::NTupleSize_t GetNElements() const { return fNElements; }
      void SetNElements(ROOT::NTupleSize_t n) { fNElements = n; }
      void IncrementNElements(ROOT::NTupleSize_t by) { fNElements += by; }

      std::optional<std::uint32_t> GetCompressionSettings() const { return fCompressionSettings; }
      void SetCompressionSettings(std::optional<std::uint32_t> comp) { fCompressionSettings = comp; }

      bool IsSuppressed() const { return fIsSuppressed; }
      void SetIsSuppressed(bool suppressed) { fIsSuppressed = suppressed; }

      bool operator==(const RColumnRange &other) const
      {
         return fPhysicalColumnId == other.fPhysicalColumnId && fFirstElementIndex == other.fFirstElementIndex &&
                fNElements == other.fNElements && fCompressionSettings == other.fCompressionSettings &&
                fIsSuppressed == other.fIsSuppressed;
      }

      bool Contains(ROOT::NTupleSize_t index) const
      {
         return (fFirstElementIndex <= index && (fFirstElementIndex + fNElements) > index);
      }
   };

   // clang-format off
   /**
   \class ROOT::RClusterDescriptor::RPageInfo
   \ingroup NTuple
   \brief Information about a single page in the context of a cluster's page range.
   */
   // clang-format on
   // NOTE: We do not need to store the element size / uncompressed page size because we know to which column
   // the page belongs
   struct RPageInfo {
   private:
      /// The sum of the elements of all the pages must match the corresponding `fNElements` field in `fColumnRanges`
      std::uint32_t fNElements = std::uint32_t(-1);
      /// The meaning of `fLocator` depends on the storage backend.
      RNTupleLocator fLocator;
      /// If true, the 8 bytes following the serialized page are an xxhash of the on-disk page data
      bool fHasChecksum = false;

   public:
      RPageInfo() = default;
      RPageInfo(std::uint32_t nElements, const RNTupleLocator &locator, bool hasChecksum)
         : fNElements(nElements), fLocator(locator), fHasChecksum(hasChecksum)
      {
      }

      bool operator==(const RPageInfo &other) const
      {
         return fNElements == other.fNElements && fLocator == other.fLocator;
      }

      std::uint32_t GetNElements() const { return fNElements; }
      void SetNElements(std::uint32_t n) { fNElements = n; }

      const RNTupleLocator &GetLocator() const { return fLocator; }
      RNTupleLocator &GetLocator() { return fLocator; }
      void SetLocator(const RNTupleLocator &locator) { fLocator = locator; }

      bool HasChecksum() const { return fHasChecksum; }
      void SetHasChecksum(bool hasChecksum) { fHasChecksum = hasChecksum; }
   };

   // clang-format off
   /**
   \class ROOT::RClusterDescriptor::RPageInfoExtended
   \ingroup NTuple
   \brief Additional information about a page in an in-memory RPageRange.

   Used by RPageRange::Find() to return information relative to the RPageRange.  This information is not stored on disk
   and we don't need to keep it in memory because it can be easily recomputed.
   */
   // clang-format on
   struct RPageInfoExtended final : RPageInfo {
   private:
      /// Index (in cluster) of the first element in page.
      ROOT::NTupleSize_t fFirstElementIndex = 0;
      /// Page number in the corresponding RPageRange.
      ROOT::NTupleSize_t fPageNumber = 0;

   public:
      RPageInfoExtended() = default;
      RPageInfoExtended(const RPageInfo &pageInfo, ROOT::NTupleSize_t firstElementIndex, ROOT::NTupleSize_t pageNumber)
         : RPageInfo(pageInfo), fFirstElementIndex(firstElementIndex), fPageNumber(pageNumber)
      {
      }

      ROOT::NTupleSize_t GetFirstElementIndex() const { return fFirstElementIndex; }
      void SetFirstElementIndex(ROOT::NTupleSize_t firstInPage) { fFirstElementIndex = firstInPage; }

      ROOT::NTupleSize_t GetPageNumber() const { return fPageNumber; }
      void SetPageNumber(ROOT::NTupleSize_t pageNumber) { fPageNumber = pageNumber; }
   };

   // clang-format off
   /**
   \class ROOT::RClusterDescriptor::RPageRange
   \ingroup NTuple
   \brief Records the partition of data into pages for a particular column in a particular cluster
   */
   // clang-format on
   class RPageRange final {
      friend class Internal::RClusterDescriptorBuilder;

   private:
      /// \brief Extend this RPageRange to fit the given RColumnRange.
      ///
      /// To do so, prepend as many synthetic RPageInfos as needed to cover the range in `columnRange`.
      /// RPageInfos are constructed to contain as many elements of type `element` given a page size
      /// limit of `pageSize` (in bytes); the locator for the referenced pages is `kTypePageZero`.
      /// This function is used to make up RPageRanges for clusters that contain deferred columns.
      /// \return The number of column elements covered by the synthesized RPageInfos
      std::size_t ExtendToFitColumnRange(const RColumnRange &columnRange,
                                         const ROOT::Internal::RColumnElementBase &element, std::size_t pageSize);

      /// Has the same length than fPageInfos and stores the sum of the number of elements of all the pages
      /// up to and including a given index. Used for binary search in Find().
      std::vector<ROOT::NTupleSize_t> fCumulativeNElements;

      ROOT::DescriptorId_t fPhysicalColumnId = ROOT::kInvalidDescriptorId;
      std::vector<RPageInfo> fPageInfos;

   public:
      RPageRange() = default;
      RPageRange(const RPageRange &other) = delete;
      RPageRange &operator=(const RPageRange &other) = delete;
      RPageRange(RPageRange &&other) = default;
      RPageRange &operator=(RPageRange &&other) = default;

      RPageRange Clone() const
      {
         RPageRange clone;
         clone.fPhysicalColumnId = fPhysicalColumnId;
         clone.fPageInfos = fPageInfos;
         clone.fCumulativeNElements = fCumulativeNElements;
         return clone;
      }

      /// Find the page in the RPageRange that contains the given element. The element must exist.
      RPageInfoExtended Find(ROOT::NTupleSize_t idxInCluster) const;

      ROOT::DescriptorId_t GetPhysicalColumnId() const { return fPhysicalColumnId; }
      void SetPhysicalColumnId(ROOT::DescriptorId_t id) { fPhysicalColumnId = id; }

      const std::vector<RPageInfo> &GetPageInfos() const { return fPageInfos; }
      std::vector<RPageInfo> &GetPageInfos() { return fPageInfos; }

      bool operator==(const RPageRange &other) const
      {
         return fPhysicalColumnId == other.fPhysicalColumnId && fPageInfos == other.fPageInfos;
      }
   };

private:
   ROOT::DescriptorId_t fClusterId = ROOT::kInvalidDescriptorId;
   /// Clusters can be swapped by adjusting the entry offsets of the cluster and all ranges
   ROOT::NTupleSize_t fFirstEntryIndex = ROOT::kInvalidNTupleIndex;
   ROOT::NTupleSize_t fNEntries = ROOT::kInvalidNTupleIndex;

   std::unordered_map<ROOT::DescriptorId_t, RColumnRange> fColumnRanges;
   std::unordered_map<ROOT::DescriptorId_t, RPageRange> fPageRanges;

public:
   class RColumnRangeIterable;

   RClusterDescriptor() = default;
   RClusterDescriptor(const RClusterDescriptor &other) = delete;
   RClusterDescriptor &operator=(const RClusterDescriptor &other) = delete;
   RClusterDescriptor(RClusterDescriptor &&other) = default;
   RClusterDescriptor &operator=(RClusterDescriptor &&other) = default;

   RClusterDescriptor Clone() const;

   bool operator==(const RClusterDescriptor &other) const;

   ROOT::DescriptorId_t GetId() const { return fClusterId; }
   ROOT::NTupleSize_t GetFirstEntryIndex() const { return fFirstEntryIndex; }
   ROOT::NTupleSize_t GetNEntries() const { return fNEntries; }
   const RColumnRange &GetColumnRange(ROOT::DescriptorId_t physicalId) const { return fColumnRanges.at(physicalId); }
   const RPageRange &GetPageRange(ROOT::DescriptorId_t physicalId) const { return fPageRanges.at(physicalId); }
   /// Returns an iterator over pairs { columnId, columnRange }. The iteration order is unspecified.
   RColumnRangeIterable GetColumnRangeIterable() const;
   bool ContainsColumn(ROOT::DescriptorId_t physicalId) const
   {
      return fColumnRanges.find(physicalId) != fColumnRanges.end();
   }
   std::uint64_t GetNBytesOnStorage() const;
};

class RClusterDescriptor::RColumnRangeIterable final {
private:
   const RClusterDescriptor &fDesc;

public:
   class RIterator final {
   private:
      using Iter_t = std::unordered_map<ROOT::DescriptorId_t, RColumnRange>::const_iterator;
      /// The wrapped map iterator
      Iter_t fIter;

   public:
      using iterator_category = std::forward_iterator_tag;
      using iterator = RIterator;
      using value_type = RColumnRange;
      using difference_type = std::ptrdiff_t;
      using pointer = const RColumnRange *;
      using reference = const RColumnRange &;

      RIterator(Iter_t iter) : fIter(iter) {}
      iterator operator++()
      {
         ++fIter;
         return *this;
      }
      reference operator*() { return fIter->second; }
      pointer operator->() { return &fIter->second; }
      bool operator!=(const iterator &rh) const { return fIter != rh.fIter; }
      bool operator==(const iterator &rh) const { return fIter == rh.fIter; }
   };

   explicit RColumnRangeIterable(const RClusterDescriptor &desc) : fDesc(desc) {}

   RIterator begin() { return RIterator{fDesc.fColumnRanges.cbegin()}; }
   RIterator end() { return fDesc.fColumnRanges.cend(); }
   size_t size() { return fDesc.fColumnRanges.size(); }
};

// clang-format off
/**
\class ROOT::RClusterGroupDescriptor
\ingroup NTuple
\brief Clusters are bundled in cluster groups.

Very large RNTuples can contain multiple cluster groups to organize cluster metadata.
Every RNTuple has at least one cluster group.  The clusters in a cluster group are ordered
corresponding to their first entry number.
*/
// clang-format on
class RClusterGroupDescriptor final {
   friend class Internal::RClusterGroupDescriptorBuilder;

private:
   ROOT::DescriptorId_t fClusterGroupId = ROOT::kInvalidDescriptorId;
   /// The cluster IDs can be empty if the corresponding page list is not loaded.
   /// Otherwise, cluster ids are sorted by first entry number.
   std::vector<ROOT::DescriptorId_t> fClusterIds;
   /// The page list that corresponds to the cluster group
   RNTupleLocator fPageListLocator;
   /// Uncompressed size of the page list
   std::uint64_t fPageListLength = 0;
   /// The minimum first entry number of the clusters in the cluster group
   std::uint64_t fMinEntry = 0;
   /// Number of entries that are (partially for sharded clusters) covered by this cluster group.
   std::uint64_t fEntrySpan = 0;
   /// Number of clusters is always known even if the cluster IDs are not (yet) populated
   std::uint32_t fNClusters = 0;

public:
   RClusterGroupDescriptor() = default;
   RClusterGroupDescriptor(const RClusterGroupDescriptor &other) = delete;
   RClusterGroupDescriptor &operator=(const RClusterGroupDescriptor &other) = delete;
   RClusterGroupDescriptor(RClusterGroupDescriptor &&other) = default;
   RClusterGroupDescriptor &operator=(RClusterGroupDescriptor &&other) = default;

   RClusterGroupDescriptor Clone() const;
   /// Creates a clone without the cluster IDs
   RClusterGroupDescriptor CloneSummary() const;

   bool operator==(const RClusterGroupDescriptor &other) const;

   ROOT::DescriptorId_t GetId() const { return fClusterGroupId; }
   std::uint32_t GetNClusters() const { return fNClusters; }
   RNTupleLocator GetPageListLocator() const { return fPageListLocator; }
   std::uint64_t GetPageListLength() const { return fPageListLength; }
   const std::vector<ROOT::DescriptorId_t> &GetClusterIds() const { return fClusterIds; }
   std::uint64_t GetMinEntry() const { return fMinEntry; }
   std::uint64_t GetEntrySpan() const { return fEntrySpan; }
   /// A cluster group is loaded in two stages. Stage one loads only the summary information.
   /// Stage two loads the list of cluster IDs.
   bool HasClusterDetails() const { return !fClusterIds.empty(); }
};

/// Used in RExtraTypeInfoDescriptor
enum class EExtraTypeInfoIds {
   kInvalid,
   kStreamerInfo
};

// clang-format off
/**
\class ROOT::RExtraTypeInfoDescriptor
\ingroup NTuple
\brief Field specific extra type information from the header / extenstion header

Currently only used by streamer fields to store RNTuple-wide list of streamer info records.
*/
// clang-format on
class RExtraTypeInfoDescriptor final {
   friend class Internal::RExtraTypeInfoDescriptorBuilder;

private:
   /// Specifies the meaning of the extra information
   EExtraTypeInfoIds fContentId = EExtraTypeInfoIds::kInvalid;
   /// Type version the extra type information is bound to
   std::uint32_t fTypeVersion = 0;
   /// The type name the extra information refers to; empty for RNTuple-wide extra information
   std::string fTypeName;
   /// The content format depends on the content ID and may be binary
   std::string fContent;

public:
   RExtraTypeInfoDescriptor() = default;
   RExtraTypeInfoDescriptor(const RExtraTypeInfoDescriptor &other) = delete;
   RExtraTypeInfoDescriptor &operator=(const RExtraTypeInfoDescriptor &other) = delete;
   RExtraTypeInfoDescriptor(RExtraTypeInfoDescriptor &&other) = default;
   RExtraTypeInfoDescriptor &operator=(RExtraTypeInfoDescriptor &&other) = default;

   bool operator==(const RExtraTypeInfoDescriptor &other) const;

   RExtraTypeInfoDescriptor Clone() const;

   EExtraTypeInfoIds GetContentId() const { return fContentId; }
   std::uint32_t GetTypeVersion() const { return fTypeVersion; }
   const std::string &GetTypeName() const { return fTypeName; }
   const std::string &GetContent() const { return fContent; }
};

// clang-format off
/**
\class ROOT::RNTupleDescriptor
\ingroup NTuple
\brief The on-storage metadata of an RNTuple

Represents the on-disk (on storage) information about an RNTuple. The metadata consists of a header, a footer, and
potentially multiple page lists.
The header carries the RNTuple schema, i.e. the fields and the associated columns and their relationships.
The footer carries information about one or several cluster groups and links to their page lists.
For every cluster group, a page list envelope stores cluster summaries and page locations.
For every cluster, it stores for every column the range of element indexes as well as a list of pages and page
locations.

The descriptor provides machine-independent (de-)serialization of headers and footers, and it provides lookup routines
for RNTuple objects (pages, clusters, ...).  It is supposed to be usable by all RPageStorage implementations.

The serialization does not use standard ROOT streamers in order to not let it depend on libCore. The serialization uses
the concept of envelopes and frames: header, footer, and page list envelopes have a preamble with a type ID and length.
Substructures are serialized in frames and have a size and number of items (for list frames). This allows for forward
and backward compatibility when the metadata evolves.
*/
// clang-format on
class RNTupleDescriptor final {
   friend class Internal::RNTupleDescriptorBuilder;
   friend RNTupleDescriptor Internal::CloneDescriptorSchema(const RNTupleDescriptor &desc);

public:
   class RHeaderExtension;

private:
   /// The RNTuple name needs to be unique in a given storage location (file)
   std::string fName;
   /// Free text from the user
   std::string fDescription;

   ROOT::DescriptorId_t fFieldZeroId = ROOT::kInvalidDescriptorId; ///< Set by the descriptor builder

   std::uint64_t fNPhysicalColumns = 0; ///< Updated by the descriptor builder when columns are added

   std::set<unsigned int> fFeatureFlags;
   std::unordered_map<ROOT::DescriptorId_t, RFieldDescriptor> fFieldDescriptors;
   std::unordered_map<ROOT::DescriptorId_t, RColumnDescriptor> fColumnDescriptors;

   std::vector<RExtraTypeInfoDescriptor> fExtraTypeInfoDescriptors;
   std::unique_ptr<RHeaderExtension> fHeaderExtension;

   //// All fields above are part of the schema and are cloned when creating a new descriptor from a given one
   //// (see CloneSchema())

   std::uint64_t fOnDiskHeaderSize = 0;    ///< Set by the descriptor builder when deserialized
   std::uint64_t fOnDiskHeaderXxHash3 = 0; ///< Set by the descriptor builder when deserialized
   std::uint64_t fOnDiskFooterSize = 0; ///< Like fOnDiskHeaderSize, contains both cluster summaries and page locations

   std::uint64_t fNEntries = 0;  ///< Updated by the descriptor builder when the cluster groups are added
   std::uint64_t fNClusters = 0; ///< Updated by the descriptor builder when the cluster groups are added

   /// \brief The generation of the descriptor
   ///
   /// Once constructed by an RNTupleDescriptorBuilder, the descriptor is mostly immutable except for the set of
   /// active page locations.  During the lifetime of the descriptor, page location information for clusters
   /// can be added or removed.  When this happens, the generation should be increased, so that users of the
   /// descriptor know that the information changed.  The generation is increased, e.g., by the page source's
   /// exclusive lock guard around the descriptor.  It is used, e.g., by the descriptor cache in RNTupleReader.
   std::uint64_t fGeneration = 0;

   std::unordered_map<ROOT::DescriptorId_t, RClusterGroupDescriptor> fClusterGroupDescriptors;
   /// References cluster groups sorted by entry range and thus allows for binary search.
   /// Note that this list is empty during the descriptor building process and will only be
   /// created when the final descriptor is extracted from the builder.
   std::vector<ROOT::DescriptorId_t> fSortedClusterGroupIds;
   /// Potentially a subset of all the available clusters
   std::unordered_map<ROOT::DescriptorId_t, RClusterDescriptor> fClusterDescriptors;

   // We don't expose this publicly because when we add sharded clusters, this interface does not make sense anymore
   ROOT::DescriptorId_t FindClusterId(ROOT::NTupleSize_t entryIdx) const;

   /// Creates a descriptor containing only the schema information about this RNTuple, i.e. all the information needed
   /// to create a new RNTuple with the same schema as this one but not necessarily the same clustering. This is used
   /// when merging two RNTuples.
   RNTupleDescriptor CloneSchema() const;

public:
   static constexpr unsigned int kFeatureFlagTest = 137; // Bit reserved for forward-compatibility testing

   class RColumnDescriptorIterable;
   class RFieldDescriptorIterable;
   class RClusterGroupDescriptorIterable;
   class RClusterDescriptorIterable;
   class RExtraTypeInfoDescriptorIterable;

   /// Modifiers passed to CreateModel()
   struct RCreateModelOptions {
   private:
      /// If set to true, projected fields will be reconstructed as such. This will prevent the model to be used
      /// with an RNTupleReader, but it is useful, e.g., to accurately merge data.
      bool fReconstructProjections = false;
      /// If this option is enabled, the model will be created and all fields containing unknown data (directly
      /// or indirectly) will be skipped instead.
      /// Normally creating a model will fail if any of the reconstructed fields contains an unknown column type.
      bool fForwardCompatible = false;
      /// If true, the model will be created without a default entry (bare model).
      bool fCreateBare = false;
      /// If true, fields with a user defined type that have no available dictionaries will be reconstructed
      /// as record fields from the on-disk information; otherwise, they will cause an error.
      bool fEmulateUnknownTypes = false;

   public:
      RCreateModelOptions() {} // Work around compiler bug, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88165

      void SetReconstructProjections(bool v) { fReconstructProjections = v; }
      bool GetReconstructProjections() const { return fReconstructProjections; }

      void SetForwardCompatible(bool v) { fForwardCompatible = v; }
      bool GetForwardCompatible() const { return fForwardCompatible; }

      void SetCreateBare(bool v) { fCreateBare = v; }
      bool GetCreateBare() const { return fCreateBare; }

      void SetEmulateUnknownTypes(bool v) { fEmulateUnknownTypes = v; }
      bool GetEmulateUnknownTypes() const { return fEmulateUnknownTypes; }
   };

   RNTupleDescriptor() = default;
   RNTupleDescriptor(const RNTupleDescriptor &other) = delete;
   RNTupleDescriptor &operator=(const RNTupleDescriptor &other) = delete;
   RNTupleDescriptor(RNTupleDescriptor &&other) = default;
   RNTupleDescriptor &operator=(RNTupleDescriptor &&other) = default;

   RNTupleDescriptor Clone() const;

   bool operator==(const RNTupleDescriptor &other) const;

   std::uint64_t GetOnDiskHeaderXxHash3() const { return fOnDiskHeaderXxHash3; }
   std::uint64_t GetOnDiskHeaderSize() const { return fOnDiskHeaderSize; }
   std::uint64_t GetOnDiskFooterSize() const { return fOnDiskFooterSize; }

   const RFieldDescriptor &GetFieldDescriptor(ROOT::DescriptorId_t fieldId) const
   {
      return fFieldDescriptors.at(fieldId);
   }
   const RColumnDescriptor &GetColumnDescriptor(ROOT::DescriptorId_t columnId) const
   {
      return fColumnDescriptors.at(columnId);
   }
   const RClusterGroupDescriptor &GetClusterGroupDescriptor(ROOT::DescriptorId_t clusterGroupId) const
   {
      return fClusterGroupDescriptors.at(clusterGroupId);
   }
   const RClusterDescriptor &GetClusterDescriptor(ROOT::DescriptorId_t clusterId) const
   {
      return fClusterDescriptors.at(clusterId);
   }

   RFieldDescriptorIterable GetFieldIterable(const RFieldDescriptor &fieldDesc) const;
   RFieldDescriptorIterable
   GetFieldIterable(const RFieldDescriptor &fieldDesc,
                    const std::function<bool(ROOT::DescriptorId_t, ROOT::DescriptorId_t)> &comparator) const;
   RFieldDescriptorIterable GetFieldIterable(ROOT::DescriptorId_t fieldId) const;
   RFieldDescriptorIterable
   GetFieldIterable(ROOT::DescriptorId_t fieldId,
                    const std::function<bool(ROOT::DescriptorId_t, ROOT::DescriptorId_t)> &comparator) const;

   RFieldDescriptorIterable GetTopLevelFields() const;
   RFieldDescriptorIterable
   GetTopLevelFields(const std::function<bool(ROOT::DescriptorId_t, ROOT::DescriptorId_t)> &comparator) const;

   RColumnDescriptorIterable GetColumnIterable() const;
   RColumnDescriptorIterable GetColumnIterable(const RFieldDescriptor &fieldDesc) const;
   RColumnDescriptorIterable GetColumnIterable(ROOT::DescriptorId_t fieldId) const;

   RClusterGroupDescriptorIterable GetClusterGroupIterable() const;

   RClusterDescriptorIterable GetClusterIterable() const;

   RExtraTypeInfoDescriptorIterable GetExtraTypeInfoIterable() const;

   const std::string &GetName() const { return fName; }
   const std::string &GetDescription() const { return fDescription; }

   std::size_t GetNFields() const { return fFieldDescriptors.size(); }
   std::size_t GetNLogicalColumns() const { return fColumnDescriptors.size(); }
   std::size_t GetNPhysicalColumns() const { return fNPhysicalColumns; }
   std::size_t GetNClusterGroups() const { return fClusterGroupDescriptors.size(); }
   std::size_t GetNClusters() const { return fNClusters; }
   std::size_t GetNActiveClusters() const { return fClusterDescriptors.size(); }
   std::size_t GetNExtraTypeInfos() const { return fExtraTypeInfoDescriptors.size(); }

   /// We know the number of entries from adding the cluster summaries
   ROOT::NTupleSize_t GetNEntries() const { return fNEntries; }
   ROOT::NTupleSize_t GetNElements(ROOT::DescriptorId_t physicalColumnId) const;

   /// Returns the logical parent of all top-level RNTuple data fields.
   ROOT::DescriptorId_t GetFieldZeroId() const { return fFieldZeroId; }
   const RFieldDescriptor &GetFieldZero() const { return GetFieldDescriptor(GetFieldZeroId()); }
   ROOT::DescriptorId_t FindFieldId(std::string_view fieldName, ROOT::DescriptorId_t parentId) const;
   /// Searches for a top-level field
   ROOT::DescriptorId_t FindFieldId(std::string_view fieldName) const;
   ROOT::DescriptorId_t FindLogicalColumnId(ROOT::DescriptorId_t fieldId, std::uint32_t columnIndex,
                                            std::uint16_t representationIndex) const;
   ROOT::DescriptorId_t FindPhysicalColumnId(ROOT::DescriptorId_t fieldId, std::uint32_t columnIndex,
                                             std::uint16_t representationIndex) const;
   ROOT::DescriptorId_t FindClusterId(ROOT::DescriptorId_t physicalColumnId, ROOT::NTupleSize_t index) const;
   ROOT::DescriptorId_t FindNextClusterId(ROOT::DescriptorId_t clusterId) const;
   ROOT::DescriptorId_t FindPrevClusterId(ROOT::DescriptorId_t clusterId) const;

   /// Walks up the parents of the field ID and returns a field name of the form a.b.c.d
   /// In case of invalid field ID, an empty string is returned.
   std::string GetQualifiedFieldName(ROOT::DescriptorId_t fieldId) const;

   bool HasFeature(unsigned int flag) const { return fFeatureFlags.count(flag) > 0; }
   std::vector<std::uint64_t> GetFeatureFlags() const;

   /// Return header extension information; if the descriptor does not have a header extension, return `nullptr`
   const RHeaderExtension *GetHeaderExtension() const { return fHeaderExtension.get(); }

   /// Methods to load and drop cluster group details (cluster IDs and page locations)
   RResult<void>
   AddClusterGroupDetails(ROOT::DescriptorId_t clusterGroupId, std::vector<RClusterDescriptor> &clusterDescs);
   RResult<void> DropClusterGroupDetails(ROOT::DescriptorId_t clusterGroupId);

   std::uint64_t GetGeneration() const { return fGeneration; }
   void IncGeneration() { fGeneration++; }

   /// Re-create the C++ model from the stored metadata
   std::unique_ptr<ROOT::RNTupleModel> CreateModel(const RCreateModelOptions &options = RCreateModelOptions()) const;
   void PrintInfo(std::ostream &output) const;
};

// clang-format off
/**
\class ROOT::RNTupleDescriptor::RColumnDescriptorIterable
\ingroup NTuple
\brief Used to loop over a field's associated columns
*/
// clang-format on
class RNTupleDescriptor::RColumnDescriptorIterable final {
private:
   /// The associated RNTuple for this range.
   const RNTupleDescriptor &fNTuple;
   /// The descriptor ids of the columns ordered by field, representation, and column index
   std::vector<ROOT::DescriptorId_t> fColumns = {};

public:
   class RIterator final {
   private:
      /// The enclosing range's RNTuple.
      const RNTupleDescriptor &fNTuple;
      /// The enclosing range's descriptor id list.
      const std::vector<ROOT::DescriptorId_t> &fColumns;
      std::size_t fIndex = 0;

   public:
      using iterator_category = std::forward_iterator_tag;
      using iterator = RIterator;
      using value_type = RFieldDescriptor;
      using difference_type = std::ptrdiff_t;
      using pointer = const RColumnDescriptor *;
      using reference = const RColumnDescriptor &;

      RIterator(const RNTupleDescriptor &ntuple, const std::vector<ROOT::DescriptorId_t> &columns, std::size_t index)
         : fNTuple(ntuple), fColumns(columns), fIndex(index)
      {
      }
      iterator operator++()
      {
         ++fIndex;
         return *this;
      }
      reference operator*() { return fNTuple.GetColumnDescriptor(fColumns.at(fIndex)); }
      pointer operator->() { return &fNTuple.GetColumnDescriptor(fColumns.at(fIndex)); }
      bool operator!=(const iterator &rh) const { return fIndex != rh.fIndex; }
      bool operator==(const iterator &rh) const { return fIndex == rh.fIndex; }
   };

   RColumnDescriptorIterable(const RNTupleDescriptor &ntuple, const RFieldDescriptor &fieldDesc);
   RColumnDescriptorIterable(const RNTupleDescriptor &ntuple);

   RIterator begin() { return RIterator(fNTuple, fColumns, 0); }
   RIterator end() { return RIterator(fNTuple, fColumns, fColumns.size()); }
   size_t size() { return fColumns.size(); }
};

// clang-format off
/**
\class ROOT::RNTupleDescriptor::RFieldDescriptorIterable
\ingroup NTuple
\brief Used to loop over a field's child fields
*/
// clang-format on
class RNTupleDescriptor::RFieldDescriptorIterable final {
private:
   /// The associated RNTuple for this range.
   const RNTupleDescriptor &fNTuple;
   /// The descriptor IDs of the child fields. These may be sorted using
   /// a comparison function.
   std::vector<ROOT::DescriptorId_t> fFieldChildren = {};

public:
   class RIterator final {
   private:
      /// The enclosing range's RNTuple.
      const RNTupleDescriptor &fNTuple;
      /// The enclosing range's descriptor id list.
      const std::vector<ROOT::DescriptorId_t> &fFieldChildren;
      std::size_t fIndex = 0;

   public:
      using iterator_category = std::forward_iterator_tag;
      using iterator = RIterator;
      using value_type = RFieldDescriptor;
      using difference_type = std::ptrdiff_t;
      using pointer = RFieldDescriptor *;
      using reference = const RFieldDescriptor &;

      RIterator(const RNTupleDescriptor &ntuple, const std::vector<ROOT::DescriptorId_t> &fieldChildren,
                std::size_t index)
         : fNTuple(ntuple), fFieldChildren(fieldChildren), fIndex(index)
      {
      }
      iterator operator++()
      {
         ++fIndex;
         return *this;
      }
      reference operator*() { return fNTuple.GetFieldDescriptor(fFieldChildren.at(fIndex)); }
      bool operator!=(const iterator &rh) const { return fIndex != rh.fIndex; }
      bool operator==(const iterator &rh) const { return fIndex == rh.fIndex; }
   };
   RFieldDescriptorIterable(const RNTupleDescriptor &ntuple, const RFieldDescriptor &field)
      : fNTuple(ntuple), fFieldChildren(field.GetLinkIds())
   {
   }
   /// Sort the range using an arbitrary comparison function.
   RFieldDescriptorIterable(const RNTupleDescriptor &ntuple, const RFieldDescriptor &field,
                            const std::function<bool(ROOT::DescriptorId_t, ROOT::DescriptorId_t)> &comparator)
      : fNTuple(ntuple), fFieldChildren(field.GetLinkIds())
   {
      std::sort(fFieldChildren.begin(), fFieldChildren.end(), comparator);
   }
   RIterator begin() { return RIterator(fNTuple, fFieldChildren, 0); }
   RIterator end() { return RIterator(fNTuple, fFieldChildren, fFieldChildren.size()); }
};

// clang-format off
/**
\class ROOT::RNTupleDescriptor::RClusterGroupDescriptorIterable
\ingroup NTuple
\brief Used to loop over all the cluster groups of an RNTuple (in unspecified order)

Enumerate all cluster group IDs from the descriptor.  No specific order can be assumed.
*/
// clang-format on
class RNTupleDescriptor::RClusterGroupDescriptorIterable final {
private:
   /// The associated RNTuple for this range.
   const RNTupleDescriptor &fNTuple;

public:
   class RIterator final {
   private:
      /// The enclosing range's RNTuple.
      const RNTupleDescriptor &fNTuple;
      std::size_t fIndex = 0;

   public:
      using iterator_category = std::forward_iterator_tag;
      using iterator = RIterator;
      using value_type = RClusterGroupDescriptor;
      using difference_type = std::ptrdiff_t;
      using pointer = RClusterGroupDescriptor *;
      using reference = const RClusterGroupDescriptor &;

      RIterator(const RNTupleDescriptor &ntuple, std::size_t index) : fNTuple(ntuple), fIndex(index) {}
      iterator operator++()
      {
         ++fIndex;
         return *this;
      }
      reference operator*()
      {
         auto it = fNTuple.fClusterGroupDescriptors.begin();
         std::advance(it, fIndex);
         return it->second;
      }
      bool operator!=(const iterator &rh) const { return fIndex != rh.fIndex; }
      bool operator==(const iterator &rh) const { return fIndex == rh.fIndex; }
   };

   RClusterGroupDescriptorIterable(const RNTupleDescriptor &ntuple) : fNTuple(ntuple) {}
   RIterator begin() { return RIterator(fNTuple, 0); }
   RIterator end() { return RIterator(fNTuple, fNTuple.GetNClusterGroups()); }
};

// clang-format off
/**
\class ROOT::RNTupleDescriptor::RClusterDescriptorIterable
\ingroup NTuple
\brief Used to loop over all the clusters of an RNTuple (in unspecified order)

Enumerate all cluster IDs from all cluster descriptors.  No specific order can be assumed, use
RNTupleDescriptor::FindNextClusterId() and RNTupleDescriptor::FindPrevClusterId() to traverse
clusters by entry number.
*/
// clang-format on
class RNTupleDescriptor::RClusterDescriptorIterable final {
private:
   /// The associated RNTuple for this range.
   const RNTupleDescriptor &fNTuple;

public:
   class RIterator final {
   private:
      /// The enclosing range's RNTuple.
      const RNTupleDescriptor &fNTuple;
      std::size_t fIndex = 0;

   public:
      using iterator_category = std::forward_iterator_tag;
      using iterator = RIterator;
      using value_type = RClusterDescriptor;
      using difference_type = std::ptrdiff_t;
      using pointer = RClusterDescriptor *;
      using reference = const RClusterDescriptor &;

      RIterator(const RNTupleDescriptor &ntuple, std::size_t index) : fNTuple(ntuple), fIndex(index) {}
      iterator operator++()
      {
         ++fIndex;
         return *this;
      }
      reference operator*()
      {
         auto it = fNTuple.fClusterDescriptors.begin();
         std::advance(it, fIndex);
         return it->second;
      }
      bool operator!=(const iterator &rh) const { return fIndex != rh.fIndex; }
      bool operator==(const iterator &rh) const { return fIndex == rh.fIndex; }
   };

   RClusterDescriptorIterable(const RNTupleDescriptor &ntuple) : fNTuple(ntuple) {}
   RIterator begin() { return RIterator(fNTuple, 0); }
   RIterator end() { return RIterator(fNTuple, fNTuple.GetNActiveClusters()); }
};

// clang-format off
/**
\class ROOT::RNTupleDescriptor::RExtraTypeInfoDescriptorIterable
\ingroup NTuple
\brief Used to loop over all the extra type info record of an RNTuple (in unspecified order)
*/
// clang-format on
class RNTupleDescriptor::RExtraTypeInfoDescriptorIterable final {
private:
   /// The associated RNTuple for this range.
   const RNTupleDescriptor &fNTuple;

public:
   class RIterator final {
   private:
      /// The enclosing range's RNTuple.
      const RNTupleDescriptor &fNTuple;
      std::size_t fIndex = 0;

   public:
      using iterator_category = std::forward_iterator_tag;
      using iterator = RIterator;
      using value_type = RExtraTypeInfoDescriptor;
      using difference_type = std::ptrdiff_t;
      using pointer = RExtraTypeInfoDescriptor *;
      using reference = const RExtraTypeInfoDescriptor &;

      RIterator(const RNTupleDescriptor &ntuple, std::size_t index) : fNTuple(ntuple), fIndex(index) {}
      iterator operator++()
      {
         ++fIndex;
         return *this;
      }
      reference operator*()
      {
         auto it = fNTuple.fExtraTypeInfoDescriptors.begin();
         std::advance(it, fIndex);
         return *it;
      }
      bool operator!=(const iterator &rh) const { return fIndex != rh.fIndex; }
      bool operator==(const iterator &rh) const { return fIndex == rh.fIndex; }
   };

   RExtraTypeInfoDescriptorIterable(const RNTupleDescriptor &ntuple) : fNTuple(ntuple) {}
   RIterator begin() { return RIterator(fNTuple, 0); }
   RIterator end() { return RIterator(fNTuple, fNTuple.GetNExtraTypeInfos()); }
};

// clang-format off
/**
\class ROOT::RNTupleDescriptor::RHeaderExtension
\ingroup NTuple
\brief Summarizes information about fields and the corresponding columns that were added after the header has been serialized
*/
// clang-format on
class RNTupleDescriptor::RHeaderExtension final {
   friend class Internal::RNTupleDescriptorBuilder;

private:
   /// All field IDs of late model extensions, in the order of field addition. This is necessary to serialize the
   /// the fields in that order.
   std::vector<ROOT::DescriptorId_t> fFieldIdsOrder;
   /// All field IDs of late model extensions for efficient lookup. When a column gets added to the extension
   /// header, this enables us to determine if the column belongs to a field of the header extension of if it
   /// belongs to a field of the regular header that gets extended by additional column representations.
   std::unordered_set<ROOT::DescriptorId_t> fFieldIdsLookup;
   /// All logical column IDs of columns that extend, with additional column representations, fields of the regular
   /// header. During serialization, these columns are not picked up as columns of `fFieldIdsOrder`. But instead
   /// these columns need to be serialized in the extension header without re-serializing the field.
   std::vector<ROOT::DescriptorId_t> fExtendedColumnRepresentations;
   /// Number of logical and physical columns; updated by the descriptor builder when columns are added
   std::uint32_t fNLogicalColumns = 0;
   std::uint32_t fNPhysicalColumns = 0;

   /// Marks `fieldDesc` as an extended field, i.e. a field that appears in the Header Extension (e.g. having been added
   /// through late model extension). Note that the field descriptor should also have been added to the RNTuple
   /// Descriptor alongside non-extended fields.
   void MarkExtendedField(const RFieldDescriptor &fieldDesc)
   {
      fFieldIdsOrder.emplace_back(fieldDesc.GetId());
      fFieldIdsLookup.insert(fieldDesc.GetId());
   }

   /// Marks `columnDesc` as an extended column, i.e. a column that appears in the Header Extension (e.g. having been
   /// added through late model extension as an additional representation of an existing column). Note that the column
   /// descriptor should also have been added to the RNTuple Descriptor alongside non-extended columns.
   void MarkExtendedColumn(const RColumnDescriptor &columnDesc)
   {
      fNLogicalColumns++;
      if (!columnDesc.IsAliasColumn())
         fNPhysicalColumns++;
      if (fFieldIdsLookup.count(columnDesc.GetFieldId()) == 0) {
         fExtendedColumnRepresentations.emplace_back(columnDesc.GetLogicalId());
      }
   }

public:
   std::size_t GetNFields() const { return fFieldIdsOrder.size(); }
   std::size_t GetNLogicalColumns() const { return fNLogicalColumns; }
   std::size_t GetNPhysicalColumns() const { return fNPhysicalColumns; }
   const std::vector<ROOT::DescriptorId_t> &GetExtendedColumnRepresentations() const
   {
      return fExtendedColumnRepresentations;
   }
   /// Return a vector containing the IDs of the top-level fields defined in the extension header, in the order
   /// of their addition.
   /// We cannot create this vector when building the fFields because at the time when AddExtendedField is called,
   /// the field is not yet linked into the schema tree.
   std::vector<ROOT::DescriptorId_t> GetTopLevelFields(const RNTupleDescriptor &desc) const;

   bool ContainsField(ROOT::DescriptorId_t fieldId) const
   {
      return fFieldIdsLookup.find(fieldId) != fFieldIdsLookup.end();
   }
   bool ContainsExtendedColumnRepresentation(ROOT::DescriptorId_t columnId) const
   {
      return std::find(fExtendedColumnRepresentations.begin(), fExtendedColumnRepresentations.end(), columnId) !=
             fExtendedColumnRepresentations.end();
   }
};

namespace Internal {

// clang-format off
/**
\class ROOT::Internal::RColumnDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RColumnDescriptor

Dangling column descriptors can become actual descriptors when added to an
RNTupleDescriptorBuilder instance and then linked to their fields.
*/
// clang-format on
class RColumnDescriptorBuilder final {
private:
   RColumnDescriptor fColumn = RColumnDescriptor();

public:
   /// Make an empty column descriptor builder.
   RColumnDescriptorBuilder() = default;

   RColumnDescriptorBuilder &LogicalColumnId(ROOT::DescriptorId_t logicalColumnId)
   {
      fColumn.fLogicalColumnId = logicalColumnId;
      return *this;
   }
   RColumnDescriptorBuilder &PhysicalColumnId(ROOT::DescriptorId_t physicalColumnId)
   {
      fColumn.fPhysicalColumnId = physicalColumnId;
      return *this;
   }
   RColumnDescriptorBuilder &BitsOnStorage(std::uint16_t bitsOnStorage)
   {
      fColumn.fBitsOnStorage = bitsOnStorage;
      return *this;
   }
   RColumnDescriptorBuilder &Type(ROOT::ENTupleColumnType type)
   {
      fColumn.fType = type;
      return *this;
   }
   RColumnDescriptorBuilder &FieldId(ROOT::DescriptorId_t fieldId)
   {
      fColumn.fFieldId = fieldId;
      return *this;
   }
   RColumnDescriptorBuilder &Index(std::uint32_t index)
   {
      fColumn.fIndex = index;
      return *this;
   }
   RColumnDescriptorBuilder &FirstElementIndex(std::uint64_t firstElementIdx)
   {
      fColumn.fFirstElementIndex = firstElementIdx;
      return *this;
   }
   RColumnDescriptorBuilder &SetSuppressedDeferred()
   {
      R__ASSERT(fColumn.fFirstElementIndex != 0);
      if (fColumn.fFirstElementIndex > 0)
         fColumn.fFirstElementIndex = -fColumn.fFirstElementIndex;
      return *this;
   }
   RColumnDescriptorBuilder &RepresentationIndex(std::uint16_t representationIndex)
   {
      fColumn.fRepresentationIndex = representationIndex;
      return *this;
   }
   RColumnDescriptorBuilder &ValueRange(double min, double max)
   {
      fColumn.fValueRange = {min, max};
      return *this;
   }
   RColumnDescriptorBuilder &ValueRange(std::optional<RColumnDescriptor::RValueRange> valueRange)
   {
      fColumn.fValueRange = valueRange;
      return *this;
   }
   ROOT::DescriptorId_t GetFieldId() const { return fColumn.fFieldId; }
   ROOT::DescriptorId_t GetRepresentationIndex() const { return fColumn.fRepresentationIndex; }
   /// Attempt to make a column descriptor. This may fail if the column
   /// was not given enough information to make a proper descriptor.
   RResult<RColumnDescriptor> MakeDescriptor() const;
};

// clang-format off
/**
\class ROOT::Internal::RFieldDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RFieldDescriptor

Dangling field descriptors describe a single field in isolation. They are
missing the necessary relationship information (parent field, any child fields)
required to describe a real RNTuple field.

Dangling field descriptors can only become actual descriptors when added to an
RNTupleDescriptorBuilder instance and then linked to other fields.
*/
// clang-format on
class RFieldDescriptorBuilder final {
private:
   RFieldDescriptor fField = RFieldDescriptor();

public:
   /// Make an empty dangling field descriptor.
   RFieldDescriptorBuilder() = default;
   /// Make a new RFieldDescriptorBuilder based off an existing descriptor.
   /// Relationship information is lost during the conversion to a
   /// dangling descriptor:
   /// * Parent id is reset to an invalid id.
   /// * Field children ids are forgotten.
   ///
   /// These properties must be set using RNTupleDescriptorBuilder::AddFieldLink().
   explicit RFieldDescriptorBuilder(const RFieldDescriptor &fieldDesc);

   /// Make a new RFieldDescriptorBuilder based off a live RNTuple field.
   static RFieldDescriptorBuilder FromField(const ROOT::RFieldBase &field);

   RFieldDescriptorBuilder &FieldId(ROOT::DescriptorId_t fieldId)
   {
      fField.fFieldId = fieldId;
      return *this;
   }
   RFieldDescriptorBuilder &FieldVersion(std::uint32_t fieldVersion)
   {
      fField.fFieldVersion = fieldVersion;
      return *this;
   }
   RFieldDescriptorBuilder &TypeVersion(std::uint32_t typeVersion)
   {
      fField.fTypeVersion = typeVersion;
      return *this;
   }
   RFieldDescriptorBuilder &ParentId(ROOT::DescriptorId_t id)
   {
      fField.fParentId = id;
      return *this;
   }
   RFieldDescriptorBuilder &ProjectionSourceId(ROOT::DescriptorId_t id)
   {
      fField.fProjectionSourceId = id;
      return *this;
   }
   RFieldDescriptorBuilder &FieldName(const std::string &fieldName)
   {
      fField.fFieldName = fieldName;
      return *this;
   }
   RFieldDescriptorBuilder &FieldDescription(const std::string &fieldDescription)
   {
      fField.fFieldDescription = fieldDescription;
      return *this;
   }
   RFieldDescriptorBuilder &TypeName(const std::string &typeName)
   {
      fField.fTypeName = typeName;
      return *this;
   }
   RFieldDescriptorBuilder &TypeAlias(const std::string &typeAlias)
   {
      fField.fTypeAlias = typeAlias;
      return *this;
   }
   RFieldDescriptorBuilder &NRepetitions(std::uint64_t nRepetitions)
   {
      fField.fNRepetitions = nRepetitions;
      return *this;
   }
   RFieldDescriptorBuilder &Structure(const ROOT::ENTupleStructure &structure)
   {
      fField.fStructure = structure;
      return *this;
   }
   RFieldDescriptorBuilder &TypeChecksum(const std::optional<std::uint32_t> typeChecksum)
   {
      fField.fTypeChecksum = typeChecksum;
      return *this;
   }
   ROOT::DescriptorId_t GetParentId() const { return fField.fParentId; }
   /// Attempt to make a field descriptor. This may fail if the dangling field
   /// was not given enough information to make a proper descriptor.
   RResult<RFieldDescriptor> MakeDescriptor() const;
};

// clang-format off
/**
\class ROOT::Internal::RClusterDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RClusterDescriptor

The cluster descriptor builder starts from a summary-only cluster descriptor and allows for the
piecewise addition of page locations.
*/
// clang-format on
class RClusterDescriptorBuilder final {
private:
   RClusterDescriptor fCluster;

public:
   RClusterDescriptorBuilder &ClusterId(ROOT::DescriptorId_t clusterId)
   {
      fCluster.fClusterId = clusterId;
      return *this;
   }

   RClusterDescriptorBuilder &FirstEntryIndex(std::uint64_t firstEntryIndex)
   {
      fCluster.fFirstEntryIndex = firstEntryIndex;
      return *this;
   }

   RClusterDescriptorBuilder &NEntries(std::uint64_t nEntries)
   {
      fCluster.fNEntries = nEntries;
      return *this;
   }

   RResult<void> CommitColumnRange(ROOT::DescriptorId_t physicalId, std::uint64_t firstElementIndex,
                                   std::uint32_t compressionSettings, const RClusterDescriptor::RPageRange &pageRange);

   /// Books the given column ID as being suppressed in this cluster. The correct first element index and number of
   /// elements need to be set by CommitSuppressedColumnRanges() once all the calls to CommitColumnRange() and
   /// MarkSuppressedColumnRange() took place.
   RResult<void> MarkSuppressedColumnRange(ROOT::DescriptorId_t physicalId);

   /// Sets the first element index and number of elements for all the suppressed column ranges.
   /// The information is taken from the corresponding columns from the primary representation.
   /// Needs to be called when all the columns (suppressed and regular) where added.
   RResult<void> CommitSuppressedColumnRanges(const RNTupleDescriptor &desc);

   /// Add column and page ranges for columns created during late model extension missing in this cluster.  The locator
   /// type for the synthesized page ranges is `kTypePageZero`.  All the page sources must be able to populate the
   /// 'zero' page from such locator. Any call to CommitColumnRange() and CommitSuppressedColumnRanges()
   /// should happen before calling this function.
   RClusterDescriptorBuilder &AddExtendedColumnRanges(const RNTupleDescriptor &desc);

   const RClusterDescriptor::RColumnRange &GetColumnRange(ROOT::DescriptorId_t physicalId)
   {
      return fCluster.GetColumnRange(physicalId);
   }

   /// Move out the full cluster descriptor including page locations
   RResult<RClusterDescriptor> MoveDescriptor();
};

// clang-format off
/**
\class ROOT::Internal::RClusterGroupDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RClusterGroupDescriptor
*/
// clang-format on
class RClusterGroupDescriptorBuilder final {
private:
   RClusterGroupDescriptor fClusterGroup;

public:
   RClusterGroupDescriptorBuilder() = default;
   static RClusterGroupDescriptorBuilder FromSummary(const RClusterGroupDescriptor &clusterGroupDesc);

   RClusterGroupDescriptorBuilder &ClusterGroupId(ROOT::DescriptorId_t clusterGroupId)
   {
      fClusterGroup.fClusterGroupId = clusterGroupId;
      return *this;
   }
   RClusterGroupDescriptorBuilder &PageListLocator(const RNTupleLocator &pageListLocator)
   {
      fClusterGroup.fPageListLocator = pageListLocator;
      return *this;
   }
   RClusterGroupDescriptorBuilder &PageListLength(std::uint64_t pageListLength)
   {
      fClusterGroup.fPageListLength = pageListLength;
      return *this;
   }
   RClusterGroupDescriptorBuilder &MinEntry(std::uint64_t minEntry)
   {
      fClusterGroup.fMinEntry = minEntry;
      return *this;
   }
   RClusterGroupDescriptorBuilder &EntrySpan(std::uint64_t entrySpan)
   {
      fClusterGroup.fEntrySpan = entrySpan;
      return *this;
   }
   RClusterGroupDescriptorBuilder &NClusters(std::uint32_t nClusters)
   {
      fClusterGroup.fNClusters = nClusters;
      return *this;
   }
   void AddSortedClusters(const std::vector<ROOT::DescriptorId_t> &clusterIds)
   {
      if (clusterIds.size() != fClusterGroup.GetNClusters())
         throw RException(R__FAIL("mismatch of number of clusters"));
      fClusterGroup.fClusterIds = clusterIds;
   }

   RResult<RClusterGroupDescriptor> MoveDescriptor();
};

// clang-format off
/**
\class ROOT::Internal::RExtraTypeInfoDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RExtraTypeInfoDescriptor
*/
// clang-format on
class RExtraTypeInfoDescriptorBuilder final {
private:
   RExtraTypeInfoDescriptor fExtraTypeInfo;

public:
   RExtraTypeInfoDescriptorBuilder() = default;

   RExtraTypeInfoDescriptorBuilder &ContentId(EExtraTypeInfoIds contentId)
   {
      fExtraTypeInfo.fContentId = contentId;
      return *this;
   }
   RExtraTypeInfoDescriptorBuilder &TypeVersion(std::uint32_t typeVersion)
   {
      fExtraTypeInfo.fTypeVersion = typeVersion;
      return *this;
   }
   RExtraTypeInfoDescriptorBuilder &TypeName(const std::string &typeName)
   {
      fExtraTypeInfo.fTypeName = typeName;
      return *this;
   }
   RExtraTypeInfoDescriptorBuilder &Content(const std::string &content)
   {
      fExtraTypeInfo.fContent = content;
      return *this;
   }

   RResult<RExtraTypeInfoDescriptor> MoveDescriptor();
};

// clang-format off
/**
\class ROOT::Internal::RNTupleDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RNTupleDescriptor

Used by RPageStorage implementations in order to construct the RNTupleDescriptor from the various header parts.
*/
// clang-format on
class RNTupleDescriptorBuilder final {
private:
   RNTupleDescriptor fDescriptor;
   RResult<void> EnsureFieldExists(ROOT::DescriptorId_t fieldId) const;

public:
   /// Checks whether invariants hold:
   /// * RNTuple name is valid
   /// * Fields have valid parents
   /// * Number of columns is constant across column representations
   RResult<void> EnsureValidDescriptor() const;
   const RNTupleDescriptor &GetDescriptor() const { return fDescriptor; }
   RNTupleDescriptor MoveDescriptor();

   /// Copies the "schema" part of `descriptor` into the builder's descriptor.
   /// This resets the builder's descriptor.
   void SetSchemaFromExisting(const RNTupleDescriptor &descriptor);

   void SetNTuple(const std::string_view name, const std::string_view description);
   void SetFeature(unsigned int flag);

   void SetOnDiskHeaderXxHash3(std::uint64_t xxhash3) { fDescriptor.fOnDiskHeaderXxHash3 = xxhash3; }
   void SetOnDiskHeaderSize(std::uint64_t size) { fDescriptor.fOnDiskHeaderSize = size; }
   /// The real footer size also include the page list envelopes
   void AddToOnDiskFooterSize(std::uint64_t size) { fDescriptor.fOnDiskFooterSize += size; }

   void AddField(const RFieldDescriptor &fieldDesc);
   RResult<void> AddFieldLink(ROOT::DescriptorId_t fieldId, ROOT::DescriptorId_t linkId);
   RResult<void> AddFieldProjection(ROOT::DescriptorId_t sourceId, ROOT::DescriptorId_t targetId);

   // The field that the column belongs to has to be already available. For fields with multiple columns,
   // the columns need to be added in order of the column index
   RResult<void> AddColumn(RColumnDescriptor &&columnDesc);

   RResult<void> AddClusterGroup(RClusterGroupDescriptor &&clusterGroup);
   RResult<void> AddCluster(RClusterDescriptor &&clusterDesc);

   RResult<void> AddExtraTypeInfo(RExtraTypeInfoDescriptor &&extraTypeInfoDesc);
   void ReplaceExtraTypeInfo(RExtraTypeInfoDescriptor &&extraTypeInfoDesc);

   /// Clears so-far stored clusters, fields, and columns and return to a pristine RNTupleDescriptor
   void Reset();

   /// Mark the beginning of the header extension; any fields and columns added after a call to this function are
   /// annotated as begin part of the header extension.
   void BeginHeaderExtension();

   /// \brief Shift column IDs of alias columns by `offset`
   ///
   /// If the descriptor is constructed in pieces consisting of physical and alias columns
   /// (regular and projected fields), the natural column order would be
   ///   - Physical and alias columns of piece one
   ///   - Physical and alias columns of piece two
   ///   - etc.
   /// What we want, however, are first all physical column IDs and then all alias column IDs.
   /// This method adds `offset` to the logical column IDs of all alias columns and fixes up the corresponding
   /// column IDs in the projected field descriptors.  In this way, a new piece of physical and alias columns can
   /// first shift the existing alias columns by the number of new physical columns, resulting in the following order
   ///   - Physical columns of piece one
   ///   - Physical columns of piece two
   ///   - ...
   //    - Logical columns of piece one
   ///   - Logical columns of piece two
   ///   - ...
   void ShiftAliasColumns(std::uint32_t offset);

   /// Get the streamer info records for custom classes. Currently requires the corresponding dictionaries to be loaded.
   ROOT::Internal::RNTupleSerializer::StreamerInfoMap_t BuildStreamerInfos() const;
};

inline RNTupleDescriptor CloneDescriptorSchema(const RNTupleDescriptor &desc)
{
   return desc.CloneSchema();
}

} // namespace Internal
} // namespace ROOT

#endif // ROOT_RNTupleDescriptor
