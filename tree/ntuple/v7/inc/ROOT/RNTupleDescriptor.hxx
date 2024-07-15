/// \file ROOT/RNTupleDescriptor.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Javier Lopez-Gomez <javier.lopez.gomez@cern.ch>
/// \date 2018-07-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleDescriptor
#define ROOT7_RNTupleDescriptor

#include <ROOT/RError.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RSpan.hxx>
#include <string_view>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <vector>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ROOT {
namespace Experimental {

class RFieldBase;
class RNTupleDescriptor;
class RNTupleModel;

namespace Internal {
class RColumnElementBase;
} // namespace Internal

namespace Internal {
class RColumnDescriptorBuilder;
class RColumnGroupDescriptorBuilder;
class RClusterDescriptorBuilder;
class RClusterGroupDescriptorBuilder;
class RExtraTypeInfoDescriptorBuilder;
class RFieldDescriptorBuilder;
class RNTupleDescriptorBuilder;
} // namespace Internal

// clang-format off
/**
\class ROOT::Experimental::RFieldDescriptor
\ingroup NTuple
\brief Meta-data stored for every field of an ntuple
*/
// clang-format on
class RFieldDescriptor {
   friend class Internal::RNTupleDescriptorBuilder;
   friend class Internal::RFieldDescriptorBuilder;

private:
   DescriptorId_t fFieldId = kInvalidDescriptorId;
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
   ENTupleStructure fStructure = ENTupleStructure::kInvalid;
   /// Establishes sub field relationships, such as classes and collections
   DescriptorId_t fParentId = kInvalidDescriptorId;
   /// For projected fields, the source field ID
   DescriptorId_t fProjectionSourceId = kInvalidDescriptorId;
   /// The pointers in the other direction from parent to children. They are serialized, too, to keep the
   /// order of sub fields.
   std::vector<DescriptorId_t> fLinkIds;
   /// The ordered list of columns attached to this field
   std::vector<DescriptorId_t> fLogicalColumnIds;

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
   /// access to sub fields, which is provided by the ntuple descriptor argument.
   std::unique_ptr<RFieldBase> CreateField(const RNTupleDescriptor &ntplDesc) const;

   DescriptorId_t GetId() const { return fFieldId; }
   std::uint32_t GetFieldVersion() const { return fFieldVersion; }
   std::uint32_t GetTypeVersion() const { return fTypeVersion; }
   const std::string &GetFieldName() const { return fFieldName; }
   const std::string &GetFieldDescription() const { return fFieldDescription; }
   const std::string &GetTypeName() const { return fTypeName; }
   const std::string &GetTypeAlias() const { return fTypeAlias; }
   std::uint64_t GetNRepetitions() const { return fNRepetitions; }
   ENTupleStructure GetStructure() const { return fStructure; }
   DescriptorId_t GetParentId() const { return fParentId; }
   DescriptorId_t GetProjectionSourceId() const { return fProjectionSourceId; }
   const std::vector<DescriptorId_t> &GetLinkIds() const { return fLinkIds; }
   const std::vector<DescriptorId_t> &GetLogicalColumnIds() const { return fLogicalColumnIds; }
   bool IsProjectedField() const { return fProjectionSourceId != kInvalidDescriptorId; }
};

// clang-format off
/**
\class ROOT::Experimental::RColumnDescriptor
\ingroup NTuple
\brief Meta-data stored for every column of an ntuple
*/
// clang-format on
class RColumnDescriptor {
   friend class Internal::RColumnDescriptorBuilder;
   friend class Internal::RNTupleDescriptorBuilder;

private:
   /// The actual column identifier, which is the link to the corresponding field
   DescriptorId_t fLogicalColumnId = kInvalidDescriptorId;
   /// Usually identical to the logical column ID, except for alias columns where it references the shadowed column
   DescriptorId_t fPhysicalColumnId = kInvalidDescriptorId;
   /// The on-disk column type
   EColumnType fType;
   /// Every column belongs to one and only one field
   DescriptorId_t fFieldId = kInvalidDescriptorId;
   /// A field can be serialized into several columns, which are numbered from zero to $n$
   std::uint32_t fIndex;
   /// Specifies the index for the first stored element for this column. For deferred columns the value is greater
   /// than 0
   std::uint64_t fFirstElementIndex = 0U;

public:
   RColumnDescriptor() = default;
   RColumnDescriptor(const RColumnDescriptor &other) = delete;
   RColumnDescriptor &operator=(const RColumnDescriptor &other) = delete;
   RColumnDescriptor(RColumnDescriptor &&other) = default;
   RColumnDescriptor &operator=(RColumnDescriptor &&other) = default;

   bool operator==(const RColumnDescriptor &other) const;
   /// Get a copy of the descriptor
   RColumnDescriptor Clone() const;

   DescriptorId_t GetLogicalId() const { return fLogicalColumnId; }
   DescriptorId_t GetPhysicalId() const { return fPhysicalColumnId; }
   EColumnType GetType() const { return fType; }
   std::uint32_t GetIndex() const { return fIndex; }
   DescriptorId_t GetFieldId() const { return fFieldId; }
   bool IsAliasColumn() const { return fPhysicalColumnId != fLogicalColumnId; }
   std::uint64_t GetFirstElementIndex() const { return fFirstElementIndex; }
   bool IsDeferredColumn() const { return fFirstElementIndex > 0; }
};

// clang-format off
/**
\class ROOT::Experimental::RColumnGroupDescriptor
\ingroup NTuple
\brief Meta-data for a sets of columns; non-trivial column groups are used for sharded clusters

Clusters can span a subset of columns. Such subsets are described as a column group. An empty column group
is used to denote the column group of all the columns. Every ntuple has at least one column group.
*/
// clang-format on
class RColumnGroupDescriptor {
   friend class Internal::RColumnGroupDescriptorBuilder;

private:
   DescriptorId_t fColumnGroupId = kInvalidDescriptorId;
   std::unordered_set<DescriptorId_t> fPhysicalColumnIds;

public:
   RColumnGroupDescriptor() = default;
   RColumnGroupDescriptor(const RColumnGroupDescriptor &other) = delete;
   RColumnGroupDescriptor &operator=(const RColumnGroupDescriptor &other) = delete;
   RColumnGroupDescriptor(RColumnGroupDescriptor &&other) = default;
   RColumnGroupDescriptor &operator=(RColumnGroupDescriptor &&other) = default;

   bool operator==(const RColumnGroupDescriptor &other) const;

   DescriptorId_t GetId() const { return fColumnGroupId; }
   const std::unordered_set<DescriptorId_t> &GetPhysicalColumnIds() const { return fPhysicalColumnIds; }
   bool Contains(DescriptorId_t physicalId) const
   {
      return fPhysicalColumnIds.empty() || fPhysicalColumnIds.count(physicalId) > 0;
   }
   bool HasAllColumns() const { return fPhysicalColumnIds.empty(); }
};

// clang-format off
/**
\class ROOT::Experimental::RClusterDescriptor
\ingroup NTuple
\brief Meta-data for a set of ntuple clusters

The cluster descriptor is built in two phases.  In a first phase, the descriptor has only an ID.
In a second phase, the event range, column group, page locations and column ranges are added.
Both phases are populated by the RClusterDescriptorBuilder.
Clusters usually span across all available columns but in some cases they can describe only a subset of the columns,
for instance when describing friend ntuples.
*/
// clang-format on
class RClusterDescriptor {
   friend class Internal::RClusterDescriptorBuilder;

public:
   /// The window of element indexes of a particular column in a particular cluster
   struct RColumnRange {
      DescriptorId_t fPhysicalColumnId = kInvalidDescriptorId;
      /// The global index of the first column element in the cluster
      NTupleSize_t fFirstElementIndex = kInvalidNTupleIndex;
      /// The number of column elements in the cluster
      ClusterSize_t fNElements = kInvalidClusterIndex;
      /// The usual format for ROOT compression settings (see Compression.h).
      /// The pages of a particular column in a particular cluster are all compressed with the same settings.
      int fCompressionSettings = kUnknownCompressionSettings;

      // TODO(jblomer): we perhaps want to store summary information, such as average, min/max, etc.
      // Should this be done on the field level?

      bool operator==(const RColumnRange &other) const
      {
         return fPhysicalColumnId == other.fPhysicalColumnId && fFirstElementIndex == other.fFirstElementIndex &&
                fNElements == other.fNElements && fCompressionSettings == other.fCompressionSettings;
      }

      bool Contains(NTupleSize_t index) const
      {
         return (fFirstElementIndex <= index && (fFirstElementIndex + fNElements) > index);
      }
   };

   /// Records the parition of data into pages for a particular column in a particular cluster
   class RPageRange {
      friend class Internal::RClusterDescriptorBuilder;
      /// Extend this RPageRange to fit the given RColumnRange, i.e. prepend as many synthetic RPageInfos as needed to
      /// cover the range in `columnRange`. `RPageInfo`s are constructed to contain as many elements of type `element`
      /// given a page size limit of `pageSize` (in bytes); the locator for the referenced pages is `kTypePageZero`.
      /// This function is used to make up `RPageRange`s for clusters that contain deferred columns.
      /// \return The number of column elements covered by the synthesized RPageInfos
      std::size_t ExtendToFitColumnRange(const RColumnRange &columnRange, const Internal::RColumnElementBase &element,
                                         std::size_t pageSize);

   public:
      /// We do not need to store the element size / uncompressed page size because we know to which column
      /// the page belongs
      struct RPageInfo {
         /// The sum of the elements of all the pages must match the corresponding fNElements field in fColumnRanges
         std::uint32_t fNElements = std::uint32_t(-1);
         /// The meaning of fLocator depends on the storage backend.
         RNTupleLocator fLocator;
         /// If true, the 8 bytes following the serialized page are an xxhash of the on-disk page data
         bool fHasChecksum = false;

         bool operator==(const RPageInfo &other) const
         {
            return fNElements == other.fNElements && fLocator == other.fLocator;
         }
      };
      struct RPageInfoExtended : RPageInfo {
         /// Index (in cluster) of the first element in page.
         ClusterSize_t::ValueType fFirstInPage = 0;
         /// Page number in the corresponding RPageRange.
         NTupleSize_t fPageNo = 0;

         RPageInfoExtended() = default;
         RPageInfoExtended(const RPageInfo &pi, ClusterSize_t::ValueType i, NTupleSize_t n)
            : RPageInfo(pi), fFirstInPage(i), fPageNo(n)
         {
         }
      };

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
         return clone;
      }

      /// Find the page in the RPageRange that contains the given element. The element must exist.
      RPageInfoExtended Find(ClusterSize_t::ValueType idxInCluster) const;

      DescriptorId_t fPhysicalColumnId = kInvalidDescriptorId;
      std::vector<RPageInfo> fPageInfos;

      bool operator==(const RPageRange &other) const
      {
         return fPhysicalColumnId == other.fPhysicalColumnId && fPageInfos == other.fPageInfos;
      }
   };

private:
   DescriptorId_t fClusterId = kInvalidDescriptorId;
   /// Clusters can be swapped by adjusting the entry offsets
   NTupleSize_t fFirstEntryIndex = kInvalidNTupleIndex;
   // TODO(jblomer): change to std::uint64_t
   ClusterSize_t fNEntries = kInvalidClusterIndex;

   std::unordered_map<DescriptorId_t, RColumnRange> fColumnRanges;
   std::unordered_map<DescriptorId_t, RPageRange> fPageRanges;

public:
   class RColumnRangeIterable {
   private:
      const RClusterDescriptor &fDesc;

   public:
      class RIterator {
      private:
         using Iter_t = std::unordered_map<DescriptorId_t, RColumnRange>::const_iterator;
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
      size_t count() { return fDesc.fColumnRanges.size(); }
   };

   RClusterDescriptor() = default;
   RClusterDescriptor(const RClusterDescriptor &other) = delete;
   RClusterDescriptor &operator=(const RClusterDescriptor &other) = delete;
   RClusterDescriptor(RClusterDescriptor &&other) = default;
   RClusterDescriptor &operator=(RClusterDescriptor &&other) = default;

   RClusterDescriptor Clone() const;

   bool operator==(const RClusterDescriptor &other) const;

   DescriptorId_t GetId() const { return fClusterId; }
   NTupleSize_t GetFirstEntryIndex() const { return fFirstEntryIndex; }
   ClusterSize_t GetNEntries() const { return fNEntries; }
   const RColumnRange &GetColumnRange(DescriptorId_t physicalId) const { return fColumnRanges.at(physicalId); }
   const RPageRange &GetPageRange(DescriptorId_t physicalId) const { return fPageRanges.at(physicalId); }
   /// Returns an iterator over pairs { columnId, columnRange }. The iteration order is unspecified.
   RColumnRangeIterable GetColumnRangeIterable() const { return RColumnRangeIterable(*this); }
   bool ContainsColumn(DescriptorId_t physicalId) const
   {
      return fColumnRanges.find(physicalId) != fColumnRanges.end();
   }
   std::uint64_t GetBytesOnStorage() const;
};

// clang-format off
/**
\class ROOT::Experimental::RClusterGroupDescriptor
\ingroup NTuple
\brief Clusters are bundled in cluster groups.

Very large ntuples or combined ntuples (chains, friends) contain multiple cluster groups. The cluster groups
may contain sharded clusters.
Every ntuple has at least one cluster group.  The clusters in a cluster group are ordered corresponding to
the order of page locations in the page list envelope that belongs to the cluster group (see format specification)
*/
// clang-format on
class RClusterGroupDescriptor {
   friend class Internal::RClusterGroupDescriptorBuilder;

private:
   DescriptorId_t fClusterGroupId = kInvalidDescriptorId;
   /// The cluster IDs can be empty if the corresponding page list is not loaded.
   std::vector<DescriptorId_t> fClusterIds;
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
   // Creates a clone without the cluster IDs
   RClusterGroupDescriptor CloneSummary() const;

   bool operator==(const RClusterGroupDescriptor &other) const;

   DescriptorId_t GetId() const { return fClusterGroupId; }
   std::uint32_t GetNClusters() const { return fNClusters; }
   RNTupleLocator GetPageListLocator() const { return fPageListLocator; }
   std::uint64_t GetPageListLength() const { return fPageListLength; }
   const std::vector<DescriptorId_t> &GetClusterIds() const { return fClusterIds; }
   std::uint64_t GetMinEntry() const { return fMinEntry; }
   std::uint64_t GetEntrySpan() const { return fEntrySpan; }
   /// A cluster group is loaded in two stages. Stage one loads only the summary information.
   /// Stage two loads the list of cluster IDs.
   bool HasClusterDetails() const { return !fClusterIds.empty(); }
};

/// Used in RExtraTypeInfoDescriptor
enum class EExtraTypeInfoIds { kInvalid, kStreamerInfo };

// clang-format off
/**
\class ROOT::Experimental::RExtraTypeInfoDescriptor
\ingroup NTuple
\brief Field specific extra type information from the header / extenstion header

Currently only used by unsplit fields to store RNTuple-wide list of streamer info records.
*/
// clang-format on
class RExtraTypeInfoDescriptor {
   friend class Internal::RExtraTypeInfoDescriptorBuilder;

private:
   /// Specifies the meaning of the extra information
   EExtraTypeInfoIds fContentId = EExtraTypeInfoIds::kInvalid;
   /// Extra type information restricted to a certain version range of the type
   std::uint32_t fTypeVersionFrom = 0;
   std::uint32_t fTypeVersionTo = 0;
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
   std::uint32_t GetTypeVersionFrom() const { return fTypeVersionFrom; }
   std::uint32_t GetTypeVersionTo() const { return fTypeVersionTo; }
   const std::string &GetTypeName() const { return fTypeName; }
   const std::string &GetContent() const { return fContent; }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleDescriptor
\ingroup NTuple
\brief The on-storage meta-data of an ntuple

Represents the on-disk (on storage) information about an ntuple. The meta-data consists of a header and one or
several footers. The header carries the ntuple schema, i.e. the fields and the associated columns and their
relationships. The footer(s) carry information about one or several clusters. For every cluster, a footer stores
its location and size, and for every column the range of element indexes as well as a list of pages and page
locations.

The descriptor provide machine-independent (de-)serialization of headers and footers, and it provides lookup routines
for ntuple objects (pages, clusters, ...).  It is supposed to be usable by all RPageStorage implementations.

The serialization does not use standard ROOT streamers in order to not let it depend on libCore. The serialization uses
the concept of frames: header, footer, and substructures have a preamble with version numbers and the size of the
writte struct. This allows for forward and backward compatibility when the meta-data evolves.
*/
// clang-format on
class RNTupleDescriptor {
   friend class Internal::RNTupleDescriptorBuilder;

public:
   class RHeaderExtension;

private:
   /// The ntuple name needs to be unique in a given storage location (file)
   std::string fName;
   /// Free text from the user
   std::string fDescription;

   std::uint64_t fOnDiskHeaderXxHash3 = 0; ///< Set by the descriptor builder when deserialized
   std::uint64_t fOnDiskHeaderSize = 0;    ///< Set by the descriptor builder when deserialized
   std::uint64_t fOnDiskFooterSize = 0; ///< Like fOnDiskHeaderSize, contains both cluster summaries and page locations

   std::uint64_t fNEntries = 0;         ///< Updated by the descriptor builder when the cluster groups are added
   std::uint64_t fNClusters = 0;        ///< Updated by the descriptor builder when the cluster groups are added
   std::uint64_t fNPhysicalColumns = 0; ///< Updated by the descriptor builder when columns are added

   DescriptorId_t fFieldZeroId = kInvalidDescriptorId; ///< Set by the descriptor builder

   /**
    * Once constructed by an RNTupleDescriptorBuilder, the descriptor is mostly immutable except for set of
    * active the page locations.  During the lifetime of the descriptor, page location information for clusters
    * can be added or removed.  When this happens, the generation should be increased, so that users of the
    * descriptor know that the information changed.  The generation is increased, e.g., by the page source's
    * exclusive lock guard around the descriptor.  It is used, e.g., by the descriptor cache in RNTupleReader.
    */
   std::uint64_t fGeneration = 0;

   std::set<unsigned int> fFeatureFlags;
   std::unordered_map<DescriptorId_t, RFieldDescriptor> fFieldDescriptors;
   std::unordered_map<DescriptorId_t, RColumnDescriptor> fColumnDescriptors;
   std::unordered_map<DescriptorId_t, RClusterGroupDescriptor> fClusterGroupDescriptors;
   /// May contain only a subset of all the available clusters, e.g. the clusters of the current file
   /// from a chain of files
   std::unordered_map<DescriptorId_t, RClusterDescriptor> fClusterDescriptors;
   std::vector<RExtraTypeInfoDescriptor> fExtraTypeInfoDescriptors;
   std::unique_ptr<RHeaderExtension> fHeaderExtension;

public:
   static constexpr unsigned int kFeatureFlagTest = 137; // Bit reserved for forward-compatibility testing

   // clang-format off
   /**
   \class ROOT::Experimental::RNTupleDescriptor::RHeaderExtension
   \ingroup NTuple
   \brief Summarizes information about fields and the corresponding columns that were added after the header has been serialized
   */
   // clang-format on
   class RHeaderExtension {
      friend class Internal::RNTupleDescriptorBuilder;

   private:
      /// Contains the list of field IDs that are part of the header extension; the corresponding columns are
      /// available via `GetColumnIterable()`.
      std::vector<DescriptorId_t> fFields;
      /// Number of logical and physical columns; updated by the descriptor builder when columns are added
      std::uint64_t fNLogicalColumns = 0;
      std::uint64_t fNPhysicalColumns = 0;

      void AddFieldId(DescriptorId_t id) { fFields.push_back(id); }
      void AddColumn(bool isAliasColumn)
      {
         fNLogicalColumns++;
         if (!isAliasColumn)
            fNPhysicalColumns++;
      }

   public:
      std::size_t GetNFields() const { return fFields.size(); }
      std::size_t GetNLogicalColumns() const { return fNLogicalColumns; }
      std::size_t GetNPhysicalColumns() const { return fNPhysicalColumns; }
      /// Return a vector containing the IDs of the top-level fields defined in the extension header
      std::vector<DescriptorId_t> GetTopLevelFields(const RNTupleDescriptor &desc) const;
   };

   // clang-format off
   /**
   \class ROOT::Experimental::RNTupleDescriptor::RColumnDescriptorIterable
   \ingroup NTuple
   \brief Used to loop over a field's associated columns
   */
   // clang-format on
   class RColumnDescriptorIterable {
   private:
      /// The associated NTuple for this range.
      const RNTupleDescriptor &fNTuple;
      /// The descriptor ids of the columns ordered by index id
      std::vector<DescriptorId_t> fColumns = {};

      void CollectColumnIds(DescriptorId_t fieldId);

   public:
      class RIterator {
      private:
         /// The enclosing range's NTuple.
         const RNTupleDescriptor &fNTuple;
         /// The enclosing range's descriptor id list.
         const std::vector<DescriptorId_t> &fColumns;
         std::size_t fIndex = 0;

      public:
         using iterator_category = std::forward_iterator_tag;
         using iterator = RIterator;
         using value_type = RFieldDescriptor;
         using difference_type = std::ptrdiff_t;
         using pointer = RColumnDescriptor *;
         using reference = const RColumnDescriptor &;

         RIterator(const RNTupleDescriptor &ntuple, const std::vector<DescriptorId_t> &columns, std::size_t index)
            : fNTuple(ntuple), fColumns(columns), fIndex(index)
         {
         }
         iterator operator++()
         {
            ++fIndex;
            return *this;
         }
         reference operator*() { return fNTuple.GetColumnDescriptor(fColumns.at(fIndex)); }
         bool operator!=(const iterator &rh) const { return fIndex != rh.fIndex; }
         bool operator==(const iterator &rh) const { return fIndex == rh.fIndex; }
      };

      RColumnDescriptorIterable(const RNTupleDescriptor &ntuple, const RFieldDescriptor &field);
      RColumnDescriptorIterable(const RNTupleDescriptor &ntuple);

      RIterator begin() { return RIterator(fNTuple, fColumns, 0); }
      RIterator end() { return RIterator(fNTuple, fColumns, fColumns.size()); }
      size_t count() { return fColumns.size(); }
   };

   // clang-format off
   /**
   \class ROOT::Experimental::RNTupleDescriptor::RFieldDescriptorIterable
   \ingroup NTuple
   \brief Used to loop over a field's child fields
   */
   // clang-format on
   class RFieldDescriptorIterable {
   private:
      /// The associated NTuple for this range.
      const RNTupleDescriptor &fNTuple;
      /// The descriptor ids of the child fields. These may be sorted using
      /// a comparison function.
      std::vector<DescriptorId_t> fFieldChildren = {};

   public:
      class RIterator {
      private:
         /// The enclosing range's NTuple.
         const RNTupleDescriptor &fNTuple;
         /// The enclosing range's descriptor id list.
         const std::vector<DescriptorId_t> &fFieldChildren;
         std::size_t fIndex = 0;

      public:
         using iterator_category = std::forward_iterator_tag;
         using iterator = RIterator;
         using value_type = RFieldDescriptor;
         using difference_type = std::ptrdiff_t;
         using pointer = RFieldDescriptor *;
         using reference = const RFieldDescriptor &;

         RIterator(const RNTupleDescriptor &ntuple, const std::vector<DescriptorId_t> &fieldChildren, std::size_t index)
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
                               const std::function<bool(DescriptorId_t, DescriptorId_t)> &comparator)
         : fNTuple(ntuple), fFieldChildren(field.GetLinkIds())
      {
         std::sort(fFieldChildren.begin(), fFieldChildren.end(), comparator);
      }
      RIterator begin() { return RIterator(fNTuple, fFieldChildren, 0); }
      RIterator end() { return RIterator(fNTuple, fFieldChildren, fFieldChildren.size()); }
   };

   // clang-format off
   /**
   \class ROOT::Experimental::RNTupleDescriptor::RClusterGroupDescriptorIterable
   \ingroup NTuple
   \brief Used to loop over all the cluster groups of an ntuple (in unspecified order)

   Enumerate all cluster group IDs from the cluster group descriptor.  No specific order can be assumed, use
   FindNextClusterGroupId and FindPrevClusterGroupId to traverse clusters groups by entry number.
   */
   // clang-format on
   class RClusterGroupDescriptorIterable {
   private:
      /// The associated NTuple for this range.
      const RNTupleDescriptor &fNTuple;

   public:
      class RIterator {
      private:
         /// The enclosing range's NTuple.
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
   \class ROOT::Experimental::RNTupleDescriptor::RClusterDescriptorIterable
   \ingroup NTuple
   \brief Used to loop over all the clusters of an ntuple (in unspecified order)

   Enumerate all cluster IDs from the cluster descriptor.  No specific order can be assumed, use
   FindNextClusterId and FindPrevClusterId to travers clusters by entry number.
   */
   // clang-format on
   class RClusterDescriptorIterable {
   private:
      /// The associated NTuple for this range.
      const RNTupleDescriptor &fNTuple;

   public:
      class RIterator {
      private:
         /// The enclosing range's NTuple.
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
   \class ROOT::Experimental::RNTupleDescriptor::RExtraTypeInfoDescriptorIterable
   \ingroup NTuple
   \brief Used to loop over all the extra type info record of an ntuple (in unspecified order)
   */
   // clang-format on
   class RExtraTypeInfoDescriptorIterable {
   private:
      /// The associated NTuple for this range.
      const RNTupleDescriptor &fNTuple;

   public:
      class RIterator {
      private:
         /// The enclosing range's NTuple.
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

   /// Modifiers passed to `CreateModel`
   struct RCreateModelOptions {
      RCreateModelOptions() {} // Work around compiler bug, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88165
      /// If set to true, projected fields will be reconstructed as such. This will prevent the model to be used
      /// with an RNTupleReader, but it is useful, e.g., to accurately merge data.
      bool fReconstructProjections = false;
   };

   RNTupleDescriptor() = default;
   RNTupleDescriptor(const RNTupleDescriptor &other) = delete;
   RNTupleDescriptor &operator=(const RNTupleDescriptor &other) = delete;
   RNTupleDescriptor(RNTupleDescriptor &&other) = default;
   RNTupleDescriptor &operator=(RNTupleDescriptor &&other) = default;

   std::unique_ptr<RNTupleDescriptor> Clone() const;

   bool operator==(const RNTupleDescriptor &other) const;

   std::uint64_t GetOnDiskHeaderXxHash3() const { return fOnDiskHeaderXxHash3; }
   std::uint64_t GetOnDiskHeaderSize() const { return fOnDiskHeaderSize; }
   std::uint64_t GetOnDiskFooterSize() const { return fOnDiskFooterSize; }

   const RFieldDescriptor &GetFieldDescriptor(DescriptorId_t fieldId) const { return fFieldDescriptors.at(fieldId); }
   const RColumnDescriptor &GetColumnDescriptor(DescriptorId_t columnId) const
   {
      return fColumnDescriptors.at(columnId);
   }
   const RClusterGroupDescriptor &GetClusterGroupDescriptor(DescriptorId_t clusterGroupId) const
   {
      return fClusterGroupDescriptors.at(clusterGroupId);
   }
   const RClusterDescriptor &GetClusterDescriptor(DescriptorId_t clusterId) const
   {
      return fClusterDescriptors.at(clusterId);
   }

   RFieldDescriptorIterable GetFieldIterable(const RFieldDescriptor &fieldDesc) const
   {
      return RFieldDescriptorIterable(*this, fieldDesc);
   }
   RFieldDescriptorIterable
   GetFieldIterable(const RFieldDescriptor &fieldDesc,
                    const std::function<bool(DescriptorId_t, DescriptorId_t)> &comparator) const
   {
      return RFieldDescriptorIterable(*this, fieldDesc, comparator);
   }
   RFieldDescriptorIterable GetFieldIterable(DescriptorId_t fieldId) const
   {
      return GetFieldIterable(GetFieldDescriptor(fieldId));
   }
   RFieldDescriptorIterable
   GetFieldIterable(DescriptorId_t fieldId, const std::function<bool(DescriptorId_t, DescriptorId_t)> &comparator) const
   {
      return GetFieldIterable(GetFieldDescriptor(fieldId), comparator);
   }
   RFieldDescriptorIterable GetTopLevelFields() const { return GetFieldIterable(GetFieldZeroId()); }
   RFieldDescriptorIterable
   GetTopLevelFields(const std::function<bool(DescriptorId_t, DescriptorId_t)> &comparator) const
   {
      return GetFieldIterable(GetFieldZeroId(), comparator);
   }

   RColumnDescriptorIterable GetColumnIterable() const { return RColumnDescriptorIterable(*this); }
   RColumnDescriptorIterable GetColumnIterable(const RFieldDescriptor &fieldDesc) const
   {
      return RColumnDescriptorIterable(*this, fieldDesc);
   }
   RColumnDescriptorIterable GetColumnIterable(DescriptorId_t fieldId) const
   {
      return RColumnDescriptorIterable(*this, GetFieldDescriptor(fieldId));
   }

   RClusterGroupDescriptorIterable GetClusterGroupIterable() const { return RClusterGroupDescriptorIterable(*this); }

   RClusterDescriptorIterable GetClusterIterable() const { return RClusterDescriptorIterable(*this); }

   RExtraTypeInfoDescriptorIterable GetExtraTypeInfoIterable() const { return RExtraTypeInfoDescriptorIterable(*this); }

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
   NTupleSize_t GetNEntries() const { return fNEntries; }
   NTupleSize_t GetNElements(DescriptorId_t physicalColumnId) const;

   /// Returns the logical parent of all top-level NTuple data fields.
   DescriptorId_t GetFieldZeroId() const { return fFieldZeroId; }
   const RFieldDescriptor &GetFieldZero() const { return GetFieldDescriptor(GetFieldZeroId()); }
   DescriptorId_t FindFieldId(std::string_view fieldName, DescriptorId_t parentId) const;
   /// Searches for a top-level field
   DescriptorId_t FindFieldId(std::string_view fieldName) const;
   DescriptorId_t FindLogicalColumnId(DescriptorId_t fieldId, std::uint32_t columnIndex) const;
   DescriptorId_t FindPhysicalColumnId(DescriptorId_t fieldId, std::uint32_t columnIndex) const;
   DescriptorId_t FindClusterId(DescriptorId_t physicalColumnId, NTupleSize_t index) const;
   DescriptorId_t FindNextClusterId(DescriptorId_t clusterId) const;
   DescriptorId_t FindPrevClusterId(DescriptorId_t clusterId) const;

   /// Walks up the parents of the field ID and returns a field name of the form a.b.c.d
   /// In case of invalid field ID, an empty string is returned.
   std::string GetQualifiedFieldName(DescriptorId_t fieldId) const;

   bool HasFeature(unsigned int flag) const { return fFeatureFlags.count(flag) > 0; }
   std::vector<std::uint64_t> GetFeatureFlags() const;

   /// Return header extension information; if the descriptor does not have a header extension, return `nullptr`
   const RHeaderExtension *GetHeaderExtension() const { return fHeaderExtension.get(); }

   /// Methods to load and drop cluster group details (cluster IDs and page locations)
   RResult<void> AddClusterGroupDetails(DescriptorId_t clusterGroupId, std::vector<RClusterDescriptor> &clusterDescs);
   RResult<void> DropClusterGroupDetails(DescriptorId_t clusterGroupId);

   std::uint64_t GetGeneration() const { return fGeneration; }
   void IncGeneration() { fGeneration++; }

   /// Re-create the C++ model from the stored meta-data
   std::unique_ptr<RNTupleModel> CreateModel(const RCreateModelOptions &options = RCreateModelOptions()) const;
   void PrintInfo(std::ostream &output) const;
};

namespace Internal {

// clang-format off
/**
\class ROOT::Experimental::Internal::RColumnDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RColumnDescriptor

Dangling column descriptors can become actual descriptors when added to an
RNTupleDescriptorBuilder instance and then linked to their fields.
*/
// clang-format on
class RColumnDescriptorBuilder {
private:
   RColumnDescriptor fColumn = RColumnDescriptor();

public:
   /// Make an empty column descriptor builder.
   RColumnDescriptorBuilder() = default;

   RColumnDescriptorBuilder &LogicalColumnId(DescriptorId_t logicalColumnId)
   {
      fColumn.fLogicalColumnId = logicalColumnId;
      return *this;
   }
   RColumnDescriptorBuilder &PhysicalColumnId(DescriptorId_t physicalColumnId)
   {
      fColumn.fPhysicalColumnId = physicalColumnId;
      return *this;
   }
   RColumnDescriptorBuilder &Type(EColumnType type)
   {
      fColumn.fType = type;
      return *this;
   }
   RColumnDescriptorBuilder &FieldId(DescriptorId_t fieldId)
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
   DescriptorId_t GetFieldId() const { return fColumn.fFieldId; }
   /// Attempt to make a column descriptor. This may fail if the column
   /// was not given enough information to make a proper descriptor.
   RResult<RColumnDescriptor> MakeDescriptor() const;
};

// clang-format off
/**
\class ROOT::Experimental::Internal::RFieldDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RFieldDescriptor

Dangling field descriptors describe a single field in isolation. They are
missing the necessary relationship information (parent field, any child fields)
required to describe a real NTuple field.

Dangling field descriptors can only become actual descriptors when added to an
RNTupleDescriptorBuilder instance and then linked to other fields.
*/
// clang-format on
class RFieldDescriptorBuilder {
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

   /// Make a new RFieldDescriptorBuilder based off a live NTuple field.
   static RFieldDescriptorBuilder FromField(const RFieldBase &field);

   RFieldDescriptorBuilder &FieldId(DescriptorId_t fieldId)
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
   RFieldDescriptorBuilder &ParentId(DescriptorId_t id)
   {
      fField.fParentId = id;
      return *this;
   }
   RFieldDescriptorBuilder &ProjectionSourceId(DescriptorId_t id)
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
   RFieldDescriptorBuilder &Structure(const ENTupleStructure &structure)
   {
      fField.fStructure = structure;
      return *this;
   }
   DescriptorId_t GetParentId() const { return fField.fParentId; }
   /// Attempt to make a field descriptor. This may fail if the dangling field
   /// was not given enough information to make a proper descriptor.
   RResult<RFieldDescriptor> MakeDescriptor() const;
};

// clang-format off
/**
\class ROOT::Experimental::Internal::RClusterDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RClusterDescriptor

The cluster descriptor builder starts from a summary-only cluster descriptor and allows for the
piecewise addition of page locations.
*/
// clang-format on
class RClusterDescriptorBuilder {
private:
   RClusterDescriptor fCluster;

public:
   RClusterDescriptorBuilder &ClusterId(DescriptorId_t clusterId)
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

   RResult<void> CommitColumnRange(DescriptorId_t physicalId, std::uint64_t firstElementIndex,
                                   std::uint32_t compressionSettings, const RClusterDescriptor::RPageRange &pageRange);

   /// Add column and page ranges for columns created during late model extension missing in this cluster.  The locator
   /// type for the synthesized page ranges is `kTypePageZero`.  All the page sources must be able to populate the
   /// 'zero' page from such locator. Any call to `CommitColumnRange()` should happen before calling this function.
   RClusterDescriptorBuilder &AddExtendedColumnRanges(const RNTupleDescriptor &desc);

   /// Move out the full cluster descriptor including page locations
   RResult<RClusterDescriptor> MoveDescriptor();
};

// clang-format off
/**
\class ROOT::Experimental::Internal::RClusterGroupDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RClusterGroupDescriptor
*/
// clang-format on
class RClusterGroupDescriptorBuilder {
private:
   RClusterGroupDescriptor fClusterGroup;

public:
   RClusterGroupDescriptorBuilder() = default;
   static RClusterGroupDescriptorBuilder FromSummary(const RClusterGroupDescriptor &clusterGroupDesc);

   RClusterGroupDescriptorBuilder &ClusterGroupId(DescriptorId_t clusterGroupId)
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
   void AddClusters(const std::vector<DescriptorId_t> &clusterIds)
   {
      if (clusterIds.size() != fClusterGroup.GetNClusters())
         throw RException(R__FAIL("mismatch of number of clusters"));
      fClusterGroup.fClusterIds = clusterIds;
   }

   RResult<RClusterGroupDescriptor> MoveDescriptor();
};

// clang-format off
/**
\class ROOT::Experimental::Internal::RColumnGroupDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RColumnGroupDescriptor
*/
// clang-format on
class RColumnGroupDescriptorBuilder {
private:
   RColumnGroupDescriptor fColumnGroup;

public:
   RColumnGroupDescriptorBuilder() = default;

   RColumnGroupDescriptorBuilder &ColumnGroupId(DescriptorId_t columnGroupId)
   {
      fColumnGroup.fColumnGroupId = columnGroupId;
      return *this;
   }
   void AddColumn(DescriptorId_t physicalId) { fColumnGroup.fPhysicalColumnIds.insert(physicalId); }

   RResult<RColumnGroupDescriptor> MoveDescriptor();
};

// clang-format off
/**
\class ROOT::Experimental::Internal::RExtraTypeInfoDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RExtraTypeInfoDescriptor
*/
// clang-format on
class RExtraTypeInfoDescriptorBuilder {
private:
   RExtraTypeInfoDescriptor fExtraTypeInfo;

public:
   RExtraTypeInfoDescriptorBuilder() = default;

   RExtraTypeInfoDescriptorBuilder &ContentId(EExtraTypeInfoIds contentId)
   {
      fExtraTypeInfo.fContentId = contentId;
      return *this;
   }
   RExtraTypeInfoDescriptorBuilder &TypeVersionFrom(std::uint32_t typeVersionFrom)
   {
      fExtraTypeInfo.fTypeVersionFrom = typeVersionFrom;
      return *this;
   }
   RExtraTypeInfoDescriptorBuilder &TypeVersionTo(std::uint32_t typeVersionTo)
   {
      fExtraTypeInfo.fTypeVersionTo = typeVersionTo;
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
\class ROOT::Experimental::Internal::RNTupleDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RNTupleDescriptor

Used by RPageStorage implementations in order to construct the RNTupleDescriptor from the various header parts.
*/
// clang-format on
class RNTupleDescriptorBuilder {
private:
   RNTupleDescriptor fDescriptor;
   RResult<void> EnsureFieldExists(DescriptorId_t fieldId) const;
   // Called by AddColumn() to populate the fLogicalFieldIds member of the field descriptor
   RResult<void> AttachColumn(DescriptorId_t fieldId, const RColumnDescriptor &columnDesc);

public:
   /// Checks whether invariants hold:
   /// * NTuple name is valid
   /// * Fields have valid parent and child ids
   RResult<void> EnsureValidDescriptor() const;
   const RNTupleDescriptor &GetDescriptor() const { return fDescriptor; }
   RNTupleDescriptor MoveDescriptor();

   void SetNTuple(const std::string_view name, const std::string_view description);
   void SetFeature(unsigned int flag);

   void SetOnDiskHeaderXxHash3(std::uint64_t xxhash3) { fDescriptor.fOnDiskHeaderXxHash3 = xxhash3; }
   void SetOnDiskHeaderSize(std::uint64_t size) { fDescriptor.fOnDiskHeaderSize = size; }
   /// The real footer size also include the page list envelopes
   void AddToOnDiskFooterSize(std::uint64_t size) { fDescriptor.fOnDiskFooterSize += size; }

   void AddField(const RFieldDescriptor &fieldDesc);
   RResult<void> AddFieldLink(DescriptorId_t fieldId, DescriptorId_t linkId);
   RResult<void> AddFieldProjection(DescriptorId_t sourceId, DescriptorId_t targetId);

   // The field that the column belongs to has to be already available. For fields with multiple columns,
   // the columns need to be added in order of the column index
   RResult<void> AddColumn(RColumnDescriptor &&columnDesc);

   RResult<void> AddClusterGroup(RClusterGroupDescriptor &&clusterGroup);
   RResult<void> AddCluster(RClusterDescriptor &&clusterDesc);

   RResult<void> AddExtraTypeInfo(RExtraTypeInfoDescriptor &&extraTypeInfoDesc);

   /// Clears so-far stored clusters, fields, and columns and return to a pristine ntuple descriptor
   void Reset();

   /// Mark the beginning of the header extension; any fields and columns added after a call to this function are
   /// annotated as begin part of the header extension.
   void BeginHeaderExtension();
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleDescriptor
