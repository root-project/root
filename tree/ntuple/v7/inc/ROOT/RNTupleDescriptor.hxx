/// \file ROOT/RNTupleDescriptor.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
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

#include <ROOT/RColumnModel.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RSpan.hxx>
#include <ROOT/RStringView.hxx>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ROOT {
namespace Experimental {

class RFieldDescriptorBuilder;
class RNTupleDescriptor;
class RNTupleDescriptorBuilder;
class RNTupleModel;

namespace Detail {
   class RFieldBase;
}


// clang-format off
/**
\class ROOT::Experimental::RFieldDescriptor
\ingroup NTuple
\brief Meta-data stored for every field of an ntuple
*/
// clang-format on
class RFieldDescriptor {
   friend class RNTupleDescriptorBuilder;
   friend class RFieldDescriptorBuilder;

private:
   DescriptorId_t fFieldId = kInvalidDescriptorId;
   /// The version of the C++-type-to-column translation mechanics
   RNTupleVersion fFieldVersion = RNTupleVersion();
   /// The version of the C++ type itself
   RNTupleVersion fTypeVersion = RNTupleVersion();
   /// The leaf name, not including parent fields
   std::string fFieldName;
   /// Free text set by the user
   std::string fFieldDescription;
   /// The C++ type that was used when writing the field
   std::string fTypeName;
   /// The number of elements per entry for fixed-size arrays
   std::uint64_t fNRepetitions = 0;
   /// The structural information carried by this field in the data model tree
   ENTupleStructure fStructure = ENTupleStructure::kInvalid;
   /// Establishes sub field relationships, such as classes and collections
   DescriptorId_t fParentId = kInvalidDescriptorId;
   /// The pointers in the other direction from parent to children. They are serialized, too, to keep the
   /// order of sub fields.
   std::vector<DescriptorId_t> fLinkIds;

public:
   RFieldDescriptor() = default;
   RFieldDescriptor(const RFieldDescriptor &other) = delete;
   RFieldDescriptor &operator =(const RFieldDescriptor &other) = delete;
   RFieldDescriptor(RFieldDescriptor &&other) = default;
   RFieldDescriptor &operator =(RFieldDescriptor &&other) = default;

   /// In order to handle changes to the serialization routine in future ntuple versions
   static constexpr std::uint16_t kFrameVersionCurrent = 0;
   static constexpr std::uint16_t kFrameVersionMin = 0;

   bool operator==(const RFieldDescriptor &other) const;
   /// Get a copy of the descriptor
   RFieldDescriptor Clone() const;
   /// In general, we create a field simply from the C++ type name. For untyped fields, however, we potentially need
   /// access to sub fields, which is provided by the ntuple descriptor argument.
   std::unique_ptr<Detail::RFieldBase> CreateField(const RNTupleDescriptor &ntplDesc) const;

   DescriptorId_t GetId() const { return fFieldId; }
   RNTupleVersion GetFieldVersion() const { return fFieldVersion; }
   RNTupleVersion GetTypeVersion() const { return fTypeVersion; }
   std::string GetFieldName() const { return fFieldName; }
   std::string GetFieldDescription() const { return fFieldDescription; }
   std::string GetTypeName() const { return fTypeName; }
   std::uint64_t GetNRepetitions() const { return fNRepetitions; }
   ENTupleStructure GetStructure() const { return fStructure; }
   DescriptorId_t GetParentId() const { return fParentId; }
   const std::vector<DescriptorId_t> &GetLinkIds() const { return fLinkIds; }
};


// clang-format off
/**
\class ROOT::Experimental::RColumnDescriptor
\ingroup NTuple
\brief Meta-data stored for every column of an ntuple
*/
// clang-format on
class RColumnDescriptor {
   friend class RColumnDescriptorBuilder;
   friend class RNTupleDescriptorBuilder;

private:
   DescriptorId_t fColumnId = kInvalidDescriptorId;
   /// Versions can change, e.g., when new column types are added
   RNTupleVersion fVersion;
   /// Contains the column type and whether it is sorted
   RColumnModel fModel;
   /// Every column belongs to one and only one field
   DescriptorId_t fFieldId = kInvalidDescriptorId;
   /// A field can be serialized into several columns, which are numbered from zero to $n$
   std::uint32_t fIndex;

public:
   /// In order to handle changes to the serialization routine in future ntuple versions
   static constexpr std::uint16_t kFrameVersionCurrent = 0;
   static constexpr std::uint16_t kFrameVersionMin = 0;

   RColumnDescriptor() = default;
   RColumnDescriptor(const RColumnDescriptor &other) = delete;
   RColumnDescriptor &operator =(const RColumnDescriptor &other) = delete;
   RColumnDescriptor(RColumnDescriptor &&other) = default;
   RColumnDescriptor &operator =(RColumnDescriptor &&other) = default;

   bool operator==(const RColumnDescriptor &other) const;
   /// Get a copy of the descriptor
   RColumnDescriptor Clone() const;

   DescriptorId_t GetId() const { return fColumnId; }
   RNTupleVersion GetVersion() const { return fVersion; }
   RColumnModel GetModel() const { return fModel; }
   std::uint32_t GetIndex() const { return fIndex; }
   DescriptorId_t GetFieldId() const { return fFieldId; }
};


// clang-format off
/**
\class ROOT::Experimental::RClusterDescriptor
\ingroup NTuple
\brief Meta-data for a set of ntuple clusters

The cluster descriptor might carry information of only a subset of available clusters, for instance if multiple
files are chained and not all of them have been processed yet. Clusters usually span across all available columns but
in some cases they can describe only a subset of the columns, for instance when describing friend ntuples.
*/
// clang-format on
class RClusterDescriptor {
   friend class RNTupleDescriptorBuilder;
   friend class RClusterDescriptorBuilder;

public:
   /// The window of element indexes of a particular column in a particular cluster
   struct RColumnRange {
      DescriptorId_t fColumnId = kInvalidDescriptorId;
      /// A 64bit element index
      NTupleSize_t fFirstElementIndex = kInvalidNTupleIndex;
      /// A 32bit value for the number of column elements in the cluster
      ClusterSize_t fNElements = kInvalidClusterIndex;
      /// The usual format for ROOT compression settings (see Compression.h).
      /// The pages of a particular column in a particular cluster are all compressed with the same settings.
      std::int64_t fCompressionSettings = 0;

      // TODO(jblomer): we perhaps want to store summary information, such as average, min/max, etc.
      // Should this be done on the field level?

      bool operator==(const RColumnRange &other) const {
         return fColumnId == other.fColumnId && fFirstElementIndex == other.fFirstElementIndex &&
                fNElements == other.fNElements && fCompressionSettings == other.fCompressionSettings;
      }

      bool Contains(NTupleSize_t index) const {
         return (fFirstElementIndex <= index && (fFirstElementIndex + fNElements) > index);
      }
   };

   /// Records the parition of data into pages for a particular column in a particular cluster
   struct RPageRange {
      /// We do not need to store the element size / uncompressed page size because we know to which column
      /// the page belongs
      struct RPageInfo {
         /// The sum of the elements of all the pages must match the corresponding fNElements field in fColumnRanges
         ClusterSize_t fNElements = kInvalidClusterIndex;
         /// The meaning of fLocator depends on the storage backend.
         RNTupleLocator fLocator;

         bool operator==(const RPageInfo &other) const {
            return fNElements == other.fNElements && fLocator == other.fLocator;
         }
      };
      struct RPageInfoExtended : RPageInfo {
         /// Index (in cluster) of the first element in page.
         RClusterSize::ValueType fFirstInPage;
         /// Page number in the corresponding RPageRange.
         NTupleSize_t fPageNo;

         RPageInfoExtended(const RPageInfo &pi, RClusterSize::ValueType i, NTupleSize_t n)
            : RPageInfo(pi), fFirstInPage(i), fPageNo(n) {}
      };

      RPageRange() = default;
      RPageRange(const RPageRange &other) = delete;
      RPageRange &operator =(const RPageRange &other) = delete;
      RPageRange(RPageRange &&other) = default;
      RPageRange &operator =(RPageRange &&other) = default;

      RPageRange Clone() const {
         RPageRange clone;
         clone.fColumnId = fColumnId;
         clone.fPageInfos = fPageInfos;
         return clone;
      }

      /// Find the page in the RPageRange that contains the given element. The element must exist.
      RPageInfoExtended Find(RClusterSize::ValueType idxInCluster) const;

      DescriptorId_t fColumnId = kInvalidDescriptorId;
      std::vector<RPageInfo> fPageInfos;

      bool operator==(const RPageRange &other) const {
         return fColumnId == other.fColumnId && fPageInfos == other.fPageInfos;
      }
   };

private:
   DescriptorId_t fClusterId = kInvalidDescriptorId;
   /// Future versions of the cluster descriptor might add more meta-data, e.g. a semantic checksum
   RNTupleVersion fVersion;
   /// Clusters can be swapped by adjusting the entry offsets
   NTupleSize_t fFirstEntryIndex = kInvalidNTupleIndex;
   ClusterSize_t fNEntries = kInvalidClusterIndex;

   std::unordered_map<DescriptorId_t, RColumnRange> fColumnRanges;
   std::unordered_map<DescriptorId_t, RPageRange> fPageRanges;

public:
   /// In order to handle changes to the serialization routine in future ntuple versions
   static constexpr std::uint16_t kFrameVersionCurrent = 0;
   static constexpr std::uint16_t kFrameVersionMin = 0;

   RClusterDescriptor() = default;
   RClusterDescriptor(const RClusterDescriptor &other) = delete;
   RClusterDescriptor &operator =(const RClusterDescriptor &other) = delete;
   RClusterDescriptor(RClusterDescriptor &&other) = default;
   RClusterDescriptor &operator =(RClusterDescriptor &&other) = default;

   bool operator==(const RClusterDescriptor &other) const;

   DescriptorId_t GetId() const { return fClusterId; }
   RNTupleVersion GetVersion() const { return fVersion; }
   NTupleSize_t GetFirstEntryIndex() const { return fFirstEntryIndex; }
   ClusterSize_t GetNEntries() const { return fNEntries; }
   const RColumnRange &GetColumnRange(DescriptorId_t columnId) const { return fColumnRanges.at(columnId); }
   const RPageRange &GetPageRange(DescriptorId_t columnId) const { return fPageRanges.at(columnId); }
   bool ContainsColumn(DescriptorId_t columnId) const;
   std::unordered_set<DescriptorId_t> GetColumnIds() const;
   std::uint64_t GetBytesOnStorage() const;
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
   friend class RNTupleDescriptorBuilder;

private:
   /// The ntuple name needs to be unique in a given storage location (file)
   std::string fName;
   /// Free text from the user
   std::string fDescription;
   /// The origin of the data
   std::string fAuthor;
   /// The current responsible for storing the data
   std::string fCustodian;
   /// The time stamp of the ntuple data (immutable)
   std::chrono::system_clock::time_point fTimeStampData;
   /// The time stamp of writing the data to storage, which gets updated when re-written
   std::chrono::system_clock::time_point fTimeStampWritten;
   /// The version evolves with the ntuple summary meta-data
   RNTupleVersion fVersion;
   /// Every NTuple gets a unique identifier
   RNTupleUuid fOwnUuid;
   /// Column sets that are created as derived sets from existing NTuples share the same group id.
   /// NTuples in the same group have the same number of entries and are supposed to contain associated data.
   RNTupleUuid fGroupUuid;

   std::uint64_t fOnDiskHeaderSize = 0; ///< Set by the descriptor builder when deserialized
   std::uint64_t fOnDiskFooterSize = 0; ///< Like fOnDiskHeaderSize, contains both cluster summaries and page locations

   std::unordered_map<DescriptorId_t, RFieldDescriptor> fFieldDescriptors;
   std::unordered_map<DescriptorId_t, RColumnDescriptor> fColumnDescriptors;
   /// May contain only a subset of all the available clusters, e.g. the clusters of the current file
   /// from a chain of files
   std::unordered_map<DescriptorId_t, RClusterDescriptor> fClusterDescriptors;

public:
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
            : fNTuple(ntuple), fColumns(columns), fIndex(index) {}
         iterator operator++() { ++fIndex; return *this; }
         reference operator*() { return fNTuple.GetColumnDescriptor(fColumns.at(fIndex)); }
         bool operator!=(const iterator &rh) const { return fIndex != rh.fIndex; }
         bool operator==(const iterator &rh) const { return fIndex == rh.fIndex; }
      };

      RColumnDescriptorIterable(const RNTupleDescriptor &ntuple, const RFieldDescriptor &field)
         : fNTuple(ntuple)
      {
         for (unsigned int i = 0; true; ++i) {
            auto columnId = ntuple.FindColumnId(field.GetId(), i);
            if (columnId == kInvalidDescriptorId)
               break;
            fColumns.emplace_back(columnId);
         }
      }
      RIterator begin() { return RIterator(fNTuple, fColumns, 0); }
      RIterator end() { return RIterator(fNTuple, fColumns, fColumns.size()); }
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
      const RNTupleDescriptor& fNTuple;
      /// The descriptor ids of the child fields. These may be sorted using
      /// a comparison function.
      std::vector<DescriptorId_t> fFieldChildren = {};
   public:
      class RIterator {
      private:
         /// The enclosing range's NTuple.
         const RNTupleDescriptor& fNTuple;
         /// The enclosing range's descriptor id list.
         const std::vector<DescriptorId_t>& fFieldChildren;
         std::size_t fIndex = 0;
      public:
         using iterator_category = std::forward_iterator_tag;
         using iterator = RIterator;
         using value_type = RFieldDescriptor;
         using difference_type = std::ptrdiff_t;
         using pointer = RFieldDescriptor*;
         using reference = const RFieldDescriptor&;

         RIterator(const RNTupleDescriptor& ntuple, const std::vector<DescriptorId_t>& fieldChildren,
            std::size_t index) : fNTuple(ntuple), fFieldChildren(fieldChildren), fIndex(index) {}
         iterator operator++() { ++fIndex; return *this; }
         reference operator*() {
            return fNTuple.GetFieldDescriptor(
               fFieldChildren.at(fIndex)
            );
         }
         bool operator!=(const iterator& rh) const { return fIndex != rh.fIndex; }
         bool operator==(const iterator& rh) const { return fIndex == rh.fIndex; }
      };
      RFieldDescriptorIterable(const RNTupleDescriptor& ntuple, const RFieldDescriptor& field)
         : fNTuple(ntuple), fFieldChildren(field.GetLinkIds()) {}
      /// Sort the range using an arbitrary comparison function.
      RFieldDescriptorIterable(const RNTupleDescriptor& ntuple, const RFieldDescriptor& field,
         const std::function<bool(DescriptorId_t, DescriptorId_t)>& comparator)
         : fNTuple(ntuple), fFieldChildren(field.GetLinkIds())
      {
         std::sort(fFieldChildren.begin(), fFieldChildren.end(), comparator);
      }
      RIterator begin() {
         return RIterator(fNTuple, fFieldChildren, 0);
      }
      RIterator end() {
         return RIterator(fNTuple, fFieldChildren, fFieldChildren.size());
      }
   };

   // clang-format off
   /**
   \class ROOT::Experimental::RNTupleDescriptor::RClusterDescriptorIterable
   \ingroup NTuple
   \brief Used to loop over all the clusters of an ntuple (in unspecified order)

   Enumerate all cluster IDs from the cluster descriptor.  No specific order can be assumed, use
   FindNextClusterId and FindPrevClusterId to travers clusters by entry number.
   TODO(jblomer): review naming of *Range classes and possibly rename consistently to *Iterable
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
         iterator operator++() { ++fIndex; return *this; }
         reference operator*() {
            auto it = fNTuple.fClusterDescriptors.begin();
            std::advance(it, fIndex);
            return it->second;
         }
         bool operator!=(const iterator &rh) const { return fIndex != rh.fIndex; }
         bool operator==(const iterator &rh) const { return fIndex == rh.fIndex; }
      };

      RClusterDescriptorIterable(const RNTupleDescriptor &ntuple) : fNTuple(ntuple) { }
      RIterator begin() { return RIterator(fNTuple, 0); }
      RIterator end() { return RIterator(fNTuple, fNTuple.GetNClusters()); }
   };

   /// In order to handle changes to the serialization routine in future ntuple versions
   static constexpr std::uint16_t kFrameVersionCurrent = 0;
   static constexpr std::uint16_t kFrameVersionMin = 0;
   /// The preamble is sufficient to get the length of the header
   static constexpr unsigned int kNBytesPreamble = 8;
   /// The last few bytes after the footer store the length of footer and header
   static constexpr unsigned int kNBytesPostscript = 16;

   RNTupleDescriptor() = default;
   RNTupleDescriptor(const RNTupleDescriptor &other) = delete;
   RNTupleDescriptor &operator=(const RNTupleDescriptor &other) = delete;
   RNTupleDescriptor(RNTupleDescriptor &&other) = default;
   RNTupleDescriptor &operator=(RNTupleDescriptor &&other) = default;

   bool operator ==(const RNTupleDescriptor &other) const;

   /// We deliberately do not use ROOT's built-in serialization in order to allow for use of RNTuple's without libCore
   /// Serializes the global ntuple information as well as the column and field schemata
   /// Returns the number of bytes and fills buffer if it is not nullptr.
   /// TODO(jblomer): instead of runtime testing for nullptr, there should be a template for the case where
   /// only the size of the buffer is required.
   std::uint32_t SerializeHeader(void* buffer) const;
   /// Serializes cluster meta data. Returns the number of bytes and fills buffer if it is not nullptr.
   std::uint32_t SerializeFooter(void* buffer) const;
   /// Given kNBytesPostscript bytes, extract the header and footer lengths in bytes
   static void LocateMetadata(const void *postscript, std::uint32_t &szHeader, std::uint32_t &szFooter);

   std::uint32_t GetHeaderSize() const {
      return SerializeHeader(nullptr);
   }
   std::uint32_t GetFooterSize() const {
      return SerializeFooter(nullptr);
   }

   std::uint64_t GetOnDiskHeaderSize() const { return fOnDiskHeaderSize; }
   std::uint64_t GetOnDiskFooterSize() const { return fOnDiskFooterSize; }

   const RFieldDescriptor& GetFieldDescriptor(DescriptorId_t fieldId) const {
      return fFieldDescriptors.at(fieldId);
   }
   const RColumnDescriptor& GetColumnDescriptor(DescriptorId_t columnId) const {
      return fColumnDescriptors.at(columnId);
   }
   const RClusterDescriptor& GetClusterDescriptor(DescriptorId_t clusterId) const {
      return fClusterDescriptors.at(clusterId);
   }

   RFieldDescriptorIterable GetFieldIterable(const RFieldDescriptor& fieldDesc) const {
      return RFieldDescriptorIterable(*this, fieldDesc);
   }
   RFieldDescriptorIterable GetFieldIterable(const RFieldDescriptor& fieldDesc,
      const std::function<bool(DescriptorId_t, DescriptorId_t)>& comparator) const
   {
      return RFieldDescriptorIterable(*this, fieldDesc, comparator);
   }
   RFieldDescriptorIterable GetFieldIterable(DescriptorId_t fieldId) const {
      return GetFieldIterable(GetFieldDescriptor(fieldId));
   }
   RFieldDescriptorIterable GetFieldIterable(DescriptorId_t fieldId,
      const std::function<bool(DescriptorId_t, DescriptorId_t)>& comparator) const
   {
      return GetFieldIterable(GetFieldDescriptor(fieldId), comparator);
   }
   RFieldDescriptorIterable GetTopLevelFields() const {
      return GetFieldIterable(GetFieldZeroId());
   }
   RFieldDescriptorIterable GetTopLevelFields(
      const std::function<bool(DescriptorId_t, DescriptorId_t)>& comparator) const
   {
      return GetFieldIterable(GetFieldZeroId(), comparator);
   }

   RColumnDescriptorIterable GetColumnIterable(const RFieldDescriptor &fieldDesc) const
   {
      return RColumnDescriptorIterable(*this, fieldDesc);
   }
   RColumnDescriptorIterable GetColumnIterable(DescriptorId_t fieldId) const
   {
      return RColumnDescriptorIterable(*this, GetFieldDescriptor(fieldId));
   }

   RClusterDescriptorIterable GetClusterIterable() const
   {
      return RClusterDescriptorIterable(*this);
   }

   std::string GetName() const { return fName; }
   std::string GetDescription() const { return fDescription; }
   std::string GetAuthor() const { return fAuthor; }
   std::string GetCustodian() const { return fCustodian; }
   std::chrono::system_clock::time_point GetTimeStampData() const { return fTimeStampData; }
   std::chrono::system_clock::time_point GetTimeStampWritten() const { return fTimeStampWritten; }
   RNTupleVersion GetVersion() const { return fVersion; }
   RNTupleUuid GetOwnUuid() const { return fOwnUuid; }
   RNTupleUuid GetGroupUuid() const { return fGroupUuid; }

   std::size_t GetNFields() const { return fFieldDescriptors.size(); }
   std::size_t GetNColumns() const { return fColumnDescriptors.size(); }
   std::size_t GetNClusters() const { return fClusterDescriptors.size(); }

   // The number of entries as seen with the currently loaded cluster meta-data; there might be more
   NTupleSize_t GetNEntries() const;
   NTupleSize_t GetNElements(DescriptorId_t columnId) const;

   /// Returns the logical parent of all top-level NTuple data fields.
   DescriptorId_t GetFieldZeroId() const;
   const RFieldDescriptor &GetFieldZero() const { return GetFieldDescriptor(GetFieldZeroId()); }
   DescriptorId_t FindFieldId(std::string_view fieldName, DescriptorId_t parentId) const;
   /// Searches for a top-level field
   DescriptorId_t FindFieldId(std::string_view fieldName) const;
   DescriptorId_t FindColumnId(DescriptorId_t fieldId, std::uint32_t columnIndex) const;
   DescriptorId_t FindClusterId(DescriptorId_t columnId, NTupleSize_t index) const;
   DescriptorId_t FindNextClusterId(DescriptorId_t clusterId) const;
   DescriptorId_t FindPrevClusterId(DescriptorId_t clusterId) const;

   /// Walks up the parents of the field ID and returns a field name of the form a.b.c.d
   /// In case of invalid field ID, an empty string is returned.
   std::string GetQualifiedFieldName(DescriptorId_t fieldId) const;

   /// Re-create the C++ model from the stored meta-data
   std::unique_ptr<RNTupleModel> GenerateModel() const;
   void PrintInfo(std::ostream &output) const;
};


// clang-format off
/**
\class ROOT::Experimental::RColumnDescriptorBuilder
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

   RColumnDescriptorBuilder& ColumnId(DescriptorId_t columnId) {
      fColumn.fColumnId = columnId;
      return *this;
   }
   RColumnDescriptorBuilder& Model(const RColumnModel &model) {
      fColumn.fModel = model;
      return *this;
   }
   RColumnDescriptorBuilder& FieldId(DescriptorId_t fieldId) {
      fColumn.fFieldId = fieldId;
      return *this;
   }
   RColumnDescriptorBuilder& Index(std::uint32_t index) {
      fColumn.fIndex = index;
      return *this;
   }
   DescriptorId_t GetFieldId() const { return fColumn.fFieldId; }
   /// Attempt to make a column descriptor. This may fail if the column
   /// was not given enough information to make a proper descriptor.
   RResult<RColumnDescriptor> MakeDescriptor() const;
};


// clang-format off
/**
\class ROOT::Experimental::RFieldDescriptorBuilder
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
   explicit RFieldDescriptorBuilder(const RFieldDescriptor& fieldDesc);

   /// Make a new RFieldDescriptorBuilder based off a live NTuple field.
   static RFieldDescriptorBuilder FromField(const Detail::RFieldBase& field);

   RFieldDescriptorBuilder& FieldId(DescriptorId_t fieldId) {
      fField.fFieldId = fieldId;
      return *this;
   }
   RFieldDescriptorBuilder& FieldVersion(const RNTupleVersion& fieldVersion) {
      fField.fFieldVersion = fieldVersion;
      return *this;
   }
   RFieldDescriptorBuilder& TypeVersion(const RNTupleVersion& typeVersion) {
      fField.fTypeVersion = typeVersion;
      return *this;
   }
   RFieldDescriptorBuilder& ParentId(DescriptorId_t id) {
      fField.fParentId = id;
      return *this;
   }
   RFieldDescriptorBuilder& FieldName(const std::string& fieldName) {
      fField.fFieldName = fieldName;
      return *this;
   }
   RFieldDescriptorBuilder& FieldDescription(const std::string& fieldDescription) {
      fField.fFieldDescription = fieldDescription;
      return *this;
   }
   RFieldDescriptorBuilder& TypeName(const std::string& typeName) {
      fField.fTypeName = typeName;
      return *this;
   }
   RFieldDescriptorBuilder& NRepetitions(std::uint64_t nRepetitions) {
      fField.fNRepetitions = nRepetitions;
      return *this;
   }
   RFieldDescriptorBuilder& Structure(const ENTupleStructure& structure) {
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
\class ROOT::Experimental::RClusterDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RClusterDescriptor

Dangling cluster descriptors can become actual descriptors when added to an
RNTupleDescriptorBuilder instance.
*/
// clang-format on
class RClusterDescriptorBuilder {
private:
   RClusterDescriptor fCluster = RClusterDescriptor();
public:
   /// Make an empty cluster descriptor builder.
   RClusterDescriptorBuilder() = default;

   RClusterDescriptorBuilder& ClusterId(DescriptorId_t clusterId) {
      fCluster.fClusterId = clusterId;
      return *this;
   }
   RClusterDescriptorBuilder& FirstEntryIndex(std::uint64_t firstEntryIndex) {
      fCluster.fFirstEntryIndex = firstEntryIndex;
      return *this;
   }
   RClusterDescriptorBuilder& NEntries(std::uint64_t nEntries) {
      fCluster.fNEntries = nEntries;
      return *this;
   }

   RResult<void> CommitColumnRange(DescriptorId_t columnId,
                                   std::uint64_t firstElementIndex,
                                   std::uint32_t compressionSettings,
                                   const RClusterDescriptor::RPageRange &pageRange);

   /// Attempt to make a cluster descriptor. This may fail if the cluster
   /// was not given enough information to make a proper descriptor.
   RResult<RClusterDescriptor> MoveDescriptor();
};


// clang-format off
/**
\class ROOT::Experimental::RNTupleDescriptorBuilder
\ingroup NTuple
\brief A helper class for piece-wise construction of an RNTupleDescriptor

Used by RPageStorage implementations in order to construct the RNTupleDescriptor from the various header parts.
*/
// clang-format on
class RNTupleDescriptorBuilder {
private:
   RNTupleDescriptor fDescriptor;
   std::uint32_t fHeaderCRC32 = 0;
   std::vector<Internal::RNTupleSerializer::RClusterSummary> fClusterSummaries;
   std::vector<Internal::RNTupleSerializer::RClusterGroup> fClusterGroups;

   RResult<void> EnsureFieldExists(DescriptorId_t fieldId) const;
public:
   /// Checks whether invariants hold:
   /// * NTuple name is valid
   /// * Fields have valid parent and child ids
   RResult<void> EnsureValidDescriptor() const;
   const RNTupleDescriptor& GetDescriptor() const { return fDescriptor; }
   RNTupleDescriptor MoveDescriptor();

   void SetNTuple(const std::string_view name, const std::string_view description, const std::string_view author,
                  const RNTupleVersion &version, const RNTupleUuid &uuid);
   void SetHeaderCRC32(std::uint32_t crc32) { fHeaderCRC32 = crc32; }
   std::uint32_t GetHeaderCRC32() const { return fHeaderCRC32; }

   void SetOnDiskHeaderSize(std::uint64_t size) { fDescriptor.fOnDiskHeaderSize = size; }
   void SetOnDiskFooterSize(std::uint64_t size) { fDescriptor.fOnDiskFooterSize = size; }

   void AddField(const RFieldDescriptor& fieldDesc);
   RResult<void> AddFieldLink(DescriptorId_t fieldId, DescriptorId_t linkId);

   void AddColumn(DescriptorId_t columnId, DescriptorId_t fieldId,
                  const RNTupleVersion &version, const RColumnModel &model, std::uint32_t index);
   RResult<void> AddColumn(RColumnDescriptor &&columnDesc);

   void SetFromHeader(void* headerBuffer);

   void AddCluster(DescriptorId_t clusterId, RNTupleVersion version,
                   NTupleSize_t firstEntryIndex, ClusterSize_t nEntries);
   void AddClusterColumnRange(DescriptorId_t clusterId, const RClusterDescriptor::RColumnRange &columnRange);
   void AddClusterPageRange(DescriptorId_t clusterId, RClusterDescriptor::RPageRange &&pageRange);

   void AddClustersFromFooter(void* footerBuffer);

   void AddClusterSummary(Internal::RNTupleSerializer::RClusterSummary &clusterSummary);
   void AddClusterGroup(Internal::RNTupleSerializer::RClusterGroup &clusterGroup);
   Internal::RNTupleSerializer::RClusterGroup GetClusterGroup(std::uint32_t id) const { return fClusterGroups.at(id); }
   RResult<void> AddCluster(DescriptorId_t clusterId, RClusterDescriptorBuilder &&partialCluster);

   /// Clears so-far stored clusters, fields, and columns and return to a pristine ntuple descriptor
   void Reset();
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleDescriptor
