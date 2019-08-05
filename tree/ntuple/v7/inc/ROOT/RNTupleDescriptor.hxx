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
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RStringView.hxx>

#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace ROOT {
namespace Experimental {

class RNTupleDescriptorBuilder;
class RNTupleModel;

class RFieldDescriptor {
   friend class RNTupleDescriptorBuilder;

private:
   DescriptorId_t fFieldId = kInvalidDescriptorId;
   RNTupleVersion fFieldVersion;
   RNTupleVersion fTypeVersion;
   /// The leaf name, not including parent fields
   std::string fFieldName;
   /// Free text set by the user
   std::string fFieldDescription;
   /// The C++ type that was used when writing the field
   std::string fTypeName;
   /// The number of elements for fixed-size arrays
   std::uint64_t fNRepetitions;
   /// The structural information carried by this field in the data model tree
   ENTupleStructure fStructure;
   /// Establishes sub field relationships, such as classes and collections
   DescriptorId_t fParentId = kInvalidDescriptorId;
   /// For pointers and optional/variant fields, the pointee field(s)
   std::vector<DescriptorId_t> fLinkIds;

public:
   /// In order to handle changes to the serialization routine in future ntuple versions
   static constexpr std::uint16_t kFrameVersionCurrent = 0;
   static constexpr std::uint16_t kFrameVersionMin = 0;

   bool operator==(const RFieldDescriptor &other) const;

   DescriptorId_t GetId() const { return fFieldId; }
   RNTupleVersion GetFieldVersion() const { return fFieldVersion; }
   RNTupleVersion GetTypeVersion() const { return fTypeVersion; }
   std::string GetFieldName() const { return fFieldName; }
   std::string GetFieldDescription() const { return fFieldDescription; }
   std::string GetTypeName() const { return fTypeName; }
   std::uint64_t GetNRepetitions() const { return fNRepetitions; }
   ENTupleStructure GetStructure() const { return fStructure; }
   DescriptorId_t GetParentId() const { return fParentId; }
   std::vector<DescriptorId_t> GetLinkIds() const { return fLinkIds; }
};


class RColumnDescriptor {
   friend class RNTupleDescriptorBuilder;

private:
   DescriptorId_t fColumnId = kInvalidDescriptorId;;
   RNTupleVersion fVersion;
   RColumnModel fModel;
   /// Every column belongs to one and only one field
   DescriptorId_t fFieldId = kInvalidDescriptorId;
   /// A field can be serialized into several columns, which are numbered from zero to $n$
   std::uint32_t fIndex;
   /// Pointer to the parent column with offsets
   DescriptorId_t fOffsetId = kInvalidDescriptorId;
   /// For index and offset columns of collections, pointers and variants, the pointee field(s)
   std::vector<DescriptorId_t> fLinkIds;

public:
   /// In order to handle changes to the serialization routine in future ntuple versions
   static constexpr std::uint16_t kFrameVersionCurrent = 0;
   static constexpr std::uint16_t kFrameVersionMin = 0;

   bool operator==(const RColumnDescriptor &other) const;

   DescriptorId_t GetId() const { return fColumnId; }
   RNTupleVersion GetVersion() const { return fVersion; }
   RColumnModel GetModel() const { return fModel; }
   std::uint32_t GetIndex() const { return fIndex; }
   DescriptorId_t GetFieldId() const { return fFieldId; }
   DescriptorId_t GetOffsetId() const { return fOffsetId; }
   std::vector<DescriptorId_t> GetLinkIds() const { return fLinkIds; }
};


class RClusterDescriptor {
   friend class RNTupleDescriptorBuilder;

public:
   struct RColumnRange {
      DescriptorId_t fColumnId = kInvalidDescriptorId;
      NTupleSize_t fFirstElementIndex = kInvalidNTupleIndex;
      ClusterSize_t fNElements = kInvalidClusterIndex;
      // TODO(jblomer): we perhaps want to store summary information, such as average, min/max, etc.
      // Should this be done on the field level?

      bool operator==(const RColumnRange &other) const {
         return fColumnId == other.fColumnId && fFirstElementIndex == other.fFirstElementIndex &&
                fNElements == other.fNElements;
      }

      bool Contains(NTupleSize_t index) const {
         return (fFirstElementIndex <= index && (fFirstElementIndex + fNElements) > index);
      }
   };

   struct RPageRange {
      /// We do not need to store the element size / page size because we know to which column
      /// the page belongs
      struct RPageInfo {
         /// The sum of the elements of all the pages must match the corresponding fNElements field in fColumnRanges
         ClusterSize_t fNElements = kInvalidClusterIndex;
         /// The meaning of fLocator depends on the storage backend.  It indicates where on the storage
         /// medium the page resides.  For file based storage, for instance, it can be the offset in the file.
         std::int64_t fLocator = 0;

         bool operator==(const RPageInfo &other) const {
            return fNElements == other.fNElements && fLocator == other.fLocator;
         }
      };

      DescriptorId_t fColumnId = kInvalidDescriptorId;
      std::vector<RPageInfo> fPageInfos;

      bool operator==(const RPageRange &other) const {
         return fColumnId == other.fColumnId && fPageInfos == other.fPageInfos;
      }
   };

private:
   DescriptorId_t fClusterId = kInvalidDescriptorId;;
   RNTupleVersion fVersion;
   NTupleSize_t fFirstEntryIndex = kInvalidNTupleIndex;
   ClusterSize_t fNEntries = kInvalidClusterIndex;
   std::unordered_map<DescriptorId_t, RColumnRange> fColumnRanges;
   std::unordered_map<DescriptorId_t, RPageRange> fPageRanges;

public:
   /// In order to handle changes to the serialization routine in future ntuple versions
   static constexpr std::uint16_t kFrameVersionCurrent = 0;
   static constexpr std::uint16_t kFrameVersionMin = 0;

   bool operator==(const RClusterDescriptor &other) const;

   DescriptorId_t GetId() const { return fClusterId; }
   RNTupleVersion GetVersion() const { return fVersion; }
   NTupleSize_t GetFirstEntryIndex() const { return fFirstEntryIndex; }
   ClusterSize_t GetNEntries() const { return fNEntries; }
   RColumnRange GetColumnRange(DescriptorId_t columnId) const { return fColumnRanges.at(columnId); }
   RPageRange GetPageRange(DescriptorId_t columnId) const { return fPageRanges.at(columnId); }
};


/**
 * Represents the on-disk (on storage) information about an ntuple.  This can, for instance, be used
 * by 3rd party utilies.
 */
class RNTupleDescriptor {
   friend class RNTupleDescriptorBuilder;

private:
   std::string fName;
   std::string fDescription;
   std::string fAuthor;
   std::string fCustodian;
   std::chrono::system_clock::time_point fTimeStampData;
   std::chrono::system_clock::time_point fTimeStampWritten;
   RNTupleVersion fVersion;
   /// Every NTuple gets a unique identifier
   RNTupleUuid fOwnUuid;
   /// Column sets that are created as derived sets from existing NTuples share the same group id.
   /// NTuples in the same group have the same number of entries and are supposed to contain associated data.
   RNTupleUuid fGroupUuid;

   std::unordered_map<DescriptorId_t, RFieldDescriptor> fFieldDescriptors;
   std::unordered_map<DescriptorId_t, RColumnDescriptor> fColumnDescriptors;
   /// May contain only a subset of all the available clusters, e.g. the clusters of the current file
   /// from a chain of files
   std::unordered_map<DescriptorId_t, RClusterDescriptor> fClusterDescriptors;

public:
   /// In order to handle changes to the serialization routine in future ntuple versions
   static constexpr std::uint16_t kFrameVersionCurrent = 0;
   static constexpr std::uint16_t kFrameVersionMin = 0;

   bool operator ==(const RNTupleDescriptor &other) const;

   // We deliberately do not use ROOT's built-in serialization in order to allow for use of RNTuple's outside ROOT
   /**
    * Serializes the global ntuple information as well as the column and field schemata
    * Returns the number of bytes and fills buffer if it is not nullptr.
    */
   std::uint32_t SerializeHeader(void* buffer) const;
   /**
    * Serializes cluster meta data. Returns the number of bytes and fills buffer if it is not nullptr.
    */
   std::uint32_t SerializeFooter(void* buffer) const;

   const RFieldDescriptor& GetFieldDescriptor(DescriptorId_t fieldId) const { return fFieldDescriptors.at(fieldId); }
   const RColumnDescriptor& GetColumnDescriptor(DescriptorId_t columnId) const {
      return fColumnDescriptors.at(columnId);
   }
   const RClusterDescriptor& GetClusterDescriptor(DescriptorId_t clusterId) const {
      return fClusterDescriptors.at(clusterId);
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

   // Note that this is the number of entries as seen with the currently loaded cluster meta-data; there might be more
   NTupleSize_t GetNEntries() const;
   NTupleSize_t GetNElements(DescriptorId_t columnId) const;

   DescriptorId_t FindFieldId(std::string_view fieldName, DescriptorId_t parentId) const;
   DescriptorId_t FindColumnId(DescriptorId_t fieldId, std::uint32_t columnIndex) const;
   DescriptorId_t FindClusterId(DescriptorId_t columnId, NTupleSize_t index) const;

   /// Re-create the C++ model from the stored meta-data
   std::unique_ptr<RNTupleModel> GenerateModel() const;
};


/**
 * Used by RPageStorage implementations in order to construct the RNTupleDescriptor from the various header parts.
 */
class RNTupleDescriptorBuilder {
private:
   RNTupleDescriptor fDescriptor;

public:
   bool IsValid() const { return true; /* TODO(jblomer) */}
   const RNTupleDescriptor& GetDescriptor() const { return fDescriptor; }

   void SetNTuple(const std::string_view name, const std::string_view description, const std::string_view author,
                  const RNTupleVersion &version, const RNTupleUuid &uuid);

   void AddField(DescriptorId_t fieldId, const RNTupleVersion &fieldVersion, const RNTupleVersion &typeVersion,
                 std::string_view fieldName, std::string_view typeName, std::uint64_t nRepetitions,
                 ENTupleStructure structure);
   void SetFieldParent(DescriptorId_t fieldId, DescriptorId_t parentId);
   void AddFieldLink(DescriptorId_t fieldId, DescriptorId_t linkId);

   void AddColumn(DescriptorId_t columnId, DescriptorId_t fieldId,
                  const RNTupleVersion &version, const RColumnModel &model, std::uint32_t index);
   void SetColumnOffset(DescriptorId_t columnId, DescriptorId_t offsetId);
   void AddColumnLink(DescriptorId_t columnId, DescriptorId_t linkId);

   void SetFromHeader(void* headerBuffer);

   void AddCluster(DescriptorId_t clusterId, RNTupleVersion version,
                   NTupleSize_t firstEntryIndex, ClusterSize_t nEntries);
   void AddClusterColumnRange(DescriptorId_t clusterId, const RClusterDescriptor::RColumnRange &columnRange);
   void AddClusterPageRange(DescriptorId_t clusterId, const RClusterDescriptor::RPageRange &pageRange);

   void AddClustersFromFooter(void* footerBuffer);
};

} // namespace Experimental
} // namespace ROOT

#endif
