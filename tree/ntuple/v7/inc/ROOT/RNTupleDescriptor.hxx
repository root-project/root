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

#include <vector>
#include <string>
#include <unordered_map>

namespace ROOT {
namespace Experimental {

class RNTupleDescriptorBuilder;

class RFieldDescriptor {
   friend class RNTupleDescriptorBuilder;

private:
   DescriptorId_t fFieldId = kInvalidDescriptorId;
   RNTupleVersion fFieldVersion;
   RNTupleVersion fTypeVersion;
   /// The leaf name, not including parent fields
   std::string fFieldName;
   /// The C++ type that was used when writing the field
   std::string fTypeName;
   /// The structural information carried by this field in the data model tree
   ENTupleStructure fStructure;
   /// Establishes sub field relationships, such as classes and collections
   DescriptorId_t fParentId = kInvalidDescriptorId;
   /// For pointers and optional/variant fields, the pointee field(s)
   std::vector<DescriptorId_t> fLinkIds;

public:
   bool operator==(const RFieldDescriptor &other) const;

   DescriptorId_t GetId() const { return fFieldId; }
   RNTupleVersion GetFieldVersion() const { return fFieldVersion; }
   RNTupleVersion GetTypeVersion() const { return fTypeVersion; }
   std::string GetFieldName() const { return fFieldName; }
   std::string GetTypeName() const { return fTypeName; }
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
   DescriptorId_t fFieldId = kInvalidDescriptorId;;
   /// Pointer to the parent column with offsets
   DescriptorId_t fOffsetId = kInvalidDescriptorId;;
   /// For index and offset columns of collections, pointers and variants, the pointee field(s)
   std::vector<DescriptorId_t> fLinkIds;

public:
   bool operator==(const RColumnDescriptor &other) const;

   DescriptorId_t GetId() const { return fColumnId; }
   RNTupleVersion GetVersion() const { return fVersion; }
   RColumnModel GetModel() const { return fModel; }
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
   };

private:
   DescriptorId_t fClusterId = kInvalidDescriptorId;;
   RNTupleVersion fVersion;
   NTupleSize_t fFirstEntryIndex = kInvalidNTupleIndex;
   ClusterSize_t fNEntries = kInvalidClusterIndex;
   std::unordered_map<DescriptorId_t, RColumnRange> fColumnRanges;

public:
   bool operator==(const RClusterDescriptor &other) const;

   DescriptorId_t GetId() const { return fClusterId; }
   RNTupleVersion GetVersion() const { return fVersion; }
   NTupleSize_t GetFirstEntryIndex() const { return fFirstEntryIndex; }
   ClusterSize_t GetNEntries() const { return fNEntries; }
   RColumnRange GetColumnRanges(DescriptorId_t columnId) const { return fColumnRanges.at(columnId); }
};


/**
 * Represents the on-disk (on storage) information about an ntuple.  This can, for instance, be used
 * by 3rd party utilies.
 */
class RNTupleDescriptor {
   friend class RNTupleDescriptorBuilder;

private:
   /// In order to handle changes to the serialization routine in future ntuple versions
   static constexpr std::uint32_t kByteProtocol = 0;

   std::string fName;
   RNTupleVersion fVersion;
   /// Every NTuple gets a unique identifier
   Uuid_t fOwnUuid;
   /// Column sets that are created as derived sets from existing NTuples share the same group id.
   /// NTuples in the same group have the same number of entries and are supposed to contain associated data.
   Uuid_t fGroupUuid;

   std::unordered_map<DescriptorId_t, RFieldDescriptor> fFieldDescriptors;
   std::unordered_map<DescriptorId_t, RColumnDescriptor> fColumnDescriptors;
   /// May contain only a subset of all the available clusters, e.g. the clusters of the current file
   /// from a chain of files
   std::unordered_map<DescriptorId_t, RClusterDescriptor> fClusterDescriptors;

public:
   bool operator ==(const RNTupleDescriptor &other) const;

   /// We deliberately do not use ROOT's built-in serialization in order to allow for use of RNTuple's outside ROOT
   /**
    * Serializes the global ntuple information as well as the column and field schemata
    * Returns the number of bytes and fills buffer if it is not nullptr.
    */
   std::uint32_t SerializeHeader(void* buffer);
   /**
    * Serializes cluster meta data. Returns the number of bytes and fills buffer if it is not nullptr.
    */
   std::uint32_t SerializeFooter(void* buffer);

   const RFieldDescriptor& GetFieldDescriptor(DescriptorId_t fieldId) const { return fFieldDescriptors.at(fieldId); }
   const RColumnDescriptor& GetColumnDescriptor(DescriptorId_t columnId) const {
      return fColumnDescriptors.at(columnId);
   }
   const RClusterDescriptor& GetClusterDescriptor(DescriptorId_t clusterId) const {
      return fClusterDescriptors.at(clusterId);
   }
   std::string GetName() const { return fName; }
   RNTupleVersion GetVersion() const { return fVersion; }
   Uuid_t GetOwnUuid() const { return fOwnUuid; }
   Uuid_t GetGroupUuid() const { return fGroupUuid; }
};


/**
 * Used by RPageStorage implementations in order to construct the RNTupleDescriptor from the various header parts.
 */
class RNTupleDescriptorBuilder {
private:
   RNTupleDescriptor fDescriptor;

public:
   const RNTupleDescriptor& GetDescriptor() const { return fDescriptor; }

   void SetNTuple(const std::string_view &name, const RNTupleVersion &version, const Uuid_t &uuid);

   void AddField(DescriptorId_t fieldId, const RNTupleVersion &fieldVersion, const RNTupleVersion &typeVersion,
                 std::string_view fieldName, std::string_view typeName, ENTupleStructure structure);
   void SetFieldParent(DescriptorId_t fieldId, DescriptorId_t parentId);
   void AddFieldLink(DescriptorId_t fieldId, DescriptorId_t linkId);

   void AddColumn(DescriptorId_t columnId, DescriptorId_t fieldId,
                  const RNTupleVersion &version, const RColumnModel &model);
   void SetColumnOffset(DescriptorId_t columnId, DescriptorId_t offsetId);
   void AddColumnLink(DescriptorId_t columnId, DescriptorId_t linkId);

   void SetFromHeader(void* headerBuffer);

   void AddCluster(DescriptorId_t clusterId, RNTupleVersion version,
                   NTupleSize_t firstEntryIndex, ClusterSize_t nEntries);
   void AddClusterColumnRange(DescriptorId_t clusterId, const RClusterDescriptor::RColumnRange &columnRange);

   void AddClustersFromFooter(void* footerBuffer);
};

} // namespace Experimental
} // namespace ROOT

#endif
