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
   RForestVersion fFieldVersion;
   RForestVersion fTypeVersion;
   /// The leaf name, not including parent fields
   std::string fFieldName;
   /// The C++ type that was used when writing the field
   std::string fTypeName;
   /// The structural information carried by this field in the data model tree
   EForestStructure fStructure;
   /// Establishes sub field trees, such as classes and collections
   DescriptorId_t fParentId = kInvalidDescriptorId;
   /// For pointers and optional/variant fields, the pointee field(s)
   std::vector<DescriptorId_t> fLinkIds;

public:
   DescriptorId_t GetId() const { return fFieldId; }
   RForestVersion GetFieldVersion() const { return fFieldVersion; }
   RForestVersion GetTypeVersion() const { return fTypeVersion; }
   std::string GetFieldName() const { return fFieldName; }
   std::string GetTypeName() const { return fTypeName; }
   EForestStructure GetStructure() const { return fStructure; }
   DescriptorId_t GetParentId() const { return fParentId; }
   std::vector<DescriptorId_t> GetLinkIds() const { return fLinkIds; }
};


class RColumnDescriptor {
   friend class RNTupleDescriptorBuilder;

private:
   DescriptorId_t fColumnId = kInvalidDescriptorId;;
   RForestVersion fVersion;
   RColumnModel fModel;
   /// Every column belongs to one and only one field
   DescriptorId_t fFieldId = kInvalidDescriptorId;;
   /// Pointer to the parent column with offsets
   DescriptorId_t fOffsetId = kInvalidDescriptorId;;
   /// For index and offset columns of collections, pointers and variants, the pointee field(s)
   std::vector<DescriptorId_t> fLinkIds;

public:
   DescriptorId_t GetId() const { return fColumnId; }
   RForestVersion GetVersion() const { return fVersion; }
   RColumnModel GetModel() const { return fModel; }
   DescriptorId_t GetFieldId() const { return fFieldId; }
   DescriptorId_t GetOffsetId() const { return fOffsetId; }
   std::vector<DescriptorId_t> GetLinkIds() { return fLinkIds; }
};


class RClusterDescriptor {
   friend class RNTupleDescriptorBuilder;

public:
   struct RColumnInfo {
      DescriptorId_t fColumnId = kInvalidDescriptorId;
      ForestSize_t fFirstElementIndex = kInvalidForestIndex;
      ClusterSize_t fNElements = kInvalidClusterIndex;
   };

private:
   DescriptorId_t fClusterId = kInvalidDescriptorId;;
   RForestVersion fVersion;
   ForestSize_t fFirstEntryIndex = kInvalidForestIndex;
   ClusterSize_t fNEntries = kInvalidClusterIndex;
   std::unordered_map<DescriptorId_t, RColumnInfo> fColumnInfos;

public:
   DescriptorId_t GetId() const { return fClusterId; }
   RForestVersion GetVersion() const { return fVersion; }
   ForestSize_t GetFirstEntryIndex() const { return fFirstEntryIndex; }
   ClusterSize_t GetNEntries() const { return fNEntries; }
   RColumnInfo GetColumnInfo(DescriptorId_t columnId) const { return fColumnInfos.at(columnId); }
};


/**
 * Represents the on-disk (on storage) information about an ntuple.  This can, for instance, be used
 * by 3rd party utilies.
 */
class RNTupleDescriptor {
   friend class RNTupleDescriptorBuilder;

private:
   RForestVersion fVersion;
   std::string fName;

   std::unordered_map<DescriptorId_t, RFieldDescriptor> fFieldDescriptors;
   std::unordered_map<DescriptorId_t, RColumnDescriptor> fColumnDescriptors;
   std::unordered_map<DescriptorId_t, RClusterDescriptor> fClusterDescriptors;

public:
   const RFieldDescriptor& GetFieldDescriptor(DescriptorId_t fieldId) const { return fFieldDescriptors.at(fieldId); }
   const RColumnDescriptor& GetColumnDescriptor(DescriptorId_t columnId) const {
      return fColumnDescriptors.at(columnId);
   }
   const RClusterDescriptor& GetClusterDescriptor(DescriptorId_t clusterId) const {
      return fClusterDescriptors.at(clusterId);
   }
   std::string GetName() const { return fName; }
};


/**
 * Used by RPageStorage implementations in order to construct the RNTupleDescriptor from the various header parts.
 */
class RNTupleDescriptorBuilder {
private:
   RNTupleDescriptor fDescriptor;

public:
   const RNTupleDescriptor& GetDescriptor() const { return fDescriptor; }

   void SetForest(std::string_view name, const RForestVersion &version);

   void AddField(DescriptorId_t fieldId, const RForestVersion &fieldVersion, const RForestVersion &typeVersion,
                 std::string_view fieldName, std::string_view typeName, EForestStructure structure);
   void SetFieldParent(DescriptorId_t fieldId, DescriptorId_t parentId);
   void AddFieldLink(DescriptorId_t fieldId, DescriptorId_t linkId);

   void AddColumn(DescriptorId_t columnId, DescriptorId_t fieldId,
                  const RForestVersion &version, const RColumnModel &model);
   void SetColumnOffset(DescriptorId_t columnId, DescriptorId_t offsetId);
   void AddColumnLink(DescriptorId_t columnId, DescriptorId_t linkId);

   void AddCluster(DescriptorId_t clusterId, RForestVersion version,
                   ForestSize_t firstEntryIndex, ClusterSize_t nEntries);
   void AddClusterColumnInfo(DescriptorId_t clusterId, const RClusterDescriptor::RColumnInfo &columnInfo);
};

} // namespace Experimental
} // namespace ROOT

#endif
