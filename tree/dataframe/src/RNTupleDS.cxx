/// \file RNTupleDS.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Enrico Guiraud <enrico.guiraud@cern.ch>
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

#include <ROOT/RDF/RColumnReaderBase.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleDS.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPageStorage.hxx>
#include <string_view>

#include <TError.h>

#include <string>
#include <vector>
#include <typeinfo>
#include <utility>

// clang-format off
/**
* \class ROOT::Experimental::RNTupleDS
* \ingroup dataframe
* \brief The RDataSource implementation for RNTuple. It lets RDataFrame read RNTuple data.
*
* An RDataFrame that reads RNTuple data can be constructed using FromRNTuple().
*
* For each column containing an array or a collection, a corresponding column `#colname` is available to access
* `colname.size()` without reading and deserializing the collection values.
*
**/
// clang-format on

namespace ROOT {
namespace Experimental {
namespace Internal {

/// An artificial field that transforms an RNTuple column that contains the offset of collections into
/// collection sizes. It is used to provide the "number of" RDF columns for collections, e.g.
/// `R_rdf_sizeof_jets` for a collection named `jets`.
///
/// This field owns the collection offset field but instead of exposing the collection offsets it exposes
/// the collection sizes (offset(N+1) - offset(N)).  For the time being, we offer this functionality only in RDataFrame.
/// TODO(jblomer): consider providing a general set of useful virtual fields as part of RNTuple.
class RRDFCardinalityField : public ROOT::Experimental::Detail::RFieldBase {
protected:
   std::unique_ptr<ROOT::Experimental::Detail::RFieldBase> CloneImpl(std::string_view /* newName */) const final
   {
      return std::make_unique<RRDFCardinalityField>();
   }
   void GenerateValue(void *where) const final { *static_cast<std::size_t *>(where) = 0; }

public:
   static std::string TypeName() { return "std::size_t"; }
   RRDFCardinalityField()
      : ROOT::Experimental::Detail::RFieldBase("", TypeName(), ENTupleStructure::kLeaf, false /* isSimple */) {}
   RRDFCardinalityField(RRDFCardinalityField &&other) = default;
   RRDFCardinalityField &operator=(RRDFCardinalityField &&other) = default;
   ~RRDFCardinalityField() = default;

   const RColumnRepresentations &GetColumnRepresentations() const final
   {
      static RColumnRepresentations representations(
         {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
         {});
      return representations;
   }
   // Field is only used for reading
   void GenerateColumnsImpl() final { assert(false && "Cardinality fields must only be used for reading"); }
   void GenerateColumnsImpl(const RNTupleDescriptor &desc) final
   {
      auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
      fColumns.emplace_back(
         ROOT::Experimental::Detail::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
   }

   size_t GetValueSize() const final { return sizeof(std::size_t); }
   size_t GetAlignment() const final { return alignof(std::size_t); }

   /// Get the number of elements of the collection identified by globalIndex
   void ReadGlobalImpl(ROOT::Experimental::NTupleSize_t globalIndex, void *to) final
   {
      RClusterIndex collectionStart;
      ClusterSize_t size;
      fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &size);
      *static_cast<std::size_t *>(to) = size;
   }

   /// Get the number of elements of the collection identified by clusterIndex
   void ReadInClusterImpl(const ROOT::Experimental::RClusterIndex &clusterIndex, void *to) final
   {
      RClusterIndex collectionStart;
      ClusterSize_t size;
      fPrincipalColumn->GetCollectionInfo(clusterIndex, &collectionStart, &size);
      *static_cast<std::size_t *>(to) = size;
   }
};

/// Every RDF column is represented by exactly one RNTuple field
class RNTupleColumnReader : public ROOT::Detail::RDF::RColumnReaderBase {
   using RFieldBase = ROOT::Experimental::Detail::RFieldBase;
   using RPageSource = ROOT::Experimental::Detail::RPageSource;

   RFieldBase *fProtoField;                    ///< The prototype field from which fField is cloned
   std::unique_ptr<RFieldBase> fField;         ///< The field backing the RDF column
   std::unique_ptr<RFieldBase::RValue> fValue; ///< The memory location used to read from fField
   unsigned int fSlot;       ///< Allows SwitchFile() to find the correct page source for reconnecting the reader
   Long64_t fLastEntry = -1; ///< Last entry number that was read
   /// For chains, the logical entry and the physical entry in a particular file are different.
   /// The entry offset stores the logical entry number (sum of all previous entries) when the current file was opened.
   ULong64_t fEntryOffset = 0;

public:
   RNTupleColumnReader(RFieldBase *protoField, unsigned int slot)
      : fProtoField(protoField),
        fField(protoField->Clone(protoField->GetName())),
        fValue(std::make_unique<RFieldBase::RValue>(fField->GenerateValue())),
        fSlot(slot)
   {
   }
   ~RNTupleColumnReader() = default;

   /// Connect the field and its subfields to the page source
   void Connect(RPageSource &source)
   {
      fField->ConnectPageSource(source);
      for (auto &f : *fField)
         f.ConnectPageSource(source);
   }

   void Disconnect()
   {
      fValue = nullptr;
      fField = nullptr;
   }

   void Reconnect(RPageSource &source, ULong64_t entryOffset)
   {
      fEntryOffset = entryOffset;
      fField = fProtoField->Clone(fProtoField->GetName());
      fValue = std::make_unique<RFieldBase::RValue>(fField->GenerateValue());
      Connect(source);
   }

   unsigned int GetSlot() const { return fSlot; }
   ROOT::Experimental::Detail::RFieldBase *GetProtoField() const { return fProtoField; }

   void *GetImpl(Long64_t entry) final
   {
      if (entry != fLastEntry) {
         fValue->Read(entry - fEntryOffset);
         fLastEntry = entry;
      }
      return fValue->GetRawPtr();
   }
};

} // namespace Internal

RNTupleDS::~RNTupleDS() = default;

void RNTupleDS::AddField(const RNTupleDescriptor &desc, std::string_view colName, DescriptorId_t fieldId,
                         std::vector<DescriptorId_t> skeinIDs)
{
   // As an example for the mapping of RNTuple fields to RDF columns, let's consider an RNTuple
   // using the following types and with a top-level field named "event" of type Event:
   //
   // struct Event {
   //    int id;
   //    std::vector<Track> tracks;
   // };
   // struct Track {
   //    std::vector<Hit> hits;
   // };
   // struct Hit {
   //    float x;
   //    float y;
   // };
   //
   // AddField() will be called from the constructor with the RNTuple root field (ENTupleStructure::kRecord).
   // From there, we recurse into the "event" sub field (also ENTupleStructure::kRecord) and further down the
   // tree of sub fields and expose the following RDF columns:
   //
   // "event"                             [Event]
   // "event.id"                          [int]
   // "event.tracks"                      [RVec<Track>]
   // "R_rdf_sizeof_event.tracks"         [unsigned int]
   // "event.tracks.hits"                 [RVec<RVec<Hit>>]
   // "R_rdf_sizeof_event.tracks.hits"    [RVec<unsigned int>]
   // "event.tracks.hits.x"               [RVec<RVec<float>>]
   // "R_rdf_sizeof_event.tracks.hits.x"  [RVec<unsigned int>]
   // "event.tracks.hits.y"               [RVec<RVec<float>>]
   // "R_rdf_sizeof_event.tracks.hits.y"  [RVec<unsigned int>]

   const auto &fieldDesc = desc.GetFieldDescriptor(fieldId);
   if (fieldDesc.GetStructure() == ENTupleStructure::kCollection) {
      // Inner fields of collections are provided as projected collections of only that inner field,
      // E.g. we provide a projected collection RVec<RVec<float>> for "event.tracks.hits.x" in the example
      // above.

      // We open a new collection scope with fieldID being the inner most collection. E.g. for "event.tracks.hits",
      // skeinIDs would already contain the fieldID of "event.tracks"
      skeinIDs.emplace_back(fieldId);

      if (fieldDesc.GetTypeName().empty()) {
         // Anonymous collection with one or several sub fields
         auto cardinalityField = std::make_unique<ROOT::Experimental::Internal::RRDFCardinalityField>();
         cardinalityField->SetOnDiskId(fieldId);
         fColumnNames.emplace_back("R_rdf_sizeof_" + std::string(colName));
         fColumnTypes.emplace_back(cardinalityField->GetType());
         fFieldPrototypes.emplace_back(std::move(cardinalityField));

         for (const auto &f : desc.GetFieldIterable(fieldDesc.GetId())) {
            AddField(desc, std::string(colName) + "." + f.GetFieldName(), f.GetId(), skeinIDs);
         }
      } else {
         // ROOT::RVec with exactly one sub field
         const auto &f = *desc.GetFieldIterable(fieldDesc.GetId()).begin();
         AddField(desc, colName, f.GetId(), skeinIDs);
      }
      // Note that at the end of the recursion, we handled the inner sub collections as well as the
      // collection as whole, so we are done.
      return;
   } else if (fieldDesc.GetStructure() == ENTupleStructure::kRecord) {
      // Inner fields of records are provided as individual RDF columns, e.g. "event.id"
      for (const auto &f : desc.GetFieldIterable(fieldDesc.GetId())) {
         auto innerName = colName.empty() ? f.GetFieldName() : (std::string(colName) + "." + f.GetFieldName());
         AddField(desc, innerName, f.GetId(), skeinIDs);
      }
   }

   // The fieldID could be the root field or the class of fieldId might not be loaded.
   // In these cases, only the inner fields are exposed as RDF columns.
   auto fieldOrException = Detail::RFieldBase::Create(fieldDesc.GetFieldName(), fieldDesc.GetTypeName());
   if (!fieldOrException)
      return;
   auto valueField = fieldOrException.Unwrap();
   valueField->SetOnDiskId(fieldId);
   for (auto &f : *valueField) {
      f.SetOnDiskId(desc.FindFieldId(f.GetName(), f.GetParent()->GetOnDiskId()));
   }
   std::unique_ptr<Detail::RFieldBase> cardinalityField;
   // Collections get the additional "number of" RDF column (e.g. "R_rdf_sizeof_tracks")
   if (!skeinIDs.empty()) {
      cardinalityField = std::make_unique<ROOT::Experimental::Internal::RRDFCardinalityField>();
      cardinalityField->SetOnDiskId(skeinIDs.back());
   }

   for (auto i = skeinIDs.rbegin(); i != skeinIDs.rend(); ++i) {
      valueField = std::make_unique<ROOT::Experimental::RRVecField>("", std::move(valueField));
      valueField->SetOnDiskId(*i);
      // Skip the inner-most collection level to construct the cardinality column
      if (i != skeinIDs.rbegin()) {
         cardinalityField = std::make_unique<ROOT::Experimental::RRVecField>("", std::move(cardinalityField));
         cardinalityField->SetOnDiskId(*i);
      }
   }

   if (cardinalityField) {
      fColumnNames.emplace_back("R_rdf_sizeof_" + std::string(colName));
      fColumnTypes.emplace_back(cardinalityField->GetType());
      fFieldPrototypes.emplace_back(std::move(cardinalityField));
   }

   skeinIDs.emplace_back(fieldId);
   fColumnNames.emplace_back(colName);
   fColumnTypes.emplace_back(valueField->GetType());
   fFieldPrototypes.emplace_back(std::move(valueField));
}

RNTupleDS::RNTupleDS(std::unique_ptr<Detail::RPageSource> pageSource)
{
   pageSource->Attach();
   auto descriptorGuard = pageSource->GetSharedDescriptorGuard();
   fSources.emplace_back(std::move(pageSource));

   AddField(descriptorGuard.GetRef(), "", descriptorGuard->GetFieldZeroId(), std::vector<DescriptorId_t>());
}

RNTupleDS::RNTupleDS(std::string_view ntupleName, const std::vector<std::string> &fileNames)
   : RNTupleDS(Detail::RPageSource::Create(ntupleName, fileNames[0]))
{
   fNTupleName = ntupleName;
   fFileNames = fileNames;
}

RDF::RDataSource::Record_t RNTupleDS::GetColumnReadersImpl(std::string_view /* name */, const std::type_info & /* ti */)
{
   // This datasource uses the GetColumnReaders2 API instead (better name in the works)
   return {};
}

std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
RNTupleDS::GetColumnReaders(unsigned int slot, std::string_view name, const std::type_info & /*tid*/)
{
   // at this point we can assume that `name` will be found in fColumnNames, RDF is in charge validation
   // TODO(jblomer): check incoming type
   const auto index = std::distance(fColumnNames.begin(), std::find(fColumnNames.begin(), fColumnNames.end(), name));
   auto field = fFieldPrototypes[index].get();
   auto reader = std::make_unique<Internal::RNTupleColumnReader>(field, slot);
   reader->Connect(*fSources[slot]);
   fActiveColumnReaders.emplace_back(reader.get());

   if (fFieldId2QualifiedName.count(field->GetOnDiskId()) == 0) {
      auto descGuard = fSources[slot]->GetSharedDescriptorGuard();
      fFieldId2QualifiedName[field->GetOnDiskId()] = descGuard->GetQualifiedFieldName(field->GetOnDiskId());
      for (const auto &s : *field) {
         if (fFieldId2QualifiedName.count(s.GetOnDiskId()) == 0) {
            fFieldId2QualifiedName[s.GetOnDiskId()] = descGuard->GetQualifiedFieldName(s.GetOnDiskId());
         }
      }
   }

   return reader;
}

bool RNTupleDS::SetEntry(unsigned int, ULong64_t)
{
   return true;
}

void RNTupleDS::SwitchFile()
{
   if (fFileNames.size() <= 1)
      return;

   for (auto r : fActiveColumnReaders) {
      r->Disconnect();
   }

   fSources[0] = Detail::RPageSource::Create(fNTupleName, fFileNames[fNextFileIndex]);
   fSources[0]->Attach();
   for (unsigned int i = 1; i < fNSlots; ++i) {
      fSources[i] = fSources[0]->Clone();
      fSources[i]->Attach();
   }

   // Reset field IDs of active field prototypes
   std::unordered_map<ROOT::Experimental::DescriptorId_t, ROOT::Experimental::DescriptorId_t> fFieldIdMap;
   {
      auto descGuard = fSources[0]->GetSharedDescriptorGuard();
      for (const auto &[oldId, qualifiedName] : fFieldId2QualifiedName) {
         fFieldIdMap[oldId] = descGuard->FindFieldId(qualifiedName);
      }
   }
   for (auto &p : fFieldPrototypes) {
      if (fFieldId2QualifiedName.count(p->GetOnDiskId()) == 0)
         continue;
      p->SetOnDiskId(fFieldIdMap[p->GetOnDiskId()]);
      for (auto &s : *p) {
         s.SetOnDiskId(fFieldIdMap[s.GetOnDiskId()]);
      }
   }
   // Update fFieldId2QualifiedName with new field IDs
   std::unordered_map<ROOT::Experimental::DescriptorId_t, std::string> newFieldId2QualifiedName;
   for (auto [oldId, newId] : fFieldIdMap) {
      newFieldId2QualifiedName[newId] = fFieldId2QualifiedName[oldId];
   }
   std::swap(newFieldId2QualifiedName, fFieldId2QualifiedName);

   for (auto r : fActiveColumnReaders) {
      r->Reconnect(*fSources[r->GetSlot()], fSeenEntries);
   }
}

std::vector<std::pair<ULong64_t, ULong64_t>> RNTupleDS::GetEntryRanges()
{
   std::vector<std::pair<ULong64_t, ULong64_t>> rangesBySlot;
   if (fHasSeenAllRanges)
      return rangesBySlot;

   // The first file is opened in the constructor or in Initialize()
   if (fNextFileIndex != 0)
      SwitchFile();

   std::vector<std::pair<ULong64_t, ULong64_t>> rangesByCluster;
   {
      auto descriptorGuard = fSources[0]->GetSharedDescriptorGuard();
      auto clusterId = descriptorGuard->FindClusterId(0, 0);
      while (clusterId != kInvalidDescriptorId) {
         const auto &clusterDesc = descriptorGuard->GetClusterDescriptor(clusterId);
         rangesByCluster.emplace_back(std::make_pair<ULong64_t, ULong64_t>(
            clusterDesc.GetFirstEntryIndex(), clusterDesc.GetFirstEntryIndex() + clusterDesc.GetNEntries()));
         clusterId = descriptorGuard->FindNextClusterId(clusterId);
      }
   }

   // Distribute slots equidistantly over the entry range, aligned on cluster boundaries
   const unsigned int nRangesByCluster = rangesByCluster.size();
   const auto chunkSize = nRangesByCluster / fNSlots;
   const auto remainder = nRangesByCluster % fNSlots;
   std::size_t iRange = 0;
   const unsigned int N = std::min(fNSlots, nRangesByCluster);
   for (unsigned int i = 0; i < N; ++i) {
      auto start = rangesByCluster[iRange].first;
      iRange += chunkSize + static_cast<int>(i < remainder);
      R__ASSERT(iRange > 0);
      auto end = rangesByCluster[iRange - 1].second;
      rangesBySlot.emplace_back(start + fSeenEntries, end + fSeenEntries);

      fSources[i]->SetEntryRange({start, end - start});
   }
   auto nEntries = fSources[0]->GetNEntries();
   if (nEntries == 0) {
      rangesBySlot.emplace_back(fSeenEntries, fSeenEntries);
   }

   fNextFileIndex++;
   fHasSeenAllRanges = (fNextFileIndex >= fFileNames.size());
   fSeenEntries += fSources[0]->GetNEntries();
   return rangesBySlot;
}

std::string RNTupleDS::GetTypeName(std::string_view colName) const
{
   const auto index = std::distance(fColumnNames.begin(), std::find(fColumnNames.begin(), fColumnNames.end(), colName));
   return fColumnTypes[index];
}

bool RNTupleDS::HasColumn(std::string_view colName) const
{
   return std::find(fColumnNames.begin(), fColumnNames.end(), colName) != fColumnNames.end();
}

void RNTupleDS::Initialize()
{
   fHasSeenAllRanges = false;
   fSeenEntries = 0;
   fNextFileIndex = 0;
   SwitchFile();
}

void RNTupleDS::Finalize() {}

void RNTupleDS::SetNSlots(unsigned int nSlots)
{
   R__ASSERT(fNSlots == 0);
   R__ASSERT(nSlots > 0);
   fNSlots = nSlots;

   for (unsigned int i = 1; i < fNSlots; ++i) {
      fSources.emplace_back(fSources[0]->Clone());
      assert(i == (fSources.size() - 1));
      fSources[i]->Attach();
   }
}
} // namespace Experimental
} // namespace ROOT

ROOT::RDataFrame ROOT::RDF::Experimental::FromRNTuple(std::string_view ntupleName, std::string_view fileName)
{
   auto pageSource = ROOT::Experimental::Detail::RPageSource::Create(ntupleName, fileName);
   ROOT::RDataFrame rdf(std::make_unique<ROOT::Experimental::RNTupleDS>(std::move(pageSource)));
   return rdf;
}

ROOT::RDataFrame ROOT::RDF::Experimental::FromRNTuple(ROOT::Experimental::RNTuple *ntuple)
{
   ROOT::RDataFrame rdf(std::make_unique<ROOT::Experimental::RNTupleDS>(ntuple->MakePageSource()));
   return rdf;
}
