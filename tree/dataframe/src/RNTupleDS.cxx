/// \file RNTupleDS.cxx
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Enrico Guiraud <enrico.guiraud@cern.ch>
/// \date 2018-10-04

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RDF/RColumnReaderBase.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDF/Utils.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldUtils.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleDS.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RPageStorage.hxx>
#include <string_view>

#include <TError.h>
#include <TSystem.h>

#include <cassert>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <typeinfo>
#include <utility>

// clang-format off
/**
* \class ROOT::RDF::RNTupleDS
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

namespace ROOT::Internal::RDF {
/// An artificial field that transforms an RNTuple column that contains the offset of collections into
/// collection sizes. It is used to provide the "number of" RDF columns for collections, e.g.
/// `R_rdf_sizeof_jets` for a collection named `jets`.
///
/// This field owns the collection offset field but instead of exposing the collection offsets it exposes
/// the collection sizes (offset(N+1) - offset(N)).  For the time being, we offer this functionality only in RDataFrame.
/// TODO(jblomer): consider providing a general set of useful virtual fields as part of RNTuple.
class RRDFCardinalityField final : public ROOT::RFieldBase {
protected:
   std::unique_ptr<ROOT::RFieldBase> CloneImpl(std::string_view /* newName */) const final
   {
      return std::make_unique<RRDFCardinalityField>();
   }
   void ConstructValue(void *where) const final { *static_cast<std::size_t *>(where) = 0; }

   // We construct these fields and know that they match the page source
   void ReconcileOnDiskField(const RNTupleDescriptor &) final {}

public:
   RRDFCardinalityField() : ROOT::RFieldBase("", "std::size_t", ROOT::ENTupleStructure::kPlain, false /* isSimple */) {}
   RRDFCardinalityField(RRDFCardinalityField &&other) = default;
   RRDFCardinalityField &operator=(RRDFCardinalityField &&other) = default;
   ~RRDFCardinalityField() override = default;

   const RColumnRepresentations &GetColumnRepresentations() const final
   {
      static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                     {ENTupleColumnType::kIndex64},
                                                     {ENTupleColumnType::kSplitIndex32},
                                                     {ENTupleColumnType::kIndex32}},
                                                    {});
      return representations;
   }
   // Field is only used for reading
   void GenerateColumns() final { throw RException(R__FAIL("Cardinality fields must only be used for reading")); }
   void GenerateColumns(const ROOT::RNTupleDescriptor &desc) final
   {
      GenerateColumnsImpl<ROOT::Internal::RColumnIndex>(desc);
   }

   size_t GetValueSize() const final { return sizeof(std::size_t); }
   size_t GetAlignment() const final { return alignof(std::size_t); }

   /// Get the number of elements of the collection identified by globalIndex
   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final
   {
      RNTupleLocalIndex collectionStart;
      ROOT::NTupleSize_t size;
      fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &size);
      *static_cast<std::size_t *>(to) = size;
   }

   /// Get the number of elements of the collection identified by clusterIndex
   void ReadInClusterImpl(ROOT::RNTupleLocalIndex localIndex, void *to) final
   {
      RNTupleLocalIndex collectionStart;
      ROOT::NTupleSize_t size;
      fPrincipalColumn->GetCollectionInfo(localIndex, &collectionStart, &size);
      *static_cast<std::size_t *>(to) = size;
   }
};

/**
 * @brief An artificial field that provides the size of a fixed-size array
 *
 * This is the implementation of `R_rdf_sizeof_column` in case `column` contains
 * fixed-size arrays on disk.
 */
class RArraySizeField final : public ROOT::RFieldBase {
private:
   std::size_t fArrayLength;

   std::unique_ptr<ROOT::RFieldBase> CloneImpl(std::string_view) const final
   {
      return std::make_unique<RArraySizeField>(fArrayLength);
   }
   void GenerateColumns() final { throw RException(R__FAIL("RArraySizeField fields must only be used for reading")); }
   void GenerateColumns(const ROOT::RNTupleDescriptor &) final {}
   void ReadGlobalImpl(ROOT::NTupleSize_t /*globalIndex*/, void *to) final
   {
      *static_cast<std::size_t *>(to) = fArrayLength;
   }
   void ReadInClusterImpl(RNTupleLocalIndex /*localIndex*/, void *to) final
   {
      *static_cast<std::size_t *>(to) = fArrayLength;
   }

   // We construct these fields and know that they match the page source
   void ReconcileOnDiskField(const RNTupleDescriptor &) final {}

public:
   RArraySizeField(std::size_t arrayLength)
      : ROOT::RFieldBase("", "std::size_t", ROOT::ENTupleStructure::kPlain, false /* isSimple */),
        fArrayLength(arrayLength)
   {
   }
   RArraySizeField(const RArraySizeField &other) = delete;
   RArraySizeField &operator=(const RArraySizeField &other) = delete;
   RArraySizeField(RArraySizeField &&other) = default;
   RArraySizeField &operator=(RArraySizeField &&other) = default;
   ~RArraySizeField() final = default;

   void ConstructValue(void *where) const final { *static_cast<std::size_t *>(where) = 0; }
   std::size_t GetValueSize() const final { return sizeof(std::size_t); }
   std::size_t GetAlignment() const final { return alignof(std::size_t); }
};

/// Every RDF column is represented by exactly one RNTuple field
class RNTupleColumnReader : public ROOT::Detail::RDF::RColumnReaderBase {
   using RFieldBase = ROOT::RFieldBase;
   using RPageSource = ROOT::Internal::RPageSource;

   RNTupleDS *fDataSource;                     ///< The data source that owns this column reader
   RFieldBase *fProtoField;                    ///< The prototype field from which fField is cloned
   std::unique_ptr<RFieldBase> fField;         ///< The field backing the RDF column
   std::unique_ptr<RFieldBase::RValue> fValue; ///< The memory location used to read from fField
   std::shared_ptr<void> fValuePtr;            ///< Used to reuse the object created by fValue when reconnecting sources
   Long64_t fLastEntry = -1;                   ///< Last entry number that was read
   /// For chains, the logical entry and the physical entry in any particular file can be different.
   /// The entry offset stores the logical entry number (sum of all previous physical entries) when a file of the
   /// corresponding data source was opened.
   Long64_t fEntryOffset = 0;

public:
   RNTupleColumnReader(RNTupleDS *ds, RFieldBase *protoField) : fDataSource(ds), fProtoField(protoField) {}
   ~RNTupleColumnReader() override = default;

   /// Connect the field and its subfields to the page source
   void Connect(RPageSource &source, Long64_t entryOffset)
   {
      assert(fLastEntry == -1);

      fEntryOffset = entryOffset;

      // Create a new, real field from the prototype and set its field ID in the context of the given page source
      fField = fProtoField->Clone(fProtoField->GetFieldName());
      {
         auto descGuard = source.GetSharedDescriptorGuard();
         // Set the on-disk field IDs for the field and the subfield
         fField->SetOnDiskId(
            descGuard->FindFieldId(fDataSource->fFieldId2QualifiedName.at(fProtoField->GetOnDiskId())));
         auto iProto = fProtoField->cbegin();
         auto iReal = fField->begin();
         for (; iReal != fField->end(); ++iProto, ++iReal) {
            iReal->SetOnDiskId(descGuard->FindFieldId(fDataSource->fFieldId2QualifiedName.at(iProto->GetOnDiskId())));
         }
      }

      try {
         ROOT::Internal::CallConnectPageSourceOnField(*fField, source);
      } catch (const ROOT::RException &err) {
         auto onDiskType = source.GetSharedDescriptorGuard()->GetFieldDescriptor(fField->GetOnDiskId()).GetTypeName();
         std::string msg = "RNTupleDS: invalid type \"" + fField->GetTypeName() + "\" for column \"" +
                           fDataSource->fFieldId2QualifiedName[fField->GetOnDiskId()] + "\" with on-disk type \"" +
                           onDiskType + "\"";
         throw std::runtime_error(msg);
      }

      if (fValuePtr) {
         // When the reader reconnects to a new file, the fValuePtr is already set
         fValue = std::make_unique<RFieldBase::RValue>(fField->BindValue(fValuePtr));
         fValuePtr = nullptr;
      } else {
         // For the first file, create a new object for this field (reader)
         fValue = std::make_unique<RFieldBase::RValue>(fField->CreateValue());
      }
   }

   void Disconnect(bool keepValue)
   {
      if (fValue && keepValue) {
         fValuePtr = fValue->GetPtr<void>();
      }
      fValue = nullptr;
      fField = nullptr;
      fLastEntry = -1;
   }

   void *GetImpl(Long64_t entry) final
   {
      if (entry != fLastEntry) {
         fValue->Read(entry - fEntryOffset);
         fLastEntry = entry;
      }
      return fValue->GetPtr<void>().get();
   }
};
} // namespace ROOT::Internal::RDF

ROOT::RDF::RNTupleDS::~RNTupleDS() = default;

void ROOT::RDF::RNTupleDS::AddField(const ROOT::RNTupleDescriptor &desc, std::string_view colName,
                                    ROOT::DescriptorId_t fieldId, std::vector<RNTupleDS::RFieldInfo> fieldInfos,
                                    bool convertToRVec)
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
   const auto &nRepetitions = fieldDesc.GetNRepetitions();
   if ((fieldDesc.GetStructure() == ROOT::ENTupleStructure::kCollection) || (nRepetitions > 0)) {
      // The field is a collection or a fixed-size array.
      // We open a new collection scope with fieldID being the inner most collection. E.g. for "event.tracks.hits",
      // fieldInfos would already contain the fieldID of "event.tracks"
      fieldInfos.emplace_back(fieldId, nRepetitions);
   }

   if (fieldDesc.GetStructure() == ROOT::ENTupleStructure::kCollection) {
      // Inner fields of collections are provided as projected collections of only that inner field,
      // E.g. we provide a projected collection RVec<RVec<float>> for "event.tracks.hits.x" in the example
      // above.
      bool representableAsRVec =
         convertToRVec && (fieldDesc.GetTypeName().substr(0, 19) == "ROOT::VecOps::RVec<" ||
                           fieldDesc.GetTypeName().substr(0, 12) == "std::vector<" || fieldDesc.GetTypeName() == "");
      const auto &f = *desc.GetFieldIterable(fieldDesc.GetId()).begin();
      AddField(desc, colName, f.GetId(), fieldInfos, representableAsRVec);

      // Note that at the end of the recursion, we handled the inner sub collections as well as the
      // collection as whole, so we are done.
      return;

   } else if (nRepetitions > 0) {
      // Fixed-size array, same logic as ROOT::RVec.
      const auto &f = *desc.GetFieldIterable(fieldDesc.GetId()).begin();
      AddField(desc, colName, f.GetId(), fieldInfos);
      return;
   } else if (fieldDesc.GetStructure() == ROOT::ENTupleStructure::kRecord) {
      // Inner fields of records are provided as individual RDF columns, e.g. "event.id"
      for (const auto &f : desc.GetFieldIterable(fieldDesc.GetId())) {
         auto innerName = colName.empty() ? f.GetFieldName() : (std::string(colName) + "." + f.GetFieldName());
         // Inner fields of collections of records are always exposed as ROOT::RVec
         AddField(desc, innerName, f.GetId(), fieldInfos);
      }
   }

   // The fieldID could be the root field or the class of fieldId might not be loaded.
   // In these cases, only the inner fields are exposed as RDF columns.
   auto fieldOrException = ROOT::RFieldBase::Create(fieldDesc.GetFieldName(), fieldDesc.GetTypeName());
   if (!fieldOrException)
      return;
   auto valueField = fieldOrException.Unwrap();
   valueField->SetOnDiskId(fieldId);
   for (auto &f : *valueField) {
      f.SetOnDiskId(desc.FindFieldId(f.GetFieldName(), f.GetParent()->GetOnDiskId()));
   }
   std::unique_ptr<ROOT::RFieldBase> cardinalityField;
   // Collections get the additional "number of" RDF column (e.g. "R_rdf_sizeof_tracks")
   if (!fieldInfos.empty()) {
      const auto &info = fieldInfos.back();
      if (info.fNRepetitions > 0) {
         cardinalityField = std::make_unique<ROOT::Internal::RDF::RArraySizeField>(info.fNRepetitions);
      } else {
         cardinalityField = std::make_unique<ROOT::Internal::RDF::RRDFCardinalityField>();
      }
      cardinalityField->SetOnDiskId(info.fFieldId);
   }

   for (auto i = fieldInfos.rbegin(); i != fieldInfos.rend(); ++i) {
      const auto &fieldInfo = *i;

      if (fieldInfo.fNRepetitions > 0) {
         // Fixed-size array, read it as ROOT::RVec in memory
         valueField = std::make_unique<ROOT::RArrayAsRVecField>("", std::move(valueField), fieldInfo.fNRepetitions);
      } else {
         // Actual collection. A std::vector or ROOT::RVec gets added as a ROOT::RVec. All other collection types keep
         // their original type.
         if (convertToRVec) {
            valueField = std::make_unique<ROOT::RRVecField>("", std::move(valueField));
         } else {
            auto outerFieldType = desc.GetFieldDescriptor(fieldInfo.fFieldId).GetTypeName();
            valueField = ROOT::RFieldBase::Create("", outerFieldType).Unwrap();
         }
      }

      valueField->SetOnDiskId(fieldInfo.fFieldId);

      // Skip the inner-most collection level to construct the cardinality column
      // It's taken care of by the `if (!fieldInfos.empty())` scope above
      if (i != fieldInfos.rbegin()) {
         if (fieldInfo.fNRepetitions > 0) {
            // This collection level refers to a fixed-size array
            cardinalityField =
               std::make_unique<ROOT::RArrayAsRVecField>("", std::move(cardinalityField), fieldInfo.fNRepetitions);
         } else {
            // This collection level refers to an RVec
            cardinalityField = std::make_unique<ROOT::RRVecField>("", std::move(cardinalityField));
         }

         cardinalityField->SetOnDiskId(fieldInfo.fFieldId);
      }
   }

   if (cardinalityField) {
      fColumnNames.emplace_back("R_rdf_sizeof_" + std::string(colName));
      fColumnTypes.emplace_back(cardinalityField->GetTypeName());
      fProtoFields.emplace_back(std::move(cardinalityField));
   }

   fieldInfos.emplace_back(fieldId, nRepetitions);
   fColumnNames.emplace_back(colName);
   fColumnTypes.emplace_back(valueField->GetTypeName());
   fProtoFields.emplace_back(std::move(valueField));
}

ROOT::RDF::RNTupleDS::RNTupleDS(std::unique_ptr<ROOT::Internal::RPageSource> pageSource)
{
   pageSource->Attach();
   fPrincipalDescriptor = pageSource->GetSharedDescriptorGuard()->Clone();
   fStagingArea.emplace_back(std::move(pageSource));

   AddField(fPrincipalDescriptor, "", fPrincipalDescriptor.GetFieldZeroId(),
            std::vector<ROOT::RDF::RNTupleDS::RFieldInfo>());
}

namespace {

const ROOT::RNTupleReadOptions &GetOpts()
{
   // The setting is for now a global one, must be decided before running the
   // program by setting the appropriate environment variable. Make sure that
   // option configuration is thread-safe and happens only once.
   static ROOT::RNTupleReadOptions opts;
   static std::once_flag flag;
   std::call_once(flag, []() {
      if (auto env = gSystem->Getenv("ROOT_RNTUPLE_CLUSTERBUNCHSIZE"); env != nullptr && strlen(env) > 0) {
         std::string envStr{env};
         auto envNum{std::stoul(envStr)};
         envNum = envNum == 0 ? 1 : envNum;
         ROOT::Internal::RNTupleReadOptionsManip::SetClusterBunchSize(opts, envNum);
      }
   });
   return opts;
}

std::unique_ptr<ROOT::Internal::RPageSource> CreatePageSource(std::string_view ntupleName, std::string_view fileName)
{
   return ROOT::Internal::RPageSource::Create(ntupleName, fileName, GetOpts());
}
} // namespace

ROOT::RDF::RNTupleDS::RNTupleDS(std::string_view ntupleName, std::string_view fileName)
   : RNTupleDS(CreatePageSource(ntupleName, fileName))
{
   fFileNames = std::vector<std::string>{std::string{fileName}};
}

ROOT::RDF::RNTupleDS::RNTupleDS(std::string_view ntupleName, const std::vector<std::string> &fileNames)
   : RNTupleDS(CreatePageSource(ntupleName, fileNames[0]))
{
   fNTupleName = ntupleName;
   fFileNames = fileNames;
   fStagingArea.resize(fFileNames.size());
}

ROOT::RDF::RNTupleDS::RNTupleDS(std::string_view ntupleName, const std::vector<std::string> &fileNames,
                                const std::pair<ULong64_t, ULong64_t> &range)
   : RNTupleDS(ntupleName, fileNames)
{
   fGlobalEntryRange = range;
}

ROOT::RDF::RDataSource::Record_t
ROOT::RDF::RNTupleDS::GetColumnReadersImpl(std::string_view /* name */, const std::type_info & /* ti */)
{
   // This datasource uses the newer GetColumnReaders() API
   return {};
}

ROOT::RFieldBase *ROOT::RDF::RNTupleDS::GetFieldWithTypeChecks(std::string_view fieldName, const std::type_info &tid)
{
   // At this point we can assume that `name` will be found in fColumnNames
   const auto index =
      std::distance(fColumnNames.begin(), std::find(fColumnNames.begin(), fColumnNames.end(), fieldName));

   // A reader was requested but we don't have RTTI for it, this is encoded with the tag UseNativeDataType. We can just
   // return the available protofield
   if (tid == typeid(ROOT::Internal::RDF::UseNativeDataType)) {
      return fProtoFields[index].get();
   }

   // The user explicitly requested a type
   const auto requestedType = ROOT::Internal::GetRenormalizedTypeName(ROOT::Internal::RDF::TypeID2TypeName(tid));

   // If the field corresponding to the provided name is not a cardinality column and the requested type is different
   // from the proto field that was created when the data source was constructed, we first have to create an
   // alternative proto field for the column reader. Otherwise, we can directly use the existing proto field.
   if (fieldName.substr(0, 13) != "R_rdf_sizeof_" && requestedType != fColumnTypes[index]) {
      auto &altProtoFields = fAlternativeProtoFields[index];

      // If we can find the requested type in the registered alternative protofields, return the corresponding field
      if (auto altProtoField = std::find_if(altProtoFields.begin(), altProtoFields.end(),
                                            [&requestedType](const std::unique_ptr<ROOT::RFieldBase> &fld) {
                                               return fld->GetTypeName() == requestedType;
                                            });
          altProtoField != altProtoFields.end()) {
         return altProtoField->get();
      }

      // Otherwise, create a new protofield and register it in the alternatives before returning
      auto newAltProtoFieldOrException = ROOT::RFieldBase::Create(std::string(fieldName), requestedType);
      if (!newAltProtoFieldOrException) {
         throw std::runtime_error("RNTupleDS: Could not create field with type \"" + requestedType +
                                  "\" for column \"" + std::string(fieldName) + "\"");
      }
      auto newAltProtoField = newAltProtoFieldOrException.Unwrap();
      newAltProtoField->SetOnDiskId(fProtoFields[index]->GetOnDiskId());
      auto *newField = newAltProtoField.get();
      altProtoFields.emplace_back(std::move(newAltProtoField));
      return newField;
   }

   // General case: there was a correspondence between the user-requested type and the corresponding column type
   return fProtoFields[index].get();
}

std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
ROOT::RDF::RNTupleDS::GetColumnReaders(unsigned int slot, std::string_view name, const std::type_info &tid)
{
   ROOT::RFieldBase *field = GetFieldWithTypeChecks(name, tid);
   assert(field != nullptr);

   // Map the field's and subfields' IDs to qualified names so that we can later connect the fields to
   // other page sources from the chain
   fFieldId2QualifiedName[field->GetOnDiskId()] = fPrincipalDescriptor.GetQualifiedFieldName(field->GetOnDiskId());
   for (const auto &s : *field) {
      fFieldId2QualifiedName[s.GetOnDiskId()] = fPrincipalDescriptor.GetQualifiedFieldName(s.GetOnDiskId());
   }

   auto reader = std::make_unique<ROOT::Internal::RDF::RNTupleColumnReader>(this, field);
   fActiveColumnReaders[slot].emplace_back(reader.get());

   return reader;
}

void ROOT::RDF::RNTupleDS::ExecStaging()
{
   while (true) {
      std::unique_lock lock(fMutexStaging);
      fCvStaging.wait(lock, [this] { return fIsReadyForStaging || fStagingThreadShouldTerminate; });
      if (fStagingThreadShouldTerminate)
         return;

      assert(!fHasNextSources);
      StageNextSources();
      fHasNextSources = true;
      fIsReadyForStaging = false;

      lock.unlock();
      fCvStaging.notify_one();
   }
}

void ROOT::RDF::RNTupleDS::StageNextSources()
{
   const auto nFiles = fFileNames.empty() ? 1 : fFileNames.size();

   for (auto i = fNextFileIndex; (i < nFiles) && ((i - fNextFileIndex) < fNSlots); ++i) {

      if (fStagingThreadShouldTerminate)
         return;

      if (fStagingArea[i]) {
         // The first file is already open and was used to read the schema
         assert(i == 0);
      } else {
         fStagingArea[i] = CreatePageSource(fNTupleName, fFileNames[i]);
         fStagingArea[i]->LoadStructure();
      }
   }
}

void ROOT::RDF::RNTupleDS::PrepareNextRanges()
{
   assert(fNextRanges.empty());

   auto nFiles = fFileNames.empty() ? 1 : fFileNames.size();
   auto nRemainingFiles = nFiles - fNextFileIndex;

   if (nRemainingFiles == 0)
      return;

   // Easy work scheduling: one file per slot. We skip empty files (files without entries).

   if ((nRemainingFiles >= fNSlots) || (fGlobalEntryRange.has_value())) {
      while ((fNextRanges.size() < fNSlots) && (fNextFileIndex < nFiles)) {
         REntryRangeDS range;

         std::swap(fStagingArea[fNextFileIndex], range.fSource);

         if (!range.fSource) {
            // Typically, the prestaged source should have been present. Only if some of the files are empty, we need
            // to open and attach files here.
            range.fSource = CreatePageSource(fNTupleName, fFileNames[fNextFileIndex]);
         }
         range.fFileName = fFileNames[fNextFileIndex];
         range.fSource->Attach();
         fNextFileIndex++;
         auto nEntries = range.fSource->GetNEntries();
         if (nEntries == 0)
            continue;
         range.fLastEntry = nEntries; // whole file per slot, i.e. entry range [0..nEntries - 1]

         fNextRanges.emplace_back(std::move(range));
      }
      return;
   }

   // Work scheduling of the tail: multiple slots work on the same file.
   // Every slot still has its own page source but these page sources may open the same file.
   // Again, we need to skip empty files.
   unsigned int nSlotsPerFile = fNSlots / nRemainingFiles;
   for (std::size_t i = 0; (fNextRanges.size() < fNSlots) && (fNextFileIndex < nFiles); ++i) {
      std::unique_ptr<ROOT::Internal::RPageSource> source;
      // Need to look for the file name to populate the sample info later
      const auto &sourceFileName = fFileNames[fNextFileIndex];
      std::swap(fStagingArea[fNextFileIndex], source);
      if (!source) {
         // Empty files trigger this condition
         source = CreatePageSource(fNTupleName, fFileNames[fNextFileIndex]);
      }
      source->Attach();
      fNextFileIndex++;

      auto nEntries = source->GetNEntries();
      if (nEntries == 0)
         continue;

      // If last file: use all remaining slots
      if (i == (nRemainingFiles - 1))
         nSlotsPerFile = fNSlots - fNextRanges.size();

      const auto rangesByCluster = [&source]() {
         // Take the shared lock of the descriptor just for the time necessary
         const auto descGuard = source->GetSharedDescriptorGuard();
         return ROOT::Internal::GetClusterBoundaries(descGuard.GetRef());
      }();

      const unsigned int nRangesByCluster = rangesByCluster.size();

      // Distribute slots equidistantly over the entry range, aligned on cluster boundaries
      const auto nClustersPerSlot = nRangesByCluster / nSlotsPerFile;
      const auto remainder = nRangesByCluster % nSlotsPerFile;
      std::size_t iRange = 0;
      unsigned int iSlot = 0;
      const unsigned int N = std::min(nSlotsPerFile, nRangesByCluster);
      for (; iSlot < N; ++iSlot) {
         auto start = rangesByCluster[iRange].fFirstEntry;
         iRange += nClustersPerSlot + static_cast<int>(iSlot < remainder);
         assert(iRange > 0);
         auto end = rangesByCluster[iRange - 1].fLastEntryPlusOne;

         REntryRangeDS range;
         range.fFileName = sourceFileName;
         // The last range for this file just takes the already opened page source. All previous ranges clone.
         if (iSlot == N - 1) {
            range.fSource = std::move(source);
         } else {
            range.fSource = source->Clone();
         }
         range.fSource->SetEntryRange({start, end - start});
         range.fFirstEntry = start;
         range.fLastEntry = end;
         fNextRanges.emplace_back(std::move(range));
      }
   } // loop over tail of remaining files
}

std::vector<std::pair<ULong64_t, ULong64_t>> ROOT::RDF::RNTupleDS::GetEntryRanges()
{
   std::vector<std::pair<ULong64_t, ULong64_t>> ranges;

   // We need to distinguish between single threaded and multi-threaded runs.
   // In single threaded mode, InitSlot is only called once and column readers have to be rewired
   // to new page sources of the chain in GetEntryRanges. In multi-threaded mode, on the other hand,
   // InitSlot is called for every returned range, thus rewiring the column readers takes place in
   // InitSlot and FinalizeSlot.

   if (fNSlots == 1) {
      for (auto r : fActiveColumnReaders[0]) {
         r->Disconnect(true /* keepValue */);
      }
   }

   // If we have fewer files than slots and we run multiple event loops, we can reuse fCurrentRanges and don't need
   // to worry about loading the fNextRanges. I.e., in this case we don't enter the if block.
   if (fCurrentRanges.empty() || fSeenEntriesNoGlobalRange > 0) {
      // Otherwise, i.e. start of the first event loop or in the middle of the event loop, prepare the next ranges
      // and swap with the current ones.
      {
         std::unique_lock lock(fMutexStaging);
         fCvStaging.wait(lock, [this] { return fHasNextSources; });
      }
      PrepareNextRanges();
      if (fNextRanges.empty()) {
         // No more data
         return ranges;
      }

      assert(fNextRanges.size() <= fNSlots);

      fCurrentRanges.clear();
      std::swap(fCurrentRanges, fNextRanges);
   }

   // Stage next batch of files for the next call to GetEntryRanges()
   {
      std::lock_guard _(fMutexStaging);
      fIsReadyForStaging = true;
      fHasNextSources = false;
   }
   fCvStaging.notify_one();

   // Create ranges for the RDF loop manager from the list of REntryRangeDS records.
   // The entry ranges that are relative to the page source in REntryRangeDS are translated into absolute
   // entry ranges, given the current state of the entry cursor.
   // We remember the connection from first absolute entry index of a range to its REntryRangeDS record
   // so that we can properly rewire the column reader in InitSlot
   fFirstEntry2RangeIdx.clear();
   fOriginalRanges.clear();

   ULong64_t nEntriesPerSource = 0;

   for (std::size_t i = 0; i < fCurrentRanges.size(); ++i) {

      // Several consecutive ranges may operate on the same file (each with their own page source clone).
      // We can detect a change of file when the first entry number jumps back to 0.
      if (fCurrentRanges[i].fFirstEntry == 0) {
         // New source
         fSeenEntriesNoGlobalRange += nEntriesPerSource;
         nEntriesPerSource = 0;
      }

      auto start = fCurrentRanges[i].fFirstEntry + fSeenEntriesNoGlobalRange;
      auto end = fCurrentRanges[i].fLastEntry + fSeenEntriesNoGlobalRange;

      nEntriesPerSource += end - start;

      if (fGlobalEntryRange.has_value()) {

         // We need to consider different scenarios for when we have GlobalRanges set by the user.
         // Consider a simple case of 3 files, with original ranges set as (consecutive entries of 3 files):
         // [0, 20], [20, 45], [45, 65]
         // we will now see what happens in each of the scenarios below when GlobalRanges can be set to different
         // values:
         // a) [2, 5] - we stay in file 1
         //  - hence we will use the 1st case and get the range [2,5], in this case we also need to quit further
         //  processing from the other files by entering case 3
         // b) [2, 21] - we start in file 1 and finish in file 2
         // - use the 2nd case first, as 21 > 20 (end of first file), then we will go to case 1, resulting in ranges:
         // [2, 20], [20, 21], c) [21 - 40] - we skip file 1, start in file 2 and stay in file 2
         // - to skip the first file, we use the 4th case, followed by the 1st case, resulting range is: [21, 40]
         // d) [21 - 65] - we skip file 1, start in file 2 and continue to file 3
         // - to skip the first file, we use the 4th case, we continue with the 2nd case, and use the 1st case at the
         // end, resulting ranges are [21, 45], [45, 65]
         // The first case
         if (fGlobalEntryRange->first >= start && fGlobalEntryRange->second <= end) {
            fOriginalRanges.emplace_back(start, end);
            fFirstEntry2RangeIdx[fGlobalEntryRange->first] = i;
            ranges.emplace_back(fGlobalEntryRange->first, fGlobalEntryRange->second);
         }

         // The second case:
         // The `fGlobalEntryRange->first < end` condition is to distinguish this case from the 4th case.
         else if (fGlobalEntryRange->second > end && fGlobalEntryRange->first < end) {
            fOriginalRanges.emplace_back(start, end);
            fFirstEntry2RangeIdx[fGlobalEntryRange->first] = i;
            ranges.emplace_back(fGlobalEntryRange->first, end);
            std::optional<std::pair<ULong64_t, ULong64_t>> newvalues({end, fGlobalEntryRange->second});
            fGlobalEntryRange.swap(newvalues);
         }
         // The third case, needed to correctly quit processing if we only stay in the first file
         else if (fGlobalEntryRange->second < start) {
            return ranges;
         }

         // The fourth case:
         else if (fGlobalEntryRange->first >= end) {
            fOriginalRanges.emplace_back(start, end);
            fFirstEntry2RangeIdx[start] = i;
            ranges.emplace_back(start, start);
         }
      }

      else {
         fFirstEntry2RangeIdx[start] = i;
         fOriginalRanges.emplace_back(start, end);
         ranges.emplace_back(start, end);
      }
   }

   fSeenEntriesNoGlobalRange += nEntriesPerSource;

   if ((fNSlots == 1) && (fCurrentRanges[0].fSource)) {
      for (auto r : fActiveColumnReaders[0]) {
         r->Connect(*fCurrentRanges[0].fSource, fOriginalRanges[0].first);
      }
   }

   return ranges;
}

void ROOT::RDF::RNTupleDS::InitSlot(unsigned int slot, ULong64_t firstEntry)
{
   if (fNSlots == 1) {
      // Ensure the connection between slot and range is valid also in single-thread mode
      fSlotsToRangeIdxs[0] = 0;
      return;
   }

   // The same slot ID could be picked multiple times in the same execution, thus
   // ending up processing different page sources. Here we re-establish the
   // connection between the slot and the correct page source by finding which
   // range index corresponds to the first entry passed.
   auto idxRange = fFirstEntry2RangeIdx.at(firstEntry);

   // We also remember this connection so it can later be retrieved in CreateSampleInfo
   fSlotsToRangeIdxs[slot * ROOT::Internal::RDF::CacheLineStep<std::size_t>()] = idxRange;

   for (auto r : fActiveColumnReaders[slot]) {
      r->Connect(*fCurrentRanges[idxRange].fSource,
                 fOriginalRanges[idxRange].first - fCurrentRanges[idxRange].fFirstEntry);
   }
}

void ROOT::RDF::RNTupleDS::FinalizeSlot(unsigned int slot)
{
   if (fNSlots == 1)
      return;

   for (auto r : fActiveColumnReaders[slot]) {
      r->Disconnect(true /* keepValue */);
   }
}

std::string ROOT::RDF::RNTupleDS::GetTypeName(std::string_view colName) const
{
   auto colNamePos = std::find(fColumnNames.begin(), fColumnNames.end(), colName);

   if (colNamePos == fColumnNames.end()) {
      auto msg = std::string("RNTupleDS: There is no column with name \"") + std::string(colName) + "\"";
      throw std::runtime_error(msg);
   }

   const auto index = std::distance(fColumnNames.begin(), colNamePos);
   return fColumnTypes[index];
}

bool ROOT::RDF::RNTupleDS::HasColumn(std::string_view colName) const
{
   return std::find(fColumnNames.begin(), fColumnNames.end(), colName) != fColumnNames.end();
}

void ROOT::RDF::RNTupleDS::Initialize()
{
   fSeenEntriesNoGlobalRange = 0;
   fNextFileIndex = 0;
   fIsReadyForStaging = fHasNextSources = fStagingThreadShouldTerminate = false;
   fThreadStaging = std::thread(&RNTupleDS::ExecStaging, this);
   assert(fNextRanges.empty());

   if (fCurrentRanges.empty() || (fFileNames.size() > fNSlots)) {
      // First event loop or large number of files: start the staging process.
      {
         std::lock_guard _(fMutexStaging);
         fIsReadyForStaging = true;
      }
      fCvStaging.notify_one();
   } else {
      // Otherwise, we will reuse fCurrentRanges. Make sure that staging and preparing next ranges will be a noop
      // (already at the end of the list of files).
      fNextFileIndex = std::max(fFileNames.size(), std::size_t(1));
   }
}

void ROOT::RDF::RNTupleDS::Finalize()
{
   for (unsigned int i = 0; i < fNSlots; ++i) {
      for (auto r : fActiveColumnReaders[i]) {
         r->Disconnect(false /* keepValue */);
      }
   }
   {
      std::lock_guard _(fMutexStaging);
      fStagingThreadShouldTerminate = true;
   }
   fCvStaging.notify_one();
   fThreadStaging.join();
   // If we have a chain with more files than the number of slots, the files opened at the end of the
   // event loop won't be reused when the event loop restarts, so we can close them.
   if (fFileNames.size() > fNSlots) {
      fCurrentRanges.clear();
      fNextRanges.clear();
      fStagingArea.clear();
      fStagingArea.resize(fFileNames.size());
   }
}

void ROOT::RDF::RNTupleDS::SetNSlots(unsigned int nSlots)
{
   assert(fNSlots == 0);
   assert(nSlots > 0);
   fNSlots = nSlots;
   fActiveColumnReaders.resize(fNSlots);
   fSlotsToRangeIdxs.resize(fNSlots * ROOT::Internal::RDF::CacheLineStep<std::size_t>());
}

ROOT::RDataFrame ROOT::RDF::FromRNTuple(std::string_view ntupleName, std::string_view fileName)
{
   return ROOT::RDataFrame(std::make_unique<ROOT::RDF::RNTupleDS>(ntupleName, fileName));
}

ROOT::RDataFrame ROOT::RDF::FromRNTuple(std::string_view ntupleName, const std::vector<std::string> &fileNames)
{
   return ROOT::RDataFrame(std::make_unique<ROOT::RDF::RNTupleDS>(ntupleName, fileNames));
}

ROOT::RDF::RSampleInfo ROOT::Internal::RDF::RNTupleDS::CreateSampleInfo(
   unsigned int slot, const std::unordered_map<std::string, ROOT::RDF::Experimental::RSample *> &sampleMap) const
{
   // The same slot ID could be picked multiple times in the same execution, thus
   // ending up processing different page sources. Here we re-establish the
   // connection between the slot and the correct page source by retrieving
   // which range is connected currently to the slot

   const auto &rangeIdx = fSlotsToRangeIdxs.at(slot * ROOT::Internal::RDF::CacheLineStep<std::size_t>());

   // Missing source if a file does not exist
   if (!fCurrentRanges[rangeIdx].fSource)
      return ROOT::RDF::RSampleInfo{};

   const auto &ntupleName = fCurrentRanges[rangeIdx].fSource->GetNTupleName();
   const auto &ntuplePath = fCurrentRanges[rangeIdx].fFileName;
   const auto ntupleID = std::string(ntuplePath) + '/' + ntupleName;

   if (sampleMap.empty())
      return ROOT::RDF::RSampleInfo(
         ntupleID, std::make_pair(fCurrentRanges[rangeIdx].fFirstEntry, fCurrentRanges[rangeIdx].fLastEntry));

   if (sampleMap.find(ntupleID) == sampleMap.end())
      throw std::runtime_error("Full sample identifier '" + ntupleID + "' cannot be found in the available samples.");

   return ROOT::RDF::RSampleInfo(
      ntupleID, std::make_pair(fCurrentRanges[rangeIdx].fFirstEntry, fCurrentRanges[rangeIdx].fLastEntry),
      sampleMap.at(ntupleID));
}

ROOT::RDataFrame ROOT::Internal::RDF::FromRNTuple(std::string_view ntupleName,
                                                  const ROOT::RDF::ColumnNames_t &fileNames,
                                                  const std::pair<ULong64_t, ULong64_t> &range)
{
   std::unique_ptr<ROOT::RDF::RNTupleDS> ds{new ROOT::RDF::RNTupleDS(ntupleName, fileNames, range)};
   return ROOT::RDataFrame(std::move(ds));
}

std::pair<std::vector<ROOT::Internal::RNTupleClusterBoundaries>, ROOT::NTupleSize_t>
ROOT::Internal::RDF::GetClustersAndEntries(std::string_view ntupleName, std::string_view location)
{
   auto source = ROOT::Internal::RPageSource::Create(ntupleName, location);
   source->Attach();
   const auto descGuard = source->GetSharedDescriptorGuard();
   return std::make_pair(ROOT::Internal::GetClusterBoundaries(descGuard.GetRef()), descGuard->GetNEntries());
}
