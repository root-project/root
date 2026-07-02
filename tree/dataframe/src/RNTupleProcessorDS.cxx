/// \file RNTupleProcessorDS.cxx
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
#include <ROOT/RNTupleProcessorDS.hxx>
#include <ROOT/RPageStorage.hxx>
#include <string_view>

#include <TError.h>
#include <TSystem.h>

#include <cassert>
#include <memory>
#include <string>

// namespace ROOT::Experimental::RDF {
// class RNTupleProcessorDS;
// }

// clang-format off
/**
* \class ROOT::Experimental::RDF::RNTupleProcessorDS
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
namespace ROOT::Experimental::Internal::RDF {
class RRDFCardinalityFieldBase : public ROOT::RFieldBase {
protected:
   // We construct these fields and know that they match the page source
   void ReconcileOnDiskField(const RNTupleDescriptor &) final {}

   RRDFCardinalityFieldBase(std::string_view name, std::string_view type)
      : ROOT::RFieldBase(name, type, ROOT::ENTupleStructure::kPlain, false /* isSimple */)
   {
   }

   // Field is only used for reading
   void GenerateColumns() final { throw RException(R__FAIL("Cardinality fields must only be used for reading")); }
   void GenerateColumns(const ROOT::RNTupleDescriptor &desc) final
   {
      GenerateColumnsImpl<ROOT::Internal::RColumnIndex>(desc);
   }

public:
   RRDFCardinalityFieldBase(const RRDFCardinalityFieldBase &other) = delete;
   RRDFCardinalityFieldBase &operator=(const RRDFCardinalityFieldBase &other) = delete;
   RRDFCardinalityFieldBase(RRDFCardinalityFieldBase &&other) = default;
   RRDFCardinalityFieldBase &operator=(RRDFCardinalityFieldBase &&other) = default;
   ~RRDFCardinalityFieldBase() override = default;

   const RColumnRepresentations &GetColumnRepresentations() const final
   {
      static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                     {ENTupleColumnType::kIndex64},
                                                     {ENTupleColumnType::kSplitIndex32},
                                                     {ENTupleColumnType::kIndex32}},
                                                    {});
      return representations;
   }
};

/// An artificial field that transforms an RNTuple column that contains the offset of collections into
/// collection sizes. It is used to provide the "number of" RDF columns for collections, e.g.
/// `R_rdf_sizeof_jets` for a collection named `jets`.
///
/// This is similar to the RCardinalityField but it presents itself as an integer type.
/// The template argument T must be an integral type.
template <typename T>
class RRDFCardinalityField final : public RRDFCardinalityFieldBase {
   static_assert(std::is_integral_v<T>, "T must be an integral type");

   inline void CheckSize(ROOT::NTupleSize_t size) const
   {
      if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, std::uint64_t>)
         return;
      if (size > static_cast<ROOT::NTupleSize_t>(std::numeric_limits<T>::max())) {
         throw RException(R__FAIL(std::string("integer overflow in field ") + GetFieldName() +
                                  ". Please read the column with a larger-sized integral type."));
      }
   }

protected:
   std::unique_ptr<ROOT::RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RRDFCardinalityField>(newName);
   }
   void ConstructValue(void *where) const final { *static_cast<T *>(where) = 0; }

public:
   RRDFCardinalityField(std::string_view name)
      : RRDFCardinalityFieldBase(name, ROOT::Internal::GetRenormalizedTypeName(typeid(T)))
   {
   }
   RRDFCardinalityField(const RRDFCardinalityField &other) = delete;
   RRDFCardinalityField &operator=(const RRDFCardinalityField &other) = delete;
   RRDFCardinalityField(RRDFCardinalityField &&other) = default;
   RRDFCardinalityField &operator=(RRDFCardinalityField &&other) = default;
   ~RRDFCardinalityField() override = default;

   std::size_t GetValueSize() const final { return sizeof(T); }
   std::size_t GetAlignment() const final { return alignof(T); }

   /// Get the number of elements of the collection identified by globalIndex
   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final
   {
      RNTupleLocalIndex collectionStart;
      ROOT::NTupleSize_t size;
      fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &size);
      CheckSize(size);
      *static_cast<T *>(to) = size;
   }

   /// Get the number of elements of the collection identified by clusterIndex
   void ReadInClusterImpl(ROOT::RNTupleLocalIndex localIndex, void *to) final
   {
      RNTupleLocalIndex collectionStart;
      ROOT::NTupleSize_t size;
      fPrincipalColumn->GetCollectionInfo(localIndex, &collectionStart, &size);
      CheckSize(size);
      *static_cast<T *>(to) = size;
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

   std::unique_ptr<ROOT::RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RArraySizeField>(newName, fArrayLength);
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
   RArraySizeField(std::string_view name, std::size_t arrayLength)
      : ROOT::RFieldBase(name, ROOT::Internal::GetRenormalizedTypeName(typeid(std::size_t)),
                         ROOT::ENTupleStructure::kPlain, false /* isSimple */),
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
class RNTupleProcessorColumnReader : public ROOT::Detail::RDF::RColumnReaderBase {
   RNTupleProcessor *fProcessor;                ///< The processor managed by the data source
   RNTupleProcessorOptionalPtr<void> fValuePtr; ///< Container for the value of the column

public:
   RNTupleProcessorColumnReader(RNTupleProcessor &processor, RNTupleProcessorOptionalPtr<void> &valuePtr)
      : fProcessor(&processor), fValuePtr(std::move(valuePtr))
   {
   }
   ~RNTupleProcessorColumnReader() override = default;

   void *GetImpl(Long64_t entry) final
   {
      assert(entry == static_cast<Long64_t>(fProcessor->GetCurrentEntryNumber()));
      return fValuePtr.GetRawPtr();
   }
};
} // namespace ROOT::Experimental::Internal::RDF

ROOT::Experimental::RDF::RNTupleProcessorDS::~RNTupleProcessorDS() = default;

void ROOT::Experimental::RDF::RNTupleProcessorDS::AddField(const ROOT::RFieldBase &field, std::string_view colName,
                                                           Internal::RNTupleProcessorProvenance procProvenance,
                                                           std::vector<RNTupleProcessorDS::RFieldInfo> fieldInfos,
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

   const auto &nRepetitions = field.GetNRepetitions();
   if ((field.GetStructure() == ROOT::ENTupleStructure::kCollection) || (nRepetitions > 0)) {
      // The field is a collection or a fixed-size array.
      // We open a new collection scope with fieldID being the inner most collection. E.g. for "event.tracks.hits",
      // fieldInfos would already contain the fieldID of "event.tracks"
      fieldInfos.emplace_back(field.GetOnDiskId(), field.GetFieldName(), field.GetTypeName(), nRepetitions);
   }

   if (field.GetStructure() == ROOT::ENTupleStructure::kCollection) {
      // Inner fields of collections are provided as projected collections of only that inner field,
      // E.g. we provide a projected collection RVec<RVec<float>> for "event.tracks.hits.x" in the example
      // above.
      bool representableAsRVec =
         convertToRVec && (field.GetTypeName().substr(0, 19) == "ROOT::VecOps::RVec<" ||
                           field.GetTypeName().substr(0, 12) == "std::vector<" || field.GetTypeName() == "");
      const auto *f = field.GetConstSubfields()[0];
      AddField(*f, colName, procProvenance, fieldInfos, representableAsRVec);

      // Note that at the end of the recursion, we handled the inner sub collections as well as the
      // collection as whole, so we are done.
      return;

   } else if (nRepetitions > 0) {
      // Fixed-size array, same logic as ROOT::RVec.
      const auto *f = field.GetConstSubfields()[0];
      AddField(*f, colName, procProvenance, fieldInfos);
      return;
   } else if (field.GetStructure() == ROOT::ENTupleStructure::kRecord) {
      // Inner fields of records are provided as individual RDF columns, e.g. "event.id"
      for (const auto &f : field.GetConstSubfields()) {
         auto innerName = colName.empty() ? f->GetFieldName() : (std::string(colName) + "." + f->GetFieldName());
         // Inner fields of collections of records are always exposed as ROOT::RVec
         AddField(*f, innerName, procProvenance, fieldInfos);
      }

      // Do not add untyped record fields
      if (field.GetTypeName() == "")
         return;
   }

   // The fieldID could be the root field or the class of fieldId might not be loaded.
   // In these cases, only the inner fields are exposed as RDF columns.
   auto fieldOrException = ROOT::RFieldBase::Create(field.GetFieldName(), field.GetTypeName());
   if (!fieldOrException)
      return;
   auto valueField = fieldOrException.Unwrap();
   if (const auto cardinalityField = dynamic_cast<const ROOT::RCardinalityField *>(valueField.get())) {
      // Cardinality fields in RDataFrame are presented as integers
      if (cardinalityField->As32Bit()) {
         valueField = std::make_unique<Internal::RDF::RRDFCardinalityField<std::uint32_t>>(field.GetFieldName());
      } else if (cardinalityField->As64Bit()) {
         valueField = std::make_unique<Internal::RDF::RRDFCardinalityField<std::uint64_t>>(field.GetFieldName());
      } else {
         R__ASSERT(false && "cardinality field stored with an unexpected integer type");
      }
   }

   valueField->SetOnDiskId(field.GetOnDiskId());
   auto valueSubfields = valueField->GetMutableSubfields();
   const auto fieldSubfields = field.GetConstSubfields();
   for (unsigned i = 0; i < valueSubfields.size(); ++i) {
      valueSubfields[i]->SetOnDiskId(fieldSubfields[i]->GetOnDiskId());
   }

   std::unique_ptr<ROOT::RFieldBase> cardinalityField;
   // Collections get the additional "number of" RDF column (e.g. "R_rdf_sizeof_tracks")
   if (!fieldInfos.empty()) {
      const auto &info = fieldInfos.back();
      const std::string name =
         (procProvenance.Empty() ? "R_rdf_sizeof_" : procProvenance.Get() + "_R_rdf_sizeof_") + info.fFieldName;
      if (info.fNRepetitions > 0) {
         cardinalityField = std::make_unique<Internal::RDF::RArraySizeField>(name, info.fNRepetitions);
      } else {
         cardinalityField = std::make_unique<Internal::RDF::RRDFCardinalityField<std::size_t>>(name);
      }
      cardinalityField->SetOnDiskId(info.fFieldId);
   }

   for (auto i = fieldInfos.rbegin(); i != fieldInfos.rend(); ++i) {
      const auto &fieldInfo = *i;

      const auto valueFieldName = valueField->GetFieldName();

      if (fieldInfo.fNRepetitions > 0) {
         // Fixed-size array, read it as ROOT::RVec in memory
         valueField =
            std::make_unique<ROOT::RArrayAsRVecField>(valueFieldName, valueField->Clone("_0"), fieldInfo.fNRepetitions);
      } else {
         // Actual collection. A std::vector or ROOT::RVec gets added as a ROOT::RVec. All other collection types keep
         // their original type.
         if (convertToRVec) {
            valueField = std::make_unique<ROOT::RRVecField>(valueFieldName, valueField->Clone("_0"));
         } else {
            auto outerFieldType = fieldInfo.fTypeName;
            valueField = ROOT::RFieldBase::Create(valueFieldName, outerFieldType).Unwrap();
         }
      }

      valueField->SetOnDiskId(fieldInfo.fFieldId);

      // Skip the inner-most collection level to construct the cardinality column
      // It's taken care of by the `if (!fieldInfos.empty())` scope above
      if (i != fieldInfos.rbegin()) {
         const auto cardinalityFieldName = cardinalityField->GetFieldName();
         if (fieldInfo.fNRepetitions > 0) {
            // This collection level refers to a fixed-size array
            cardinalityField = std::make_unique<ROOT::RArrayAsRVecField>(
               cardinalityFieldName, cardinalityField->Clone("_0"), fieldInfo.fNRepetitions);
         } else {
            // This collection level refers to an RVec
            cardinalityField = std::make_unique<ROOT::RRVecField>(cardinalityFieldName, cardinalityField->Clone("_0"));
         }

         cardinalityField->SetOnDiskId(fieldInfo.fFieldId);
      }
   }

   if (cardinalityField) {
      std::string cardinalityFieldName =
         (procProvenance.Empty() ? "R_rdf_sizeof_" : procProvenance.Get() + "_R_rdf_sizeof_") + std::string(colName);
      fColumnNames.emplace_back(cardinalityFieldName);
      fColumnTypes.emplace_back(cardinalityField->GetTypeName());
      fProcessor->AddFieldToEntry(std::move(cardinalityField), cardinalityFieldName, nullptr,
                                  Internal::RNTupleProcessorProvenance(), /*isJoinField=*/false);
   }

   fieldInfos.emplace_back(field.GetOnDiskId(), field.GetFieldName(), field.GetTypeName(), nRepetitions);
   std::string canonicalName = (procProvenance.Empty() ? "" : procProvenance.Get() + ".") + std::string(colName);
   fColumnNames.emplace_back(canonicalName);
   fColumnTypes.emplace_back(valueField->GetTypeName());

   // Add nested fields explicitly to the entry, because they may be mapped differently.
   if (fieldInfos.size() > 1) {
      fProcessor->AddFieldToEntry(std::move(valueField), canonicalName, nullptr, Internal::RNTupleProcessorProvenance(),
                                  /*isJoinField=*/false);
   }
}

ROOT::Experimental::RDF::RNTupleProcessorDS::RNTupleProcessorDS(
   std::unique_ptr<ROOT::Experimental::RNTupleProcessor> processor)
   : fProcessor(std::move(processor))
{
   // Do not add the subfields now, this is handled by AddField.
   fProcessor->AddAllFieldsToEntry(Internal::RNTupleProcessorProvenance(), /*addPrefixProvenance=*/false,
                                   /*includeSubfields=*/false);
   const auto &entry = fProcessor->GetEntry();
   for (auto fieldIdx : entry.GetFieldIndices()) {
      const auto &field = entry.GetValue(fieldIdx).GetField();
      AddField(field, entry.GetQualifiedFieldName(fieldIdx), entry.GetFieldProvenance(fieldIdx),
               std::vector<ROOT::Experimental::RDF::RNTupleProcessorDS::RFieldInfo>());
   }
}

ROOT::RDF::RDataSource::Record_t
ROOT::Experimental::RDF::RNTupleProcessorDS::GetColumnReadersImpl(std::string_view /* name */,
                                                                  const std::type_info & /* ti */)
{
   // This datasource uses the newer GetColumnReaders() API
   return {};
}

std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
ROOT::Experimental::RDF::RNTupleProcessorDS::GetColumnReaders(unsigned int /* slot */, std::string_view name,
                                                              const std::type_info &tid)
{
   // At this point we can assume that `name` will be found in fColumnNames
   const auto requestedType = ROOT::Internal::GetRenormalizedTypeName(ROOT::Internal::RDF::TypeID2TypeName(tid));

   // First check if a field with the requested type already exists. If that is the case, we can immediately create a
   // column reader for it.
   if (fProcessor->GetEntry().FindFieldIndex(name, requestedType)) {
      auto valuePtr = fProcessor->RequestField(std::string(name), requestedType);
      auto reader = std::make_unique<Internal::RDF::RNTupleProcessorColumnReader>(*fProcessor, valuePtr);
      fActiveColumnReaders.emplace_back(reader.get());

      return reader;
   }

   // Secondly, check whether a field with the same name, but a different on-disk type exists. If this is not the case,
   // first try to add the field. If the field cannot be added, throw an exception.
   // Otherwise, we create a new field with the requested type and add it to the processor entry
   // before creating a column reader for it.
   auto fieldIdx = fProcessor->GetEntry().FindFieldIndex(name);
   if (!fieldIdx) {
      try {
         auto valuePtr = fProcessor->RequestField(std::string(name), requestedType);
         auto reader = std::make_unique<Internal::RDF::RNTupleProcessorColumnReader>(*fProcessor, valuePtr);
         fActiveColumnReaders.emplace_back(reader.get());

         return reader;
      } catch (const ROOT::RException &) {
         throw std::runtime_error("RNTupleProcessorDS: Column \"" + std::string(name) + "\" does not exist");
      }
   }

   const auto &field = fProcessor->GetEntry().GetField(*fieldIdx);
   const std::string strName = std::string(name);
   std::unique_ptr<ROOT::RFieldBase> newField;
   if (dynamic_cast<const ROOT::RCardinalityField *>(&field)) {
      if (requestedType == "bool") {
         newField = std::make_unique<ROOT::Experimental::Internal::RDF::RRDFCardinalityField<bool>>(strName);
      } else if (requestedType == "char") {
         newField = std::make_unique<ROOT::Experimental::Internal::RDF::RRDFCardinalityField<char>>(strName);
      } else if (requestedType == "std::int8_t") {
         newField = std::make_unique<ROOT::Experimental::Internal::RDF::RRDFCardinalityField<std::int8_t>>(strName);
      } else if (requestedType == "std::uint8_t") {
         newField = std::make_unique<ROOT::Experimental::Internal::RDF::RRDFCardinalityField<std::uint8_t>>(strName);
      } else if (requestedType == "std::int16_t") {
         newField = std::make_unique<ROOT::Experimental::Internal::RDF::RRDFCardinalityField<std::int16_t>>(strName);
      } else if (requestedType == "std::uint16_t") {
         newField = std::make_unique<ROOT::Experimental::Internal::RDF::RRDFCardinalityField<std::uint16_t>>(strName);
      } else if (requestedType == "std::int32_t") {
         newField = std::make_unique<ROOT::Experimental::Internal::RDF::RRDFCardinalityField<std::int32_t>>(strName);
      } else if (requestedType == "std::uint32_t") {
         newField = std::make_unique<ROOT::Experimental::Internal::RDF::RRDFCardinalityField<std::uint32_t>>(strName);
      } else if (requestedType == "std::int64_t") {
         newField = std::make_unique<ROOT::Experimental::Internal::RDF::RRDFCardinalityField<std::int64_t>>(strName);
      } else if (requestedType == "std::uint64_t") {
         newField = std::make_unique<ROOT::Experimental::Internal::RDF::RRDFCardinalityField<std::uint64_t>>(strName);
      } else {
         throw std::runtime_error("RNTupleProcessorDS: Could not create field with type \"" + requestedType +
                                  "\" for column \"" + strName + "\"");
      }
   } else {
      auto newAltProtoFieldOrException = ROOT::RFieldBase::Create(strName, requestedType);
      if (!newAltProtoFieldOrException) {
         throw std::runtime_error("RNTupleProcessorDS: Could not create field with type \"" + requestedType +
                                  "\" for column \"" + strName + "\"");
      }
      newField = newAltProtoFieldOrException.Unwrap();
   }
   newField->SetOnDiskId(field.GetOnDiskId());

   try {
      fProcessor->AddFieldToEntry(std::move(newField), strName, nullptr, Internal::RNTupleProcessorProvenance(),
                                  /*isJoinField=*/false);
   } catch (const ROOT::RException &) {
      std::string msg = "RNTupleProcessorDS: invalid type \"" + requestedType + "\" for column \"" + strName +
                        "\" with on-disk type \"" + field.GetTypeName() + "\"";
      throw std::runtime_error(msg);
   }

   auto valuePtr = fProcessor->RequestField(std::string(name), requestedType);
   auto reader = std::make_unique<Internal::RDF::RNTupleProcessorColumnReader>(*fProcessor, valuePtr);
   fActiveColumnReaders.emplace_back(reader.get());

   return reader;
}

std::vector<std::pair<ULong64_t, ULong64_t>> ROOT::Experimental::RDF::RNTupleProcessorDS::GetEntryRanges()
{
   std::vector<std::pair<ULong64_t, ULong64_t>> ranges;
   if (fProcessor->GetNEntries() == fProcessor->GetNEntriesProcessed())
      return ranges;
   ranges.emplace_back(0, fProcessor->GetNEntries());
   return ranges;
}

void ROOT::Experimental::RDF::RNTupleProcessorDS::InitSlot(unsigned int /* slot */, ULong64_t /* firstEntry */)
{
   assert(fNSlots == 1 && "MT not supported");
}

void ROOT::Experimental::RDF::RNTupleProcessorDS::FinalizeSlot(unsigned int /* slot */)
{
   assert(fNSlots == 1 && "MT not supported");
}

std::string ROOT::Experimental::RDF::RNTupleProcessorDS::GetTypeName(std::string_view colName) const
{
   auto colNamePos = std::find(fColumnNames.begin(), fColumnNames.end(), colName);

   if (colNamePos == fColumnNames.end()) {
      auto msg = std::string("RNTupleProcessorDS: There is no column with name \"") + std::string(colName) + "\"";
      throw std::runtime_error(msg);
   }

   const auto index = std::distance(fColumnNames.begin(), colNamePos);
   return fColumnTypes[index];
}

bool ROOT::Experimental::RDF::RNTupleProcessorDS::HasColumn(std::string_view colName) const
{
   return std::find(fColumnNames.begin(), fColumnNames.end(), colName) != fColumnNames.end();
}

void ROOT::Experimental::RDF::RNTupleProcessorDS::Initialize()
{
   fProcessor->Reset();
   fProcessor->Connect(fProcessor->GetEntry().GetFieldIndices(), Internal::RNTupleProcessorProvenance(),
                       /*updateFields=*/false);
   return;
}

void ROOT::Experimental::RDF::RNTupleProcessorDS::Finalize()
{
   return;
}

void ROOT::Experimental::RDF::RNTupleProcessorDS::SetNSlots(unsigned int nSlots)
{
   assert(fNSlots == 0);
   assert(nSlots == 1);
   fNSlots = nSlots;
}

bool ROOT::Experimental::RDF::RNTupleProcessorDS::SetEntry(unsigned int /* slot */, ULong64_t entry)
{
   if (fProcessor->GetCurrentEntryNumber() != entry)
      return fProcessor->LoadEntry(entry) != ROOT::kInvalidNTupleIndex;
   return true;
}

ROOT::RDataFrame
ROOT::Experimental::RDF::FromRNTupleProcessor(std::unique_ptr<ROOT::Experimental::RNTupleProcessor> processor)
{
   return ROOT::RDataFrame(std::make_unique<ROOT::Experimental::RDF::RNTupleProcessorDS>(std::move(processor)));
}

ROOT::RDF::RSampleInfo ROOT::Experimental::RDF::RNTupleProcessorDS::CreateSampleInfo(
   unsigned int /* slot */, const std::unordered_map<std::string, ROOT::RDF::Experimental::RSample *> &sampleMap) const
{
   const auto &ntupleID = fProcessor->GetProcessorName();

   // TODO: There is no support for RNTuple in RDatasetSpec, thus the sample map
   // is always empty at the moment.
   if (sampleMap.empty())
      return ROOT::RDF::RSampleInfo(ntupleID, std::make_pair(0, fProcessor->GetNEntries()));

   if (sampleMap.find(ntupleID) == sampleMap.end())
      throw std::runtime_error("Full sample identifier '" + ntupleID + "' cannot be found in the available samples.");

   return ROOT::RDF::RSampleInfo(ntupleID, std::make_pair(0, fProcessor->GetNEntries()), sampleMap.at(ntupleID));
}
