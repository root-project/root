/// \file RNTupleProcessor.cxx
/// \ingroup NTuple
/// \author Florine de Geus <florine.de.geus@cern.ch>
/// \date 2024-03-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleProcessor.hxx>

#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/StringUtils.hxx>

#include <TDirectory.h>

#include <iomanip>

std::unique_ptr<ROOT::Internal::RPageSource> ROOT::Experimental::RNTupleOpenSpec::CreatePageSource() const
{
   if (const std::string *storagePath = std::get_if<std::string>(&fStorage))
      return ROOT::Internal::RPageSource::Create(fNTupleName, *storagePath);

   auto dir = std::get<TDirectory *>(fStorage);
   auto ntuple = std::unique_ptr<ROOT::RNTuple>(dir->Get<ROOT::RNTuple>(fNTupleName.c_str()));
   return ROOT::Internal::RPageSourceFile::CreateFromAnchor(*ntuple);
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::Create(RNTupleOpenSpec ntuple, std::string_view processorName)
{
   return std::unique_ptr<RNTupleSingleProcessor>(new RNTupleSingleProcessor(std::move(ntuple), processorName));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(std::vector<RNTupleOpenSpec> ntuples, std::string_view processorName)
{
   if (ntuples.empty())
      throw RException(R__FAIL("at least one RNTuple must be provided"));

   std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors;
   innerProcessors.reserve(ntuples.size());

   for (auto &ntuple : ntuples) {
      innerProcessors.emplace_back(Create(std::move(ntuple)));
   }

   return CreateChain(std::move(innerProcessors), processorName);
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors,
                                                  std::string_view processorName)
{
   if (innerProcessors.empty())
      throw RException(R__FAIL("at least one inner processor must be provided"));

   return std::unique_ptr<RNTupleChainProcessor>(new RNTupleChainProcessor(std::move(innerProcessors), processorName));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateJoin(RNTupleOpenSpec primaryNTuple, RNTupleOpenSpec auxNTuple,
                                                 const std::vector<std::string> &joinFields,
                                                 std::string_view processorName)
{
   if (joinFields.size() > 4) {
      throw RException(R__FAIL("a maximum of four join fields is allowed"));
   }

   if (std::unordered_set(joinFields.begin(), joinFields.end()).size() < joinFields.size()) {
      throw RException(R__FAIL("join fields must be unique"));
   }

   std::unique_ptr<RNTupleProcessor> primaryProcessor = Create(primaryNTuple, processorName);

   std::unique_ptr<RNTupleProcessor> auxProcessor = Create(auxNTuple);

   return CreateJoin(std::move(primaryProcessor), std::move(auxProcessor), joinFields, processorName);
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateJoin(std::unique_ptr<RNTupleProcessor> primaryProcessor,
                                                 std::unique_ptr<RNTupleProcessor> auxProcessor,
                                                 const std::vector<std::string> &joinFields,
                                                 std::string_view processorName)
{
   if (joinFields.size() > 4) {
      throw RException(R__FAIL("a maximum of four join fields is allowed"));
   }

   if (std::unordered_set(joinFields.begin(), joinFields.end()).size() < joinFields.size()) {
      throw RException(R__FAIL("join fields must be unique"));
   }

   return std::unique_ptr<RNTupleJoinProcessor>(
      new RNTupleJoinProcessor(std::move(primaryProcessor), std::move(auxProcessor), joinFields, processorName));
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleSingleProcessor::RNTupleSingleProcessor(RNTupleOpenSpec ntuple,
                                                                   std::string_view processorName)
   : RNTupleProcessor(processorName), fNTupleSpec(std::move(ntuple))
{
   if (fProcessorName.empty()) {
      fProcessorName = fNTupleSpec.fNTupleName;
   }
}

void ROOT::Experimental::RNTupleSingleProcessor::Initialize()
{
   // The processor has already been initialized.
   if (IsInitialized())
      return;
   fPageSource = fNTupleSpec.CreatePageSource();
   fPageSource->Attach();
   ROOT::RNTupleDescriptor::RCreateModelOptions opts;
   opts.SetCreateBare(true);
   fProtoModel = fPageSource->GetSharedDescriptorGuard()->CreateModel(opts);
   fProtoModel->Unfreeze();
   fEntry = std::make_shared<Internal::RNTupleProcessorEntry>();
}

bool ROOT::Experimental::RNTupleSingleProcessor::FieldExists(std::string_view fieldName)
{
   Initialize();
   auto desc = fPageSource->GetSharedDescriptorGuard();
   auto fieldZeroId = desc->GetFieldZeroId();

   // TODO handle subfields
   return desc->FindFieldId(fieldName, fieldZeroId) != ROOT::kInvalidDescriptorId;
}

ROOT::RResult<ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t>
ROOT::Experimental::RNTupleSingleProcessor::AddFieldToEntry(std::string_view fieldName, void *valuePtr,
                                                            std::string_view auxFieldPrefix)
{
   auto fieldIdx = fEntry->FindFieldIndex(fieldName);
   if (!fieldIdx) {
      try {
         if (!auxFieldPrefix.empty()) {
            auto strippedFieldName = fieldName.substr(auxFieldPrefix.size() + 1);
            auto &field = fProtoModel->GetMutableField(strippedFieldName);
            fieldIdx = fEntry->AddField(fieldName, field, valuePtr, /*isAuxiliary=*/true);
            return *fieldIdx;
         }

         auto &field = fProtoModel->GetMutableField(fieldName);
         fieldIdx = fEntry->AddField(fieldName, field, valuePtr);
         return *fieldIdx;
      } catch (const ROOT::RException &) {
         return R__FAIL("cannot register field with name \"" + std::string(fieldName) +
                        "\" because it is not present in the on-disk information of the RNTuple(s) this "
                        "processor is created from");
      }
   } else {
      return *fieldIdx;
   }
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleSingleProcessor::LoadEntry(ROOT::NTupleSize_t entryNumber)
{
   if (entryNumber >= fNEntries || !fEntry)
      return kInvalidNTupleIndex;

   fEntry->Read(entryNumber);

   fNEntriesProcessed++;
   fCurrentEntryNumber = entryNumber;
   return entryNumber;
}

void ROOT::Experimental::RNTupleSingleProcessor::Connect(
   const std::vector<ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t> &fieldIdxs, bool updateEntry)
{
   Initialize();

   // The processor has already been connected.
   if (fNEntries != kInvalidNTupleIndex && !updateEntry)
      return;

   fNEntries = fPageSource->GetNEntries();

   auto desc = fPageSource->GetSharedDescriptorGuard();
   auto fieldZeroId = desc->GetFieldZeroId();

   for (const auto &fieldIdx : fieldIdxs) {
      const auto &entryField = fEntry->GetField(fieldIdx);

      // TODO handle subfields
      auto onDiskId = desc->FindFieldId(entryField.GetQualifiedFieldName(), fieldZeroId);
      // The field we are trying to connect is not present in the ntuple
      if (onDiskId == kInvalidDescriptorId) {
         fEntry->SetFieldValidity(fieldIdx, false);
         continue;
      }

      auto &modelField = fProtoModel->GetMutableField(entryField.GetQualifiedFieldName());

      if (entryField.GetState() == RFieldBase::EState::kConnectedToSource && &entryField != &modelField) {
         fEntry->UpdateField(fieldIdx, modelField);
      }

      if (modelField.GetState() == RFieldBase::EState::kUnconnected) {
         modelField.SetOnDiskId(onDiskId);
         ROOT::Internal::CallConnectPageSourceOnField(modelField, *fPageSource);
      }

      fEntry->SetFieldValidity(fieldIdx, true);
   }
}

void ROOT::Experimental::RNTupleSingleProcessor::AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable,
                                                                       ROOT::NTupleSize_t entryOffset)
{
   Connect(fEntry->GetFieldIndices(), /*updateEntry=*/false);
   joinTable.Add(*fPageSource, Internal::RNTupleJoinTable::kDefaultPartitionKey, entryOffset);
}

void ROOT::Experimental::RNTupleSingleProcessor::PrintStructureImpl(std::ostream &output) const
{
   static constexpr int width = 32;

   std::string ntupleNameTrunc = fNTupleSpec.fNTupleName.substr(0, width - 4);
   if (ntupleNameTrunc.size() < fNTupleSpec.fNTupleName.size())
      ntupleNameTrunc = fNTupleSpec.fNTupleName.substr(0, width - 6) + "..";

   output << "+" << std::setfill('-') << std::setw(width - 1) << "+\n";
   output << std::setfill(' ') << "| " << ntupleNameTrunc << std::setw(width - 2 - ntupleNameTrunc.size()) << " |\n";

   if (const std::string *storage = std::get_if<std::string>(&fNTupleSpec.fStorage)) {
      std::string storageTrunc = storage->substr(0, width - 5);
      if (storageTrunc.size() < storage->size())
         storageTrunc = storage->substr(0, width - 8) + "...";

      output << std::setfill(' ') << "| " << storageTrunc << std::setw(width - 2 - storageTrunc.size()) << " |\n";
   } else {
      output << "| " << std::setw(width - 2) << " |\n";
   }

   output << "+" << std::setfill('-') << std::setw(width - 1) << "+\n";
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleChainProcessor::RNTupleChainProcessor(
   std::vector<std::unique_ptr<RNTupleProcessor>> processors, std::string_view processorName)
   : RNTupleProcessor(processorName), fInnerProcessors(std::move(processors))
{
   if (fProcessorName.empty()) {
      // `CreateChain` ensures there is at least one inner processor.
      fProcessorName = fInnerProcessors[0]->GetProcessorName();
   }

   fInnerNEntries.assign(fInnerProcessors.size(), kInvalidNTupleIndex);
}

void ROOT::Experimental::RNTupleChainProcessor::Initialize()
{
   if (IsInitialized())
      return;
   fInnerProcessors[0]->Initialize();
   fProtoModel = fInnerProcessors[0]->GetProtoModel().Clone();
   fEntry = std::make_shared<Internal::RNTupleProcessorEntry>();
   fInnerProcessors[0]->SetEntry(fEntry);
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleChainProcessor::GetNEntries()
{
   if (fNEntries == kInvalidNTupleIndex) {
      fNEntries = 0;

      for (unsigned i = 0; i < fInnerProcessors.size(); ++i) {
         if (fInnerNEntries[i] == kInvalidNTupleIndex) {
            fInnerNEntries[i] = fInnerProcessors[i]->GetNEntries();
         }

         fNEntries += fInnerNEntries[i];
      }
   }

   return fNEntries;
}

void ROOT::Experimental::RNTupleChainProcessor::Connect(
   const std::vector<ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t> &fieldIdxs,
   bool /* updateEntry */)
{
   Initialize();
   fFieldIdxs = fieldIdxs;
   ConnectInnerProcessor(fCurrentProcessorNumber);
}

void ROOT::Experimental::RNTupleChainProcessor::ConnectInnerProcessor(std::size_t processorNumber)
{
   auto &innerProc = fInnerProcessors[processorNumber];
   innerProc->Initialize();
   innerProc->SetEntry(fEntry);
   innerProc->Connect(fFieldIdxs, /*updateEntry=*/true);
}

ROOT::RResult<ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t>
ROOT::Experimental::RNTupleChainProcessor::AddFieldToEntry(std::string_view fieldName, void *valuePtr,
                                                           std::string_view auxFieldPrefix)
{
   return R__FORWARD_RESULT(
      fInnerProcessors[fCurrentProcessorNumber]->AddFieldToEntry(fieldName, valuePtr, auxFieldPrefix));
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleChainProcessor::LoadEntry(ROOT::NTupleSize_t entryNumber)
{
   ROOT::NTupleSize_t localEntryNumber = entryNumber;
   std::size_t currProcessorNumber = 0;
   if (entryNumber < fCurrentEntryNumber) {
      fCurrentProcessorNumber = 0;
      ConnectInnerProcessor(fCurrentProcessorNumber);
   }

   // As long as the entry fails to load from the current processor, we decrement the local entry number with the number
   // of entries in this processor and try with the next processor until we find the correct local entry number.
   while (fInnerProcessors[currProcessorNumber]->LoadEntry(localEntryNumber) == kInvalidNTupleIndex) {
      if (fInnerNEntries[currProcessorNumber] == kInvalidNTupleIndex) {
         fInnerNEntries[currProcessorNumber] = fInnerProcessors[currProcessorNumber]->GetNEntries();
      }

      localEntryNumber -= fInnerNEntries[currProcessorNumber];

      // The provided global entry number is larger than the number of available entries.
      if (++currProcessorNumber >= fInnerProcessors.size())
         return kInvalidNTupleIndex;

      ConnectInnerProcessor(currProcessorNumber);
   }

   fCurrentProcessorNumber = currProcessorNumber;
   fNEntriesProcessed++;
   fCurrentEntryNumber = entryNumber;
   return entryNumber;
}

void ROOT::Experimental::RNTupleChainProcessor::AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable,
                                                                      ROOT::NTupleSize_t entryOffset)
{
   for (unsigned i = 0; i < fInnerProcessors.size(); ++i) {
      const auto &innerProc = fInnerProcessors[i];
      // TODO can this be done (more) lazily? I.e. only when a match cannot be found in the current inner proc?
      innerProc->Initialize();
      innerProc->AddEntriesToJoinTable(joinTable, entryOffset);
      entryOffset += innerProc->GetNEntries();
   }
}

void ROOT::Experimental::RNTupleChainProcessor::PrintStructureImpl(std::ostream &output) const
{
   for (const auto &innerProc : fInnerProcessors) {
      innerProc->PrintStructure(output);
   }
}

//------------------------------------------------------------------------------

namespace ROOT::Experimental::Internal {
class RAuxiliaryProcessorField final : public ROOT::RRecordField {
private:
   using RFieldBase::GenerateColumns;
   void GenerateColumns() final
   {
      throw RException(R__FAIL("RAuxiliaryProcessorField fields must only be used for reading"));
   }

public:
   RAuxiliaryProcessorField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> itemFields)
      : ROOT::RRecordField(fieldName, "RAuxiliaryProcessorField")
   {
      fOffsets.reserve(itemFields.size());
      for (auto &item : itemFields) {
         fOffsets.push_back(GetItemPadding(fSize, item->GetAlignment()));
      }
      AttachItemFields(std::move(itemFields));
   }
};
} // namespace ROOT::Experimental::Internal

namespace {
bool IsAuxiliaryField(std::string_view fieldName, std::string_view auxProcessorName)
{
   return fieldName.find(std::string(auxProcessorName) + ".") == 0;
}
} // anonymous namespace

ROOT::Experimental::RNTupleJoinProcessor::RNTupleJoinProcessor(std::unique_ptr<RNTupleProcessor> primaryProcessor,
                                                               std::unique_ptr<RNTupleProcessor> auxProcessor,
                                                               const std::vector<std::string> &joinFields,
                                                               std::string_view processorName)
   : RNTupleProcessor(processorName),
     fPrimaryProcessor(std::move(primaryProcessor)),
     fAuxiliaryProcessor(std::move(auxProcessor)),
     fJoinFieldNames(joinFields)
{
   if (fProcessorName.empty()) {
      fProcessorName = fPrimaryProcessor->GetProcessorName();
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::Initialize()
{
   if (IsInitialized())
      return;

   fPrimaryProcessor->Initialize();
   fAuxiliaryProcessor->Initialize();

   // fProtoModel = fPrimaryProcessor->GetProtoModel().Clone();
   // fAuxiliaryProtoModel = fAuxiliaryProcessor->GetProtoModel().Clone();

   // If the primaryProcessor has a field with the name of the auxProcessor (either as a "proper" field or because the
   // primary processor itself is a join where its auxProcessor bears the same name as the current auxProcessor), there
   // will be name conflicts, so error out.
   if (auto &primaryModel = fPrimaryProcessor->GetProtoModel();
       primaryModel.GetFieldNames().find(fAuxiliaryProcessor->GetProcessorName()) !=
       primaryModel.GetFieldNames().end()) {
      throw RException(R__FAIL("a field or nested auxiliary processor named \"" +
                               fAuxiliaryProcessor->GetProcessorName() +
                               "\" is already present as a field in the primary processor; rename the auxiliary "
                               "processor to avoid conflicts"));
   }

   SetProtoModel(fPrimaryProcessor->GetProtoModel().Clone(), fAuxiliaryProcessor->GetProtoModel().Clone());

   fEntry = std::make_shared<Internal::RNTupleProcessorEntry>();

   fPrimaryProcessor->SetEntry(fEntry);
   fAuxiliaryProcessor->SetEntry(fEntry);

   if (!fJoinFieldNames.empty()) {
      for (const auto &joinField : fJoinFieldNames) {
         if (!fAuxiliaryProcessor->FieldExists(joinField)) {
            throw RException(R__FAIL("could not find join field \"" + joinField + "\" in auxiliary processor \"" +
                                     fAuxiliaryProcessor->GetProcessorName() + "\""));
         }
         auto fieldIdx = AddFieldToEntry(joinField);
         if (!fieldIdx)
            throw RException(R__FAIL("could not find join field \"" + joinField + "\" in primary processor \"" +
                                     fPrimaryProcessor->GetProcessorName() + "\""));
         fJoinFieldIdxs.emplace_back(fieldIdx.Unwrap());
      }

      fJoinTable = Internal::RNTupleJoinTable::Create(fJoinFieldNames);
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::Connect(
   const std::vector<ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t> &fieldIdxs, bool updateEntry)
{
   Initialize();

   for (const auto &fieldIdx : fieldIdxs) {
      if (fEntry->FieldIsAuxiliary(fieldIdx))
         fAuxiliaryFieldIdxs.push_back(fieldIdx);
      else
         fPrimaryFieldIdxs.push_back(fieldIdx);
   }

   fPrimaryProcessor->Connect(fPrimaryFieldIdxs, updateEntry);
   fAuxiliaryProcessor->Connect(fAuxiliaryFieldIdxs, updateEntry);
}

void ROOT::Experimental::RNTupleJoinProcessor::SetProtoModel(std::unique_ptr<ROOT::RNTupleModel> primaryModel,
                                                             std::unique_ptr<RNTupleModel> auxModel)
{
   fProtoModel = std::move(primaryModel);
   fProtoModel->Unfreeze();

   // Create an anonymous record field for the auxiliary processor, containing its top-level fields. These original
   // top-level fields are registered as subfields in this processor's proto-model, such that they can be accessed as
   // `auxNTupleName.fieldName`.
   std::vector<std::unique_ptr<ROOT::RFieldBase>> auxFields;
   auxFields.reserve(auxModel->GetFieldNames().size());

   for (const auto &fieldName : auxModel->GetFieldNames()) {
      auxFields.emplace_back(auxModel->GetConstField(fieldName).Clone(fieldName));
   }

   auto auxParentField = std::make_unique<Internal::RAuxiliaryProcessorField>(fAuxiliaryProcessor->GetProcessorName(),
                                                                              std::move(auxFields));
   const auto &subFields = auxParentField->GetConstSubfields();
   fProtoModel->AddField(std::move(auxParentField));

   for (const auto &field : subFields) {
      fProtoModel->RegisterSubfield(field->GetQualifiedFieldName());

      if (field->GetTypeName() == "RAuxiliaryProcessorField") {
         for (const auto &auxSubField : field->GetConstSubfields()) {
            fProtoModel->RegisterSubfield(auxSubField->GetQualifiedFieldName());
         }
      }
   }
}

ROOT::RResult<ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t>
ROOT::Experimental::RNTupleJoinProcessor::AddFieldToEntry(std::string_view fieldName, void *valuePtr,
                                                          std::string_view auxFieldPrefix)
{
   std::string updatedPrefix = auxFieldPrefix.empty()
                                  ? fAuxiliaryProcessor->GetProcessorName()
                                  : std::string(auxFieldPrefix) + "." + fAuxiliaryProcessor->GetProcessorName();
   if (IsAuxiliaryField(fieldName, updatedPrefix)) {
      std::string newPrefix = auxFieldPrefix.empty()
                                 ? fAuxiliaryProcessor->GetProcessorName()
                                 : fAuxiliaryProcessor->GetProcessorName() + "." + std::string(auxFieldPrefix);
      auto fieldIdx = fAuxiliaryProcessor->AddFieldToEntry(fieldName, valuePtr, newPrefix);
      if (fieldIdx)
         fAuxiliaryFieldIdxs.push_back(fieldIdx.Unwrap());
      return R__FORWARD_RESULT(fieldIdx);
   } else {
      auto fieldIdx = fPrimaryProcessor->AddFieldToEntry(fieldName, valuePtr, auxFieldPrefix);
      if (fieldIdx)
         fPrimaryFieldIdxs.push_back(fieldIdx.Unwrap());
      return R__FORWARD_RESULT(fieldIdx);
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::SetAuxiliaryFieldValidity(bool isValid)
{
   for (const auto &fieldIdx : fAuxiliaryFieldIdxs) {
      fEntry->SetFieldValidity(fieldIdx, isValid);
   }
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleJoinProcessor::LoadEntry(ROOT::NTupleSize_t entryNumber)
{
   if (fPrimaryProcessor->LoadEntry(entryNumber) == kInvalidNTupleIndex)
      return kInvalidNTupleIndex;

   fCurrentEntryNumber = entryNumber;
   fNEntriesProcessed++;

   if (!fJoinTable) {
      // The auxiliary processor's fields are valid if the entry could be loaded.
      SetAuxiliaryFieldValidity(fAuxiliaryProcessor->LoadEntry(entryNumber) != kInvalidNTupleIndex);
      return entryNumber;
   }

   if (!fJoinTableIsBuilt) {
      fAuxiliaryProcessor->AddEntriesToJoinTable(*fJoinTable);
      fJoinTableIsBuilt = true;
   }

   // Collect the values of the join fields for this entry.
   std::vector<void *> valPtrs;
   valPtrs.reserve(fJoinFieldIdxs.size());
   for (const auto &fieldIdx : fJoinFieldIdxs) {
      auto ptr = fEntry->GetPtr<void>(fieldIdx);
      valPtrs.push_back(ptr.get());
   }

   // Find the entry index corresponding to the join field values for each auxiliary processor and load the
   // corresponding entry.
   const auto entryIdx = fJoinTable->GetEntryIndex(valPtrs);

   if (entryIdx == kInvalidNTupleIndex) {
      SetAuxiliaryFieldValidity(false);
   } else {
      SetAuxiliaryFieldValidity(true);
      for (const auto &fieldIdx : fAuxiliaryFieldIdxs) {
         fEntry->ReadValue(fieldIdx, entryIdx);
      }
   }

   return entryNumber;
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleJoinProcessor::GetNEntries()
{
   if (fNEntries == kInvalidNTupleIndex)
      fNEntries = fPrimaryProcessor->GetNEntries();
   return fNEntries;
}

void ROOT::Experimental::RNTupleJoinProcessor::AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable,
                                                                     ROOT::NTupleSize_t entryOffset)
{
   fPrimaryProcessor->AddEntriesToJoinTable(joinTable, entryOffset);
}

void ROOT::Experimental::RNTupleJoinProcessor::PrintStructureImpl(std::ostream &output) const
{
   std::ostringstream primaryStructureStr;
   fPrimaryProcessor->PrintStructure(primaryStructureStr);
   const auto primaryStructure = ROOT::Split(primaryStructureStr.str(), "\n", /*skipEmpty=*/true);
   const auto primaryStructureWidth = primaryStructure.front().size();

   std::ostringstream auxStructureStr;
   fAuxiliaryProcessor->PrintStructure(auxStructureStr);
   const auto auxStructure = ROOT::Split(auxStructureStr.str(), "\n", /*skipEmpty=*/true);

   const auto maxLength = std::max(primaryStructure.size(), auxStructure.size());
   for (unsigned i = 0; i < maxLength; i++) {
      if (i < primaryStructure.size())
         output << primaryStructure[i];
      else
         output << std::setw(primaryStructureWidth) << "";

      if (i < auxStructure.size())
         output << " " << auxStructure[i];

      output << "\n";
   }
}
