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
ROOT::Experimental::RNTupleProcessor::Create(RNTupleOpenSpec ntuple, std::unique_ptr<ROOT::RNTupleModel> model,
                                             std::string_view processorName)
{
   return std::unique_ptr<RNTupleSingleProcessor>(
      new RNTupleSingleProcessor(std::move(ntuple), std::move(model), processorName));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(std::vector<RNTupleOpenSpec> ntuples,
                                                  std::unique_ptr<ROOT::RNTupleModel> model,
                                                  std::string_view processorName)
{
   if (ntuples.empty())
      throw RException(R__FAIL("at least one RNTuple must be provided"));

   std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors;
   innerProcessors.reserve(ntuples.size());

   // If no model is provided, infer it from the first ntuple.
   if (!model) {
      auto firstPageSource = ntuples[0].CreatePageSource();
      firstPageSource->Attach();
      model = firstPageSource->GetSharedDescriptorGuard()->CreateModel();
   }

   for (auto &ntuple : ntuples) {
      innerProcessors.emplace_back(Create(std::move(ntuple), model->Clone()));
   }

   return CreateChain(std::move(innerProcessors), std::move(model), processorName);
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors,
                                                  std::unique_ptr<ROOT::RNTupleModel> model,
                                                  std::string_view processorName)
{
   if (innerProcessors.empty())
      throw RException(R__FAIL("at least one inner processor must be provided"));

   // If no model is provided, infer it from the first inner processor.
   if (!model) {
      model = innerProcessors[0]->GetModel().Clone();
   }

   return std::unique_ptr<RNTupleChainProcessor>(
      new RNTupleChainProcessor(std::move(innerProcessors), std::move(model), processorName));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateJoin(RNTupleOpenSpec primaryNTuple, RNTupleOpenSpec auxNTuple,
                                                 const std::vector<std::string> &joinFields,
                                                 std::unique_ptr<ROOT::RNTupleModel> primaryModel,
                                                 std::unique_ptr<ROOT::RNTupleModel> auxModel,
                                                 std::string_view processorName)
{
   if (joinFields.size() > 4) {
      throw RException(R__FAIL("a maximum of four join fields is allowed"));
   }

   if (std::unordered_set(joinFields.begin(), joinFields.end()).size() < joinFields.size()) {
      throw RException(R__FAIL("join fields must be unique"));
   }

   std::unique_ptr<RNTupleProcessor> primaryProcessor;
   if (primaryModel)
      primaryProcessor = Create(primaryNTuple, primaryModel->Clone(), processorName);
   else
      primaryProcessor = Create(primaryNTuple, nullptr, processorName);

   std::unique_ptr<RNTupleProcessor> auxProcessor;
   if (auxModel)
      auxProcessor = Create(auxNTuple, auxModel->Clone());
   else
      auxProcessor = Create(auxNTuple);

   return CreateJoin(std::move(primaryProcessor), std::move(auxProcessor), joinFields, std::move(primaryModel),
                     std::move(auxModel), processorName);
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor> ROOT::Experimental::RNTupleProcessor::CreateJoin(
   std::unique_ptr<RNTupleProcessor> primaryProcessor, std::unique_ptr<RNTupleProcessor> auxProcessor,
   const std::vector<std::string> &joinFields, std::unique_ptr<ROOT::RNTupleModel> primaryModel,
   std::unique_ptr<ROOT::RNTupleModel> auxModel, std::string_view processorName)
{
   if (joinFields.size() > 4) {
      throw RException(R__FAIL("a maximum of four join fields is allowed"));
   }

   if (std::unordered_set(joinFields.begin(), joinFields.end()).size() < joinFields.size()) {
      throw RException(R__FAIL("join fields must be unique"));
   }

   return std::unique_ptr<RNTupleJoinProcessor>(
      new RNTupleJoinProcessor(std::move(primaryProcessor), std::move(auxProcessor), joinFields,
                               std::move(primaryModel), std::move(auxModel), processorName));
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleSingleProcessor::RNTupleSingleProcessor(RNTupleOpenSpec ntuple,
                                                                   std::unique_ptr<ROOT::RNTupleModel> model,
                                                                   std::string_view processorName)
   : RNTupleProcessor(processorName, std::move(model)), fNTupleSpec(std::move(ntuple))
{
   if (!fModel) {
      fPageSource = fNTupleSpec.CreatePageSource();
      fPageSource->Attach();
      fModel = fPageSource->GetSharedDescriptorGuard()->CreateModel();
   }

   if (fProcessorName.empty()) {
      fProcessorName = fNTupleSpec.fNTupleName;
   }

   fModel->Freeze();
   fEntry = fModel->CreateEntry();

   for (const auto &value : *fEntry) {
      auto &field = value.GetField();
      auto token = fEntry->GetToken(field.GetFieldName());

      // If the model has a default entry, use the value pointers from the entry in the entry managed by the
      // processor. This way, the pointers returned by RNTupleModel::MakeField can be used in the processor loop
      // to access the corresponding field values.
      if (!fModel->IsBare()) {
         auto valuePtr = fModel->GetDefaultEntry().GetPtr<void>(token);
         fEntry->BindValue(token, valuePtr);
      }
   }
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleSingleProcessor::LoadEntry(ROOT::NTupleSize_t entryNumber)
{
   Connect();

   if (entryNumber >= fNEntries)
      return kInvalidNTupleIndex;

   fEntry->Read(entryNumber);

   fNEntriesProcessed++;
   fCurrentEntryNumber = entryNumber;
   return entryNumber;
}

void ROOT::Experimental::RNTupleSingleProcessor::SetEntryPointers(const ROOT::REntry &entry,
                                                                  std::string_view fieldNamePrefix)
{
   for (const auto &value : *fEntry) {
      std::string fieldName = value.GetField().GetQualifiedFieldName();
      auto valuePtr = fieldNamePrefix.empty() ? entry.GetPtr<void>(fieldName)
                                              : entry.GetPtr<void>(std::string(fieldNamePrefix) + "." + fieldName);

      fEntry->BindValue(fieldName, valuePtr);
   }
}

void ROOT::Experimental::RNTupleSingleProcessor::Connect()
{
   // The processor has already been connected.
   if (fNEntries != kInvalidNTupleIndex)
      return;

   if (!fPageSource)
      fPageSource = fNTupleSpec.CreatePageSource();
   fPageSource->Attach();
   fNEntries = fPageSource->GetNEntries();

   auto desc = fPageSource->GetSharedDescriptorGuard();
   auto &fieldZero = ROOT::Internal::GetFieldZeroOfModel(*fModel);
   auto fieldZeroId = desc->GetFieldZeroId();
   fieldZero.SetOnDiskId(fieldZeroId);

   for (auto &field : fieldZero.GetMutableSubfields()) {
      auto onDiskId = desc->FindFieldId(field->GetQualifiedFieldName(), fieldZeroId);
      // The field we are trying to connect is not present in the ntuple
      if (onDiskId == kInvalidDescriptorId) {
         throw RException(R__FAIL("field \"" + field->GetQualifiedFieldName() + "\" not found in the current RNTuple"));
      }

      // The field will already have an on-disk ID when the model was inferred from the page source because the user
      // didn't provide a model themselves.
      if (field->GetOnDiskId() == kInvalidDescriptorId)
         field->SetOnDiskId(onDiskId);

      ROOT::Internal::CallConnectPageSourceOnField(*field, *fPageSource);
   }
}

void ROOT::Experimental::RNTupleSingleProcessor::AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable,
                                                                       ROOT::NTupleSize_t entryOffset)
{
   Connect();
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
   std::vector<std::unique_ptr<RNTupleProcessor>> processors, std::unique_ptr<ROOT::RNTupleModel> model,
   std::string_view processorName)
   : RNTupleProcessor(processorName, std::move(model)), fInnerProcessors(std::move(processors))
{
   if (fProcessorName.empty()) {
      // `CreateChain` ensures there is at least one inner processor.
      fProcessorName = fInnerProcessors[0]->GetProcessorName();
   }

   fInnerNEntries.assign(fInnerProcessors.size(), kInvalidNTupleIndex);

   fModel->Freeze();
   fEntry = fModel->CreateEntry();

   for (const auto &value : *fEntry) {
      auto &field = value.GetField();
      auto token = fEntry->GetToken(field.GetQualifiedFieldName());

      // If the model has a default entry, use the value pointers from the entry in the entry managed by the
      // processor. This way, the pointers returned by RNTupleModel::MakeField can be used in the processor loop
      // to access the corresponding field values.
      if (!fModel->IsBare()) {
         auto valuePtr = fModel->GetDefaultEntry().GetPtr<void>(token);
         fEntry->BindValue(token, valuePtr);
      }
   }

   for (auto &innerProc : fInnerProcessors) {
      innerProc->SetEntryPointers(*fEntry);
   }
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

void ROOT::Experimental::RNTupleChainProcessor::SetEntryPointers(const ROOT::REntry &entry,
                                                                 std::string_view fieldNamePrefix)
{
   for (const auto &value : *fEntry) {
      std::string fieldName = value.GetField().GetQualifiedFieldName();
      auto valuePtr = fieldNamePrefix.empty() ? entry.GetPtr<void>(fieldName)
                                              : entry.GetPtr<void>(std::string(fieldNamePrefix) + "." + fieldName);

      fEntry->BindValue(fieldName, valuePtr);
   }

   for (auto &innerProc : fInnerProcessors) {
      innerProc->SetEntryPointers(entry, fieldNamePrefix);
   }
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleChainProcessor::LoadEntry(ROOT::NTupleSize_t entryNumber)
{
   ROOT::NTupleSize_t localEntryNumber = entryNumber;
   size_t currProcessor = 0;

   // As long as the entry fails to load from the current processor, we decrement the local entry number with the number
   // of entries in this processor and try with the next processor until we find the correct local entry number.
   while (fInnerProcessors[currProcessor]->LoadEntry(localEntryNumber) == kInvalidNTupleIndex) {
      if (fInnerNEntries[currProcessor] == kInvalidNTupleIndex) {
         fInnerNEntries[currProcessor] = fInnerProcessors[currProcessor]->GetNEntries();
      }

      localEntryNumber -= fInnerNEntries[currProcessor];

      // The provided global entry number is larger than the number of available entries.
      if (++currProcessor >= fInnerProcessors.size())
         return kInvalidNTupleIndex;
   }

   if (currProcessor != fCurrentProcessorNumber)
      fCurrentProcessorNumber = currProcessor;

   fNEntriesProcessed++;
   fCurrentEntryNumber = entryNumber;
   return entryNumber;
}

void ROOT::Experimental::RNTupleChainProcessor::AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable,
                                                                      ROOT::NTupleSize_t entryOffset)
{
   for (unsigned i = 0; i < fInnerProcessors.size(); ++i) {
      const auto &innerProc = fInnerProcessors[i];
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

ROOT::Experimental::RNTupleJoinProcessor::RNTupleJoinProcessor(std::unique_ptr<RNTupleProcessor> primaryProcessor,
                                                               std::unique_ptr<RNTupleProcessor> auxProcessor,
                                                               const std::vector<std::string> &joinFields,
                                                               std::unique_ptr<ROOT::RNTupleModel> primaryModel,
                                                               std::unique_ptr<ROOT::RNTupleModel> auxModel,
                                                               std::string_view processorName)
   : RNTupleProcessor(processorName, nullptr),
     fPrimaryProcessor(std::move(primaryProcessor)),
     fAuxiliaryProcessor(std::move(auxProcessor))
{
   if (fProcessorName.empty()) {
      fProcessorName = fPrimaryProcessor->GetProcessorName();
   }

   if (!primaryModel)
      primaryModel = fPrimaryProcessor->GetModel().Clone();
   if (!auxModel) {
      auxModel = fAuxiliaryProcessor->GetModel().Clone();
   }

   // If the primaryProcessor has a field with the name of the auxProcessor (either as a "proper" field or because the
   // primary processor itself is a join where its auxProcessor bears the same name as the current auxProcessor), there
   // will be name conflicts, so error out.
   if (primaryModel->GetFieldNames().find(fAuxiliaryProcessor->GetProcessorName()) !=
       primaryModel->GetFieldNames().end()) {
      throw RException(R__FAIL("a field or nested auxiliary processor named \"" +
                               fAuxiliaryProcessor->GetProcessorName() +
                               "\" is already present in the model of the primary processor; rename the auxiliary "
                               "processor to avoid conflicts"));
   }

   SetModel(std::move(primaryModel), std::move(auxModel));

   fModel->Freeze();
   fEntry = fModel->CreateEntry();

   for (const auto &value : *fEntry) {
      auto &field = value.GetField();
      const auto &fieldName = field.GetQualifiedFieldName();

      // If the model provided by the user has a default entry, use the value pointers from the default entry of the
      // model that was passed to this constructor. This way, the pointers returned by RNTupleModel::MakeField can be
      // used in the processor loop to access the corresponding field values.
      if (!fModel->IsBare()) {
         auto valuePtr = fModel->GetDefaultEntry().GetPtr<void>(fieldName);
         fEntry->BindValue(fieldName, valuePtr);
      }
   }

   fPrimaryProcessor->SetEntryPointers(*fEntry);
   // FIXME(fdegeus): for nested auxiliary processors, simply passing the processor name is not sufficient because we
   // also need the name(s) of the *inner* processor(s) (e.g., "ntuple0.ntuple1"). This means either (1) recursively
   // infer this full name or (2) rethink the way fields in auxiliary processors together with how entries are
   // currently set altogether.
   fAuxiliaryProcessor->SetEntryPointers(*fEntry, fAuxiliaryProcessor->GetProcessorName());

   if (!joinFields.empty()) {
      for (const auto &joinField : joinFields) {
         auto token = fEntry->GetToken(joinField);
         fJoinFieldTokens.emplace_back(token);
      }

      fJoinTable = Internal::RNTupleJoinTable::Create(joinFields);
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::SetModel(std::unique_ptr<ROOT::RNTupleModel> primaryModel,
                                                        std::unique_ptr<RNTupleModel> auxModel)
{
   fModel = std::move(primaryModel);
   fModel->Unfreeze();

   // Create an anonymous record field for the auxiliary processor, containing its top-level fields. These original
   // top-level fields are registered as subfields in the join model, such that they can be accessed as
   // `auxNTupleName.fieldName`.
   std::vector<std::unique_ptr<ROOT::RFieldBase>> auxFields;
   auxFields.reserve(auxModel->GetFieldNames().size());

   for (const auto &fieldName : auxModel->GetFieldNames()) {
      auxFields.emplace_back(auxModel->GetConstField(fieldName).Clone(fieldName));
   }

   auto auxParentField = std::make_unique<Internal::RAuxiliaryProcessorField>(fAuxiliaryProcessor->GetProcessorName(),
                                                                              std::move(auxFields));
   const auto &subFields = auxParentField->GetConstSubfields();
   fModel->AddField(std::move(auxParentField));

   for (const auto &field : subFields) {
      fModel->RegisterSubfield(field->GetQualifiedFieldName());

      if (field->GetTypeName() == "RAuxiliaryProcessorField") {
         for (const auto &auxSubField : field->GetConstSubfields()) {
            fModel->RegisterSubfield(auxSubField->GetQualifiedFieldName());
         }
      }
   }

   // If the model has a default entry, adopt its value pointers. This way, the pointers returned by
   // RNTupleModel::MakeField can be used in the processor loop to access the corresponding field values.
   if (!fModel->IsBare() && !auxModel->IsBare()) {
      const auto &auxDefaultEntry = auxModel->GetDefaultEntry();
      auto &joinDefaultEntry = fModel->GetDefaultEntry();
      for (const auto &fieldName : auxModel->GetFieldNames()) {
         auto valuePtr = auxDefaultEntry.GetPtr<void>(fieldName);
         joinDefaultEntry.BindValue(fAuxiliaryProcessor->GetProcessorName() + "." + fieldName, valuePtr);
      }
   }

   fModel->Freeze();
}

void ROOT::Experimental::RNTupleJoinProcessor::SetEntryPointers(const ROOT::REntry &entry,
                                                                std::string_view fieldNamePrefix)
{
   for (const auto &value : *fEntry) {
      std::string fieldName = value.GetField().GetQualifiedFieldName();
      auto valuePtr = fieldNamePrefix.empty() ? entry.GetPtr<void>(fieldName)
                                              : entry.GetPtr<void>(std::string(fieldNamePrefix) + "." + fieldName);

      fEntry->BindValue(fieldName, valuePtr);
   }

   fPrimaryProcessor->SetEntryPointers(*fEntry);
   fAuxiliaryProcessor->SetEntryPointers(*fEntry, fAuxiliaryProcessor->GetProcessorName());
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleJoinProcessor::LoadEntry(ROOT::NTupleSize_t entryNumber)
{
   if (fPrimaryProcessor->LoadEntry(entryNumber) == kInvalidNTupleIndex)
      return kInvalidNTupleIndex;

   fCurrentEntryNumber = entryNumber;
   fNEntriesProcessed++;

   if (!fJoinTable) {
      if (fAuxiliaryProcessor->LoadEntry(entryNumber) == kInvalidNTupleIndex) {
         throw RException(R__FAIL("entry " + std::to_string(entryNumber) +
                                  " in the primary processor has no corresponding entry in auxiliary processor \"" +
                                  fAuxiliaryProcessor->GetProcessorName() + "\""));
      }

      return entryNumber;
   }

   if (!fJoinTableIsBuilt) {
      fAuxiliaryProcessor->AddEntriesToJoinTable(*fJoinTable);
      fJoinTableIsBuilt = true;
   }

   // Collect the values of the join fields for this entry.
   std::vector<void *> valPtrs;
   valPtrs.reserve(fJoinFieldTokens.size());
   for (const auto &token : fJoinFieldTokens) {
      auto ptr = fEntry->GetPtr<void>(token);
      valPtrs.push_back(ptr.get());
   }

   // Find the entry index corresponding to the join field values for each auxiliary processor and load the
   // corresponding entry.
   const auto entryIdx = fJoinTable->GetEntryIndex(valPtrs);

   if (entryIdx == kInvalidNTupleIndex)
      throw RException(R__FAIL("entry " + std::to_string(entryNumber) +
                               " in the primary processor has no corresponding entry in auxiliary processor \"" +
                               fAuxiliaryProcessor->GetProcessorName() + "\""));

   fAuxiliaryProcessor->LoadEntry(entryIdx);

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
