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

#include <TDirectory.h>

std::unique_ptr<ROOT::Internal::RPageSource> ROOT::Experimental::RNTupleOpenSpec::CreatePageSource() const
{
   if (const std::string *storagePath = std::get_if<std::string>(&fStorage))
      return ROOT::Internal::RPageSource::Create(fNTupleName, *storagePath);

   auto dir = std::get<TDirectory *>(fStorage);
   auto ntuple = std::unique_ptr<ROOT::RNTuple>(dir->Get<ROOT::RNTuple>(fNTupleName.c_str()));
   return ROOT::Internal::RPageSourceFile::CreateFromAnchor(*ntuple);
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::Create(RNTupleOpenSpec ntuple, std::unique_ptr<ROOT::RNTupleModel> model)
{
   auto processorName = ntuple.fNTupleName;
   return Create(std::move(ntuple), processorName, std::move(model));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::Create(RNTupleOpenSpec ntuple, std::string_view processorName,
                                             std::unique_ptr<ROOT::RNTupleModel> model)
{
   return std::unique_ptr<RNTupleSingleProcessor>(
      new RNTupleSingleProcessor(std::move(ntuple), processorName, std::move(model)));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(std::vector<RNTupleOpenSpec> ntuples,
                                                  std::unique_ptr<ROOT::RNTupleModel> model)
{
   if (ntuples.empty())
      throw RException(R__FAIL("at least one RNTuple must be provided"));

   auto processorName = ntuples[0].fNTupleName;
   return CreateChain(std::move(ntuples), processorName, std::move(model));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(std::vector<RNTupleOpenSpec> ntuples, std::string_view processorName,
                                                  std::unique_ptr<ROOT::RNTupleModel> model)
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

   return CreateChain(std::move(innerProcessors), processorName, std::move(model));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors,
                                                  std::unique_ptr<ROOT::RNTupleModel> model)
{
   if (innerProcessors.empty())
      throw RException(R__FAIL("at least one inner processor must be provided"));

   auto processorName = innerProcessors[0]->GetProcessorName();
   return CreateChain(std::move(innerProcessors), processorName, std::move(model));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors,
                                                  std::string_view processorName,
                                                  std::unique_ptr<ROOT::RNTupleModel> model)
{
   if (innerProcessors.empty())
      throw RException(R__FAIL("at least one inner processor must be provided"));

   // If no model is provided, infer it from the first inner processor.
   if (!model) {
      model = innerProcessors[0]->GetModel().Clone();
   }

   return std::unique_ptr<RNTupleChainProcessor>(
      new RNTupleChainProcessor(std::move(innerProcessors), processorName, std::move(model)));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateJoin(RNTupleOpenSpec primaryNTuple, std::vector<RNTupleOpenSpec> auxNTuples,
                                                 const std::vector<std::string> &joinFields,
                                                 std::unique_ptr<ROOT::RNTupleModel> primaryModel,
                                                 std::vector<std::unique_ptr<ROOT::RNTupleModel>> auxModels)
{
   auto processorName = primaryNTuple.fNTupleName;
   return CreateJoin(std::move(primaryNTuple), std::move(auxNTuples), joinFields, processorName,
                     std::move(primaryModel), std::move(auxModels));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateJoin(RNTupleOpenSpec primaryNTuple, std::vector<RNTupleOpenSpec> auxNTuples,
                                                 const std::vector<std::string> &joinFields,
                                                 std::string_view processorName,
                                                 std::unique_ptr<ROOT::RNTupleModel> primaryModel,
                                                 std::vector<std::unique_ptr<ROOT::RNTupleModel>> auxModels)
{
   if (!auxModels.empty() && auxModels.size() != auxNTuples.size())
      throw RException(R__FAIL("number of auxiliary models and auxiliary RNTuples does not match"));

   if (joinFields.size() > 4) {
      throw RException(R__FAIL("a maximum of four join fields is allowed"));
   }

   if (std::set(joinFields.begin(), joinFields.end()).size() < joinFields.size()) {
      throw RException(R__FAIL("join fields must be unique"));
   }

   // Ensure that all ntuples are uniquely named to prevent name clashes.
   // TODO(fdegeus) allow for the provision of aliases for ntuples with the same name, removing the constraint of
   // uniquely-named ntuples.
   std::unordered_set<std::string> uniqueNTupleNames{primaryNTuple.fNTupleName};
   for (const auto &ntuple : auxNTuples) {
      auto res = uniqueNTupleNames.emplace(ntuple.fNTupleName);
      if (!res.second) {
         throw ROOT::RException(R__FAIL("joining RNTuples with the same name is not allowed"));
      }
   }

   std::unique_ptr<RNTupleProcessor> primaryProcessor;
   if (primaryModel)
      primaryProcessor = Create(primaryNTuple, processorName, primaryModel->Clone());
   else
      primaryProcessor = Create(primaryNTuple, processorName);

   std::vector<std::unique_ptr<RNTupleProcessor>> auxProcessors;
   for (unsigned i = 0; i < auxNTuples.size(); ++i) {
      if (!auxModels.empty() && auxModels[i])
         auxProcessors.emplace_back(Create(auxNTuples[i], auxModels[i]->Clone()));
      else
         auxProcessors.emplace_back(Create(auxNTuples[i]));
   }

   std::unique_ptr<RNTupleJoinProcessor> processor = std::unique_ptr<RNTupleJoinProcessor>(
      new RNTupleJoinProcessor(std::move(primaryProcessor), std::move(auxProcessors), joinFields, processorName,
                               std::move(primaryModel), std::move(auxModels)));

   return processor;
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateJoin(std::unique_ptr<RNTupleProcessor> primaryProcessor,
                                                 std::vector<std::unique_ptr<RNTupleProcessor>> auxProcessors,
                                                 const std::vector<std::string> &joinFields,
                                                 std::unique_ptr<ROOT::RNTupleModel> primaryModel,
                                                 std::vector<std::unique_ptr<ROOT::RNTupleModel>> auxModels)
{
   auto processorName = primaryProcessor->GetProcessorName();
   return CreateJoin(std::move(primaryProcessor), std::move(auxProcessors), joinFields, processorName,
                     std::move(primaryModel), std::move(auxModels));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor> ROOT::Experimental::RNTupleProcessor::CreateJoin(
   std::unique_ptr<RNTupleProcessor> primaryProcessor, std::vector<std::unique_ptr<RNTupleProcessor>> auxProcessors,
   const std::vector<std::string> &joinFields, std::string_view processorName,
   std::unique_ptr<ROOT::RNTupleModel> primaryModel, std::vector<std::unique_ptr<ROOT::RNTupleModel>> auxModels)
{
   if (!auxModels.empty() && auxModels.size() != auxProcessors.size())
      throw RException(R__FAIL("number of auxiliary models and auxiliary processors does not match"));

   if (joinFields.size() > 4) {
      throw RException(R__FAIL("a maximum of four join fields is allowed"));
   }

   if (std::set(joinFields.begin(), joinFields.end()).size() < joinFields.size()) {
      throw RException(R__FAIL("join fields must be unique"));
   }

   std::unique_ptr<RNTupleJoinProcessor> processor = std::unique_ptr<RNTupleJoinProcessor>(
      new RNTupleJoinProcessor(std::move(primaryProcessor), std::move(auxProcessors), joinFields, processorName,
                               std::move(primaryModel), std::move(auxModels)));

   return processor;
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleSingleProcessor::RNTupleSingleProcessor(RNTupleOpenSpec ntuple,
                                                                   std::string_view processorName,
                                                                   std::unique_ptr<ROOT::RNTupleModel> model)
   : RNTupleProcessor(processorName, std::move(model)), fNTupleSpec(std::move(ntuple))
{
   if (!fModel) {
      fPageSource = fNTupleSpec.CreatePageSource();
      fPageSource->Attach();
      fModel = fPageSource->GetSharedDescriptorGuard()->CreateModel();
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

      auto fieldContext = RFieldContext(field.Clone(field.GetFieldName()), token);
      fFieldContexts.try_emplace(field.GetFieldName(), std::move(fieldContext));
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
   for (auto &[_, fieldContext] : fFieldContexts) {

      const auto fieldId = desc->FindFieldId(fieldContext.GetProtoField().GetFieldName());
      if (fieldId == ROOT::kInvalidDescriptorId) {
         throw RException(
            R__FAIL("field \"" + fieldContext.GetProtoField().GetFieldName() + "\" not found in current RNTuple"));
      }

      fieldContext.SetConcreteField();
      fieldContext.fConcreteField->SetOnDiskId(fieldId);
      ROOT::Internal::CallConnectPageSourceOnField(*fieldContext.fConcreteField, *fPageSource);

      auto valuePtr = fEntry->GetPtr<void>(fieldContext.fToken);
      auto value = fieldContext.fConcreteField->BindValue(valuePtr);
      fEntry->UpdateValue(fieldContext.fToken, value);
   }
}

void ROOT::Experimental::RNTupleSingleProcessor::AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable,
                                                                       ROOT::NTupleSize_t entryOffset)
{
   Connect();
   joinTable.Add(*fPageSource, Internal::RNTupleJoinTable::kDefaultPartitionKey, entryOffset);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleChainProcessor::RNTupleChainProcessor(
   std::vector<std::unique_ptr<RNTupleProcessor>> processors, std::string_view processorName,
   std::unique_ptr<ROOT::RNTupleModel> model)
   : RNTupleProcessor(processorName, std::move(model)), fInnerProcessors(std::move(processors))
{
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

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleJoinProcessor::RNTupleJoinProcessor(
   std::unique_ptr<RNTupleProcessor> primaryProcessor, std::vector<std::unique_ptr<RNTupleProcessor>> auxProcessors,
   const std::vector<std::string> &joinFields, std::string_view processorName,
   std::unique_ptr<ROOT::RNTupleModel> primaryModel, std::vector<std::unique_ptr<ROOT::RNTupleModel>> auxModels)
   : RNTupleProcessor(processorName, nullptr),
     fPrimaryProcessor(std::move(primaryProcessor)),
     fAuxiliaryProcessors(std::move(auxProcessors))
{
   // FIXME(fdegeus): this check is not complete, e.g. the situation where the auxiliary processor is a chain of joins
   // would pass. It would be better to fix the underlying issue (how to access their fields), so this check would
   // become unecessary altogether.
   for (const auto &auxProc : fAuxiliaryProcessors) {
      if (dynamic_cast<RNTupleJoinProcessor *>(auxProc.get())) {
         throw RException(R__FAIL("auxiliary RNTupleJoinProcessors are currently not supported"));
      }
   }

   if (!primaryModel)
      primaryModel = fPrimaryProcessor->GetModel().Clone();
   if (auxModels.empty()) {
      auxModels.resize(fAuxiliaryProcessors.size());
   }
   for (unsigned i = 0; i < fAuxiliaryProcessors.size(); ++i) {
      if (!auxModels[i])
         auxModels[i] = fAuxiliaryProcessors[i]->GetModel().Clone();
   }

   SetModel(std::move(primaryModel), std::move(auxModels));

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
   for (auto &auxProcessor : fAuxiliaryProcessors) {
      // FIXME(fdegeus): for nested auxiliary processors, simply passing the processor name is not sufficient because we
      // also need the name(s) of the *inner* processor(s) (e.g., "ntuple0.ntuple1"). This means either (1) recursively
      // infer this full name or (2) rethink the way fields in auxiliary processors together with how entries are
      // currently set altogether.
      auxProcessor->SetEntryPointers(*fEntry, auxProcessor->GetProcessorName());
   }

   if (!joinFields.empty()) {
      for (const auto &joinField : joinFields) {
         auto token = fEntry->GetToken(joinField);
         fJoinFieldTokens.emplace_back(token);
      }

      for (unsigned i = 0; i < fAuxiliaryProcessors.size(); ++i) {
         fJoinTables.emplace_back(Internal::RNTupleJoinTable::Create(joinFields));
      }
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::SetModel(std::unique_ptr<ROOT::RNTupleModel> primaryModel,
                                                        std::vector<std::unique_ptr<ROOT::RNTupleModel>> auxModels)
{
   fModel = std::move(primaryModel);
   fModel->Unfreeze();

   // Create an anonymous record field for each auxiliary ntuple, containing their top-level fields. These original
   // top-level fields are registered as subfields in the join model, such that they can be accessed as
   // `auxNTupleName.fieldName`.
   for (unsigned i = 0; i < auxModels.size(); ++i) {
      std::vector<std::unique_ptr<ROOT::RFieldBase>> auxFields;
      auxFields.reserve(auxModels[i]->GetFieldNames().size());

      for (const auto &fieldName : auxModels[i]->GetFieldNames()) {
         auxFields.emplace_back(auxModels[i]->GetConstField(fieldName).Clone(fieldName));
      }

      auto auxParentField =
         std::make_unique<ROOT::RRecordField>(fAuxiliaryProcessors[i]->GetProcessorName(), std::move(auxFields));
      const auto &subFields = auxParentField->GetConstSubfields();
      fModel->AddField(std::move(auxParentField));

      for (const auto &field : subFields) {
         fModel->RegisterSubfield(field->GetQualifiedFieldName());
      }

      // If the model has a default entry, adopt its value pointers. This way, the pointers returned by
      // RNTupleModel::MakeField can be used in the processor loop to access the corresponding field values.
      if (!auxModels[i]->IsBare()) {
         const auto &auxDefaultEntry = auxModels[i]->GetDefaultEntry();
         auto &joinDefaultEntry = fModel->GetDefaultEntry();
         for (const auto &fieldName : auxModels[i]->GetFieldNames()) {
            auto valuePtr = auxDefaultEntry.GetPtr<void>(fieldName);
            joinDefaultEntry.BindValue(fAuxiliaryProcessors[i]->GetProcessorName() + "." + fieldName, valuePtr);
         }
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
   for (auto &auxProc : fAuxiliaryProcessors) {
      auxProc->SetEntryPointers(*fEntry, auxProc->GetProcessorName());
   }
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleJoinProcessor::LoadEntry(ROOT::NTupleSize_t entryNumber)
{
   if (fPrimaryProcessor->LoadEntry(entryNumber) == kInvalidNTupleIndex)
      return kInvalidNTupleIndex;

   fCurrentEntryNumber = entryNumber;
   fNEntriesProcessed++;

   if (!HasJoinTable()) {
      for (auto &auxProcessor : fAuxiliaryProcessors) {
         if (auxProcessor->LoadEntry(entryNumber) == kInvalidNTupleIndex) {
            throw RException(R__FAIL("entry " + std::to_string(entryNumber) +
                                     " in the primary processor has no corresponding entry in auxiliary processor \"" +
                                     auxProcessor->GetProcessorName() + "\""));
         }
      }
   }

   if (!fJoinTablesAreBuilt) {
      for (unsigned i = 0; i < fJoinTables.size(); ++i) {
         fAuxiliaryProcessors[i]->AddEntriesToJoinTable(*fJoinTables[i]);
      }
      fJoinTablesAreBuilt = true;
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
   for (unsigned i = 0; i < fJoinTables.size(); ++i) {
      const auto entryIdx = fJoinTables[i]->GetEntryIndex(valPtrs);

      if (entryIdx == kInvalidNTupleIndex)
         throw RException(R__FAIL("entry " + std::to_string(entryNumber) +
                                  " in the primary processor has no corresponding entry in auxiliary processor \"" +
                                  fAuxiliaryProcessors[i]->GetProcessorName() + "\""));

      fAuxiliaryProcessors[i]->LoadEntry(entryIdx);
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
