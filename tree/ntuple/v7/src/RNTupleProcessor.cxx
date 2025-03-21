/// \file RNTupleProcessor.cxx
/// \ingroup NTuple ROOT7
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

std::unique_ptr<ROOT::Experimental::Internal::RPageSource> ROOT::Experimental::RNTupleOpenSpec::CreatePageSource() const
{
   if (const std::string *storagePath = std::get_if<std::string>(&fStorage))
      return ROOT::Experimental::Internal::RPageSource::Create(fNTupleName, *storagePath);

   auto dir = std::get<TDirectory *>(fStorage);
   auto ntuple = std::unique_ptr<ROOT::RNTuple>(dir->Get<ROOT::RNTuple>(fNTupleName.c_str()));
   return ROOT::Experimental::Internal::RPageSourceFile::CreateFromAnchor(*ntuple);
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
ROOT::Experimental::RNTupleProcessor::CreateJoin(const RNTupleOpenSpec &primaryNTuple,
                                                 const std::vector<RNTupleOpenSpec> &auxNTuples,
                                                 const std::vector<std::string> &joinFields,
                                                 std::unique_ptr<ROOT::RNTupleModel> primaryModel,
                                                 std::vector<std::unique_ptr<ROOT::RNTupleModel>> auxModels)
{
   return CreateJoin(primaryNTuple, auxNTuples, joinFields, primaryNTuple.fNTupleName, std::move(primaryModel),
                     std::move(auxModels));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor> ROOT::Experimental::RNTupleProcessor::CreateJoin(
   const RNTupleOpenSpec &primaryNTuple, const std::vector<RNTupleOpenSpec> &auxNTuples,
   const std::vector<std::string> &joinFields, std::string_view processorName,
   std::unique_ptr<ROOT::RNTupleModel> primaryModel, std::vector<std::unique_ptr<ROOT::RNTupleModel>> auxModels)
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

   std::unique_ptr<RNTupleJoinProcessor> processor = std::unique_ptr<RNTupleJoinProcessor>(new RNTupleJoinProcessor(
      primaryNTuple, auxNTuples, joinFields, processorName, std::move(primaryModel), std::move(auxModels)));

   processor->SetJoinFieldTokens(joinFields);
   processor->ConnectFields();

   return processor;
}

void ROOT::Experimental::RNTupleProcessor::ConnectField(RFieldContext &fieldContext, Internal::RPageSource &pageSource,
                                                        ROOT::REntry &entry)
{
   pageSource.Attach();
   auto desc = pageSource.GetSharedDescriptorGuard();

   const auto fieldId = desc->FindFieldId(fieldContext.GetProtoField().GetFieldName());
   if (fieldId == ROOT::kInvalidDescriptorId) {
      throw RException(
         R__FAIL("field \"" + fieldContext.GetProtoField().GetFieldName() + "\" not found in current RNTuple"));
   }

   fieldContext.SetConcreteField();
   fieldContext.fConcreteField->SetOnDiskId(fieldId);
   ROOT::Internal::CallConnectPageSourceOnField(*fieldContext.fConcreteField, pageSource);

   auto valuePtr = entry.GetPtr<void>(fieldContext.fToken);
   auto value = fieldContext.fConcreteField->BindValue(valuePtr);
   entry.UpdateValue(fieldContext.fToken, value);
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

void ROOT::Experimental::RNTupleSingleProcessor::SetEntryPointers(const ROOT::REntry &entry)
{
   for (const auto &value : *fEntry) {
      auto &field = value.GetField();
      auto valuePtr = entry.GetPtr<void>(field.GetQualifiedFieldName());

      fEntry->BindValue(field.GetQualifiedFieldName(), valuePtr);
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

   for (auto &[_, fieldContext] : fFieldContexts) {
      ConnectField(fieldContext, *fPageSource, *fEntry);
   }
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

void ROOT::Experimental::RNTupleChainProcessor::SetEntryPointers(const ROOT::REntry &entry)
{
   for (const auto &value : *fEntry) {
      auto &field = value.GetField();
      auto valuePtr = entry.GetPtr<void>(field.GetQualifiedFieldName());

      fEntry->BindValue(field.GetQualifiedFieldName(), valuePtr);
   }

   for (auto &innerProc : fInnerProcessors) {
      innerProc->SetEntryPointers(*fEntry);
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

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleJoinProcessor::RNTupleJoinProcessor(
   const RNTupleOpenSpec &mainNTuple, const std::vector<RNTupleOpenSpec> &auxNTuples,
   const std::vector<std::string> &joinFields, std::string_view processorName,
   std::unique_ptr<ROOT::RNTupleModel> primaryModel, std::vector<std::unique_ptr<ROOT::RNTupleModel>> auxModels)
   : RNTupleProcessor(processorName, nullptr)
{
   fNTuples.emplace_back(mainNTuple);
   fNTuples.insert(fNTuples.end(), auxNTuples.begin(), auxNTuples.end());

   fPageSource = mainNTuple.CreatePageSource();
   fPageSource->Attach();

   if (fPageSource->GetNEntries() == 0) {
      throw RException(R__FAIL("provided RNTuple is empty"));
   }

   fNEntries = fPageSource->GetNEntries();

   for (const auto &auxNTuple : auxNTuples) {
      fAuxiliaryPageSources.emplace_back(auxNTuple.CreatePageSource());
      if (!joinFields.empty())
         fJoinTables.emplace_back(Internal::RNTupleJoinTable::Create(joinFields));
   }

   if (!primaryModel)
      primaryModel = fPageSource->GetSharedDescriptorGuard()->CreateModel();
   if (auxModels.empty()) {
      auxModels.resize(fAuxiliaryPageSources.size());
   }
   for (unsigned i = 0; i < fAuxiliaryPageSources.size(); ++i) {
      if (!auxModels[i]) {
         fAuxiliaryPageSources[i]->Attach();
         auxModels[i] = fAuxiliaryPageSources[i]->GetSharedDescriptorGuard()->CreateModel();
      }
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

      auto auxNTupleName = std::find_if(auxNTuples.cbegin(), auxNTuples.cend(), [&fieldName](const RNTupleOpenSpec &n) {
         return fieldName.substr(0, n.fNTupleName.size()) == n.fNTupleName;
      });

      // If the current field name does not begin with the name of one of the auxiliary ntuples, we are dealing with a
      // field from the primary ntuple, so it can be added as a field context. Otherwise, if it does begin with the
      // name, but is not equal to just the name (e.g. it is a subfield of `auxNTupleName`, which means it is a proper
      // field in the corresponding auxiliary ntuple) we also need to add it as a field context. If it is exactly equal
      // to an auxiliary ntuple name, it is the untyped record field containing the auxiliary fields itself. This one we
      // don't want to add as a field context, because there is nothing to read from.
      // TODO(fdegeus) handle the case where a primary field has the name of an auxiliary ntuple.
      if (auxNTupleName == auxNTuples.end()) {
         fFieldContexts.try_emplace(fieldName, field.Clone(field.GetFieldName()), fEntry->GetToken(fieldName));
      } else if (fieldName != auxNTupleName->fNTupleName) {
         // Add 1 because we also have to take into account the primary ntuple.
         auto ntupleIdx = std::distance(auxNTuples.begin(), auxNTupleName) + 1;
         fFieldContexts.try_emplace(fieldName, field.Clone(field.GetFieldName()), fEntry->GetToken(fieldName),
                                    ntupleIdx);
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

      auto auxParentField = std::make_unique<ROOT::RRecordField>(fNTuples[i + 1].fNTupleName, std::move(auxFields));
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
            joinDefaultEntry.BindValue(fNTuples[i + 1].fNTupleName + "." + fieldName, valuePtr);
         }
      }
   }

   fModel->Freeze();
}

void ROOT::Experimental::RNTupleJoinProcessor::ConnectFields()
{
   for (auto &[_, fieldContext] : fFieldContexts) {
      Internal::RPageSource &pageSource =
         fieldContext.IsAuxiliary() ? *fAuxiliaryPageSources.at(fieldContext.fNTupleIdx - 1) : *fPageSource;
      ConnectField(fieldContext, pageSource, *fEntry);
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::SetEntryPointers(const ROOT::REntry &entry)
{
   for (const auto &[_, fieldContext] : fFieldContexts) {
      auto fieldName = fieldContext.GetProtoField().GetQualifiedFieldName();
      if (fieldContext.IsAuxiliary()) {
         fieldName = fNTuples[fieldContext.fNTupleIdx].fNTupleName + "." + fieldName;
      }
      auto valuePtr = entry.GetPtr<void>(fieldName);
      fEntry->BindValue(fieldName, valuePtr);
   }
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleJoinProcessor::LoadEntry(ROOT::NTupleSize_t entryNumber)
{
   if (entryNumber >= fPageSource->GetNEntries())
      return ROOT::kInvalidNTupleIndex;

   // Read the values of the primary ntuple. If no join table is used (i.e., the join is aligned), also read the values
   // of auxiliary ntuples.
   for (const auto &[_, fieldContext] : fFieldContexts) {
      if (!fieldContext.IsAuxiliary() || !HasJoinTable()) {
         auto &value = fEntry->GetValue(fieldContext.fToken);
         value.Read(entryNumber);
      }
   }

   fCurrentEntryNumber = entryNumber;
   fNEntriesProcessed++;

   // If no join table is used (i.e., the join is aligned), there's nothing left to do.
   if (!HasJoinTable())
      return entryNumber;

   // First build the join tables if this hasn't been done yet.
   if (!fJoinTablesAreBuilt) {
      for (unsigned i = 0; i < fJoinTables.size(); ++i) {
         fJoinTables[i]->Add(*fAuxiliaryPageSources[i]);
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

   // Find the entry index corresponding to the join field values for each auxiliary ntuple.
   std::vector<ROOT::NTupleSize_t> auxEntryIdxs;
   auxEntryIdxs.reserve(fJoinTables.size());
   for (const auto &joinTable : fJoinTables) {
      auto entryIdxs = joinTable->GetEntryIndexes(valPtrs);

      if (entryIdxs.empty())
         auxEntryIdxs.push_back(kInvalidNTupleIndex);
      else
         auxEntryIdxs.push_back(entryIdxs[0]);
   }

   // For each auxiliary field, load its value according to the entry number we just found of the ntuple it belongs to.
   for (const auto &[_, fieldContext] : fFieldContexts) {
      if (!fieldContext.IsAuxiliary())
         continue;

      auto &value = fEntry->GetValue(fieldContext.fToken);
      if (auxEntryIdxs[fieldContext.fNTupleIdx - 1] == ROOT::kInvalidNTupleIndex) {
         // No matching entry exists, so we reset the field's value to a default value.
         // TODO(fdegeus): further consolidate how non-existing join matches should be handled. N.B.: in case
         // ConstructValue is not used anymore in the future, remove friend in ROOT::RFieldBase.
         fieldContext.fProtoField->ConstructValue(value.GetPtr<void>().get());
      } else {
         value.Read(auxEntryIdxs[fieldContext.fNTupleIdx - 1]);
      }
   }

   return entryNumber;
}
