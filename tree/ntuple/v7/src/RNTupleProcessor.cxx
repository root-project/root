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

namespace {
using ROOT::Experimental::RNTupleOpenSpec;
void EnsureUniqueNTupleNames(const RNTupleOpenSpec &primaryNTuple, const std::vector<RNTupleOpenSpec> &auxNTuples)
{
   std::unordered_set<std::string> uniqueNTupleNames{primaryNTuple.fNTupleName};
   for (const auto &ntuple : auxNTuples) {
      auto res = uniqueNTupleNames.emplace(ntuple.fNTupleName);
      if (!res.second) {
         throw ROOT::RException(R__FAIL("joining RNTuples with the same name is not allowed"));
      }
   }
}
} // anonymous namespace

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::Create(const RNTupleOpenSpec &ntuple, std::unique_ptr<RNTupleModel> model)
{
   return Create(ntuple, ntuple.fNTupleName, std::move(model));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::Create(const RNTupleOpenSpec &ntuple, std::string_view processorName,
                                             std::unique_ptr<RNTupleModel> model)
{
   return std::unique_ptr<RNTupleSingleProcessor>(new RNTupleSingleProcessor(ntuple, processorName, std::move(model)));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(const std::vector<RNTupleOpenSpec> &ntuples,
                                                  std::unique_ptr<RNTupleModel> model)
{
   if (ntuples.empty())
      throw RException(R__FAIL("at least one RNTuple must be provided"));

   return CreateChain(ntuples, ntuples[0].fNTupleName, std::move(model));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(const std::vector<RNTupleOpenSpec> &ntuples,
                                                  std::string_view processorName, std::unique_ptr<RNTupleModel> model)
{
   if (ntuples.empty())
      throw RException(R__FAIL("at least one RNTuple must be provided"));

   std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors;
   innerProcessors.reserve(ntuples.size());

   // If no model is provided, infer it from the first ntuple.
   if (!model) {
      auto firstPageSource = Internal::RPageSource::Create(ntuples[0].fNTupleName, ntuples[0].fStorage);
      firstPageSource->Attach();
      model = firstPageSource->GetSharedDescriptorGuard()->CreateModel();
   }

   for (const auto &ntuple : ntuples) {
      innerProcessors.emplace_back(Create(ntuple, model->Clone()));
   }

   return CreateChain(std::move(innerProcessors), processorName, std::move(model));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors,
                                                  std::unique_ptr<RNTupleModel> model)
{
   if (innerProcessors.empty())
      throw RException(R__FAIL("at least one inner processor must be provided"));

   auto processorName = innerProcessors[0]->GetProcessorName();
   return CreateChain(std::move(innerProcessors), processorName, std::move(model));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors,
                                                  std::string_view processorName, std::unique_ptr<RNTupleModel> model)
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
                                                 std::vector<std::unique_ptr<RNTupleModel>> models)
{
   return CreateJoin(primaryNTuple, auxNTuples, joinFields, primaryNTuple.fNTupleName, std::move(models));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateJoin(const RNTupleOpenSpec &primaryNTuple,
                                                 const std::vector<RNTupleOpenSpec> &auxNTuples,
                                                 const std::vector<std::string> &joinFields,
                                                 std::string_view processorName,
                                                 std::vector<std::unique_ptr<RNTupleModel>> models)
{
   if (!models.empty() && models.size() != (auxNTuples.size() + 1)) {
      throw RException(R__FAIL("number of provided models must match number of specified ntuples"));
   }

   if (joinFields.size() > 4) {
      throw RException(R__FAIL("a maximum of four join fields is allowed"));
   }

   if (std::set(joinFields.begin(), joinFields.end()).size() < joinFields.size()) {
      throw RException(R__FAIL("join fields must be unique"));
   }

   // TODO(fdegeus) allow for the provision of aliases for ntuples with the same name, removing the constraint of
   // uniquely-named ntuples.
   EnsureUniqueNTupleNames(primaryNTuple, auxNTuples);

   std::unique_ptr<RNTupleJoinProcessor> processor;
   if (!models.empty()) {
      processor = std::unique_ptr<RNTupleJoinProcessor>(
         new RNTupleJoinProcessor(primaryNTuple, auxNTuples, joinFields, processorName, std::move(models)));
   } else {
      processor = std::unique_ptr<RNTupleJoinProcessor>(
         new RNTupleJoinProcessor(primaryNTuple, auxNTuples, joinFields, processorName));
   }

   processor->SetJoinFieldTokens(joinFields);
   processor->ConnectFields();

   return processor;
}

void ROOT::Experimental::RNTupleProcessor::ConnectField(RFieldContext &fieldContext, Internal::RPageSource &pageSource,
                                                        REntry &entry)
{
   auto desc = pageSource.GetSharedDescriptorGuard();

   const auto fieldId = desc->FindFieldId(fieldContext.GetProtoField().GetFieldName());
   if (fieldId == ROOT::kInvalidDescriptorId) {
      throw RException(
         R__FAIL("field \"" + fieldContext.GetProtoField().GetFieldName() + "\" not found in current RNTuple"));
   }

   fieldContext.SetConcreteField();
   fieldContext.fConcreteField->SetOnDiskId(fieldId);
   Internal::CallConnectPageSourceOnField(*fieldContext.fConcreteField, pageSource);

   auto valuePtr = entry.GetPtr<void>(fieldContext.fToken);
   auto value = fieldContext.fConcreteField->BindValue(valuePtr);
   entry.UpdateValue(fieldContext.fToken, value);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleSingleProcessor::RNTupleSingleProcessor(const RNTupleOpenSpec &ntuple,
                                                                   std::string_view processorName,
                                                                   std::unique_ptr<RNTupleModel> model)
   : RNTupleProcessor(processorName, std::move(model)), fNTupleSpec(ntuple)
{
   if (!fModel) {
      fPageSource = Internal::RPageSource::Create(fNTupleSpec.fNTupleName, fNTupleSpec.fStorage);
      fPageSource->Attach();
      fModel = fPageSource->GetSharedDescriptorGuard()->CreateModel();
   }

   fModel->Freeze();
   fEntry = fModel->CreateEntry();

   for (const auto &value : *fEntry) {
      auto &field = value.GetField();
      auto token = fEntry->GetToken(field.GetFieldName());

      // If the model has a default entry, use the value pointers from the entry in the entry managed by the
      // processor. This way, the pointers returned by RNTupleModel::MakeField can be used in the processor loop to
      // access the corresponding field values.
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

void ROOT::Experimental::RNTupleSingleProcessor::SetEntryPointers(const REntry &entry)
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
      fPageSource = Internal::RPageSource::Create(fNTupleSpec.fNTupleName, fNTupleSpec.fStorage);
   fPageSource->Attach();
   fNEntries = fPageSource->GetNEntries();

   for (auto &[_, fieldContext] : fFieldContexts) {
      ConnectField(fieldContext, *fPageSource, *fEntry);
   }
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleChainProcessor::RNTupleChainProcessor(
   std::vector<std::unique_ptr<RNTupleProcessor>> processors, std::string_view processorName,
   std::unique_ptr<RNTupleModel> model)
   : RNTupleProcessor(processorName, std::move(model)), fInnerProcessors(std::move(processors))
{
   fInnerNEntries.assign(fInnerProcessors.size(), kInvalidNTupleIndex);

   fModel->Freeze();
   fEntry = fModel->CreateEntry();

   for (const auto &value : *fEntry) {
      auto &field = value.GetField();
      auto token = fEntry->GetToken(field.GetQualifiedFieldName());

      // If the model has a default entry, use the value pointers from the entry in the entry managed by the
      // processor. This way, the pointers returned by RNTupleModel::MakeField can be used in the processor loop to
      // access the corresponding field values.
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

void ROOT::Experimental::RNTupleChainProcessor::SetEntryPointers(const REntry &entry)
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

ROOT::Experimental::RNTupleJoinProcessor::RNTupleJoinProcessor(const RNTupleOpenSpec &mainNTuple,
                                                               const std::vector<RNTupleOpenSpec> &auxNTuples,
                                                               const std::vector<std::string> &joinFields,
                                                               std::string_view processorName,
                                                               std::vector<std::unique_ptr<RNTupleModel>> models)
   : RNTupleProcessor(processorName, nullptr)
{
   fNTuples.emplace_back(mainNTuple);
   fPageSource = Internal::RPageSource::Create(mainNTuple.fNTupleName, mainNTuple.fStorage);
   fPageSource->Attach();

   if (fPageSource->GetNEntries() == 0) {
      throw RException(R__FAIL("provided RNTuple is empty"));
   }

   fNEntries = fPageSource->GetNEntries();

   if (models.empty())
      fModel = fPageSource->GetSharedDescriptorGuard()->CreateModel();
   else
      fModel = models[0]->Clone();

   fModel->Freeze();
   fEntry = fModel->CreateEntry();

   for (const auto &value : *fEntry) {
      auto &field = value.GetField();
      const auto &fieldName = field.GetQualifiedFieldName();

      // If the model provided by the user has a default entry, use the value pointers from the default entry of the
      // model that was passed to this constructor. This way, the pointers returned by RNTupleModel::MakeField can be
      // used in the processor loop to access the corresponding field values.
      if (!models.empty() && !models[0]->IsBare()) {
         auto valuePtr = models[0]->GetDefaultEntry().GetPtr<void>(fieldName);
         fEntry->BindValue(fieldName, valuePtr);
      }

      const auto &[fieldContext, _] =
         fFieldContexts.try_emplace(fieldName, field.Clone(fieldName), fEntry->GetToken(fieldName));
      ConnectField(fieldContext->second, *fPageSource, *fEntry);
   }

   for (unsigned i = 0; i < auxNTuples.size(); ++i) {
      if (models.empty()) {
         AddAuxiliary(auxNTuples[i], joinFields);
      } else {
         // The size of `models` is checked in `CreateJoin`; at this point we can safely assume that `models.size() ==
         // auxNTuples.size() + 1`.
         AddAuxiliary(auxNTuples[i], joinFields, std::move(models[i + 1]));
      }
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::AddAuxiliary(const RNTupleOpenSpec &auxNTuple,
                                                            const std::vector<std::string> &joinFields,
                                                            std::unique_ptr<RNTupleModel> model)
{
   assert(fNEntriesProcessed == 0 && "cannot add auxiliary ntuples after processing has started");

   fNTuples.emplace_back(auxNTuple);

   auto pageSource = Internal::RPageSource::Create(auxNTuple.fNTupleName, auxNTuple.fStorage);
   pageSource->Attach();

   if (pageSource->GetNEntries() == 0) {
      throw RException(R__FAIL("provided RNTuple is empty"));
   }

   if (!model)
      model = pageSource->GetSharedDescriptorGuard()->CreateModel();

   model->Freeze();
   auto entry = model->CreateBareEntry();

   // Append the auxiliary fields to the join model
   fModel->Unfreeze();

   // The fields of the auxiliary ntuple are contained in an anonymous record field and subsequently registered as
   // subfields to the join model. This way they can be accessed through the processor as `auxNTupleName.fieldName`,
   // which is necessary in case there are duplicate field names between the main ntuple and (any of the) auxiliary
   // ntuples.
   std::vector<std::unique_ptr<RFieldBase>> auxFields;
   auxFields.reserve(entry->fValues.size());
   for (const auto &val : *entry) {
      auto &field = val.GetField();

      auxFields.emplace_back(field.Clone(field.GetQualifiedFieldName()));
   }
   std::unique_ptr<RFieldBase> auxParentField =
      std::make_unique<RRecordField>(auxNTuple.fNTupleName, std::move(auxFields));

   if (!auxParentField) {
      throw RException(R__FAIL("could not create auxiliary RNTuple parent field"));
   }

   const auto &subfields = auxParentField->GetConstSubfields();
   fModel->AddField(std::move(auxParentField));
   for (const auto &field : subfields) {
      fModel->RegisterSubfield(field->GetQualifiedFieldName());
   }

   fModel->Freeze();
   // After modifying the join model, we need to create a new entry since the old one is invalidated. However, we do
   // want to carry over the value pointers, so the pointers returned by `MakeField` during the creation of the original
   // model by the user can be used in the processor loop.
   auto newEntry = fModel->CreateEntry();

   for (const auto &value : *newEntry) {
      const auto &field = value.GetField();

      // Skip if the field is the untyped record that holds the fields of auxiliary ntuples.
      const auto fnIsNTuple = [&field](RNTupleOpenSpec n) { return n.fNTupleName == field.GetFieldName(); };
      if (std::find_if(fNTuples.cbegin(), fNTuples.cend(), fnIsNTuple) != fNTuples.end()) {
         continue;
      }

      auto fieldContext = fFieldContexts.find(field.GetQualifiedFieldName());
      // If the field belongs to the auxiliary ntuple currently being added, apart from assigning its entry value the
      // correct pointer, we also have to create a field context for it.
      if (fieldContext == fFieldContexts.end()) {
         // If the model has a default entry, use the value pointers from the entry in the entry managed by the
         // processor. This way, the pointers returned by RNTupleModel::MakeField can be used in the processor loop to
         // access the corresponding field values.
         if (!model->IsBare()) {
            auto valuePtr = model->GetDefaultEntry().GetPtr<void>(field.GetFieldName());
            newEntry->BindValue(field.GetQualifiedFieldName(), valuePtr);
         }

         auto token = newEntry->GetToken(field.GetQualifiedFieldName());
         fFieldContexts.try_emplace(field.GetQualifiedFieldName(), field.Clone(field.GetFieldName()), token,
                                    fNTuples.size() - 1);
      } else {
         auto valuePtr = fEntry->GetPtr<void>(fieldContext->second.fToken);
         auto newToken = newEntry->GetToken(field.GetQualifiedFieldName());
         newEntry->BindValue(newToken, valuePtr);
         fieldContext->second.fToken = std::move(newToken);
      }
   }

   fEntry.swap(newEntry);

   // If no join fields have been specified, an aligned join is assumed and an join table won't be created.
   if (!joinFields.empty())
      fJoinTables.emplace_back(Internal::RNTupleJoinTable::Create(joinFields));

   fAuxiliaryPageSources.emplace_back(std::move(pageSource));
}

void ROOT::Experimental::RNTupleJoinProcessor::ConnectFields()
{
   for (auto &[_, fieldContext] : fFieldContexts) {
      Internal::RPageSource &pageSource =
         fieldContext.IsAuxiliary() ? *fAuxiliaryPageSources.at(fieldContext.fNTupleIdx - 1) : *fPageSource;
      ConnectField(fieldContext, pageSource, *fEntry);
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::SetEntryPointers(const REntry &entry)
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
   for (unsigned i = 0; i < fJoinTables.size(); ++i) {
      auto &joinTable = fJoinTables[i];
      if (!joinTable->IsBuilt())
         joinTable->Build(*fAuxiliaryPageSources[i]);

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
         // ConstructValue is not used anymore in the future, remove friend in RFieldBase.
         fieldContext.fProtoField->ConstructValue(value.GetPtr<void>().get());
      } else {
         value.Read(auxEntryIdxs[fieldContext.fNTupleIdx - 1]);
      }
   }

   return entryNumber;
}
