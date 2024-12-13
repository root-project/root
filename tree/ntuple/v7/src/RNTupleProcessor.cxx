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
void EnsureUniqueNTupleNames(const std::vector<RNTupleOpenSpec> &ntuples)
{
   std::unordered_set<std::string> uniqueNTupleNames;
   for (const auto &ntuple : ntuples) {
      auto res = uniqueNTupleNames.emplace(ntuple.fNTupleName);
      if (!res.second) {
         throw ROOT::RException(R__FAIL("horizontal joining of RNTuples with the same name is not allowed"));
      }
   }
}
} // anonymous namespace

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::Create(const RNTupleOpenSpec &ntuple)
{
   auto pageSource = Internal::RPageSource::Create(ntuple.fNTupleName, ntuple.fStorage);
   pageSource->Attach();
   auto model = pageSource->GetSharedDescriptorGuard()->CreateModel();
   return RNTupleProcessor::Create(ntuple, *model);
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::Create(const RNTupleOpenSpec &ntuple, RNTupleModel &model)
{
   return std::unique_ptr<RNTupleSingleProcessor>(new RNTupleSingleProcessor(ntuple, model));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(const std::vector<RNTupleOpenSpec> &ntuples,
                                                  std::unique_ptr<RNTupleModel> model)
{
   return std::unique_ptr<RNTupleChainProcessor>(new RNTupleChainProcessor(ntuples, std::move(model)));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateJoin(const std::vector<RNTupleOpenSpec> &ntuples,
                                                 const std::vector<std::string> &joinFields,
                                                 std::vector<std::unique_ptr<RNTupleModel>> models)
{
   if (ntuples.size() < 1)
      throw RException(R__FAIL("at least one RNTuple must be provided"));

   if (models.size() > 0 && models.size() != ntuples.size()) {
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
   EnsureUniqueNTupleNames(ntuples);

   std::unique_ptr<RNTupleJoinProcessor> processor;
   if (models.size() > 0) {
      processor = std::unique_ptr<RNTupleJoinProcessor>(new RNTupleJoinProcessor(ntuples[0], std::move(models[0])));
   } else {
      processor = std::unique_ptr<RNTupleJoinProcessor>(new RNTupleJoinProcessor(ntuples[0]));
   }

   for (unsigned i = 1; i < ntuples.size(); ++i) {
      if (models.size() > 0)
         processor->AddAuxiliary(ntuples[i], joinFields, std::move(models[i]));
      else
         processor->AddAuxiliary(ntuples[i], joinFields);
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
   if (fieldId == kInvalidDescriptorId) {
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

ROOT::Experimental::RNTupleSingleProcessor::RNTupleSingleProcessor(const RNTupleOpenSpec &ntuple, RNTupleModel &model)
   : RNTupleProcessor({ntuple})
{
   fPageSource = Internal::RPageSource::Create(ntuple.fNTupleName, ntuple.fStorage);
   fPageSource->Attach();

   model.Freeze();
   fEntry = model.CreateEntry();

   for (const auto &value : *fEntry) {
      auto &field = value.GetField();
      auto token = fEntry->GetToken(field.GetFieldName());

      // If the model has a default entry, use the value pointers from the entry in the entry managed by the
      // processor. This way, the pointers returned by RNTupleModel::MakeField can be used in the processor loop to
      // access the corresponding field values.
      if (!model.IsBare()) {
         auto valuePtr = model.GetDefaultEntry().GetPtr<void>(token);
         fEntry->BindValue(token, valuePtr);
      }

      auto fieldContext = RFieldContext(field.Clone(field.GetFieldName()), token);
      ConnectField(fieldContext, *fPageSource, *fEntry);
      fFieldContexts.try_emplace(field.GetFieldName(), std::move(fieldContext));
   }
}

ROOT::Experimental::NTupleSize_t ROOT::Experimental::RNTupleSingleProcessor::Advance()
{
   if (fLocalEntryNumber == kInvalidNTupleIndex || fLocalEntryNumber >= fPageSource->GetNEntries()) {
      return kInvalidNTupleIndex;
   }

   LoadEntry();

   fNEntriesProcessed++;
   return fLocalEntryNumber;
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleChainProcessor::RNTupleChainProcessor(const std::vector<RNTupleOpenSpec> &ntuples,
                                                                 std::unique_ptr<RNTupleModel> model)
   : RNTupleProcessor(ntuples)
{
   if (fNTuples.empty())
      throw RException(R__FAIL("at least one RNTuple must be provided"));

   fPageSource = Internal::RPageSource::Create(fNTuples[0].fNTupleName, fNTuples[0].fStorage);
   fPageSource->Attach();

   if (fPageSource->GetNEntries() == 0) {
      throw RException(R__FAIL("first RNTuple does not contain any entries"));
   }

   if (!model)
      model = fPageSource->GetSharedDescriptorGuard()->CreateModel();

   model->Freeze();
   fEntry = model->CreateEntry();

   for (const auto &value : *fEntry) {
      auto &field = value.GetField();
      auto token = fEntry->GetToken(field.GetFieldName());

      // If the model has a default entry, use the value pointers from the entry in the entry managed by the
      // processor. This way, the pointers returned by RNTupleModel::MakeField can be used in the processor loop to
      // access the corresponding field values.
      if (!model->IsBare()) {
         auto valuePtr = model->GetDefaultEntry().GetPtr<void>(token);
         fEntry->BindValue(token, valuePtr);
      }

      const auto &[fieldContext, _] =
         fFieldContexts.try_emplace(field.GetFieldName(), field.Clone(field.GetFieldName()), token);
      ConnectField(fieldContext->second, *fPageSource, *fEntry);
   }
}

ROOT::Experimental::NTupleSize_t ROOT::Experimental::RNTupleChainProcessor::ConnectNTuple(const RNTupleOpenSpec &ntuple)
{
   // Before destroying the current page source and replacing it by the new one, we need to reset the concrete fields
   // belonging to the current page source. Otherwise, these concrete fields become invalid.
   for (auto &[_, fieldContext] : fFieldContexts) {
      fieldContext.ResetConcreteField();
   }
   // Replace the current page source with a new one, belonging to the provided ntuple.
   fPageSource = Internal::RPageSource::Create(ntuple.fNTupleName, ntuple.fStorage);
   fPageSource->Attach();

   // Now that the new page source has been created and attached, we can create and connect the concrete fields again.
   for (auto &[_, fieldContext] : fFieldContexts) {
      ConnectField(fieldContext, *fPageSource, *fEntry);
   }

   return fPageSource->GetNEntries();
}

ROOT::Experimental::NTupleSize_t ROOT::Experimental::RNTupleChainProcessor::Advance()
{
   if (fLocalEntryNumber == kInvalidNTupleIndex)
      return kInvalidNTupleIndex;

   if (fLocalEntryNumber >= fPageSource->GetNEntries()) {
      do {
         if (++fCurrentNTupleNumber >= fNTuples.size()) {
            return kInvalidNTupleIndex;
         }
         // Skip over empty ntuples we might encounter.
      } while (ConnectNTuple(fNTuples.at(fCurrentNTupleNumber)) == 0);

      fLocalEntryNumber = 0;
   }

   LoadEntry();

   fNEntriesProcessed++;
   return fLocalEntryNumber;
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleJoinProcessor::RNTupleJoinProcessor(const RNTupleOpenSpec &mainNTuple,
                                                               std::unique_ptr<RNTupleModel> model)
   : RNTupleProcessor({mainNTuple})
{
   fPageSource = Internal::RPageSource::Create(mainNTuple.fNTupleName, mainNTuple.fStorage);
   fPageSource->Attach();

   if (fPageSource->GetNEntries() == 0) {
      throw RException(R__FAIL("provided RNTuple is empty"));
   }

   if (!model)
      model = fPageSource->GetSharedDescriptorGuard()->CreateModel();

   fJoinModel = model->Clone();
   fJoinModel->Freeze();
   fEntry = fJoinModel->CreateEntry();

   for (const auto &value : *fEntry) {
      auto &field = value.GetField();
      const auto &fieldName = field.GetQualifiedFieldName();

      // If the model has a default entry, use the value pointers from the default entry of the model that was passed to
      // this constructor. This way, the pointers returned by RNTupleModel::MakeField can be used in the processor loop
      // to access the corresponding field values.
      if (!fJoinModel->IsBare()) {
         auto valuePtr = model->GetDefaultEntry().GetPtr<void>(fieldName);
         fEntry->BindValue(fieldName, valuePtr);
      }

      const auto &[fieldContext, _] =
         fFieldContexts.try_emplace(fieldName, field.Clone(fieldName), fEntry->GetToken(fieldName));
      ConnectField(fieldContext->second, *fPageSource, *fEntry);
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
   fJoinModel->Unfreeze();

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

   const auto &subFields = auxParentField->GetSubFields();
   fJoinModel->AddField(std::move(auxParentField));
   for (const auto &field : subFields) {
      fJoinModel->RegisterSubfield(field->GetQualifiedFieldName());
   }

   fJoinModel->Freeze();
   // After modifying the join model, we need to create a new entry since the old one is invalidated. However, we do
   // want to carry over the value pointers, so the pointers returned by `MakeField` during the creation of the original
   // model by the user can be used in the processor loop.
   auto newEntry = fJoinModel->CreateEntry();

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

   // If no join fields have been specified, an aligned join is assumed and an index won't be necessary.
   if (joinFields.size() > 0)
      fJoinIndices.emplace_back(Internal::RNTupleIndex::Create(joinFields, *pageSource, true /* deferBuild */));

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

ROOT::Experimental::NTupleSize_t ROOT::Experimental::RNTupleJoinProcessor::Advance()
{
   if (fLocalEntryNumber == kInvalidNTupleIndex || fLocalEntryNumber >= fPageSource->GetNEntries()) {
      return kInvalidNTupleIndex;
   }

   LoadEntry();

   fNEntriesProcessed++;
   return fLocalEntryNumber;
}

void ROOT::Experimental::RNTupleJoinProcessor::LoadEntry()
{
   // Read the values of the primary ntuple. If no index is used (i.e., the join is aligned), also read the values of
   // auxiliary ntuples.
   for (const auto &[_, fieldContext] : fFieldContexts) {
      if (!fieldContext.IsAuxiliary() || !IsUsingIndex()) {
         auto &value = fEntry->GetValue(fieldContext.fToken);
         value.Read(fLocalEntryNumber);
      }
   }

   // If no index is used (i.e., the join is aligned), there's nothing left to do.
   if (!IsUsingIndex())
      return;

   // Collect the values of the join fields for this entry.
   std::vector<void *> valPtrs;
   valPtrs.reserve(fJoinFieldTokens.size());
   for (const auto &token : fJoinFieldTokens) {
      auto ptr = fEntry->GetPtr<void>(token);
      valPtrs.push_back(ptr.get());
   }

   // Find the index entry number corresponding to the join field values for each auxiliary ntuple.
   std::vector<NTupleSize_t> indexEntryNumbers;
   indexEntryNumbers.reserve(fJoinIndices.size());
   for (unsigned i = 0; i < fJoinIndices.size(); ++i) {
      auto &joinIndex = fJoinIndices[i];
      if (!joinIndex->IsBuilt())
         joinIndex->Build();

      indexEntryNumbers.push_back(joinIndex->GetFirstEntryNumber(valPtrs));
   }

   // For each auxiliary field, load its value according to the entry number we just found of the ntuple it belongs to.
   for (const auto &[_, fieldContext] : fFieldContexts) {
      if (!fieldContext.IsAuxiliary())
         continue;

      auto &value = fEntry->GetValue(fieldContext.fToken);
      if (indexEntryNumbers[fieldContext.fNTupleIdx - 1] == kInvalidNTupleIndex) {
         // No matching entry exists, so we reset the field's value to a default value.
         // TODO(fdegeus): further consolidate how non-existing join matches should be handled. N.B.: in case
         // ConstructValue is not used anymore in the future, remove friend in RFieldBase.
         fieldContext.fProtoField->ConstructValue(value.GetPtr<void>().get());
      } else {
         value.Read(indexEntryNumbers[fieldContext.fNTupleIdx - 1]);
      }
   }
}
