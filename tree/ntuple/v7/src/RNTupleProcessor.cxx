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
using ROOT::Experimental::RNTupleSourceSpec;
void EnsureUniqueNTupleNames(std::vector<RNTupleSourceSpec> ntuples)
{
   std::sort(ntuples.begin(), ntuples.end(),
             [](const RNTupleSourceSpec &left, const RNTupleSourceSpec &right) { return left.fName < right.fName; });
   auto newEnd =
      std::unique(ntuples.begin(), ntuples.end(), [](const RNTupleSourceSpec &left, const RNTupleSourceSpec &right) {
         return left.fName == right.fName;
      });
   if (newEnd != ntuples.end()) {
      throw ROOT::Experimental::RException(R__FAIL("horizontal joining of RNTuples with the same name is not allowed"));
   }
}
} // anonymous namespace

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(const std::vector<RNTupleSourceSpec> &ntuples,
                                                  std::unique_ptr<RNTupleModel> model)
{
   // TODO make_unique
   return std::unique_ptr<RNTupleChainProcessor>(new RNTupleChainProcessor(ntuples, std::move(model)));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateJoin(const std::vector<RNTupleSourceSpec> &ntuples,
                                                 const std::vector<std::string> &joinFields)
{
   if (ntuples.size() < 1)
      throw RException(R__FAIL("at least one RNTuple must be provided"));

   EnsureUniqueNTupleNames(ntuples);

   auto processor = std::unique_ptr<RNTupleJoinProcessor>(new RNTupleJoinProcessor(ntuples[0], joinFields));

   for (unsigned i = 1; i < ntuples.size(); ++i) {
      processor->AddAuxiliary(ntuples[i]);
   }

   processor->ConnectFields();

   return processor;
}

void ROOT::Experimental::RNTupleProcessor::ConnectField(RFieldContext &fieldContext, Internal::RPageSource &pageSource,
                                                        REntry &entry)
{
   auto desc = pageSource.GetSharedDescriptorGuard();

   auto fieldId = desc->FindFieldId(fieldContext.GetProtoField().GetFieldName());
   if (fieldId == kInvalidDescriptorId) {
      throw RException(
         R__FAIL("field \"" + fieldContext.GetProtoField().GetFieldName() + "\" not found in current RNTuple"));
   }

   fieldContext.SetConcreteField();
   fieldContext.fConcreteField->SetOnDiskId(desc->FindFieldId(fieldContext.GetProtoField().GetFieldName()));
   Internal::CallConnectPageSourceOnField(*fieldContext.fConcreteField, pageSource);

   auto token = entry.GetToken(fieldContext.GetQualifiedFieldName());
   auto valuePtr = entry.GetPtr<void>(token);
   auto value = fieldContext.fConcreteField->CreateValue();
   value.Bind(valuePtr);
   entry.UpdateValue(token, value);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleChainProcessor::RNTupleChainProcessor(const std::vector<RNTupleSourceSpec> &ntuples,
                                                                 std::unique_ptr<RNTupleModel> model)
   : RNTupleProcessor(ntuples)
{
   if (fNTuples.empty())
      throw RException(R__FAIL("at least one RNTuple must be provided"));

   fPageSource = Internal::RPageSource::Create(fNTuples[0].fName, fNTuples[0].fLocation);
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

      const auto &[fieldContext, _] = fFieldContexts.try_emplace(
         field.GetFieldName(), field.Clone(field.GetFieldName()), token, fPageSource->GetNTupleName());
      ConnectField(fieldContext->second, *fPageSource, *fEntry);
   }
}

ROOT::Experimental::NTupleSize_t
ROOT::Experimental::RNTupleChainProcessor::ConnectNTuple(const RNTupleSourceSpec &ntuple)
{
   for (auto &[_, fieldContext] : fFieldContexts) {
      fieldContext.ResetConcreteField();
   }
   fPageSource = Internal::RPageSource::Create(ntuple.fName, ntuple.fLocation);
   fPageSource->Attach();

   for (auto &[_, fieldContext] : fFieldContexts) {
      ConnectField(fieldContext, *fPageSource, *fEntry);
   }

   return fPageSource->GetNEntries();
}

ROOT::Experimental::NTupleSize_t ROOT::Experimental::RNTupleChainProcessor::Advance()
{
   ++fNEntriesProcessed;

   if (++fLocalEntryNumber >= fPageSource->GetNEntries()) {
      do {
         if (++fCurrentNTupleNumber >= fNTuples.size()) {
            return kInvalidNTupleIndex;
         }
         // Skip over empty ntuples we might encounter.
      } while (ConnectNTuple(fNTuples.at(fCurrentNTupleNumber)) == 0);

      fLocalEntryNumber = 0;
   }

   fEntry->Read(fLocalEntryNumber);

   return fNEntriesProcessed;
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleJoinProcessor::RNTupleJoinProcessor(const RNTupleSourceSpec &mainNTuple,
                                                               const std::vector<std::string> &joinFields,
                                                               std::unique_ptr<RNTupleModel> model)
   : RNTupleProcessor({mainNTuple}), fJoinFieldNames(joinFields)
{
   fPageSource = Internal::RPageSource::Create(mainNTuple.fName, mainNTuple.fLocation);
   fPageSource->Attach();

   if (fPageSource->GetNEntries() == 0) {
      throw RException(R__FAIL("provided RNTuple is empty"));
   }

   if (!model)
      model = fPageSource->GetSharedDescriptorGuard()->CreateModel();

   fJoinModel = model->Clone();
   fJoinModel->Freeze();
   fEntry = fJoinModel->CreateEntry();
   fUseIndex = !fJoinFieldNames.empty();

   for (const auto &value : *fEntry) {
      auto &field = value.GetField();
      auto token = fEntry->GetToken(field.GetFieldName());

      // If the model has a default entry, use the value pointers from the entry in the entry managed by the
      // processor. This way, the pointers returned by RNTupleModel::MakeField can be used in the processor loop to
      // access the corresponding field values.
      if (!fJoinModel->IsBare()) {
         auto valuePtr = fJoinModel->GetDefaultEntry().GetPtr<void>(token);
         fEntry->BindValue(token, valuePtr);
      }

      const auto &[fieldContext, _] =
         fFieldContexts.try_emplace(field.GetFieldName(), field.Clone(field.GetFieldName()), token, mainNTuple.fName);
      ConnectField(fieldContext->second, *fPageSource, *fEntry);
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::AddAuxiliary(const RNTupleSourceSpec &auxNTuple,
                                                            std::unique_ptr<RNTupleModel> model)
{
   auto [res, _] = fAuxiliaryPageSources.try_emplace(
      auxNTuple.fName, Internal::RPageSource::Create(auxNTuple.fName, auxNTuple.fLocation));
   auto &pageSource = *res->second;
   pageSource.Attach();

   if (pageSource.GetNEntries() == 0) {
      throw RException(R__FAIL("provided RNTuple is empty"));
   }

   if (!model)
      model = pageSource.GetSharedDescriptorGuard()->CreateModel();

   auto entry = model->CreateEntry();

   // Append the auxiliary fields to the join model
   fJoinModel->Unfreeze();
   for (const auto &value : *entry) {
      const auto &field = value.GetField();
      // TODO figure out how we can use dots
      fJoinModel->AddField(field.Clone(auxNTuple.fName + "#" + field.GetFieldName()));
   }

   fJoinModel->Freeze();
   // After modifying the join entry, we need to create a new entry since the old one is invalidated.
   auto newEntry = fJoinModel->CreateEntry();
   fEntry.swap(newEntry);

   // Create the field contexts
   for (const auto &value : *entry) {
      auto &field = value.GetField();
      std::string fieldName = auxNTuple.fName + "#" + field.GetFieldName();

      auto token = fEntry->GetToken(fieldName);

      // If the model has a default entry, use the value pointers from the entry in the entry managed by the
      // processor. This way, the pointers returned by RNTupleModel::MakeField can be used in the processor loop to
      // access the corresponding field values.
      if (!model->IsBare()) {
         auto valuePtr = model->GetDefaultEntry().GetPtr<void>(field.GetFieldName());
         fEntry->BindValue(token, valuePtr);
      }

      fFieldContexts.try_emplace(fieldName, field.Clone(field.GetFieldName()), token, auxNTuple.fName,
                                 true /* isAuxiliary */);
   }

   if (fUseIndex) {
      fJoinIndices.try_emplace(auxNTuple.fName,
                               Internal::RNTupleIndex::Create(fJoinFieldNames, pageSource, true /* deferBuild */));
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::ConnectFields()
{
   for (auto &[_, fieldContext] : fFieldContexts) {
      Internal::RPageSource &pageSource =
         fieldContext.IsAuxiliary() ? *fAuxiliaryPageSources.at(fieldContext.GetNTupleName()) : *fPageSource;
      ConnectField(fieldContext, pageSource, *fEntry);
   }
}

ROOT::Experimental::NTupleSize_t ROOT::Experimental::RNTupleJoinProcessor::Advance()
{
   ++fNEntriesProcessed;

   if (fNEntriesProcessed >= fPageSource->GetNEntries()) {
      return kInvalidNTupleIndex;
   }

   ++fLocalEntryNumber;
   LoadEntry();

   return fNEntriesProcessed;
}

void ROOT::Experimental::RNTupleJoinProcessor::LoadEntry()
{
   std::vector<void *> valPtrs;
   valPtrs.reserve(fJoinFieldNames.size());

   for (const auto &fieldName : fJoinFieldNames) {
      auto ptr = fEntry->GetPtr<void>(fieldName);
      valPtrs.push_back(ptr.get());
   }

   for (auto &[fieldName, fieldContext] : fFieldContexts) {
      if (!fieldContext.fIsAuxiliary || !fUseIndex) {
         auto &value = fEntry->GetValue(fieldName);
         value.Read(fLocalEntryNumber);
         continue;
      }

      auto joinIndex = fJoinIndices.find(fieldContext.GetNTupleName());
      if (joinIndex != fJoinIndices.end()) {
         if (!joinIndex->second->IsBuilt())
            joinIndex->second->Build();

         auto joinIdx = joinIndex->second->GetFirstEntryNumber(valPtrs);

         auto &value = fEntry->GetValue(fieldName);
         if (joinIdx == kInvalidNTupleIndex) {
            fieldContext.fProtoField->ConstructValue(value.GetPtr<void>().get());
         } else {
            value.Read(joinIdx);
         }
      }
   }
}
