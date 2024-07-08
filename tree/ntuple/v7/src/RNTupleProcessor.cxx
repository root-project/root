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

#include <ROOT/RField.hxx>

ROOT::Experimental::NTupleSize_t
ROOT::Experimental::Internal::RNTupleProcessor::ConnectNTuple(const RNTupleSourceSpec &ntuple)
{
   for (auto &fieldContext : fFieldContexts) {
      fieldContext.ResetConcreteField();
   }
   fPageSource = Internal::RPageSource::Create(ntuple.fName, ntuple.fLocation);
   fPageSource->Attach();
   ConnectFields();
   return fPageSource->GetNEntries();
}

void ROOT::Experimental::Internal::RNTupleProcessor::ConnectFields()
{
   auto desc = fPageSource->GetSharedDescriptorGuard();

   for (auto &fieldContext : fFieldContexts) {
      auto fieldId = desc->FindFieldId(fieldContext.GetProtoField().GetFieldName());
      if (fieldId == kInvalidDescriptorId) {
         throw RException(
            R__FAIL("field \"" + fieldContext.GetProtoField().GetFieldName() + "\" not found in current RNTuple"));
      }

      fieldContext.SetConcreteField();
      fieldContext.fConcreteField->SetOnDiskId(desc->FindFieldId(fieldContext.GetProtoField().GetFieldName()));
      Internal::CallConnectPageSourceOnField(*fieldContext.fConcreteField, *fPageSource);

      auto valuePtr = fEntry->GetPtr<void>(fieldContext.fToken);
      auto value = fieldContext.fConcreteField->CreateValue();
      value.Bind(valuePtr);
      fEntry->UpdateValue(fieldContext.fToken, value);
   }
}

ROOT::Experimental::Internal::RNTupleProcessor::RNTupleProcessor(const std::vector<RNTupleSourceSpec> &ntuples,
                                                                 std::unique_ptr<RNTupleModel> model)
   : fNTuples(ntuples)
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

      fFieldContexts.emplace_back(field.Clone(field.GetFieldName()), token);
   }

   ConnectFields();
}
