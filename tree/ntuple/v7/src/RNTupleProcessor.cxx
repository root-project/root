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
      fieldContext.GetConcreteField().SetOnDiskId(desc->FindFieldId(fieldContext.GetProtoField().GetFieldName()));
      Internal::CallConnectPageSourceOnField(fieldContext.GetConcreteField(), *fPageSource);

      if (fieldContext.fValuePtr) {
         fEntry->UpdateValue(fieldContext.GetToken(),
                             fieldContext.GetConcreteField().BindValue(fieldContext.fValuePtr));
      } else {
         auto value = fieldContext.GetConcreteField().CreateValue();
         fieldContext.fValuePtr = value.GetPtr<void>();
         fEntry->UpdateValue(fieldContext.GetToken(), value);
      }
   }
}

ROOT::Experimental::Internal::RNTupleProcessor::RNTupleProcessor(const std::vector<RNTupleSourceSpec> &ntuples)
   : fNTuples(ntuples)
{
   if (fNTuples.empty())
      throw RException(R__FAIL("at least one RNTuple must be provided"));

   fPageSource = Internal::RPageSource::Create(fNTuples[0].fName, fNTuples[0].fLocation);
   fPageSource->Attach();

   if (fPageSource->GetNEntries() == 0) {
      throw RException(R__FAIL("first RNTuple does not contain any entries"));
   }

   fEntry = std::unique_ptr<REntry>(new REntry());

   auto desc = fPageSource->GetSharedDescriptorGuard();

   for (const auto &fieldDesc : desc->GetTopLevelFields()) {
      auto fieldOrException = RFieldBase::Create(fieldDesc.GetFieldName(), fieldDesc.GetTypeName());
      if (fieldOrException) {
         auto field = fieldOrException.Unwrap();
         fEntry->AddValue(field->CreateValue());
         fFieldContexts.emplace_back(std::move(field), fEntry->GetToken(fieldDesc.GetFieldName()));
      } else {
         throw RException(R__FAIL("could not create field \"" + fieldDesc.GetFieldName() + "\" with type \"" +
                                  fieldDesc.GetTypeName() + "\""));
      }
   }

   ConnectFields();
}
