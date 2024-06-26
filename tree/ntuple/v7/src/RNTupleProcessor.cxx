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
   for (auto &[fieldName, fieldContext] : fFieldContexts) {
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

   for (auto &[fieldName, fieldContext] : fFieldContexts) {
      auto fieldId = desc->FindFieldId(fieldName);
      if (fieldId == kInvalidDescriptorId) {
         throw RException(R__FAIL("field \"" + fieldName + "\" not found in current RNTuple"));
      }

      auto &concreteField = fieldContext.CreateConcreteField();
      fieldContext.GetConcreteField().SetOnDiskId(desc->FindFieldId(fieldName));
      Internal::CallConnectPageSourceOnField(concreteField, *fPageSource);

      if (fieldContext.fValuePtr) {
         fEntry->UpdateValue(fieldContext.GetToken(), concreteField.BindValue(fieldContext.fValuePtr));
      } else {
         auto value = concreteField.CreateValue();
         fieldContext.fValuePtr = value.GetPtr<void>();
         fEntry->UpdateValue(fieldContext.GetToken(), value);
      }
   }
}

void ROOT::Experimental::Internal::RNTupleProcessor::ActivateField(std::string_view fieldName)
{
   auto desc = fPageSource->GetSharedDescriptorGuard();
   auto fieldId = desc->FindFieldId(fieldName);
   auto &fieldDesc = desc->GetFieldDescriptor(fieldId);

   auto fieldOrException = RFieldBase::Create(fieldDesc.GetFieldName(), fieldDesc.GetTypeName());
   if (!fieldOrException) {
      throw RException(R__FAIL("could not create field \"" + fieldDesc.GetFieldName() + "\" with type \"" +
                               fieldDesc.GetTypeName() + "\""));
   }
   auto protoField = fieldOrException.Unwrap();
   fEntry->AddValue(protoField->CreateValue());

   RFieldContext fieldContext(std::move(protoField), fEntry->GetToken(fieldDesc.GetFieldName()));
   fieldContext.fValuePtr = fEntry->GetPtr<void>(fieldContext.GetToken());

   auto &concreteField = fieldContext.CreateConcreteField();
   concreteField.SetOnDiskId(fieldId);
   Internal::CallConnectPageSourceOnField(concreteField, *fPageSource);
   fEntry->UpdateValue(fieldContext.GetToken(), concreteField.BindValue(fieldContext.fValuePtr));

   fFieldContexts.emplace(fieldName, std::move(fieldContext));
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
}

const std::vector<std::string> ROOT::Experimental::Internal::RNTupleProcessor::GetActiveFields() const
{
   std::vector<std::string> fieldNames;
   fieldNames.reserve(fFieldContexts.size());
   std::transform(fFieldContexts.cbegin(), fFieldContexts.cend(), std::back_inserter(fieldNames),
                  [](auto &fldCtx) { return fldCtx.first; });
   return fieldNames;
}
