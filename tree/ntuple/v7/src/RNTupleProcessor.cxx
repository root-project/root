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

   for (auto &[fieldName, fieldContext] : fFieldContexts) {
      ConnectField(fieldName);
   }

   return fPageSource->GetNEntries();
}

void ROOT::Experimental::Internal::RNTupleProcessor::ConnectField(std::string_view fieldName)
{
   auto &fieldContext = fFieldContexts.at(std::string(fieldName));
   auto desc = fPageSource->GetSharedDescriptorGuard();

   auto fieldId = desc->FindFieldId(fieldName);
   if (fieldId == kInvalidDescriptorId) {
      throw RException(R__FAIL("field \"" + std::string(fieldName) + "\" not found in current RNTuple"));
   }

   auto &concreteField = fieldContext.CreateConcreteField();
   concreteField.SetOnDiskId(desc->FindFieldId(fieldName));

   auto newValue = std::make_unique<RFieldBase::RValue>(concreteField.CreateValue());
   // True when the field was previously connected to another page source.
   if (fieldContext.fValue) {
      auto fieldPtr = fieldContext.fValue->GetPtr<void>();
      fieldContext.fValue.swap(newValue);
      fieldContext.fValue->Bind(fieldPtr);
   } else {
      fieldContext.fValue.swap(newValue);
   }

   Internal::CallConnectPageSourceOnField(concreteField, *fPageSource);
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

   RFieldContext fieldContext(std::move(protoField));
   fFieldContexts.emplace(fieldName, std::move(fieldContext));
   ConnectField(fieldName);
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
}

const std::vector<std::string> ROOT::Experimental::Internal::RNTupleProcessor::GetActiveFields() const
{
   std::vector<std::string> fieldNames;
   fieldNames.reserve(fFieldContexts.size());
   std::transform(fFieldContexts.cbegin(), fFieldContexts.cend(), std::back_inserter(fieldNames),
                  [](auto &fldCtx) { return fldCtx.first; });
   return fieldNames;
}
