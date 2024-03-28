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

ROOT::Experimental::RNTupleProcessor::RNTupleProcessor(std::span<RProcessorSpec> ntuples)
{
   if (ntuples.empty())
      throw RException(R__FAIL("at least one RNTuple must be provided"));

   for (const auto &ntuple : ntuples) {
      fPageSources.push_back(Internal::RPageSource::Create(ntuple.fNTupleName, ntuple.fStorage));
   }

   auto &firstPageSource = fPageSources.front();
   firstPageSource->Attach();

   auto model = firstPageSource->GetSharedDescriptorGuard()->CreateModel();
   fEntry = model->CreateBareEntry();

   auto desc = firstPageSource->GetSharedDescriptorGuard();
   SetProcessorFields(*firstPageSource, *fEntry, desc->GetFieldZeroId());
   ConnectFields(*firstPageSource);
}

void ROOT::Experimental::RNTupleProcessor::SetProcessorFields(Internal::RPageSource &pageSource, REntry &entry,
                                                              DescriptorId_t fieldId)
{
   auto desc = pageSource.GetSharedDescriptorGuard();
   auto &fieldDesc = desc->GetFieldDescriptor(fieldId);

   auto fieldOrException = RFieldBase::Create(fieldDesc.GetFieldName(), fieldDesc.GetTypeName());
   if (fieldOrException) {
      auto field = fieldOrException.Unwrap();
      field->SetOnDiskId(fieldId);
      fProcessorFields.emplace_back(std::move(field), entry.GetToken(fieldDesc.GetFieldName()));
   }

   for (const auto &subFieldDesc : desc->GetFieldIterable(fieldId)) {
      SetProcessorFields(pageSource, entry, subFieldDesc.GetId());
   }
}

void ROOT::Experimental::RNTupleProcessor::ConnectFields(Internal::RPageSource &pageSource)
{
   auto desc = pageSource.GetSharedDescriptorGuard();
   for (auto &field : fProcessorFields) {
      field.ResetConcreteField();

      field.fConcreteField->SetOnDiskId(desc->FindFieldId(field.fProtoField->GetFieldName()));
      Internal::CallConnectPageSourceOnField(*field.fConcreteField, pageSource);

      if (field.fValuePtr) {
         fEntry->UpdateValue(field.fToken, field.fConcreteField->BindValue(field.fValuePtr));
      } else {
         auto value = field.fConcreteField->CreateValue();
         field.fValuePtr = value.GetPtr<void>();
         fEntry->UpdateValue(field.fToken, value);
      }
   }
}
