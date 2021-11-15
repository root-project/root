/// \file RNTupleModel.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-15
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/StringUtils.hxx>

#include <atomic>
#include <cstdlib>
#include <memory>
#include <utility>


void ROOT::Experimental::RNTupleModel::EnsureValidFieldName(std::string_view fieldName)
{
   RResult<void> nameValid = Detail::RFieldBase::EnsureValidFieldName(fieldName);
   if (!nameValid) {
      nameValid.Throw();
   }
   auto fieldNameStr = std::string(fieldName);
   if (fFieldNames.insert(fieldNameStr).second == false) {
      throw RException(R__FAIL("field name '" + fieldNameStr + "' already exists in NTuple model"));
   }
}

ROOT::Experimental::RNTupleModel::RNTupleModel()
   : fFieldZero(std::make_unique<RFieldZero>()), fDefaultEntry(std::unique_ptr<REntry>(new REntry()))
{}

std::unique_ptr<ROOT::Experimental::RNTupleModel> ROOT::Experimental::RNTupleModel::Clone() const
{
   auto cloneModel = std::make_unique<RNTupleModel>();
   auto cloneFieldZero = fFieldZero->Clone("");
   cloneModel->fModelId = fModelId;
   cloneModel->fFieldZero = std::unique_ptr<RFieldZero>(static_cast<RFieldZero *>(cloneFieldZero.release()));
   cloneModel->fFieldNames = fFieldNames;
   cloneModel->fDescription = fDescription;
   cloneModel->fDefaultEntry = std::unique_ptr<REntry>(new REntry());
   for (const auto &f : cloneModel->fFieldZero->GetSubFields()) {
      cloneModel->fDefaultEntry->AddValue(f->GenerateValue());
   }
   return cloneModel;
}


void ROOT::Experimental::RNTupleModel::AddField(std::unique_ptr<Detail::RFieldBase> field)
{
   if (IsFrozen())
      throw RException(R__FAIL("invalid attempt to add field to frozen model"));
   if (!field)
      throw RException(R__FAIL("null field"));
   EnsureValidFieldName(field->GetName());

   fDefaultEntry->AddValue(field->GenerateValue());
   fFieldZero->Attach(std::move(field));
}


std::shared_ptr<ROOT::Experimental::RCollectionNTupleWriter> ROOT::Experimental::RNTupleModel::MakeCollection(
   std::string_view fieldName, std::unique_ptr<RNTupleModel> collectionModel)
{
   if (IsFrozen())
      throw RException(R__FAIL("invalid attempt to add collection to frozen model"));
   EnsureValidFieldName(fieldName);
   if (!collectionModel) {
      throw RException(R__FAIL("null collectionModel"));
   }
   auto collectionNTuple = std::make_shared<RCollectionNTupleWriter>(std::move(collectionModel->fDefaultEntry));
   auto field = std::make_unique<RCollectionField>(fieldName, collectionNTuple, std::move(collectionModel));
   fDefaultEntry->CaptureValue(field->CaptureValue(collectionNTuple->GetOffsetPtr()));
   fFieldZero->Attach(std::move(field));
   return collectionNTuple;
}

const ROOT::Experimental::Detail::RFieldBase *
ROOT::Experimental::RNTupleModel::GetField(std::string_view fieldName) const
{
   if (fieldName.empty())
      return nullptr;

   auto *field = static_cast<ROOT::Experimental::Detail::RFieldBase *>(fFieldZero.get());
   for (auto subfieldName : ROOT::Split(fieldName, ".")) {
      const auto subfields = field->GetSubFields();
      auto it =
         std::find_if(subfields.begin(), subfields.end(), [&](const auto *f) { return f->GetName() == subfieldName; });
      if (it != subfields.end()) {
         field = *it;
      } else {
         field = nullptr;
         break;
      }
   }

   return field;
}

ROOT::Experimental::REntry *ROOT::Experimental::RNTupleModel::GetDefaultEntry() const
{
   if (!IsFrozen())
      throw RException(R__FAIL("invalid attempt to get default entry of unfrozen model"));
   return fDefaultEntry.get();
}

std::unique_ptr<ROOT::Experimental::REntry> ROOT::Experimental::RNTupleModel::CreateEntry() const
{
   if (!IsFrozen())
      throw RException(R__FAIL("invalid attempt to create entry of unfrozen model"));

   auto entry = std::unique_ptr<REntry>(new REntry(fModelId));
   for (auto& f : *fFieldZero) {
      if (f.GetParent() != GetFieldZero())
         continue;
      entry->AddValue(f.GenerateValue());
   }
   return entry;
}

void ROOT::Experimental::RNTupleModel::Freeze()
{
   if (IsFrozen())
      return;

   static std::atomic<std::uint64_t> gLastModelId = 0;
   fModelId = ++gLastModelId;
   fDefaultEntry->fModelId = fModelId;
}

void ROOT::Experimental::RNTupleModel::SetDescription(std::string_view description)
{
   if (IsFrozen())
      throw RException(R__FAIL("invalid attempt to set description of frozen model"));
   fDescription = std::string(description);
}
