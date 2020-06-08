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

#include <ROOT/RField.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTuple.hxx>

#include <TError.h>

#include <cstdlib>
#include <memory>
#include <utility>


ROOT::Experimental::RNTupleModel::RNTupleModel()
  : fRootField(std::make_unique<RFieldRoot>())
  , fDefaultEntry(std::make_unique<REntry>())
{}

ROOT::Experimental::RNTupleModel* ROOT::Experimental::RNTupleModel::Clone()
{
   auto cloneModel = new RNTupleModel();
   auto cloneRootField = static_cast<RFieldRoot*>(fRootField->Clone(""));
   cloneModel->fRootField = std::unique_ptr<RFieldRoot>(cloneRootField);
   cloneModel->fDefaultEntry = std::unique_ptr<REntry>(cloneRootField->GenerateEntry());
   return cloneModel;
}


void ROOT::Experimental::RNTupleModel::AddField(std::unique_ptr<Detail::RFieldBase> field)
{
   fDefaultEntry->AddValue(field->GenerateValue());
   fRootField->Attach(std::move(field));
}


std::shared_ptr<ROOT::Experimental::RCollectionNTuple> ROOT::Experimental::RNTupleModel::MakeCollection(
   std::string_view fieldName, std::unique_ptr<RNTupleModel> collectionModel)
{
   auto collectionNTuple = std::make_shared<RCollectionNTuple>(std::move(collectionModel->fDefaultEntry));
   auto field = std::make_unique<RCollectionField>(fieldName, collectionNTuple, std::move(collectionModel));
   fDefaultEntry->CaptureValue(field->CaptureValue(collectionNTuple->GetOffsetPtr()));
   fRootField->Attach(std::move(field));
   return collectionNTuple;
}

std::unique_ptr<ROOT::Experimental::REntry> ROOT::Experimental::RNTupleModel::CreateEntry()
{
   auto entry = std::make_unique<REntry>();
   for (auto& f : *fRootField) {
      if (f.GetParent() != GetRootField())
         continue;
      entry->AddValue(f.GenerateValue());
   }
   return entry;
}
