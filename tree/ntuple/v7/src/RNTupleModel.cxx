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


ROOT::Experimental::RForestModel::RForestModel()
  : fRootField(std::make_unique<RFieldRoot>())
  , fDefaultEntry(std::make_unique<RForestEntry>())
{}

ROOT::Experimental::RForestModel* ROOT::Experimental::RForestModel::Clone()
{
   auto cloneModel = new RForestModel();
   auto cloneRootField = static_cast<RFieldRoot*>(fRootField->Clone(""));
   cloneModel->fRootField = std::unique_ptr<RFieldRoot>(cloneRootField);
   cloneModel->fDefaultEntry = std::unique_ptr<RForestEntry>(cloneRootField->GenerateEntry());
   return cloneModel;
}


void ROOT::Experimental::RForestModel::AddField(std::unique_ptr<Detail::RFieldBase> field)
{
   fDefaultEntry->AddValue(field->GenerateValue());
   fRootField->Attach(std::move(field));
}


std::shared_ptr<ROOT::Experimental::RCollectionForest> ROOT::Experimental::RForestModel::MakeCollection(
   std::string_view fieldName, std::unique_ptr<RForestModel> collectionModel)
{
   auto collectionForest = std::make_shared<RCollectionForest>(std::move(collectionModel->fDefaultEntry));
   auto field = std::make_unique<RFieldCollection>(fieldName, collectionForest, std::move(collectionModel));
   fDefaultEntry->CaptureValue(field->CaptureValue(collectionForest->GetOffsetPtr()));
   fRootField->Attach(std::move(field));
   return collectionForest;
}

std::unique_ptr<ROOT::Experimental::RForestEntry> ROOT::Experimental::RForestModel::CreateEntry()
{
   auto entry = std::make_unique<RForestEntry>();
   for (auto& f : *fRootField) {
      if (f.GetParent() != GetRootField())
         continue;
      entry->AddValue(f.GenerateValue());
   }
   return entry;
}
