/// \file RTree.cxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RForest.hxx"

#include "ROOT/RForestModel.hxx"
#include "ROOT/RPageStorage.hxx"
#include "ROOT/RPageStorageRoot.hxx"

#include <utility>

ROOT::Experimental::Detail::RForest::RForest(std::unique_ptr<ROOT::Experimental::RForestModel> model)
   : fModel(std::move(model))
   , fNEntries(0)
{
}

ROOT::Experimental::Detail::RForest::~RForest()
{
}

//------------------------------------------------------------------------------

ROOT::Experimental::RInputForest::RInputForest(
   std::unique_ptr<ROOT::Experimental::RForestModel> model,
   std::unique_ptr<ROOT::Experimental::Detail::RPageSource> source)
   : ROOT::Experimental::Detail::RForest(std::move(model))
   , fSource(std::move(source))
{
   fSource->Attach();
   for (auto& field : *fModel->GetRootField()) {
      field.ConnectColumns(fSource.get());
   }
   fNEntries = fSource->GetNEntries();
}

std::unique_ptr<ROOT::Experimental::RInputForest> ROOT::Experimental::RInputForest::Create(
   std::unique_ptr<RForestModel> model,
   std::string_view forestName,
   std::string_view storage)
{
   // TODO(jblomer): heuristics based on storage
   return std::make_unique<RInputForest>(
      std::move(model), std::make_unique<Detail::RPageSourceRoot>(forestName, storage));
}

ROOT::Experimental::RInputForest::RInputForest(std::unique_ptr<ROOT::Experimental::Detail::RPageSource> source)
   : ROOT::Experimental::Detail::RForest(RForestModel::Create())
   , fSource(std::move(source))
{
   fSource->Attach();
   fNEntries = fSource->GetNEntries();
}

ROOT::Experimental::RInputForest::~RInputForest()
{
}

//------------------------------------------------------------------------------

ROOT::Experimental::ROutputForest::ROutputForest(
   std::unique_ptr<ROOT::Experimental::RForestModel> model,
   std::unique_ptr<ROOT::Experimental::Detail::RPageSink> sink)
   : ROOT::Experimental::Detail::RForest(std::move(model))
   , fSink(std::move(sink))
   , fClusterSizeEntries(kDefaultClusterSizeEntries)
   , fLastCommitted(0)
{
   fSink->Create(fModel.get());
}

ROOT::Experimental::ROutputForest::~ROutputForest()
{
   CommitCluster();
   fSink->CommitDataset();
}


std::unique_ptr<ROOT::Experimental::ROutputForest> ROOT::Experimental::ROutputForest::Create(
   std::unique_ptr<RForestModel> model,
   std::string_view forestName,
   std::string_view storage)
{
   // TODO(jblomer): heuristics based on storage
   return std::make_unique<ROutputForest>(
      std::move(model), std::make_unique<Detail::RPageSinkRoot>(forestName, storage));
}


void ROOT::Experimental::ROutputForest::CommitCluster()
{
   if (fNEntries == fLastCommitted) return;
   for (auto& field : *fModel->GetRootField()) {
      field.Flush();
   }
   fSink->CommitCluster(fNEntries);
   fLastCommitted = fNEntries;
}


//------------------------------------------------------------------------------


ROOT::Experimental::RCollectionForest::RCollectionForest(std::unique_ptr<RForestEntry> defaultEntry)
   : fOffset(0), fDefaultEntry(std::move(defaultEntry))
{
}
