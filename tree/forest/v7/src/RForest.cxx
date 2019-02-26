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

#include "ROOT/RTreeModel.hxx"
#include "ROOT/RPageStorage.hxx"

#include <utility>

ROOT::Experimental::Detail::RForest::RForest(std::shared_ptr<ROOT::Experimental::RTreeModel> model)
   : fModel(model)
   , fNEntries(0)
{
}

ROOT::Experimental::Detail::RForest::~RForest()
{
}

//------------------------------------------------------------------------------

ROOT::Experimental::RInputForest::RInputForest(
   std::shared_ptr<ROOT::Experimental::RTreeModel> model,
   std::unique_ptr<ROOT::Experimental::Detail::RPageSource> source)
   : ROOT::Experimental::Detail::RForest(model)
   , fSource(std::move(source))
{
   fSource->Attach();
   for (auto& field : *model->GetRootField()) {
      field.ConnectColumns(fSource.get());
   }
   fNEntries = fSource->GetNEntries();
   fDefaultViewContext = std::unique_ptr<RTreeViewContext>(new RTreeViewContext(fSource.get()));
}

ROOT::Experimental::RInputForest::RInputForest(std::unique_ptr<ROOT::Experimental::Detail::RPageSource> source)
   : ROOT::Experimental::Detail::RForest(std::make_shared<ROOT::Experimental::RTreeModel>())
   , fSource(std::move(source))
{
   fSource->Attach();
   fNEntries = fSource->GetNEntries();
   fDefaultViewContext = std::unique_ptr<RTreeViewContext>(new RTreeViewContext(fSource.get()));
}

std::unique_ptr<ROOT::Experimental::RTreeViewContext> ROOT::Experimental::RInputForest::GetViewContext()
{
   auto ctx = new RTreeViewContext(fSource.get());
   return std::unique_ptr<RTreeViewContext>(ctx);
}

ROOT::Experimental::RInputForest::~RInputForest()
{
}

//------------------------------------------------------------------------------

ROOT::Experimental::ROutputForest::ROutputForest(
   std::shared_ptr<ROOT::Experimental::RTreeModel> model,
   std::unique_ptr<ROOT::Experimental::Detail::RPageSink> sink)
   : ROOT::Experimental::Detail::RForest(model)
   , fSink(std::move(sink))
   , fClusterSizeEntries(kDefaultClusterSizeEntries)
   , fLastCommitted(0)
{
   fSink->Create(model.get());
}

ROOT::Experimental::ROutputForest::~ROutputForest()
{
   CommitCluster();
   fSink->CommitDataset();
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
