/// \file RNTuple.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RNTuple.hxx"

#include "ROOT/RNTupleModel.hxx"
#include "ROOT/RPageStorage.hxx"
#include "ROOT/RPageStorageRoot.hxx"

#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

ROOT::Experimental::Detail::RForest::RForest(std::unique_ptr<ROOT::Experimental::RNTupleModel> model)
   : fModel(std::move(model))
   , fNEntries(0)
{
}

ROOT::Experimental::Detail::RForest::~RForest()
{
}

//------------------------------------------------------------------------------

ROOT::Experimental::RInputForest::RInputForest(
   std::unique_ptr<ROOT::Experimental::RNTupleModel> model,
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

ROOT::Experimental::RInputForest::RInputForest(std::unique_ptr<ROOT::Experimental::Detail::RPageSource> source)
   : ROOT::Experimental::Detail::RForest(nullptr)
   , fSource(std::move(source))
{
   fSource->Attach();
   fModel = fSource->GenerateModel();
   for (auto& field : *fModel->GetRootField()) {
      field.ConnectColumns(fSource.get());
   }
   fNEntries = fSource->GetNEntries();
}

ROOT::Experimental::RInputForest::~RInputForest()
{
}

std::unique_ptr<ROOT::Experimental::RInputForest> ROOT::Experimental::RInputForest::Open(
   std::unique_ptr<RNTupleModel> model,
   std::string_view forestName,
   std::string_view storage)
{
   // TODO(jblomer): heuristics based on storage
   return std::make_unique<RInputForest>(
      std::move(model), std::make_unique<Detail::RPageSourceRoot>(forestName, storage));
}

std::unique_ptr<ROOT::Experimental::RInputForest> ROOT::Experimental::RInputForest::Open(
   std::string_view forestName,
   std::string_view storage)
{
   return std::make_unique<RInputForest>(std::make_unique<Detail::RPageSourceRoot>(forestName, storage));
}

std::string ROOT::Experimental::RInputForest::GetInfo(const EForestInfo what) {
   std::ostringstream os;
   auto name = fSource->GetDescriptor().GetName();

   switch (what) {
   case EForestInfo::kSummary:
      os << "****************************** FOREST ******************************"  << std::endl
         << "* Name:    " << name << std::setw(57 - name.length())           << "*" << std::endl
         << "* Entries: " << std::setw(10) << fNEntries << std::setw(47)     << "*" << std::endl
         << "********************************************************************"  << std::endl;
      return os.str();
   default:
      // Unhandled case, internal error
      assert(false);
   }
   // Never here
   return "";
}

//------------------------------------------------------------------------------

ROOT::Experimental::ROutputForest::ROutputForest(
   std::unique_ptr<ROOT::Experimental::RNTupleModel> model,
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


std::unique_ptr<ROOT::Experimental::ROutputForest> ROOT::Experimental::ROutputForest::Recreate(
   std::unique_ptr<RNTupleModel> model,
   std::string_view forestName,
   std::string_view storage)
{
   // TODO(jblomer): heuristics based on storage
   TFile *file = TFile::Open(std::string(storage).c_str(), "RECREATE");
   Detail::RPageSinkRoot::RSettings settings;
   settings.fFile = file;
   settings.fTakeOwnership = true;
   return std::make_unique<ROutputForest>(
      std::move(model), std::make_unique<Detail::RPageSinkRoot>(forestName, settings));
}


void ROOT::Experimental::ROutputForest::CommitCluster()
{
   if (fNEntries == fLastCommitted) return;
   for (auto& field : *fModel->GetRootField()) {
      field.Flush();
      field.CommitCluster();
   }
   fSink->CommitCluster(fNEntries);
   fLastCommitted = fNEntries;
}


//------------------------------------------------------------------------------


ROOT::Experimental::RCollectionForest::RCollectionForest(std::unique_ptr<REntry> defaultEntry)
   : fOffset(0), fDefaultEntry(std::move(defaultEntry))
{
}
