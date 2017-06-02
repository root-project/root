// Author: Brian Bockelman, 2017-03-21

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <climits>

#include "TTreeReader.h"
#include "ROOT/TTreeReaderFast.hxx"

#include "TChain.h"
#include "TDirectory.h"
#include "ROOT/TTreeReaderValueFast.hxx"

using namespace ROOT::Experimental;

TTreeReaderFast::TTreeReaderFast(TTree* tree):
   fTree(tree)
{
   if (!fTree) {
      Error("TTreeReaderFast", "TTree is NULL!");
   } else {
      Initialize();
   }
}

TTreeReaderFast::TTreeReaderFast(const char* keyname, TDirectory* dir /*= NULL*/):
   fDirectory(dir)
{
   if (!fDirectory) fDirectory = gDirectory;
   fDirectory->GetObject(keyname, fTree);
   Initialize();
}

////////////////////////////////////////////////////////////////////////////////
/// Tell all value readers that the tree reader does not exist anymore.

TTreeReaderFast::~TTreeReaderFast()
{
   for (auto &reader : fValues) {
      reader->MarkTreeReaderUnavailable();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Initialization of the director.

void TTreeReaderFast::Initialize()
{
   if (!fTree) {
      MakeZombie();
      fEntryStatus = TTreeReader::kEntryNoTree;
   } else {
      fDirector = new ROOT::Internal::TBranchProxyDirector(fTree, -1);
   }

   bool IsOK = true;
   // Tell readers we now have a tree
   for (auto &reader : fValues) {
      reader->CreateProxy();
      if (reader->GetSetupStatus() != ROOT::Internal::TTreeReaderValueBase::kSetupMatch) {
         //printf("Reader setup failed.  Status: %d\n", reader->GetSetupStatus());
         IsOK = false;
      }
   }
   if (!IsOK) {
      //printf("Failed to initialize the reader.\n");
      fEntryStatus = TTreeReader::kEntryBadReader;
   }
}

TTreeReader::EEntryStatus
TTreeReaderFast::SetEntry(Long64_t entry)
{
   if (!fTree) {
      fEntryStatus =TTreeReader::kEntryNoTree;
      return fEntryStatus;
   }

   TTree* prevTree = fDirector->GetTree();

   Int_t treeNumInChainBeforeLoad = fTree->GetTreeNumber();

   TTree* treeToCallLoadOn = fTree->GetTree();
   Long64_t loadResult = treeToCallLoadOn->LoadTree(entry);

   if (loadResult == -2) {
      fEntryStatus = TTreeReader::kEntryNotFound;
      return fEntryStatus;
   }

   if (treeNumInChainBeforeLoad != fTree->GetTreeNumber()) {
      fDirector->SetTree(fTree->GetTree());
   }

   if (!prevTree || fDirector->GetReadEntry() == -1)
   {
      bool IsOK = true;
      // Tell readers we now have a tree
      for (auto &reader : fValues) {
         reader->CreateProxy();
         if (reader->GetSetupStatus() != ROOT::Internal::TTreeReaderValueBase::kSetupMatch) IsOK = false;
      }
      fEntryStatus = IsOK ? TTreeReader::kEntryValid : TTreeReader::kEntryBadReader;
   }

   return fEntryStatus;
}


////////////////////////////////////////////////////////////////////////////////
/// Add a value reader for this tree.

void TTreeReaderFast::RegisterValueReader(ROOT::Experimental::Internal::TTreeReaderValueFastBase* reader)
{
   fValues.push_back(reader);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove a value reader for this tree.

void TTreeReaderFast::DeregisterValueReader(ROOT::Experimental::Internal::TTreeReaderValueFastBase* reader)
{
   auto iReader = std::find(fValues.begin(), fValues.end(), reader);
   if (iReader == fValues.end()) {
      Error("DeregisterValueReader", "Cannot find reader of type %s for branch %s", reader->GetTypeName(), reader->fBranchName.c_str());
      return;
   }
   fValues.erase(iReader);
}

////////////////////////////////////////////////////////////////////////////////
/// Advance to the next range in the file; returns the number of events in the range.
///
/// Returned number is the number of events we can process before one of the Value
/// objects will hit the end of its buffer.

Int_t
TTreeReaderFast::GetNextRange(Int_t eventNum)
{
   Int_t remaining = INT_MAX;
   for (auto &value : fValues) {
      Int_t valueRemaining = value->GetEvents(eventNum);
      if (valueRemaining < remaining) {
          remaining = valueRemaining;
      }
   }
   //printf("TTreeReaderFast::GetNextRange: Starting at %d, remaining events %d.\n", eventNum, remaining);
   return remaining;
}

