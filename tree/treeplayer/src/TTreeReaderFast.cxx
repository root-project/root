// Author: Brian Bockelman, 2017-03-21

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTreeReader.h"
#include "TTreeReaderFast.h"

#include "TChain.h"
#include "TDirectory.h"
#include "TTreeReaderValueFast.h"

ClassImp(TTreeReaderFast)

TTreeReaderFast::TTreeReaderFast(TTree* tree):
   fTree(tree),
   fEntryStatus(TTreeReader::kEntryNotLoaded)
{
   if (!fTree) {
      Error("TTreeReader", "TTree is NULL!");
   } else {
      Initialize();
   }
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

   // Tell readers we now have a tree
   for (auto &reader : fValues) {
         reader->CreateProxy();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Remove a value reader for this tree.

void TTreeReaderFast::DeregisterValueReader(ROOT::Internal::TTreeReaderValueFastBase* reader)
{
   auto iReader = std::find(fValues.begin(), fValues.end(), reader);
   if (iReader == fValues.end()) {
      Error("DeregisterValueReader", "Cannot find reader of type %s for branch %s", reader->GetTypeName(), reader->fBranchName.c_str());
      return;
   }
   fValues.erase(iReader);
}
