// @(#)root/treeplayer:$Id$
// Author: Axel Naumann, 2011-09-28

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TTreeReaderValueFast.hxx"

#include "TTreeReader.h"
#include "TBranchClones.h"
#include "TBranchElement.h"
#include "TBranchRef.h"
#include "TBranchSTL.h"
#include "TBranchProxyDirector.h"
#include "TLeaf.h"
#include "TTreeProxyGenerator.h"
#include "TTreeReaderValue.h"
#include "TRegexp.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TNtuple.h"
#include <vector>

/** \class TTreeReaderValueFast

Extracts data from a TTree.
*/

////////////////////////////////////////////////////////////////////////////////
/// Unregister from tree reader, cleanup.

ROOT::Experimental::Internal::TTreeReaderValueFastBase::~TTreeReaderValueFastBase()
{
   if (fTreeReader) fTreeReader->DeregisterValueReader(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Attach this value to the appropriate branch on the tree.  For now, we don't
/// support the complex branch lookup of the TTreeReader -- only a fixed leaf!

void ROOT::Experimental::Internal::TTreeReaderValueFastBase::CreateProxy() {
   fReadStatus = ROOT::Internal::TTreeReaderValueBase::kReadError;
   fSetupStatus = ROOT::Internal::TTreeReaderValueBase::kSetupMissingBranch;
   if (fLeafName.size() > 0){

      Long64_t newChainOffset = fTreeReader->GetTree()->GetChainOffset();

      if (newChainOffset != fLastChainOffset){
         fLastChainOffset = newChainOffset;

         TTree *myTree = fTreeReader->GetTree();

         if (!myTree) {
            Error("TTreeReaderValueBase::GetLeaf()", "Unable to get the tree from the TTreeReader");
            return;
         }

         TBranch *myBranch = myTree->GetBranch(fBranchName.c_str());

         if (!myBranch) {
            Error("TTreeReaderValueBase::GetLeaf()", "Unable to get the branch from the tree");
            return;
         }

         fLeaf = myBranch->GetLeaf(fLeafName.c_str());
         if (!fLeaf) {
            Error("TTreeReaderValueBase::GetLeaf()", "Failed to get the leaf from the branch");
         }
         fBranch = myBranch;
      }
   }
   else {
      Error("TTreeReaderValueBase::GetLeaf()", "We are not reading a leaf");
   }
   fReadStatus = ROOT::Internal::TTreeReaderValueBase::kReadSuccess;
   fSetupStatus = ROOT::Internal::TTreeReaderValueBase::kSetupMatch;
}

