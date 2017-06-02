// @(#)root/treeplayer:$Id$
// Author: Axel Naumann, 2011-09-28

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTreeReaderValueFast.h"

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

ClassImp(ROOT::Internal::TTreeReaderValueFastBase)

////////////////////////////////////////////////////////////////////////////////
/// Construct a tree value reader and register it with the reader object.

ROOT::Internal::TTreeReaderValueFastBase::TTreeReaderValueFastBase(TTreeReader* reader /*= 0*/,
                                                 const char* branchname /*= 0*/,
                                                 TDictionary* dict /*= 0*/):
   fBranchName(branchname),
   fTreeReader(reader),
   fDict(dict),
   fProxy(NULL),
   fLeaf(NULL),
   fTreeLastOffset(-1),
   fSetupStatus(kSetupNotSetup),
   fReadStatus(kReadNothingYet)
{
   if (fTreeReader) fTreeReader->RegisterValueReader(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Unregister from tree reader, cleanup.

ROOT::Internal::TTreeReaderValueFastBase::~TTreeReaderValueFastBase()
{
   if (fTreeReader) fTreeReader->DeregisterValueReader(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Try to read the value from the TBranchProxy, returns
/// the status of the read.

ROOT::Internal::TTreeReaderValueFastBase::EReadStatus
ROOT::Internal::TTreeReaderValueFastBase::ProxyRead() {
   if (!fProxy) return kReadNothingYet;
   if (fProxy->Read()) {
      fReadStatus = kReadSuccess;
   } else {
      fReadStatus = kReadError;
   }
   return fReadStatus;
}

////////////////////////////////////////////////////////////////////////////////
/// If we are reading a leaf, return the corresponding TLeaf.

TLeaf* TTreeReaderValueFastBase::GetLeaf() {
   if (fLeafName.Length() > 0){

      Long64_t newChainOffset = fTreeReader->GetTree()->GetChainOffset();

      if (newChainOffset != fTreeLastOffset){
         fTreeLastOffset = newChainOffset;

         TTree *myTree = fTreeReader->GetTree();

         if (!myTree) {
            fReadStatus = kReadError;
            Error("TTreeReaderValueBase::GetLeaf()", "Unable to get the tree from the TTreeReader");
            return 0;
         }

         TBranch *myBranch = myTree->GetBranch(fBranchName);

         if (!myBranch) {
            fReadStatus = kReadError;
            Error("TTreeReaderValueBase::GetLeaf()", "Unable to get the branch from the tree");
            return 0;
         }

         fLeaf = myBranch->GetLeaf(fLeafName);
         if (!fLeaf) {
            Error("TTreeReaderValueBase::GetLeaf()", "Failed to get the leaf from the branch");
         }
      }
      return fLeaf;
   }
   else {
      Error("TTreeReaderValueBase::GetLeaf()", "We are not reading a leaf");
      return 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Create the proxy object for our branch.

void ROOT::Internal::TTreeReaderValueFastBase::CreateProxy() {
   TBranch* br = fTreeReader->GetTree()->GetBranch(fBranchName);


   TNamedBranchProxy* namedProxy
      = (TNamedBranchProxy*)fTreeReader->FindObject(fBranchName);
   if (namedProxy && namedProxy->GetDict() == fDict) {
      fProxy = namedProxy->GetProxy();
      return;
   }

   TBranch* branch = fTreeReader->GetTree()->GetBranch(fBranchName);
   TLeaf *myLeaf = NULL;

   const char* branchActualTypeName = GetBranchDataType(branch, branchActualType, nullptr);

   if (!branchActualType) {
      Error("TTreeReaderValueFastBase::CreateProxy()", "The branch %s contains data of type %s, which does not have a dictionary.",
                                                   fBranchName.Data(), branchActualTypeName ? branchActualTypeName : "{UNDETERMINED TYPE}");
      return;
   }

   if (!strcmp(GetTypeName(), branchActualTypeName)) {
         TDataType *dictdt = dynamic_cast<TDataType*>(fDict);
         TDataType *actualdt = dynamic_cast<TDataType*>(branchActualType);
         if (dictdt && actualdt && dictdt->GetType()>0
             && dictdt->GetType() == actualdt->GetType()) {
            // Same numerical type but different TDataType, likely Long64_t
         } else {
            Error("TTreeReaderValueFastBase::CreateProxy()",
                  "The branch %s contains data of type %s. It cannot be accessed by a TTreeReaderValueFast<%s>",
                  fBranchName.Data(), branchActualType->GetName(),
                  fDict->GetName());
            return;
         }
   }


   fBranch = branch;
}

