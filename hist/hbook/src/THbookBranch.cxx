// @(#)root/hbook:$Id$
// Author: Rene Brun   18/02/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "THbookBranch.h"
#include "THbookTree.h"

ClassImp(THbookBranch);

////////////////////////////////////////////////////////////////////////////////
/** \class THbookBranch
    \ingroup Hist
    \brief HBOOK Branch
*/

////////////////////////////////////////////////////////////////////////////////

THbookBranch::THbookBranch(TTree *tree, const char *name, void *address, const char *leaflist, Int_t basketsize, Int_t compress)
            :TBranch(tree, name,address,leaflist,basketsize,compress)
{
}

////////////////////////////////////////////////////////////////////////////////

THbookBranch::THbookBranch(TBranch *branch, const char *name, void *address, const char *leaflist, Int_t basketsize, Int_t compress)
            :TBranch(branch,name,address,leaflist,basketsize,compress)
{
}

////////////////////////////////////////////////////////////////////////////////

THbookBranch::~THbookBranch()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Browser interface.

void THbookBranch::Browse(TBrowser *b)
{
   THbookTree *tree = (THbookTree*)GetTree();
   THbookFile *file = tree->GetHbookFile();
   file->cd();

   TBranch::Browse(b);
}

////////////////////////////////////////////////////////////////////////////////
///get one entry from hbook ntuple

Int_t THbookBranch::GetEntry(Long64_t entry, Int_t /*getall*/)
{
   THbookTree *tree = (THbookTree*)GetTree();
   THbookFile *file = tree->GetHbookFile();
   if (tree->GetType() == 0) {
      return file->GetEntry(entry,tree->GetID(),0,tree->GetX());
   } else {
      tree->InitBranches(entry);
      return file->GetEntryBranch(entry,tree->GetID());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set address of this branch
/// See important remark in the header of THbookTree

void THbookBranch::SetAddress(void *add)
{
   TBranch::SetAddress(add);

   if (GetUniqueID() != 0) return; //only for first variable of the block
   THbookTree *tree = (THbookTree*)GetTree();
   THbookFile *file = tree->GetHbookFile();
   if (tree->GetType() != 0) {
      file->SetBranchAddress(tree->GetID(),GetBlockName(),add);
   }
}
