// @(#)root/hbook:$Name:$:$Id:$
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

ClassImp(THbookBranch)


//______________________________________________________________________________
THbookBranch::THbookBranch(const char *name, void *address, const char *leaflist, Int_t basketsize, Int_t compress)
            :TBranch(name,address,leaflist,basketsize,compress)
{
}


//______________________________________________________________________________
THbookBranch::~THbookBranch()
{
}


//______________________________________________________________________________
Int_t THbookBranch::GetEntry(Int_t entry, Int_t getall)
{
   THbookTree *tree = (THbookTree*)GetTree();
   THbookFile *file = tree->GetHbookFile();
   if (tree->GetType() == 0) {
      return file->GetEntry(entry,tree->GetID(),0,tree->GetX());
   } else {
      return file->GetEntryBranch(entry,tree->GetID(),fBlockName.Data(),GetName());
   }
}
