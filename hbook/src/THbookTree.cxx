// @(#)root/hbook:$Name:$:$Id:$
// Author: Rene Brun   18/02/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THbookTree                                                           //
//                                                                      //
// A wrapper class supporting Hbook ntuples (CWN and RWN).              //
// The normal TTree calls can be used, including TTree::Draw().         //
// Data read directly from the Hbook file via THbookFile.               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "THbookTree.h"
#include "THbookBranch.h"


ClassImp(THbookTree)

//______________________________________________________________________________
THbookTree::THbookTree(): TTree()
{
   fID   = 0;
   fType = 0;
   fX    = 0;
   fFile = 0;
}

//______________________________________________________________________________
THbookTree::THbookTree(const char *name,Int_t id)
    :TTree(name,name)
{
   fID   = id;
   fType = 0;
   fX    = 0;
   fFile = 0;
}


//______________________________________________________________________________
THbookTree::~THbookTree()
{
   if (fX) delete [] fX;
   if (fFile) fFile->DeleteID(fID);
}


//______________________________________________________________________________
Int_t THbookTree::GetEntry(Int_t entry, Int_t getall)
{
   return fFile->GetEntry(entry,fID,fType,GetX());
}


//______________________________________________________________________________
void THbookTree::Print(Option_t *option) const
{
   TTree::Print(option);
}

//______________________________________________________________________________
void THbookTree::SetEntries(Int_t n)
{
   fEntries = n;
   TIter next(GetListOfBranches());
   THbookBranch *branch;
   while ((branch=(THbookBranch*)next())) {
      branch->SetEntries(n);
   }
}
