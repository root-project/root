// @(#)root/hbook:$Name:  $:$Id: THbookTree.cxx,v 1.1 2002/02/18 18:02:57 rdm Exp $
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
#include "TTreeFormula.h"


ClassImp(THbookTree)

//______________________________________________________________________________
THbookTree::THbookTree(): TTree()
{
   fID   = 0;
   fType = 0;
   fX    = 0;
   fFile = 0;
   fInit = kFALSE;
}

//______________________________________________________________________________
THbookTree::THbookTree(const char *name,Int_t id)
    :TTree(name,name)
{
   fID   = id;
   fType = 0;
   fX    = 0;
   fFile = 0;
   fInit = kFALSE;
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
void THbookTree::InitBranches()
{
   Int_t nfill = GetPlayer()->GetNfill();
   if (nfill > 0) {fInit = kFALSE; return;}
   if (fInit) return;
   fInit = kTRUE;
   fFile->InitLeaves(fID, 5,GetPlayer()->GetMultiplicity());
   fFile->InitLeaves(fID, 0,GetPlayer()->GetSelect());
   fFile->InitLeaves(fID, 3,GetPlayer()->GetVar3());
   fFile->InitLeaves(fID, 2,GetPlayer()->GetVar2());
   fFile->InitLeaves(fID, 1,GetPlayer()->GetVar1());
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
