// @(#)root/hbook:$Id$
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
// IMPORTANT NOTE                                                       //
// When setting the branch address (via THbookTree::SetBranchAddress)   //
// for a branch in an Hbook block containing several names, eg          //
//    Hbook block SELEVN with the following variables:                  //
// ******************************************************************   //
//  *      1   * R*4  *         *              * SELEVN   * WGGS        //
//  *      2   * R*4  *         *              * SELEVN   * AM12        //
//  *      3   * R*4  *         *              * SELEVN   * AM34        //
//  *      4   * R*4  *         *              * SELEVN   * AM14        //
//  *      5   * R*4  *         *              * SELEVN   * AM32        //
//  *      6   * R*4  *         *              * SELEVN   * PtPI(4)     //
//  *      7   * R*4  *         *              * SELEVN   * PHIPI(4)    //
//  *      8   * R*4  *         *              * SELEVN   * THTPI(4)    //
// one must define a C struct like:                                     //
//   struct {                                                           //
//      Float_t Wggs;                                                   //
//      Float_t Am12;                                                   //
//      Float_t Am34;                                                   //
//      Float_t Am14;                                                   //
//      Float_t Am32;                                                   //
//      Float_t Ptpi[4];                                                //
//      Float_t Phipi[4];                                               //
//      Float_t Thtpi[4];                                               //
//   } event;                                                           //
//                                                                      //
// and set ONLY the first variable address with:                        //
//    h96->SetBranchAddress("Wggs",&event.Wggs);                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "THbookTree.h"
#include "THbookBranch.h"
#include "TTreeFormula.h"


ClassImp(THbookTree)

//______________________________________________________________________________
THbookTree::THbookTree(): TTree()
{
   //default constructor
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
   //constructor
   fID   = id;
   fType = 0;
   fX    = 0;
   fFile = 0;
   fInit = kFALSE;
}


//______________________________________________________________________________
THbookTree::~THbookTree()
{
   //destructor
   if (fX) delete [] fX;
   if (fFile) fFile->DeleteID(fID);
}


//______________________________________________________________________________
Int_t THbookTree::GetEntry(Long64_t entry, Int_t /*getall*/)
{
   //get one entry from the hbook ntuple
   fReadEntry = entry;
   return fFile->GetEntry(entry,fID,fType,GetX());
}


//______________________________________________________________________________
void THbookTree::InitBranches(Long64_t entry)
{
   //Initialize the branch addresses
   Int_t nfill = GetPlayer()->GetNfill();
   if (nfill > 0) {fInit = kFALSE; return;}
   if (fInit) return;
   fInit = kTRUE;
   if (!GetPlayer()->GetVar1()) {
      GetEntry(entry);
      return;
   }
   //fFile->InitLeaves(fID, 5,GetPlayer()->GetMultiplicity());
   fFile->InitLeaves(fID, 0,GetPlayer()->GetSelect());
   fFile->InitLeaves(fID, 3,GetPlayer()->GetVar3());
   fFile->InitLeaves(fID, 2,GetPlayer()->GetVar2());
   fFile->InitLeaves(fID, 1,GetPlayer()->GetVar1());
}

//______________________________________________________________________________
void THbookTree::Print(Option_t *option) const
{
   //Print an overview of the hbook ntuple
   TTree::Print(option);
}

//______________________________________________________________________________
Long64_t THbookTree::SetEntries(Long64_t n)
{
   //Set the number of entries in the tree header and its branches
   fEntries = n;
   TIter next(GetListOfBranches());
   THbookBranch *branch;
   while ((branch=(THbookBranch*)next())) {
      branch->SetEntries(n);
   }
   return n;
}
