// @(#)root/tree:$Name:  $:$Id: TBranchRef.cxx,v 1.3 2004/08/22 01:51:22 rdm Exp $
// Author: Rene Brun   19/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// A Branch for the case of an array of clone objects                   //                                                                      //
// See TTree.                                                           //                                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBranchRef.h"
#include "TTree.h"
#include "TBasket.h"
#include "TFile.h"

ClassImp(TBranchRef)

//______________________________________________________________________________
TBranchRef::TBranchRef(): TBranch()
{

   fRefTable   = 0;
}


//______________________________________________________________________________
TBranchRef::TBranchRef(TTree *tree)
    :TBranch()
{
   // main constructor called by TTree::BranchRef

   if (!tree) return;
   SetName("TRefTable");
   SetTitle("List of branch numbers with referenced objects");
   fRefTable = new TRefTable(this,100);

   fCompress       = 1;
   fBasketSize     = 32000;
   fAddress        = 0;
   fBasketBytes    = new Int_t[fMaxBaskets];
   fBasketEntry    = new Long64_t[fMaxBaskets];
   fBasketSeek     = new Long64_t[fMaxBaskets];

   for (Int_t i=0;i<fMaxBaskets;i++) {
      fBasketBytes[i] = 0;
      fBasketEntry[i] = 0;
      fBasketSeek[i]  = 0;
   }

   fTree       = tree;
   fDirectory  = fTree->GetDirectory();
   fFileName   = "";

  //  Create the first basket
   TBasket *basket = new TBasket("TRefTable",fTree->GetName(),this);
   fBaskets.Add(basket);
}


//______________________________________________________________________________
TBranchRef::~TBranchRef()
{
   delete fRefTable;
}

//______________________________________________________________________________
void TBranchRef::Clear(Option_t *option)
{
  // clear entries in the TRefTable

   fRefTable->Clear(option);
}

//______________________________________________________________________________
Int_t TBranchRef::Fill()
{
  // fill the branch basket with the referenced objects parent numbers

   Int_t nbytes = TBranch::Fill();
   return nbytes;
}

//______________________________________________________________________________
void TBranchRef::FillLeaves(TBuffer &b)
{
   // This function called by TBranch::Fill overloads TBranch::FillLeaves

    fRefTable->FillBuffer(b);
}

//______________________________________________________________________________
Bool_t TBranchRef::Notify()
{
   // This function is called by TRefTable::Notify, itself called by
   // TRef::GetObject.
   // The function reads the branch containing the object referenced
   // by the TRef.

   UInt_t uid = fRefTable->GetUID();
   GetEntry(fReadEntry);
   TBranch *branch = (TBranch*)fRefTable->GetParent(uid);
   if (branch) branch->GetEntry(fReadEntry);
   return kTRUE;
}

//______________________________________________________________________________
void TBranchRef::Print(Option_t *option) const
{
  // Print the TRefTable branch

   TBranch::Print(option);
}

//______________________________________________________________________________
void TBranchRef::ReadLeaves(TBuffer &b)
{
   // This function called by TBranch::GetEntry overloads TBranch::ReadLeaves

  fRefTable->ReadBuffer(b);
}

//______________________________________________________________________________
void TBranchRef::Reset(Option_t *option)
{
  //    Existing buffers are deleted
  //    Entries, max and min are reset
  //    TRefTable is cleared

   TBranch::Reset(option);
   fRefTable->Clear();
}

//______________________________________________________________________________
void TBranchRef::SetParent(const TObject *object)
{
   // this function is called by TBranchElement::Fill when filling
   // branches that may contain referenced objects

   TRefTable::SetRefTable(fRefTable);
   fRefTable->SetParent(object);
}
