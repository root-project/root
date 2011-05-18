// @(#)root/tree:$Id$
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
// A branch containing and managing a TRefTable for TRef autoloading.   //
// It loads the TBranch containing a referenced object when requested   //
// by TRef::GetObject(), so the reference can be resolved. The          //
// information which branch to load is stored by TRefTable. Once a      //
// TBranch has read the TBranchRef's current entry it will not be told  //
// to re-read, in case the use has changed objects read from the        //
// branch.                                                              //
//                                                                      //
//                                                                      //
// *** LIMITATION ***                                                   //
// Note that this does NOT allow for autoloading of references spanning //
// different entries. The TBranchRef's current entry has to correspond  //
// to the entry of the TBranch containing the referenced object.        //
//                                                                      //
// The TRef cannot be stored in a top-level branch which is a           //
// TBranchObject for the auto-loading to work. E.g. you cannot store    //
// the TRefs in TObjArray, and create a top-level branch storing this   //
// TObjArray.                                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBranchRef.h"
#include "TTree.h"
#include "TBasket.h"
#include "TFile.h"
#include "TFriendElement.h"

ClassImp(TBranchRef)

//______________________________________________________________________________
TBranchRef::TBranchRef(): TBranch()
{
   // Default constructor.

   fRefTable   = 0;
   fReadLeaves = (ReadLeaves_t)&TBranchRef::ReadLeavesImpl;
}


//______________________________________________________________________________
TBranchRef::TBranchRef(TTree *tree)
    :TBranch()
{
   // Main constructor called by TTree::BranchRef.

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
   fMother     = this;
   fDirectory  = fTree->GetDirectory();
   fFileName   = "";
   fReadLeaves = (ReadLeaves_t)&TBranchRef::ReadLeavesImpl;
}


//______________________________________________________________________________
TBranchRef::~TBranchRef()
{
   // Typical destructor.

   delete fRefTable;
}

//______________________________________________________________________________
void TBranchRef::Clear(Option_t *option)
{
  // Clear entries in the TRefTable.

   if (fRefTable) fRefTable->Clear(option);
}

//______________________________________________________________________________
Int_t TBranchRef::Fill()
{
  // Fill the branch basket with the referenced objects parent numbers.

   Int_t nbytes = TBranch::Fill();
   return nbytes;
}

//______________________________________________________________________________
void TBranchRef::FillLeaves(TBuffer &b)
{
   // This function called by TBranch::Fill overloads TBranch::FillLeaves.

   if (!fRefTable) fRefTable = new TRefTable(this,100);
   fRefTable->FillBuffer(b);
}

//______________________________________________________________________________
Bool_t TBranchRef::Notify()
{
   // This function is called by TRefTable::Notify, itself called by
   // TRef::GetObject.
   // The function reads the branch containing the object referenced
   // by the TRef.

   if (!fRefTable) fRefTable = new TRefTable(this,100);
   UInt_t uid = fRefTable->GetUID();
   TProcessID* context = fRefTable->GetUIDContext();
   GetEntry(fReadEntry);
   TBranch *branch = (TBranch*)fRefTable->GetParent(uid, context);
   if (branch) {
      // don't re-read, the user might have changed some object
      if (branch->GetReadEntry() != fReadEntry)
         branch->GetEntry(fReadEntry);
   } else {
      //scan the TRefTable of possible friend Trees
      TList *friends = fTree->GetListOfFriends();
      if (!friends) return kTRUE;
      TObjLink *lnk = friends->FirstLink();
      while (lnk) {
         TFriendElement* elem = (TFriendElement*)lnk->GetObject();
         TTree *tree = elem->GetTree();
         TBranchRef *bref = tree->GetBranchRef();
         if (bref) {
            bref->GetEntry(fReadEntry);
            branch = (TBranch*)bref->GetRefTable()->GetParent(uid, context);
            if (branch) {
               // don't re-read, the user might have changed some object
               if (branch->GetReadEntry() != fReadEntry)
                  branch->GetEntry(fReadEntry);
               return kTRUE;
            }
         }
         lnk = lnk->Next();
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
void TBranchRef::Print(Option_t *option) const
{
  // Print the TRefTable branch.

   TBranch::Print(option);
}

//______________________________________________________________________________
void TBranchRef::ReadLeavesImpl(TBuffer &b)
{
   // This function called by TBranch::GetEntry overloads TBranch::ReadLeaves.

   if (!fRefTable) fRefTable = new TRefTable(this,100);
   fRefTable->ReadBuffer(b);
}

//______________________________________________________________________________
void TBranchRef::Reset(Option_t *option)
{
  //    Existing buffers are deleted
  //    Entries, max and min are reset
  //    TRefTable is cleared.

   TBranch::Reset(option);
   if (!fRefTable) fRefTable = new TRefTable(this,100);
   fRefTable->Reset();
}

//______________________________________________________________________________
Int_t TBranchRef::SetParent(const TObject* object, Int_t branchID)
{
   // -- Set the current parent branch.
   //
   // This function is called by TBranchElement::GetEntry()
   // and TBranchElement::Fill() when reading or writing
   // branches that may contain referenced objects.
   //
   if (!fRefTable) {
      fRefTable = new TRefTable(this, 100);
   }
   TRefTable::SetRefTable(fRefTable);
   return fRefTable->SetParent(object, branchID);
}

