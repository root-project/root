// @(#)root/tree:$Name:  $:$Id: TBranchElement.cxx,v 1.2 2001/01/16 16:24:39 brun Exp $
// Author: Rene Brun   14/01/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBranchElement                                                       //
//                                                                      //
// A Branch for the case of an object                                   //                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TFile.h"
#include "TBranchElement.h"
#include "TBranchClones.h"
#include "TTree.h"
#include "TBasket.h"
#include "TLeafElement.h"
#include "TVirtualPad.h"
#include "TClass.h"
#include "TRealData.h"
#include "TDataType.h"
#include "TDataMember.h"
#include "TStreamerInfo.h"
#include "TBrowser.h"

R__EXTERN  TTree *gTree;

ClassImp(TBranchElement)

//______________________________________________________________________________
TBranchElement::TBranchElement(): TBranch()
{
//*-*-*-*-*-*Default constructor for BranchElement*-*-*-*-*-*-*-*-*-*
//*-*        ====================================

   fNleaves   = 1;
   fInfo = 0;
}


//______________________________________________________________________________
TBranchElement::TBranchElement(const char *name, TStreamerInfo *sinfo, Int_t id, void *addobj, Int_t basketsize, Int_t splitlevel, Int_t compress)
    :TBranch()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a BranchElement*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =====================
//
   TClass *cl    = sinfo->GetClass();
   fInfo         = sinfo;
   fID           = id;
   fType         = -1;
   fClassVersion = cl->GetClassVersion();
   if (id >= 0) {
     Int_t *types = sinfo->GetTypes();
     fType = types[fID];
  }
           
   SetName(name);
   SetTitle(name);
   fCompress = compress;
   if (compress == -1) {
      TFile *bfile = gTree->GetDirectory()->GetFile();
      if (bfile) fCompress = bfile->GetCompressionLevel();
   }
   if (basketsize < 100) basketsize = 100;
   fBasketSize     = basketsize;
   fAddress        = (char*)addobj;
   fClassName      = cl->GetName();
   fBasketEntry    = new Int_t[fMaxBaskets];
   fBasketBytes    = new Int_t[fMaxBaskets];
   fBasketSeek     = new Seek_t[fMaxBaskets];

   fBasketEntry[0] = fEntryNumber;
   fBasketBytes[0] = 0;

   TLeaf *leaf     = new TLeafElement(name,fID, fType);
   leaf->SetBranch(this);
   leaf->SetAddress(addobj);
   fNleaves = 1;
   fLeaves.Add(leaf);
   gTree->GetListOfLeaves()->Add(leaf);

// Set the bit kAutoDelete to specify that when reading
// in TLeafElement::ReadBasket, the object should be deleted
// before calling Streamer.
// It is foreseen to not set this bit in a future version.
   SetAutoDelete(kTRUE);

   fTree       = gTree;
   fDirectory  = fTree->GetDirectory();
   fFileName   = "";

//*-*-  Create the first basket
   if (splitlevel) return;
   TBasket *basket = new TBasket(name,fTree->GetName(),this);
   fBaskets.Add(basket);
}


//______________________________________________________________________________
TBranchElement::~TBranchElement()
{
//*-*-*-*-*-*Default destructor for a BranchElement*-*-*-*-*-*-*-*-*-*-*-*
//*-*        =====================================

   fBranches.Delete();
}


//______________________________________________________________________________
void TBranchElement::Browse(TBrowser *b)
{

   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches > 1) {
      fBranches.Browse( b );
   }
}

//______________________________________________________________________________
Int_t TBranchElement::Fill()
{
//*-*-*-*-*-*-*-*Loop on all leaves of this branch to fill Basket buffer*-*-*
//*-*            =======================================================

   Int_t nbytes = 0;
   Int_t i;
   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches) {
      fEntries++;
      UpdateAddress();
      for (i=0;i<nbranches;i++)  {
         TBranch *branch = (TBranch*)fBranches[i];
         if (!branch->TestBit(kDoNotProcess)) nbytes += branch->Fill();
      }
   } else {
      if (!TestBit(kDoNotProcess)) nbytes += TBranch::Fill();
   }
   return nbytes;
}

//______________________________________________________________________________
Int_t TBranchElement::GetEntry(Int_t entry, Int_t getall)
{
//*-*-*-*-*Read all branches of a BranchElement and return total number of bytes
//*-*      ====================================================================
//   If entry = 0 take current entry number + 1
//   If entry < 0 reset entry number to 0

   if (TestBit(kDoNotProcess) && !getall) return 0;
   Int_t nbytes;
   Int_t nbranches = fBranches.GetEntriesFast();

   if (nbranches) {
      if (fAddress == 0) { // try to create object
         if (!TestBit(kWarn)) {
            TClass *cl = gROOT->GetClass(fClassName);
            if (cl) {
               TObject** voidobj = (TObject**) new Long_t[1];
               *voidobj = (TObject*)cl->New();
               SetAddress(voidobj);
            } else {
               Warning("GetEntry","Cannot get class: %s",fClassName.Data());
               SetBit(kWarn);
            }
         }
      }
      nbytes = 0;
      for (Int_t i=0;i<nbranches;i++)  {
         TBranch *branch = (TBranch*)fBranches[i];
            nbytes += branch->GetEntry(entry);
      }
   } else {
      nbytes = TBranch::GetEntry(entry);
   }
   return nbytes;
}


//______________________________________________________________________________
Bool_t TBranchElement::IsFolder() const
{
//*-*-*-*-*Return TRUE if more than one leaf, FALSE otherwise*-*
//*-*      ==================================================

   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches >= 1) return kTRUE;
   else                return kFALSE;
}

//______________________________________________________________________________
void TBranchElement::Print(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*-*Print TBranch parameters*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================

   Int_t i;
   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches) {
      Printf("*Branch  :%-9s : %-54s *",GetName(),GetTitle());
      Printf("*Entries : %8d : BranchElement (see below)                               *",Int_t(fEntries));
      Printf("*............................................................................*");
      for (i=0;i<nbranches;i++)  {
         TBranch *branch = (TBranch*)fBranches.At(i);
         branch->Print(option);
      }
   } else {
      TBranch::Print(option);
   }
}

//______________________________________________________________________________
void TBranchElement::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*Reset a Branch*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ====================
//
//    Existing buffers are deleted
//    Entries, max and min are reset
//

   TBranch::Reset(option);

   Int_t i;
   Int_t nbranches = fBranches.GetEntriesFast();
   for (i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches[i];
      branch->Reset(option);
   }
}

//______________________________________________________________________________
void TBranchElement::SetAddress(void *add)
{
//*-*-*-*-*-*-*-*Set address of this branch*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ====================
//

   //special case when called from code generated by TTree::MakeClass
   if (Long_t(add) == -1) {
      SetBit(kWarn);
      return;
   }
   fReadEntry = -1;
   
   fAddress = (char*)add;
   void **ppointer = (void**)add;
   TObject *obj = (TObject*)(*ppointer);
   TClass *cl = gROOT->GetClass(fClassName.Data());
   if (!obj && cl) {
      obj = (TObject*)cl->New();
      *ppointer = (void*)obj;
   }
   Int_t *leafOffsets;
   if (!fInfo) {
      TStreamerInfo::Optimize(kFALSE);
      cl->BuildRealData((void*)obj);
      fInfo = cl->GetStreamerInfo(fClassVersion);
      leafOffsets = fInfo->GetOffsets();
      if (!leafOffsets || fInfo->IsOptimized()) fInfo->BuildOld();
   }
      
   TLeaf *leaf = (TLeaf*)fLeaves.UncheckedAt(0);
   if (leaf) {
      if (fID >= 0) {
         leafOffsets = fInfo->GetOffsets();
         char *absaddr = (char*)obj + leafOffsets[fID];
         leaf->SetAddress((void*)absaddr);
      }
   }

   Int_t i;
   Int_t nbranches = fBranches.GetEntriesFast();
   for (i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches[i];
      branch->SetAddress(add);
   }
}

//______________________________________________________________________________
void TBranchElement::SetAutoDelete(Bool_t autodel)
{
//*-*-*-*-*-*-*-*Set the AutoDelete bit
//*-*            ====================
//  This function can be used to instruct Root in TBranchElement::ReadBasket
//  to not delete the object referenced by a branchelement before reading a
//  new entry. By default, the object is deleted.
//  If autodel is kTRUE, this existing object will be deleted, a new object
//    created by the default constructor, then object->Streamer called.
//  If autodel is kFALSE, the existing object is not deleted. Root assumes
//    that the user is taking care of deleting any internal object or array
//    This can be done in Streamer itself.
//  If this branch has sub-branches, the function sets autodel for these
//  branches as well.
//  We STRONGLY suggest to activate this option by default when you create
//  the top level branch. This will make the read phase more efficient
//  because it minimizes the numbers of new/delete operations.
//  Once this option has been set and the Tree is written to a file, it is
//  not necessary to specify the option again when reading, unless you
//  want to set the opposite mode.
//

   TBranch::SetAutoDelete(autodel);
}

//______________________________________________________________________________
void TBranchElement::SetBasketSize(Int_t buffsize)
{
//*-*-*-*-*-*-*-*Reset basket size for all subbranches of this branchelement
//*-*            ==========================================================
//

   fBasketSize = buffsize;
   Int_t i;
   Int_t nbranches = fBranches.GetEntriesFast();
   for (i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches[i];
      branch->SetBasketSize(buffsize);
   }
}

//______________________________________________________________________________
void TBranchElement::UpdateAddress()
{
//*-*-*-*-*-*-*-*Update branch addresses if a new object was created*-*-*
//*-*            ===================================================
//


}
