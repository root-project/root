// @(#)root/tree:$Name:  $:$Id: TBranchClones.cxx,v 1.13 2001/10/15 06:59:52 brun Exp $
// Author: Rene Brun   11/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
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

#include "TROOT.h"
#include "TBranchClones.h"
#include "TFile.h"
#include "TTree.h"
#include "TBasket.h"
#include "TClass.h"
#include "TRealData.h"
#include "TDataType.h"
#include "TDataMember.h"
#include "TLeafI.h"

R__EXTERN TTree *gTree;

ClassImp(TBranchClones)

//______________________________________________________________________________
TBranchClones::TBranchClones(): TBranch()
{
//*-*-*-*-*-*Default constructor for BranchClones*-*-*-*-*-*-*-*-*-*
//*-*        ====================================

   fList        = 0;
   fRead        = 0;
   fN           = 0;
   fNdataMax    = 0;
   fBranchCount = 0;
}


//______________________________________________________________________________
TBranchClones::TBranchClones(const char *name, void *pointer, Int_t basketsize, Int_t compress, Int_t splitlevel)
    :TBranch()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a BranchClones*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =====================
//
   char leaflist[80];
   char branchname[80];
   char branchcount[64];
   SetName(name);
   if (compress == -1 && gTree->GetDirectory()) {
      TFile *bfile = 0;
      if (gTree->GetDirectory()) bfile = gTree->GetDirectory()->GetFile();
      if (bfile) compress = bfile->GetCompressionLevel();
   }
   char *cpointer  = (char*)pointer;
   char **ppointer = (char**)(cpointer);
   fList     = (TClonesArray*)(*ppointer);
   fAddress  = cpointer;
   fRead     = 0;
   fN        = 0;
   fNdataMax = 0;
   TClass *cl = fList->GetClass();
   if (!cl) return;
   gTree->BuildStreamerInfo(cl);

   fClassName = cl->GetName();
   
   fSplitLevel = splitlevel;

//*-*- Create a branch to store the array count
   if (basketsize < 100) basketsize = 100;
   sprintf(leaflist,"%s_/I",name);
   sprintf(branchcount,"%s_",name);
   fBranchCount = new TBranch(branchcount,&fN,leaflist,basketsize);
   fBranchCount->SetBit(kIsClone);
   TLeaf *leafcount = (TLeaf*)fBranchCount->GetListOfLeaves()->UncheckedAt(0);

   fTree       = gTree;
   fDirectory  = fTree->GetDirectory();
   fFileName   = "";

//*-*-  Create the first basket
   TBasket *basket = new TBasket(branchcount,fTree->GetName(),this);
   fBaskets.Add(basket);
   
//*-*- Loop on all public data members of the class and its base classes
   const char *itype = 0;
   TRealData *rd;
   TIter      next(cl->GetListOfRealData());
   while ((rd = (TRealData *) next())) {
      if (rd->IsObject()) continue;
      TDataMember *member = rd->GetDataMember();
      if (!member->IsPersistent()) continue; //do not process members with a ! as the first
                                             // character in the comment field
      if (!member->IsBasic() || member->IsaPointer() ) {
         Warning("BranchClones","Cannot process: %s::%s",cl->GetName(),member->GetName());
         continue;
      }
      // forget TObject part if splitlevel = 2
      if (splitlevel > 1 || fList->TestBit(TClonesArray::kForgetBits)
                         || cl->CanIgnoreTObjectStreamer()) {
         if (strcmp(member->GetName(),"fBits")     == 0) continue;
         if (strcmp(member->GetName(),"fUniqueID") == 0) continue;
      }
      
      gTree->BuildStreamerInfo(gROOT->GetClass(member->GetFullTypeName()));
      
      TDataType *membertype = member->GetDataType();
      Int_t type = membertype->GetType();
      if (type == 0) {
         Warning("BranchClones","Cannot process: %s::%s",cl->GetName(),member->GetName());
         continue;
      }
      if (type == 1)  itype = "B";
      if (type == 11) itype = "b";
      if (type == 3)  itype = "I";
      if (type == 5)  itype = "F";
      if (type == 8)  itype = "D";
      if (type == 13) itype = "i";
      if (type == 2)  itype = "S";
      if (type == 12) itype = "s";
      sprintf(leaflist,"%s[%s]/%s",member->GetName(),branchcount,itype);
      Int_t comp = compress;
      if (type == 5) comp--;
      sprintf(branchname,"%s.%s",name,rd->GetName());
      TBranch *branch  = new TBranch(branchname,this,leaflist,basketsize,comp);
      branch->SetBit(kIsClone);
      TObjArray *leaves = branch->GetListOfLeaves();
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(0);
      leaf->SetOffset(rd->GetThisOffset());
      leaf->SetLeafCount(leafcount);
      Int_t arraydim = member->GetArrayDim();
      if (arraydim) {
         Int_t maxindex=1;
         while (arraydim) {
            maxindex *= member->GetMaxIndex(--arraydim);
         }
         leaf->SetLen(maxindex);
      }
      fBranches.Add(branch);
   }
}


//______________________________________________________________________________
TBranchClones::~TBranchClones()
{
//*-*-*-*-*-*Default destructor for a BranchClones*-*-*-*-*-*-*-*-*-*-*-*
//*-*        =====================================

   delete fBranchCount;
   fBranchCount = 0;
   fBranches.Delete();
   fList = 0;
}


//______________________________________________________________________________
void TBranchClones::Browse(TBrowser *b)
{
   fBranches.Browse( b );
}

//______________________________________________________________________________
Int_t TBranchClones::Fill()
{
//*-*-*-*-*Loop on all Branches of this BranchClones to fill Basket buffer*-*
//*-*      ===============================================================

   Int_t i;
   Int_t nbytes = 0;
   Int_t nbranches = fBranches.GetEntriesFast();
   char **ppointer = (char**)(fAddress);
   if (ppointer == 0) return 0;
   fList = (TClonesArray*)(*ppointer);
   fN    = fList->GetEntriesFast();
   fEntries++;

   if (fN > fNdataMax) {
      fNdataMax = fList->GetSize();
      char branchcount[64];
      sprintf(branchcount,"%s_",GetName());
      TLeafI *leafi = (TLeafI*)fBranchCount->GetLeaf(branchcount);
      leafi->SetMaximum(fNdataMax);
      for (i=0;i<nbranches;i++)  {
         TBranch *branch = (TBranch*)fBranches.UncheckedAt(i);
         TObjArray *leaves = branch->GetListOfLeaves();
         TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(0);
         leaf->SetAddress();
      }
   }
   nbytes += fBranchCount->Fill();
   for (i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches.UncheckedAt(i);
      TObjArray *leaves = branch->GetListOfLeaves();
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(0);
      leaf->Import(fList, fN);
      nbytes += branch->Fill();
   }
   return nbytes;
}

//______________________________________________________________________________
Int_t TBranchClones::GetEntry(Int_t entry, Int_t getall)
{
//*-*-*-*-*Read all branches of a BranchClones and return total number of bytes
//*-*      ====================================================================

   if (TestBit(kDoNotProcess) && !getall) return 0;
   Int_t nbytes = fBranchCount->GetEntry(entry, getall);
   TLeaf *leafcount = (TLeaf*)fBranchCount->GetListOfLeaves()->UncheckedAt(0);
   fN = Int_t(leafcount->GetValue());
   if (fN <= 0) {
      if (fList) fList->Clear();
      return 0;
   }
   TBranch *branch;
   Int_t nbranches = fBranches.GetEntriesFast();

     // if fList exists, create clonesarray objects
   if (fList) {
     fList->Clear();
     fList->ExpandCreateFast(fN);
     for (Int_t i=0;i<nbranches;i++)  {
         branch = (TBranch*)fBranches.UncheckedAt(i);
         nbytes += branch->GetEntryExport(entry, getall, fList, fN);
      }
   } else {
      for (Int_t i=0;i<nbranches;i++)  {
         branch = (TBranch*)fBranches.UncheckedAt(i);
         nbytes += branch->GetEntry(entry, getall);
      }
   }
  return nbytes;
}

//______________________________________________________________________________
void TBranchClones::Print(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*-*Print TBranch parameters*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================

   fBranchCount->Print(option);
   Int_t i;
   Int_t nbranches = fBranches.GetEntriesFast();
   for (i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches.At(i);
      branch->Print(option);
   }
}

//______________________________________________________________________________
void TBranchClones::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*Reset a Branch*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ====================
//
//    Existing buffers are deleted
//    Entries, max and min are reset
//

   fEntries        = 0;
   fTotBytes       = 0;
   fZipBytes       = 0;
   Int_t i;
   Int_t nbranches = fBranches.GetEntriesFast();
   for (i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches.At(i);
      branch->Reset(option);
   }
   fBranchCount->Reset();
}

//______________________________________________________________________________
void TBranchClones::SetAddress(void *add)
{
//*-*-*-*-*-*-*-*Set address of this branch*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ====================
//*-*

   fReadEntry = -1;
   fAddress = (char*)add;
   char **ppointer = (char**)(fAddress);

   // test if the pointer is null
   // in this case create the correct TClonesArray
   if ( (*ppointer)==0 ) {
     *ppointer = (char*) new TClonesArray(fClassName);
     fAddress = (char*)ppointer;
   }

   fList = (TClonesArray*)(*ppointer);
   fBranchCount->SetAddress(&fN);
}

//______________________________________________________________________________
void TBranchClones::SetBasketSize(Int_t buffsize)
{
//*-*-*-*-*-*-*-*Reset basket size for all subbranches of this branchclones
//*-*            ==========================================================
//

   TBranch::SetBasketSize(buffsize);

   Int_t i;
   Int_t nbranches = fBranches.GetEntriesFast();
   for (i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches[i];
      branch->SetBasketSize(fBasketSize);
   }
}

//_______________________________________________________________________
void TBranchClones::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   UInt_t R__s, R__c;
   if (b.IsReading()) {
      b.ReadVersion(&R__s, &R__c);
      TNamed::Streamer(b);
      b >> fCompress;
      b >> fBasketSize;
      b >> fEntryOffsetLen;
      b >> fMaxBaskets;
      b >> fWriteBasket;
      b >> fEntryNumber;
      b >> fEntries;
      b >> fTotBytes;
      b >> fZipBytes;
      b >> fOffset;
      b >> fBranchCount;
      fClassName.Streamer(b);
      fBranches.Streamer(b);
      fTree = gTree;
      TBranch *branch;
      TLeaf *leaf;
      Int_t nbranches = fBranches.GetEntriesFast();
      for (Int_t i=0;i<nbranches;i++)  {
         branch = (TBranch*)fBranches[i];
         branch->SetBit(kIsClone);
         leaf = (TLeaf*)branch->GetListOfLeaves()->UncheckedAt(0);
         leaf->SetOffset(0);
      }
      fRead = 1;
      TClass *cl = gROOT->GetClass((const char*)fClassName);
      if (!cl) {
         Warning("Streamer","Unknow class: %s. Cannot read BranchClones: %s",
            fClassName.Data(),GetName());
         return;
      }
      if (!cl->GetListOfRealData())  cl->BuildRealData();
      char branchname[80];
      TRealData *rd;
      TIter      next(cl->GetListOfRealData());
      while ((rd = (TRealData *) next())) {
         TDataMember *member = rd->GetDataMember();
         if (!member->IsBasic())      continue;
         if (!member->IsPersistent()) continue;
         TDataType *membertype = member->GetDataType();
         if (membertype->GetType() == 0) continue;
         sprintf(branchname,"%s.%s",GetName(),rd->GetName());
         branch  = (TBranch*)fBranches.FindObject(branchname);
         if (!branch) continue;
         TObjArray *leaves = branch->GetListOfLeaves();
         leaf = (TLeaf*)leaves->UncheckedAt(0);
         leaf->SetOffset(rd->GetThisOffset());
      }
      b.CheckByteCount(R__s, R__c, TBranchClones::IsA());
   } else {
      R__c = b.WriteVersion(TBranchClones::IsA(), kTRUE);
      TNamed::Streamer(b);
      b << fCompress;
      b << fBasketSize;
      b << fEntryOffsetLen;
      b << fMaxBaskets;
      b << fWriteBasket;
      b << fEntryNumber;
      b << fEntries;
      b << fTotBytes;
      b << fZipBytes;
      b << fOffset;
      b << fBranchCount;
      fClassName.Streamer(b);
      fBranches.Streamer(b);
      b.SetByteCount(R__c, kTRUE);
   }
}
