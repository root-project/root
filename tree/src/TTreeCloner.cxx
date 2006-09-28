// @(#)root/tree:$Name:  $:$Id: TTreeCloner.cxx,v 1.9 2006/07/06 17:42:59 pcanal Exp $
// Author: Philippe Canal 07/11/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeCloner                                                          //
//                                                                      //
// Class implementing or helping  the various TTree cloning method      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBasket.h"
#include "TBranch.h"
#include "TBranchClones.h"
#include "TBranchElement.h"
#include "TStreamerInfo.h"
#include "TBranchRef.h"
#include "TError.h"
#include "TProcessID.h"
#include "TTree.h"
#include "TTreeCloner.h"
#include "TFile.h"
#include "TLeafI.h"
#include "TLeafL.h"

TTreeCloner::TTreeCloner(TTree *from, TTree *to, Option_t *method) :
   fIsValid(kTRUE),
   fFromTree(from),
   fToTree(to),
   fMethod(method),
   fFromBranches( from ? from->GetListOfLeaves()->GetEntries()+1 : 0),
   fToBranches( to ? from->GetListOfLeaves()->GetEntries()+1 : 0),
   fMaxBaskets(CollectBranches()),
   fBasketBranchNum(new UInt_t[fMaxBaskets]),
   fBasketNum(new UInt_t[fMaxBaskets]),
   fBasketSeek(new Long64_t[fMaxBaskets]),
   fBasketEntry(new Long64_t[fMaxBaskets]),
   fBasketIndex(new Int_t[fMaxBaskets]),
   fCloneMethod(TTreeCloner::kDefault),
   fToStartEntries(0)
{
   // Constructor.  This object would transfer the data from
   // 'from' to 'to' using the method indicated in method.

   TString opt(method);
   opt.ToLower();
   if (opt.Contains("sortbasketsbybranch")) {
      //::Info("TTreeCloner::TTreeCloner","use: kSortBasketsByBranch");
      fCloneMethod = TTreeCloner::kSortBasketsByBranch;
   } else if (opt.Contains("sortbasketsbyentry")) {
      //::Info("TTreeCloner::TTreeCloner","use: kSortBasketsByEntry");
      fCloneMethod = TTreeCloner::kSortBasketsByEntry;
   } else {
      //::Info("TTreeCloner::TTreeCloner","use: kSortBasketsByOffset");
      fCloneMethod = TTreeCloner::kSortBasketsByOffset;
   }
   if (fToTree) fToStartEntries = fToTree->GetEntries();
}

Bool_t TTreeCloner::Exec()
{
   // Execute the cloning.

   CopyStreamerInfos();
   CopyProcessIds();
   CloseOutWriteBaskets();
   CollectBaskets();
   SortBaskets();
   WriteBaskets();
   CopyMemoryBaskets();

   return kTRUE;
}

TTreeCloner::~TTreeCloner()
{
   // TTreeCloner destructor

   delete [] fBasketBranchNum;
   delete [] fBasketNum;
   delete [] fBasketSeek;
   delete [] fBasketEntry;
   delete [] fBasketIndex;
}

void TTreeCloner::CloseOutWriteBaskets()
{
   // Before we can start adding new basket, we need to flush to
   // disk the partially filled baskets (the WriteBasket)

   for(Int_t i=0; i<fToBranches.GetEntries(); ++i) {
      TBranch *to = (TBranch*)fToBranches.UncheckedAt(i);

      TObjArray *array = to->GetListOfBaskets();
      if (array->GetEntries()) {
         TBasket *basket = to->GetBasket(to->GetWriteBasket());
         if (basket) {
            if (basket->GetNevBuf()) {
               // If the basket already contains entry we need to close it
               // out. (This is because we can only transfer full compressed
               // buffer)

               if (basket->GetBufferRef()->IsReading()) {
                  basket->SetWriteMode();
               }
               to->WriteBasket(basket);
               basket = to->GetBasket(to->GetWriteBasket()); // WriteBasket create an empty basket
            }
            if (basket) {
               basket->DropBuffers();
               to->GetListOfBaskets()->RemoveAt(to->GetWriteBasket());
               delete basket;
            }
         }
      }
   }
}

UInt_t TTreeCloner::CollectBranches(TBranch *from, TBranch *to)
{
   // Fill the array of branches, adding the branch 'from' and 'to',
   // and matching the sub-branches of the 'from' and 'to' branches.
   // Returns the total number of baskets in all the from branch and
   // it sub-branches.

   // Since this is called from the constructor, this can not be a virtual function

   UInt_t numBaskets = 0;
   if (from->InheritsFrom(TBranchClones::Class())) {
      TBranchClones *fromclones = (TBranchClones*)from;
      TBranchClones *toclones = (TBranchClones*)to;
      numBaskets += CollectBranches(fromclones->fBranchCount,toclones->fBranchCount);
   
   } else if (from->InheritsFrom(TBranchElement::Class())) {
      TBranchElement *fromelem = (TBranchElement*)from;
      TBranchElement *toelem   = (TBranchElement*)to;
      if (fromelem->fMaximum > toelem->fMaximum) toelem->fMaximum = fromelem->fMaximum;
   } else {

      Int_t nb = from->GetListOfLeaves()->GetEntries();
      Int_t fnb= to->GetListOfLeaves()->GetEntries();
      if (nb!=fnb) {
         Error("TTreeCloner::CollectBranches",
            "The export branch and the import branch do not have the same number of leaves (%d vs %d)",
            fnb,nb);
         fIsValid = kFALSE;
         return 0;
      }
      for (Int_t i=0;i<nb;i++)  {
         TLeaf *fromleaf_gen = (TLeaf*)from->GetListOfLeaves()->At(i);
         if (fromleaf_gen->IsA()==TLeafI::Class()) {
            TLeafI *fromleaf = (TLeafI*)from->GetListOfLeaves()->At(i);
            TLeafI *toleaf   = (TLeafI*)to->GetListOfLeaves()->At(i);
            if (fromleaf->GetMaximum() > toleaf->GetMaximum()) 
               toleaf->SetMaximum( fromleaf->GetMaximum() );
            if (fromleaf->GetMinimum() < toleaf->GetMinimum()) 
               toleaf->SetMinimum( fromleaf->GetMinimum() );
         } else if (fromleaf_gen->IsA()==TLeafL::Class()) {
            TLeafL *fromleaf = (TLeafL*)from->GetListOfLeaves()->At(i);
            TLeafL *toleaf   = (TLeafL*)to->GetListOfLeaves()->At(i);
            if (fromleaf->GetMaximum() > toleaf->GetMaximum()) 
               toleaf->SetMaximum( fromleaf->GetMaximum() );
            if (fromleaf->GetMinimum() < toleaf->GetMinimum()) 
               toleaf->SetMinimum( fromleaf->GetMinimum() );
         }
      }

   }

   fFromBranches.AddLast(from);
   fToBranches.AddLast(to);

   numBaskets += from->GetWriteBasket();
   numBaskets += CollectBranches(from->GetListOfBranches(),to->GetListOfBranches());

   return numBaskets;
}

UInt_t TTreeCloner::CollectBranches(TObjArray *from, TObjArray *to)
{
   // Fill the array of branches, matching the branches of the 'from' and 'to' arrays.
   // Returns the total number of baskets in all the branches.

   // Since this is called from the constructor, this can not be a virtual function

   Int_t nb = from->GetEntries();
   Int_t fnb= to->GetEntries();
   if (nb!=fnb) {
      Error("TTreeCloner::CollectSubBranches",
         "The export branch and the import branch do not have the same number of branches (%d vs %d)",
         fnb,nb);
      fIsValid = kFALSE;
      return 0;
   }

   UInt_t numBasket = 0;
   for (Int_t i=0;i<nb;i++)  {
      numBasket += CollectBranches((TBranch*)from->UncheckedAt(i),(TBranch*)to->UncheckedAt(i));
   }
   return numBasket;
}


UInt_t TTreeCloner::CollectBranches()
{
   // Fill the array of branches, matching the branches of the 'from' and 'to' TTrees
   // Returns the total number of baskets in all the branches.

   // Since this is called from the constructor, this can not be a virtual function

   UInt_t numBasket = CollectBranches(fFromTree->GetListOfBranches(),
                                      fToTree->GetListOfBranches());

   if (fFromTree->GetBranchRef()) {
      fToTree->BranchRef();
      numBasket += CollectBranches(fFromTree->GetBranchRef(),fToTree->GetBranchRef());
   }
   return numBasket;
}

void TTreeCloner::CollectBaskets()
{
   // Collect the information about the on-file basket that need
   // to be copied.

   UInt_t len = fFromBranches.GetEntries();

   for(UInt_t i=0,bi=0; i<len; ++i) {
      TBranch *from = (TBranch*)fFromBranches.UncheckedAt(i);
      for(Int_t b=0; b<from->GetWriteBasket(); ++b,++bi) {
         fBasketBranchNum[bi] = i;
         fBasketNum[bi] = b;
         fBasketSeek[bi] = from->GetBasketSeek(b);
         fBasketEntry[bi] = from->GetBasketEntry()[b];
         fBasketIndex[bi] = bi;
      }
   }
}

void TTreeCloner::CopyStreamerInfos()
{
   // Make sure that all the needed TStreamerInfo are
   // present in the output file

   TFile *fromFile = fFromTree->GetDirectory()->GetFile();
   TFile *toFile = fToTree->GetDirectory()->GetFile();
   TList *l = fromFile->GetStreamerInfoList();
   TIter next(l);
   TStreamerInfo *oldInfo;
   while ( (oldInfo = (TStreamerInfo*)next()) ) {
      TStreamerInfo *curInfo = 0;
      TClass *cl = gROOT->GetClass(oldInfo->GetName());

      // Insure that the TStreamerInfo is loaded
      curInfo = cl->GetStreamerInfo(oldInfo->GetClassVersion());
      if (oldInfo->GetClassVersion()==1) {
         // We may have a Foreign class let's look using the
         // checksum:
         curInfo = cl->FindStreamerInfo(oldInfo->GetCheckSum());
      }
      curInfo->TagFile(toFile);
   }
   delete l;
}

void TTreeCloner::CopyMemoryBaskets()
{
   // Transfer the basket from the input file to the output file

   TBasket *basket = 0;
   for(Int_t i=0; i<fToBranches.GetEntries(); ++i) {
      TBranch *from = (TBranch*)fFromBranches.UncheckedAt( i );
      TBranch *to   = (TBranch*)fToBranches.UncheckedAt( i );

      basket = from->GetListOfBaskets()->GetEntries() ? from->GetBasket(from->GetWriteBasket()) : 0;
      if (basket) {
         basket = (TBasket*)basket->Clone();
         to->AddBasket(*basket, kFALSE, fToStartEntries+from->GetBasketEntry()[from->GetWriteBasket()]);
      }
      // If the branch is a TBranchElement non-terminal 'object' branch, it's basket will contain 0
      // events.
      if (from->GetEntries()!=0 && from->GetWriteBasket()==0 && basket->GetNevBuf()==0) {
         to->SetEntries(to->GetEntries()+from->GetEntries());
      }
   }
}

void TTreeCloner::CopyProcessIds()
{
   // Make sure that all the needed TStreamerInfo are
   // present in the output file

   // NOTE: We actually need to merge the ProcessId somehow :(

   TFile *fromfile = fFromTree->GetDirectory()->GetFile();
   TFile *tofile = fToTree->GetDirectory()->GetFile();

   fPidOffset = tofile->GetNProcessIDs();

   TIter next(fromfile->GetListOfKeys());
   TKey *key;
   TDirectory::TContext cur(gDirectory,fromfile);
   while ((key = (TKey*)next())) {
      if (!strcmp(key->GetClassName(),"TProcessID")) {
         TProcessID *id = (TProcessID*)key->ReadObjectAny(0);

         UShort_t out = TProcessID::WriteProcessID(id,tofile);
         if (out<fPidOffset) {
            Error("CopyProcessIDs","Copied %s from %s might already exist!\n",
                  id->GetName(),fromfile->GetName());
         }
      }
   }
}

void TTreeCloner::SortBaskets()
{
   // Sort the basket according to the user request.

   // Currently this sort __has to__ preserve the order
   // of basket for each individual branch.

   switch (fCloneMethod) {
      case kSortBasketsByBranch:
         // nothing to do, it is already sorted.
         break;
      case kSortBasketsByEntry:
         TMath::Sort( fMaxBaskets, fBasketEntry, fBasketIndex, kFALSE );
         break;
      case kSortBasketsByOffset:
      default:
         TMath::Sort( fMaxBaskets, fBasketSeek, fBasketIndex, kFALSE );
         break;
   }
}

void TTreeCloner::WriteBaskets()
{
   // Transfer the basket from the input file to the output file

   TBasket *basket = new TBasket();
   for(UInt_t j=0; j<fMaxBaskets; ++j) {
      TBranch *from = (TBranch*)fFromBranches.UncheckedAt( fBasketBranchNum[ fBasketIndex[j] ] );
      TBranch *to   = (TBranch*)fToBranches.UncheckedAt( fBasketBranchNum[ fBasketIndex[j] ] );

      TFile *tofile = to->GetFile(0);
      TFile *fromfile = from->GetFile(0);

      Int_t index = fBasketNum[ fBasketIndex[j] ];

      Long64_t pos = from->GetBasketSeek(index);
      if (from->GetBasketBytes()[index] == 0) {
         from->GetBasketBytes()[index] = basket->ReadBasketBytes(pos, fromfile);
      }
      Int_t len = from->GetBasketBytes()[index];
      Long64_t entryInBasket;
      if (index == from->GetWriteBasket()) {
         entryInBasket = from->GetEntries() - from->GetBasketEntry()[index];
      } else {
         entryInBasket = from->GetBasketEntry()[index+1] - from->GetBasketEntry()[index];
      }

      basket->LoadBasketBuffers(pos,len,fromfile);
      basket->IncrementPidOffset(fPidOffset);
      basket->CopyTo(tofile);
      to->AddBasket(*basket,kTRUE,fToStartEntries + from->GetBasketEntry()[index]);

   }
   delete basket;
}
