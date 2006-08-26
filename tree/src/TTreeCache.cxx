// @(#)root/tree:$Name:  $:$Id: TTreeCache.cxx,v 1.8 2006/08/14 12:51:40 brun Exp $
// Author: Rene Brun   04/06/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeCache                                                           //
//                                                                      //
//  A specialized TFileCacheRead object for a TTree                     //
//  This class acts as a file cache, registering automatically the      //
//  baskets from the branches being processed (TTree::Draw or           //
//  TTree::Process and TSelectors) when in the learning phase.          //
//  The learning phase is by default 100 entries.                       //
//  It can be changed via TTreeCache::SetLearnEntries.                  //
//                                                                      //
//  This cache speeds-up considerably the performance, in particular    //
//  when the Tree is accessed remotely via a high latency network.      //
//                                                                      //
//  The default cache size (10 Mbytes) may be changed via the function  //
//      TTreeCache::SetCacheSize                                        //
//                                                                      //
//  Only the baskets for the requested entry range are put in the cache //
//                                                                      //
//  For each Tree being processed a TTreeCache object is created.       //
//  This object is automatically deleted when the Tree is deleted or    //
//  when the file is deleted.                                           //
//                                                                      //
//  -Special case of a TChain                                           //
//   Once the training is done on the first Tree, the list of branches  //
//   in the cache is kept for the following files.                      //
//                                                                      //
//  -Special case of a TEventlist                                       //
//   if the Tree or TChain has a TEventlist, only the buffers           //
//   referenced by the list are put in the cache.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTreeCache.h"
#include "TChain.h"
#include "TBranch.h"
#include "TEventList.h"
#include "TObjString.h"

Int_t TTreeCache::fgLearnEntries = 100;

ClassImp(TTreeCache)

//______________________________________________________________________________
TTreeCache::TTreeCache() : TFileCacheRead(),
   fEntryMin(0),
   fEntryMax(1),
   fEntryNext(1),
   fZipBytes(0),
   fNbranches(0),
   fNReadOk(0),
   fNReadMiss(0),
   fNReadPref(0),
   fBranches(0),
   fBrNames(0),
   fOwner(0),
   fTree(0),
   fIsLearning(kTRUE)
{
   // Default Constructor.
}

//______________________________________________________________________________
TTreeCache::TTreeCache(TTree *tree, Int_t buffersize) : TFileCacheRead(tree->GetCurrentFile(),buffersize),
   fEntryMin(0),
   fEntryMax(tree->GetEntriesFast()),
   fEntryNext(0),
   fZipBytes(0),
   fNbranches(0),
   fNReadOk(0),
   fNReadMiss(0),
   fNReadPref(0),
   fBranches(0),
   fBrNames(new TList),
   fOwner(tree),
   fTree(0),
   fIsLearning(kTRUE)
{
   // Constructor.

   fEntryNext = fEntryMin + fgLearnEntries;
   Int_t nleaves = tree->GetListOfLeaves()->GetEntries();
   fBranches = new TBranch*[nleaves+10]; //add a margin just in case in a TChain?
}

//______________________________________________________________________________
TTreeCache::~TTreeCache()
{
   // destructor. (in general called by the TFile destructor

   delete [] fBranches;
   if (fBrNames) {fBrNames->Delete(); delete fBrNames;}
}

//_____________________________________________________________________________
void TTreeCache::AddBranch(TBranch *b)
{
   //add a branch to the list of branches to be stored in the cache
   //this function is called by TBranch::GetBasket

   if (!fIsLearning) return;

   //Is branch already in the cache?
   Bool_t isNew = kTRUE;
   for (int i=0;i<fNbranches;i++) {
      if (fBranches[i] == b) {isNew = kFALSE; break;}
   }
   if (isNew) {
      fTree = b->GetTree();
      fBranches[fNbranches] = b;
      fBrNames->Add(new TObjString(b->GetName()));
      fZipBytes += b->GetZipBytes();
      fNbranches++;
      if (gDebug > 0) printf("Entry: %lld, registering branch: %s\n",b->GetTree()->GetReadEntry(),b->GetName());
   }
}

//_____________________________________________________________________________
Bool_t TTreeCache::FillBuffer()
{
   //Fill the cache buffer with the branches in the cache

   if (fNbranches <= 0) return kFALSE;
   TTree *tree = fBranches[0]->GetTree();
   Long64_t entry = tree->GetReadEntry();
   if (entry < fEntryNext) return kFALSE;

   // estimate number of entries that can fit in the cache
   // compare it to the original value of fBufferSize not
   // to the real one
   fEntryNext = entry + tree->GetEntries()*fBufferSizeMin/fZipBytes;

   if (fEntryNext > fEntryMax) fEntryNext = fEntryMax+1;

   //check if owner has a TEventList set. If yes we optimize for this special case
   //reading only the baskets containing entries in the list
   TEventList *elist = fOwner->GetEventList();
   Long64_t chainOffset = 0;
   if (elist) {
      if (fOwner->IsA() ==TChain::Class()) {
         TChain *chain = (TChain*)fOwner;
         Int_t t = chain->GetTreeNumber();
         chainOffset = chain->GetTreeOffset()[t];
      }
   }

   //clear cache buffer
   TFileCacheRead::Prefetch(0,0);
   //store baskets
   Bool_t mustBreak = kFALSE;
   for (Int_t i=0;i<fNbranches;i++) {
      if (mustBreak) break;
      TBranch *b = fBranches[i];
      Int_t nb = b->GetMaxBaskets();
      Int_t *lbaskets   = b->GetBasketBytes();
      Long64_t *entries = b->GetBasketEntry();
      if (!lbaskets || !entries) continue;
      //we have found the branch. We now register all its baskets
      //from the requested offset to the basket below fEntrymax
      for (Int_t j=0;j<nb;j++) {
         Long64_t pos = b->GetBasketSeek(j);
         Int_t len = lbaskets[j];
         if (pos <= 0 || len <= 0) continue;
         if (entries[j] > fEntryNext) continue;
         if (entries[j] < entry && (j<nb-1 && entries[j+1] < entry)) continue;
         if (elist) {
            Long64_t emax = fEntryMax;
            if (j<nb-1) emax = entries[j+1]-1;
            if (!elist->ContainsRange(entries[j]+chainOffset,emax+chainOffset)) continue;
         }
         fNReadPref++;
         TFileCacheRead::Prefetch(pos,len);
         //we allow up to twice the default buffer size. When using eventlist in particular
         //it may happen that the evaluation of fEntryNext is bad, hence this protection
         if (fNtot > 2*fBufferSizeMin) {TFileCacheRead::Prefetch(0,0);mustBreak = kTRUE; break;}
      }
      if (gDebug > 0) printf("Entry: %lld, registering baskets branch %s, fEntryNext=%lld, fNseek=%d, fNtot=%d\n",entry,fBranches[i]->GetName(),fEntryNext,fNseek,fNtot);
   }
   fIsLearning = kFALSE;
   if (mustBreak) return kFALSE;
   return kTRUE;
}

//_____________________________________________________________________________
Double_t TTreeCache::GetEfficiency()
{
   // Give the total efficiency of the cache... defined as the ratio
   // of blocks found in the cache vs. the number of blocks prefetched
   // ( it could be more than 1 if we read the same block from the cache more
   //   than once )
   // Note: This should eb used at the end of the processing or we will
   //       get uncomplete stats

   if ( !fNReadPref )
      return 0;

   return ((Double_t)fNReadOk / (Double_t)fNReadPref);
}

//_____________________________________________________________________________
Double_t TTreeCache::GetEfficiencyRel()
{
   // This will indicate a sort of relative efficiency... a ratio of the
   // reads found in the cache to the number of reads so far

   if ( !fNReadOk && !fNReadMiss )
      return 0;

   return ((Double_t)fNReadOk / (Double_t)(fNReadOk + fNReadMiss));
}

//_____________________________________________________________________________
Int_t TTreeCache::GetLearnEntries()
{
   //static function returning the number of entries used to train the cache
   //see SetLearnEntries

   return fgLearnEntries;
}

//_____________________________________________________________________________
TTree *TTreeCache::GetTree() const
{
   //return Tree in the cache
   if (fNbranches <= 0) return 0;
   return fBranches[0]->GetTree();
}

//_____________________________________________________________________________
Int_t TTreeCache::ReadBuffer(char *buf, Long64_t pos, Int_t len)
{
   // Read buffer at position pos.
   // If pos is in the list of prefetched blocks read from fBuffer,
   // then try to fill the cache from the list of selected branches,
   // otherwise normal read from file. Returns -1 in case of read
   // failure, 0 in case not in cache and 1 in case read from cache.
   // This function overloads TFileCacheRead::ReadBuffer.

   //Is request already in the cache?
   if (TFileCacheRead::ReadBuffer(buf,pos,len) == 1){
      fNReadOk++;
      return 1;
   }

   //not found in cache. Do we need to fill the cache?
   Bool_t bufferFilled = FillBuffer();
   if (bufferFilled) {
      Int_t res = TFileCacheRead::ReadBuffer(buf,pos,len);

      if (res == 1)
         fNReadOk++;
      else if (res == 0)
         fNReadMiss++;

      return res;
   }
   fNReadMiss++;

   return 0;
}

//_____________________________________________________________________________
void TTreeCache::SetEntryRange(Long64_t emin, Long64_t emax)
{
   // Set the minimum and maximum entry number to be processed
   // this information helps to optimize the number of baskets to read
   // when prefetching the branch buffers.

   fEntryMin  = emin;
   fEntryMax  = emax;
   fEntryNext  = fEntryMin + fgLearnEntries;
   fIsLearning = kTRUE;
   fNbranches  = 0;
   fZipBytes   = 0;
   if (fBrNames) fBrNames->Delete();
   if (gDebug > 0) printf("SetEntryRange: fEntryMin=%lld, fEntryMax=%lld, fEntryNext=%lld\n",fEntryMin,fEntryMax,fEntryNext);

}

//_____________________________________________________________________________
void TTreeCache::SetLearnEntries(Int_t n)
{
   // Static function to set the number of entries to be used in learning mode
   // The default value for n is 10. n must be >= 1

   if (n < 1) n = 1;
   fgLearnEntries = n;
}

//_____________________________________________________________________________
void TTreeCache::UpdateBranches(TTree *tree)
{
   //update pointer to current Tree and recompute pointers to the branches in the cache

   fTree = tree;
   Prefetch(0,0);

   fEntryMin  = 0;
   fEntryMax  = fTree->GetEntries();
   fEntryNext = fEntryMin + fgLearnEntries;
   fZipBytes  = 0;
   fNbranches = 0;
   TIter next(fBrNames);
   TObjString *os;
   while ((os = (TObjString*)next())) {
      TBranch *b = fTree->GetBranch(os->GetName());
      if (!b) continue;
      fBranches[fNbranches] = b;
      fZipBytes   += b->GetZipBytes();
      fNbranches++;
   }
}
