// @(#)root/tree:$Id$
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
#include "TList.h"
#include "TBranch.h"
#include "TEventList.h"
#include "TObjString.h"
#include "TRegexp.h"
#include "TLeaf.h"
#include "TFriendElement.h"
#include "TFile.h"

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
   fIsLearning(kTRUE),
   fIsManual(kFALSE)
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
   fIsLearning(kTRUE),
   fIsManual(kFALSE)
{
   // Constructor.

   fEntryNext = fEntryMin + fgLearnEntries;
   Int_t nleaves = tree->GetListOfLeaves()->GetEntries();
   fBranches = new TObjArray(nleaves);
}

//______________________________________________________________________________
TTreeCache::~TTreeCache()
{
   // destructor. (in general called by the TFile destructor

   delete fBranches;
   if (fBrNames) {fBrNames->Delete(); delete fBrNames; fBrNames=0;}
}

//_____________________________________________________________________________
void TTreeCache::AddBranch(TBranch *b, Bool_t subbranches /*= kFALSE*/)
{
   //add a branch to the list of branches to be stored in the cache
   //this function is called by TBranch::GetBasket

   if (!fIsLearning) return;

   // Reject branch that are not from the cached tree.
   if (!b || fOwner->GetTree() != b->GetTree()) return;

   //Is branch already in the cache?
   Bool_t isNew = kTRUE;
   for (int i=0;i<fNbranches;i++) {
      if (fBranches->UncheckedAt(i) == b) {isNew = kFALSE; break;}
   }
   if (isNew) {
      fTree = b->GetTree();
      fBranches->AddAtAndExpand(b, fNbranches);
      fBrNames->Add(new TObjString(b->GetName()));
      fZipBytes += b->GetZipBytes();
      fNbranches++;
      if (gDebug > 0) printf("Entry: %lld, registering branch: %s\n",b->GetTree()->GetReadEntry(),b->GetName());
   }
   
   // process subbranches
   if (subbranches) {
      TObjArray *lb = b->GetListOfBranches();
      Int_t nb = lb->GetEntriesFast();
      for (Int_t j = 0; j < nb; j++) {
         TBranch* branch = (TBranch*) lb->UncheckedAt(j);
         if (!branch) continue;
         AddBranch(branch, subbranches);
      }
   }
}


//_____________________________________________________________________________
void TTreeCache::AddBranch(const char *bname, Bool_t subbranches /*= kFALSE*/)
{
   // Add a branch to the list of branches to be stored in the cache
   // this is to be used by user (thats why we pass the name of the branch).
   // It works in exactly the same way as TTree::SetBranchStatus so you
   // probably want to look over ther for details about the use of bname
   // with regular expresions.
   // The branches are taken with respect to the Owner of this TTreeCache
   // (i.e. the original Tree)
   
   TBranch *branch, *bcount;
   TLeaf *leaf, *leafcount;

   Int_t i;
   Int_t nleaves = (fOwner->GetListOfLeaves())->GetEntriesFast();
   TRegexp re(bname,kTRUE);
   Int_t nb = 0;

   // first pass, loop on all branches
   // for leafcount branches activate/deactivate in function of status
   for (i=0;i<nleaves;i++)  {
      leaf = (TLeaf*)(fOwner->GetListOfLeaves())->UncheckedAt(i);
      branch = (TBranch*)leaf->GetBranch();
      TString s = branch->GetName();
      if (strcmp(bname,"*")) { //Regexp gives wrong result for [] in name
         TString longname; 
         longname.Form("%s.%s",fOwner->GetName(),branch->GetName());
         if (strcmp(bname,branch->GetName()) 
             && longname != bname
             && s.Index(re) == kNPOS) continue;
      }
      nb++;
      AddBranch(branch, subbranches);
      leafcount = leaf->GetLeafCount();
      if (leafcount) {
         bcount = leafcount->GetBranch();
         AddBranch(bcount, subbranches);
      }
   }
   if (nb==0 && strchr(bname,'*')==0) {
      branch = fOwner->GetBranch(bname);
      if (branch) {
         AddBranch(branch, subbranches);
         ++nb;
      }
   }

   //search in list of friends
   UInt_t foundInFriend = 0;
   if (fOwner->GetListOfFriends()) {
      TIter nextf(fOwner->GetListOfFriends());
      TFriendElement *fe;
      TString name;
      while ((fe = (TFriendElement*)nextf())) {
         TTree *t = fe->GetTree();
         if (t==0) continue;

         // If the alias is present replace it with the real name.
         char *subbranch = (char*)strstr(bname,fe->GetName());
         if (subbranch!=bname) subbranch = 0;
         if (subbranch) {
            subbranch += strlen(fe->GetName());
            if ( *subbranch != '.' ) subbranch = 0;
            else subbranch ++;
         }
         if (subbranch) {
            name.Form("%s.%s",t->GetName(),subbranch);
         } else {
            name = bname;
         }
         AddBranch(name, subbranches);
      }
   }
   if (!nb && !foundInFriend) {
      if (gDebug > 0) printf("AddBranch: unknown branch -> %s \n", bname);
      return;
   }
}

//_____________________________________________________________________________
Bool_t TTreeCache::FillBuffer()
{
   // Fill the cache buffer with the branches in the cache.


   if (fNbranches <= 0) return kFALSE;
   TTree *tree = ((TBranch*)fBranches->UncheckedAt(0))->GetTree();
   Long64_t entry = tree->GetReadEntry();
   
   if (!fIsManual && entry < fEntryNext) return kFALSE;
   
   // Triggered by the user, not the learning phase
   if (entry == -1)  entry=0;

   // Estimate number of entries that can fit in the cache compare it
   // to the original value of fBufferSize not to the real one
   if (fZipBytes==0) {
      fEntryNext = entry + tree->GetEntries();;    
   } else {
      fEntryNext = entry + tree->GetEntries()*fBufferSizeMin/fZipBytes;
   }
   if (fEntryMax <= 0) fEntryMax = tree->GetEntries();
   if (fEntryNext > fEntryMax) fEntryNext = fEntryMax+1;

   // Check if owner has a TEventList set. If yes we optimize for this
   // Special case reading only the baskets containing entries in the
   // list.
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
      TBranch *b = (TBranch*)fBranches->UncheckedAt(i);
      Int_t nb = b->GetMaxBaskets();
      Int_t *lbaskets   = b->GetBasketBytes();
      Long64_t *entries = b->GetBasketEntry();
      if (!lbaskets || !entries) continue;
      //we have found the branch. We now register all its baskets
      //from the requested offset to the basket below fEntrymax
      Int_t blistsize = b->GetListOfBaskets()->GetSize();
      for (Int_t j=0;j<nb;j++) {
         // This basket has already been read, skip it
         if (j<blistsize && b->GetListOfBaskets()->UncheckedAt(j)) continue;

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
      if (gDebug > 0) printf("Entry: %lld, registering baskets branch %s, fEntryNext=%lld, fNseek=%d, fNtot=%d\n",entry,((TBranch*)fBranches->UncheckedAt(i))->GetName(),fEntryNext,fNseek,fNtot);
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
TTree *TTreeCache::GetOwner() const
{
   //return the owner of this cache.

   return fOwner;
}

//_____________________________________________________________________________
TTree *TTreeCache::GetTree() const
{
   //return Tree in the cache
   if (fNbranches <= 0) return 0;
   return ((TBranch*)(fBranches->UncheckedAt(0)))->GetTree();
}

//_____________________________________________________________________________
Int_t TTreeCache::ReadBuffer(char *buf, Long64_t pos, Int_t len)
{
   // Read buffer at position pos.
   // If pos is in the list of prefetched blocks read from fBuffer.
   // Otherwise try to fill the cache from the list of selected branches,
   // and recheck if pos is now in the list.
   // Returns 
   //    -1 in case of read failure, 
   //     0 in case not in cache,
   //     1 in case read from cache.
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
void TTreeCache::ResetCache()
{
   // This will simply clear the cache
   TFileCacheRead::Prefetch(0,0);
}

//_____________________________________________________________________________
void TTreeCache::SetEntryRange(Long64_t emin, Long64_t emax)
{
   // Set the minimum and maximum entry number to be processed
   // this information helps to optimize the number of baskets to read
   // when prefetching the branch buffers.

   // This is called by TTreePlayer::Process in an automatic way...
   // don't do it if the user has specified the branches.
   if(fIsManual)
      return;
   
   fEntryMin  = emin;
   fEntryMax  = emax;
   fEntryNext  = fEntryMin + fgLearnEntries;
   if (gDebug > 0) printf("SetEntryRange: fEntryMin=%lld, fEntryMax=%lld, fEntryNext=%lld\n",fEntryMin,fEntryMax,fEntryNext);
   fIsLearning = kTRUE;
   fIsManual = kFALSE;
   fNbranches  = 0;
   fZipBytes   = 0;
   if (fBrNames) fBrNames->Delete();

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
void TTreeCache::StartLearningPhase()
{
   // The name should be enough to explain the method.
   // The only additional comments is that the cache is cleaned before
   // the new learning phase.
   
   fIsLearning = kTRUE;
   fIsManual = kFALSE;
   fNbranches  = 0;
   fZipBytes   = 0;
   if (fBrNames) fBrNames->Delete();
   fIsTransferred = kFALSE;
}

//_____________________________________________________________________________
void TTreeCache::StopLearningPhase()
{
   // This is the counterpart of StartLearningPhase() and can be used to stop
   // the learning phase. It's useful when the user knows exactly what branches
   // he is going to use.
   // For the moment it's just a call to FillBuffer() since that method
   // will create the buffer lists from the specified branches.
   
   fIsLearning = kFALSE;
   fIsManual = kTRUE;
   FillBuffer();

   // If this is the first time we get here since the last FillBuffer
   // it's probable that the information about the prefetched buffers is there
   // but it hasn't actually been transfered... Is this the best place to put it??
   if (fNseek > 0 && !fIsSorted) {
      Sort();

      // Then we use the vectored read to read everything now
      fFile->ReadBuffers(fBuffer,fPos,fLen,fNb);
      fIsTransferred = kTRUE;
   }
}

//_____________________________________________________________________________
void TTreeCache::UpdateBranches(TTree *tree, Bool_t owner)
{
   //update pointer to current Tree and recompute pointers to the branches in the cache

   if (owner) {
      fOwner = tree;
      SetFile(tree->GetCurrentFile());
   }
   fTree = tree;

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
      fBranches->AddAt(b, fNbranches);
      fZipBytes   += b->GetZipBytes();
      fNbranches++;
   }
}
