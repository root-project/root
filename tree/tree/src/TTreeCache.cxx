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
//  The learning period is started or restarted when:
//     - TTree::SetCacheSize is called for the first time.
//     - TTree::SetCacheSize is called a second time with a different size.
//     - TTreeCache::StartLearningPhase is called.
//     - TTree[Cache]::SetEntryRange is called
//          * and the learning is not yet finished
//          * and has not been set to manual
//          * and the new minimun entry is different.
//
//  The learning period is stopped (and prefetching is actually started) when:
//     - TTree[Cache]::StopLearningPhase is called.
//     - An entry outside the 'learning' range is requested
//       The 'learning range is from fEntryMin (default to 0) to
//       fEntryMin + fgLearnEntries (default to 100).
//     - A 'cached' TChain switches over to a new file.
//
//     WHY DO WE NEED the TreeCache when doing data analysis?
//     ======================================================
//
//  When writing a TTree, the branch buffers are kept in memory.
//  A typical branch buffersize (before compression) is typically 32 KBytes.
//  After compression, the zipped buffer may be just a few Kbytes.
//  The branch buffers cannot be much larger in case of Trees with several
//  hundred or thousand branches.
//  When writing, this does not generate a performance problem because branch
//  buffers are always written sequentially and the OS is in general clever enough
//  to flush the data to the output file when a few MBytes of data have to be written.
//  When reading at the contrary, one may hit a performance problem when reading
//  across a network (LAN or WAN) and the network latency is high.
//  For example in a WAN with 10ms latency, reading 1000 buffers of 10 KBytes each
//  with no cache will imply 10s penalty where a local read of the 10 MBytes would
//  take about 1 second.
//  The TreeCache will try to prefetch all the buffers for the selected branches
//  such that instead of transfering 1000 buffers of 10 Kbytes, it will be able
//  to transfer one single large buffer of 10 Mbytes in one single transaction.
//  Not only the TreeCache minimizes the number of transfers, but in addition
//  it can sort the blocks to be read in increasing order such that the file
//  is read sequentially.
//  Systems like xrootd, dCache or httpd take advantage of the TreeCache in
//  reading ahead as much data as they can and return to the application
//  the maximum data specified in the cache and have the next chunk of data ready
//  when the next request comes.
//
//
//     HOW TO USE the TreeCache
//     =========================
//
//  A few use cases are discussed below. It is not simple to activate the cache
//  by default (except case1 below) because there are many possible configurations.
//  In some applications you know a priori the list of branches to read.
//  In other applications the analysis loop calls several layers of user functions
//  where it is impossible to predict a priori which branches will be used. This
//  is probably the most frequent case. In this case ROOT I/O will flag used
//  branches automatically when a branch buffer is read during the learning phase.
//  The TreeCache interface provides functions to instruct the cache about the used
//  branches if they are known a priori. In the examples below, portions of analysis
//  code are shown. The few statements involving the TreeCache are marked with //<<<
//
//  -------------------
//  1- with TTree::Draw
//  -------------------
//  the TreeCache is automatically used by TTree::Draw. The function knows
//  which branches are used in the query and it puts automatically these branches
//  in the cache. The entry range is also known automatically.
//
//  -------------------------------------
//  2- with TTree::Process and TSelectors
//  -------------------------------------
//  You must enable the cache and tell the system which branches to cache
//  and also specify the entry range. It is important to specify the entry range
//  in case you process only a subset of the events, otherwise you run the risk
//  to store in the cache entries that you do not need.
//
//      --example 2a 
//--
//   TTree *T = (TTree*)f->Get("mytree");
//   Long64_t nentries = T->GetEntries();
//   Int_t cachesize = 10000000; //10 MBytes
//   T->SetCacheSize(cachesize); //<<<
//   T->AddBranch("*",kTRUE);    //<<< add all branches to the cache
//   T->Process('myselector.C+");
//   //in the TSelector::Process function we read all branches
//   T->GetEntry(i);
//--      ... here you process your entry
//
//
//      --example 2b 
//  in the Process function we read a subset of the branches.
//  Only the branches used in the first entry will be put in the cache
//--
//   TTree *T = (TTree*)f->Get("mytree");
//   //we want to process only the 200 first entries
//   Long64_t nentries=200;
//   int efirst= 0;
//   int elast = efirst+nentries;
//   Int_t cachesize = 10000000; //10 MBytes
//   TTreeCache::SetLearnEntries(1);  //<<< we can take the decision after 1 entry
//   T->SetCacheSize(cachesize);      //<<<
//   T->SetCacheEntryRange(efirst,elast); //<<<
//   T->Process('myselector.C+","",nentries,efirst);
//   // in the TSelector::Process we read only 2 branches
//   TBranch *b1 = T->GetBranch("branch1");
//   b1->GetEntry(i);
//   if (somecondition) return;
//   TBranch *b2 = T->GetBranch("branch2");
//   b2->GetEntry(i);
//      ... here you process your entry
//--
//  ----------------------------
//  3- with your own event loop
//  ----------------------------
//    --example 3a
//      in your analysis loop, you always use 2 branches. You want to prefetch
//      the branch buffers for these 2 branches only.
//--
//   TTree *T = (TTree*)f->Get("mytree");
//   TBranch *b1 = T->GetBranch("branch1");
//   TBranch *b2 = T->GetBranch("branch2");
//   Long64_t nentries = T->GetEntries();
//   Int_t cachesize = 10000000; //10 MBytes
//   T->SetCacheSize(cachesize);     //<<<
//   T->AddBranchToCache(b1,kTRUE);  //<<<add branch1 and branch2 to the cache
//   T->AddBranchToCache(b2,kTRUE);  //<<<
//   T->StopCacheLearningPhase();    //<<<
//   for (Long64_t i=0;i<nentries;i++) {
//      T->LoadTree(i); //<<< important call when calling TBranch::GetEntry after
//      b1->GetEntry(i);
//      if (some condition not met) continue;
//      b2->GetEntry(i);
//      if (some condition not met) continue;
//      //here we read the full event only in some rare cases.
//      //there is no point in caching the other branches as it might be
//      //more economical to read only the branch buffers really used.
//      T->GetEntry(i);
//      .. process the rare but interesting cases.
//      ... here you process your entry
//   }
//--
//   --example 3b
//      in your analysis loop, you always use 2 branches in the main loop.
//      you also call some analysis functions where a few more branches will be read.
//      but you do not know a priori which ones. There is no point in prefetching 
//      branches that will be used very rarely. 
//--
//   TTree *T = (TTree*)f->Get("mytree");
//   Long64_t nentries = T->GetEntries();
//   Int_t cachesize = 10000000;   //10 MBytes
//   T->SetCacheSize(cachesize);   //<<<
//   T->SetCacheLearnEntries(5);   //<<< we can take the decision after 5 entries
//   TBranch *b1 = T->GetBranch("branch1");
//   TBranch *b2 = T->GetBranch("branch2");
//   for (Long64_t i=0;i<nentries;i++) {
//      T->LoadTree(i);
//      b1->GetEntry(i);
//      if (some condition not met) continue;
//      b2->GetEntry(i);
//      //at this point we may call a user function where a few more branches
//      //will be read conditionally. These branches will be put in the cache
//      //if they have been used in the first 10 entries
//      if (some condition not met) continue;
//      //here we read the full event only in some rare cases.
//      //there is no point in caching the other branches as it might be
//      //more economical to read only the branch buffers really used.
//      T->GetEntry(i);
//      .. process the rare but interesting cases.
//      ... here you process your entry
//   }
//--
//
//
//     SPECIAL CASES WHERE TreeCache should not be activated
//     =====================================================
//
//   When reading only a small fraction of all entries such that not all branch
//   buffers are read, it might be faster to run without a cache.
//
//
//   HOW TO VERIFY That the TreeCache has been used and check its performance
//   ========================================================================
//
//  Once your analysis loop has terminated, you can access/print the number
//  of effective system reads for a given file with a code like
//  (where TFile* f is a pointer to your file)
//
//   printf("Reading %lld bytes in %d transactions\n",f->GetBytesRead(),  f->GetReadCalls());
//
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
   fEntryCurrent(-1),
   fEntryNext(-1),
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
   fEntryCurrent(-1),
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
   // NB: if bname="*" all branches are put in the cache and the learning phase stopped
   
   TBranch *branch, *bcount;
   TLeaf *leaf, *leafcount;

   Int_t i;
   Int_t nleaves = (fOwner->GetListOfLeaves())->GetEntriesFast();
   TRegexp re(bname,kTRUE);
   Int_t nb = 0;

   // first pass, loop on all branches
   // for leafcount branches activate/deactivate in function of status
   Bool_t all = kFALSE;
   if (!strcmp(bname,"*")) all = kTRUE;
   for (i=0;i<nleaves;i++)  {
      leaf = (TLeaf*)(fOwner->GetListOfLeaves())->UncheckedAt(i);
      branch = (TBranch*)leaf->GetBranch();
      TString s = branch->GetName();
      if (!all) { //Regexp gives wrong result for [] in name
         TString longname; 
         longname.Form("%s.%s",fOwner->GetName(),branch->GetName());
         if (strcmp(bname,branch->GetName()) 
             && longname != bname
             && s.Index(re) == kNPOS) continue;
      }
      nb++;
      AddBranch(branch, subbranches);
      leafcount = leaf->GetLeafCount();
      if (leafcount && !all) {
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
            AddBranch(name, subbranches);
         }
      }
   }
   if (!nb && !foundInFriend) {
      if (gDebug > 0) printf("AddBranch: unknown branch -> %s \n", bname);
      return;
   }
   //if all branches are selected stop the learning phase
   if (*bname == '*') {
      fEntryNext = -1; // We are likely to have change the set of branches, so for the [re-]reading of the cluster.
      StopLearningPhase();
   }
}

//_____________________________________________________________________________
Bool_t TTreeCache::FillBuffer()
{
   // Fill the cache buffer with the branches in the cache.


   if (fNbranches <= 0) return kFALSE;
   TTree *tree = ((TBranch*)fBranches->UncheckedAt(0))->GetTree();
   Long64_t entry = tree->GetReadEntry();
   
   // If the entry is in the range we previously prefetched, there is 
   // no point in retrying.   Note that this will also return false
   // during the training phase (fEntryNext is then set intentional to 
   // the end of the training phase).
   if (fEntryCurrent <= entry && entry < fEntryNext) return kFALSE;
   
   // Triggered by the user, not the learning phase
   if (entry == -1)  entry = 0;

   // Estimate number of entries that can fit in the cache compare it
   // to the original value of fBufferSize not to the real one
   Long64_t autoFlush = tree->GetAutoFlush();
   if (autoFlush > 0) {
      //case when the tree autoflush has been set
      Int_t averageEntrySize = tree->GetZipBytes()/tree->GetEntries();
      if (averageEntrySize < 1) averageEntrySize = 1;
      Int_t nauto = fBufferSizeMin/(averageEntrySize*autoFlush);
      if (nauto < 1) nauto = 1;
      fEntryCurrent = entry - entry%autoFlush;
      fEntryNext = entry - entry%autoFlush + nauto*autoFlush;
   } else { 
      //case of old files before November 9 2009
      fEntryCurrent = entry;
      if (fZipBytes==0) {
         fEntryNext = entry + tree->GetEntries();;    
      } else {
         fEntryNext = entry + tree->GetEntries()*fBufferSizeMin/fZipBytes;
      }
   }
   if (fEntryCurrent < fEntryMin) fEntryCurrent = fEntryMin;
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
   for (Int_t i=0;i<fNbranches;i++) {
      TBranch *b = (TBranch*)fBranches->UncheckedAt(i);
      if (b->GetDirectory()==0) continue;
      if (b->GetDirectory()->GetFile() != fFile) continue;
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
         //important: do not try to read fEntryNext, otherwise you jump to the next autoflush
         if (entries[j] >= fEntryNext) continue;
         if (entries[j] < fEntryCurrent && (j<nb-1 && entries[j+1] <= fEntryCurrent)) continue;
         if (elist) {
            Long64_t emax = fEntryMax;
            if (j<nb-1) emax = entries[j+1]-1;
            if (!elist->ContainsRange(entries[j]+chainOffset,emax+chainOffset)) continue;
         }
         fNReadPref++;

         TFileCacheRead::Prefetch(pos,len);
      }
      if (gDebug > 0) printf("Entry: %lld, registering baskets branch %s, fEntryCurrent=%lld, fEntryNext=%lld, fNseek=%d, fNtot=%d\n",entry,((TBranch*)fBranches->UncheckedAt(i))->GetName(),fEntryCurrent,fEntryNext,fNseek,fNtot);
   }
   fIsLearning = kFALSE;
   return kTRUE;
}

//_____________________________________________________________________________
Double_t TTreeCache::GetEfficiency() const
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
Double_t TTreeCache::GetEfficiencyRel() const
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
void TTreeCache::Print(Option_t *option) const
{
   // Print cache statistics, like
   //   ******TreeCache statistics for file: cms2.root ******
   //   Number of branches in the cache ...: 1093
   //   Cache Efficiency ..................: 0.997372
   //   Cache Efficiency Rel...............: 1.000000
   //   Learn entries......................: 100
   //   Reading............................: 72761843 bytes in 7 transactions
   //   Readahead..........................: 256000 bytes with overhead = 0 bytes
   //   Average transaction................: 10394.549000 Kbytes
   //   Number of blocks in current cache..: 210, total size: 6280352
   //
   // if option = "a" the list of blocks in the cache is printed
   // see also class TTreePerfStats
   
   TString opt = option;
   opt.ToLower();
   printf("******TreeCache statistics for file: %s ******\n",fFile->GetName());
   if (fNbranches <= 0) return;
   printf("Number of branches in the cache ...: %d\n",fNbranches);
   printf("Cache Efficiency ..................: %f\n",GetEfficiency());
   printf("Cache Efficiency Rel...............: %f\n",GetEfficiencyRel());
   printf("Learn entries......................: %d\n",TTreeCache::GetLearnEntries());
   TFileCacheRead::Print(option);
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
   // don't restart it if the user has specified the branches.
   Bool_t needLearningStart = (fEntryMin != emin) && fIsLearning && !fIsManual;
   
   fEntryMin  = emin;
   fEntryMax  = emax;
   fEntryNext  = fEntryMin + fgLearnEntries;
   if (gDebug > 0)
      Info("SetEntryRange", "fEntryMin=%lld, fEntryMax=%lld, fEntryNext=%lld",
                             fEntryMin, fEntryMax, fEntryNext);

   if (needLearningStart) {
      // Restart learning
      fIsLearning = kTRUE;
      fIsManual = kFALSE;
      fNbranches  = 0;
      fZipBytes   = 0;
      if (fBrNames) fBrNames->Delete();
      fEntryCurrent = -1;
   }
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
   fEntryCurrent = -1;
}

//_____________________________________________________________________________
void TTreeCache::StopLearningPhase() 
{
   // This is the counterpart of StartLearningPhase() and can be used to stop
   // the learning phase. It's useful when the user knows exactly what branches
   // he is going to use.
   // For the moment it's just a call to FillBuffer() since that method
   // will create the buffer lists from the specified branches.
   
   if (fIsLearning) {
      // This will force FillBuffer to read the buffers.
      fEntryNext = -1;
      fIsLearning = kFALSE;
   }
   fIsManual = kTRUE;
   FillBuffer();

}

//_____________________________________________________________________________
void TTreeCache::UpdateBranches(TTree *tree, Bool_t owner)
{
   // Update pointer to current Tree and recompute pointers to the branches in the cache.

   if (owner) {
      fOwner = tree;
      SetFile(tree->GetCurrentFile());
   }
   fTree = tree;

   fEntryMin  = 0;
   fEntryMax  = fTree->GetEntries();
   
   fEntryCurrent = -1;
   
   if (fBrNames->GetEntries() == 0 && fIsLearning) {
      // We still need to learn.
      fEntryNext = fEntryMin + fgLearnEntries;
   } else {      
      // We learnt from a previous file.
      fIsLearning = kFALSE;
      fEntryNext = -1;
   }
   fZipBytes  = 0;
   fNbranches = 0;

   TIter next(fBrNames);
   TObjString *os;
   while ((os = (TObjString*)next())) {
      TBranch *b = fTree->GetBranch(os->GetName());
      if (!b) {
         continue;
      }
      fBranches->AddAt(b, fNbranches);
      fZipBytes   += b->GetZipBytes();
      fNbranches++;
   }
}
