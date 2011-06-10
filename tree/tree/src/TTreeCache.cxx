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
#include <limits.h>

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
   fIsManual(kFALSE),
   fFirstBuffer(kTRUE),
   fOneTime(kFALSE),
   fReverseRead(0),
   fFillTimes(0),
   fFirstTime(kTRUE),
   fFirstEntry(-1),
   fReadDirectionSet(kFALSE)
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
   fIsManual(kFALSE),
   fFirstBuffer(kTRUE),
   fOneTime(kFALSE),
   fReverseRead(0),
   fFillTimes(0),
   fFirstTime(kTRUE),                                                        
   fFirstEntry(-1),
   fReadDirectionSet(kFALSE)
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
   Long64_t fEntryCurrentMax = 0;
   
   if (fEnablePrefetching){ //prefetching mode
     if (fIsLearning){ //learning mode
        entry = 0;
     }
     if (fFirstTime){
        //try to detect if it is normal or reverse read
        fFirstEntry = entry;
     }
     else{
        if (fFirstEntry == entry) return kFALSE;
        //set the read direction
        if (!fReadDirectionSet) {
           if (entry < fFirstEntry){
              fReverseRead = kTRUE;
              fReadDirectionSet = kTRUE;
           }
           else if (entry > fFirstEntry){
              fReverseRead =kFALSE;
              fReadDirectionSet = kTRUE;
           }
         }
        
         if (fReverseRead){ 
            //reverse reading with prefetching
            if (fEntryCurrent >0 && entry < fEntryNext){
               //we can prefetch the next buffer
               if (entry > fEntryCurrent)
                  entry = fEntryCurrent - tree->GetAutoFlush() * fFillTimes;
               if (entry < 0) entry = 0;
            }
            else if (fEntryCurrent >= 0)
               //we are still reading from the oldest buffer, no need to prefetch a new one
               return kFALSE;
     
            if (entry < 0) return kFALSE;
            fFirstBuffer = !fFirstBuffer; 
        }
        else {
           //normal reading with prefetching
           if (fEnablePrefetching){
              if (entry < 0 && fEntryNext > 0)
                 entry = fEntryCurrent;
              else if (entry > fEntryCurrent){
                 if (entry < fEntryNext)
                    entry = fEntryNext;
              }
              else {
                //we are still reading from the oldest buffer, no need to prefetch a new one
                return kFALSE;
              }
              fFirstBuffer = !fFirstBuffer;
           }
         }
      }
   }

   // If the entry is in the range we previously prefetched, there is 
   // no point in retrying.   Note that this will also return false
   // during the training phase (fEntryNext is then set intentional to 
   // the end of the training phase).
   if (fEntryCurrent <= entry && entry < fEntryNext) return kFALSE;
   
   // Triggered by the user, not the learning phase
   if (entry == -1)  entry = 0;

   fEntryCurrentMax = fEntryCurrent;
   TTree::TClusterIterator clusterIter = tree->GetClusterIterator(entry);
   fEntryCurrent = clusterIter();
   fEntryNext = clusterIter.GetNextEntry();

   if (fEntryCurrent < fEntryMin) fEntryCurrent = fEntryMin;
   if (fEntryMax <= 0) fEntryMax = tree->GetEntries();
   if (fEntryNext > fEntryMax) fEntryNext = fEntryMax;

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
   Int_t fNtotCurrentBuf = 0;
   if (fEnablePrefetching){ //prefetching mode
      if (fFirstBuffer) {
         TFileCacheRead::Prefetch(0,0);
         fNtotCurrentBuf = fNtot;
      }
      else {
         TFileCacheRead::SecondPrefetch(0,0);
         fNtotCurrentBuf = fBNtot;
      }
   }
   else { 
      TFileCacheRead::Prefetch(0,0);
      fNtotCurrentBuf = fNtot;
   }

   //store baskets
   Int_t clusterIterations = 0;
   Long64_t minEntry = fEntryCurrent;
   Int_t prevNtot;
   Int_t minBasket = 0;  // We will use this to avoid re-checking the first baskets in the 2nd (or more) run in the while loop.
   do {
      prevNtot = fNtotCurrentBuf;
      Int_t nextMinBasket = INT_MAX;
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
         Int_t j=minBasket;  // We need this out of the loop so we can find out how far we went.
         for (;j<nb;j++) {
            // This basket has already been read, skip it
            if (j<blistsize && b->GetListOfBaskets()->UncheckedAt(j)) continue;

            Long64_t pos = b->GetBasketSeek(j);
            Int_t len = lbaskets[j];
            if (pos <= 0 || len <= 0) continue;
            //important: do not try to read fEntryNext, otherwise you jump to the next autoflush
            if (entries[j] >= fEntryNext) break; // break out of the for each branch loop.
            if (entries[j] < minEntry && (j<nb-1 && entries[j+1] <= minEntry)) continue;
            if (elist) {
               Long64_t emax = fEntryMax;
               if (j<nb-1) emax = entries[j+1]-1;
               if (!elist->ContainsRange(entries[j]+chainOffset,emax+chainOffset)) continue;
            }
            fNReadPref++;

            if (fEnablePrefetching){
               if (fFirstBuffer) {
                  TFileCacheRead::Prefetch(pos,len);
                  fNtotCurrentBuf = fNtot;
               }
               else {
                  TFileCacheRead::SecondPrefetch(pos,len);
                  fNtotCurrentBuf = fBNtot;
               }
            }
            else {
               TFileCacheRead::Prefetch(pos,len);
               fNtotCurrentBuf = fNtot;
            }
         }
         
         if (j < nextMinBasket) nextMinBasket = j;
         if (gDebug > 0) printf("Entry: %lld, registering baskets branch %s, fEntryNext=%lld, fNseek=%d, fNtotCurrentBuf=%d\n",minEntry,((TBranch*)fBranches->UncheckedAt(i))->GetName(),fEntryNext,fNseek,fNtotCurrentBuf);
      }
      clusterIterations++;
          
      minEntry = clusterIter.Next();
      if (fIsLearning) {
         fFillTimes++;
      }

      // Continue as long as we still make progress (prevNtot < fNtotCurrentBuf), that the next entry range to be looked at, 
      // which start at 'minEntry', is not past the end of the requested range (minEntry < fEntryMax)
      // and we guess that we not going to go over the requested amount of memory by asking for another set
      // of entries (fBufferSizeMin > ((Long64_t)fNtotCurrentBuf*(clusterIterations+1))/clusterIterations).
      // fNtotCurrentBuf / clusterIterations is the average size we are accumulated so far at each loop.
      // and thus (fNtotCurrentBuf / clusterIterations) * (clusterIterations+1) is a good guess at what the next total size
      // would be if we run the loop one more time.   fNtotCurrentBuf and clusterIterations are Int_t but can something
      // be 'large' (i.e. 30Mb * 300 intervals) and can overflow the numercial limit of Int_t (i.e. become
      // artificially negative).   To avoid this issue we promote fNtotCurrentBuf to a long long (64 bits rather than 32 bits) 
      if (!((fBufferSizeMin > ((Long64_t)fNtotCurrentBuf*(clusterIterations+1))/clusterIterations) && (prevNtot < fNtotCurrentBuf) && (minEntry < fEntryMax)))
         break;

      //for the reverse reading case
      if (!fIsLearning && fReverseRead){
         if (clusterIterations >= fFillTimes)
            break;
         if (minEntry >= fEntryCurrentMax && fEntryCurrentMax >0)
            break;
      }
      minBasket = nextMinBasket;
      fEntryNext = clusterIter.GetNextEntry();
      if (fEntryNext > fEntryMax) fEntryNext = fEntryMax;
   } while (kTRUE);

   //in learning mode clear cache
   if (fIsLearning) {
      fEntryNext = -1;
      fEntryCurrent = -1;
      TFileCacheRead::Prefetch(0, 0);
      TFileCacheRead::SecondPrefetch(0, 0);
      fFirstBuffer = !fFirstBuffer;
   }
   if (!fIsLearning && fFirstTime){
      // first time we add autoFlush entries , after fFillTimes * autoFlush
      // only in reverse prefetching mode
      fFirstTime = kFALSE;
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
Int_t TTreeCache::ReadBufferNormal(char *buf, Long64_t pos, Int_t len){

  //Old method ReadBuffer before the addition of the prefetch mechanism

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
Int_t TTreeCache::ReadBufferPrefetch(char *buf, Long64_t pos, Int_t len){

   // Used to read a chunk from a block previously fetched. It will call FillBuffer 
   // even if the cache lookup succeeds, because it will try to prefetch the next block 
   // as soon as we start reading from the current block.
 
   if (TFileCacheRead::ReadBuffer(buf,pos,len) == 1){
      //call FillBuffer to prefetch next block if necessary
      //(if we are currently reading from the last block available)
      FillBuffer();
      fNReadOk++;
      return 1;
   }
      
   //keep on prefetching until request is satisfied
   while (1){
     if(TFileCacheRead::ReadBuffer(buf, pos, len))
       break;
     FillBuffer();
         fNReadMiss++;
   }

   fNReadOk++;
   return 1;
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

   if (fEnablePrefetching)
      return TTreeCache::ReadBufferPrefetch(buf, pos, len);
   else
      return TTreeCache::ReadBufferNormal(buf, pos, len);
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
      StartLearningPhase();
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

   //fill the buffers only once during learning
   if (fEnablePrefetching && !fOneTime) {
      fIsLearning = kTRUE;
      FillBuffer();
      fOneTime = kTRUE;
   }
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
