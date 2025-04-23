#include "TTree.h"
#include "TLeaf.h"
#include "TFile.h"
#include "TMath.h"

TTree *createTree(Long64_t entries, Long64_t clusterSize)
{
   static int count = 0;
   TTree *tree = new TTree(TString::Format("t%d",count),"cluster size testing");
   tree->SetAutoFlush(clusterSize);
   ++count;
   int value;
   tree->Branch("value",&value);
   for(Long64_t i = 0; i < entries; ++i) {
      value = 1000*count + i;
      tree->Fill();
   }
   tree->ResetBranchAddresses();
   tree->Write();
   return tree;
}

TTree *createVarClusterTree(Long64_t entries, Long64_t clusterSize)
{
   static int count = 0;
   TTree *tree = new TTree(TString::Format("vart%d",count),"cluster size testing");
   tree->SetAutoFlush(clusterSize);
   ++count;
   int value;
   tree->Branch("value",&value);
   const int clusterSizes [] = { 2, 5, 3 };
   int cursor = 0;
   for(Long64_t i = 0; i < entries; ++i) {
      if (i>0 && i%clusterSize==0) 
      {
         clusterSize = clusterSizes[cursor];
         if (cursor == 2) {
            cursor = 0;
         } else {
            ++cursor;
         }
         tree->SetAutoFlush(clusterSize);
      }
      value = 1000*count + i;
      tree->Fill();
   }
   tree->ResetBranchAddresses();
   tree->Write();
   return tree;
}

#if 0
class ClusterIterator 
{
private:
   TTree    *fTree;         // TTree upon which we are iterating.
   Int_t     fClusterRange; // Which cluster range are we looking at.
   Long64_t  fStartEntry;   // Where does the cluster start.
   Long64_t  fNextEntry;    // Where does the cluster end (exclusive).
   
   Long64_t GetEstimatedClusterSize()
   {
      Long64_t zipBytes = fTree->GetZipBytes();
      if (zipBytes == 0) {
         return fTree->GetEntries() - 1;
      } else {
         Long64_t clusterEstimate = 1;
         Long64_t cacheSize = fTree->GetCacheSize();
         if (cacheSize > 0) {
            clusterEstimate = fTree->GetEntries() * cacheSize / zipBytes;
            if (clusterEstimate == 0)
               clusterEstimate = 1;
         }
         return clusterEstimate;
      }      
   }
   
public:
   
   ClusterIterator(TTree *tree, Long64_t firstEntry) : fTree(tree), fClusterRange(0), fStartEntry(0), fNextEntry(0)
   {
      if ( fTree->GetAutoFlush() <= 0 ) {
         // Case of old files before November 9 2009
         fStartEntry = firstEntry;
      } else if (fTree->fNClusterRange) {
         // Find the correct cluster range.
//         for(Int_t i = 0 ; i < fTree->fNClusterRange; ++i) {
//            fprintf(stdout,"%s %d %lld %lld\n", tree->GetName(), i, tree->fClusterRangeEnd[i], tree->fClusterSize[i]);
//         }
         // Since fClusterRangeEnd contains the inclusive upper end of the range, we need to search for the
         // range that was containing the previous entry and add 1 (because BinarySearch consider the values
         // to be the inclusive start of the bucket).
         fClusterRange = TMath::BinarySearch(fTree->fNClusterRange, fTree->fClusterRangeEnd, firstEntry - 1) + 1;
//         fprintf(stdout,"Found cluster %d for %lld\n",fClusterRange,firstEntry);
         Long64_t entryInRange;
         Long64_t pedestal;
         if (fClusterRange == 0) {
            pedestal = 0;
            entryInRange = firstEntry;
         } else {
            pedestal = tree->fClusterRangeEnd[fClusterRange-1] + 1;
            entryInRange = firstEntry - pedestal;
         }
         Long64_t autoflush = tree->fClusterSize[fClusterRange];
         if (autoflush == 0) {
            autoflush = GetEstimatedClusterSize();
         }
         fStartEntry = pedestal + entryInRange - entryInRange%autoflush; 
      } else {
         fStartEntry = firstEntry - firstEntry%fTree->GetAutoFlush();
      }
      fNextEntry = fStartEntry; // Position correctly for the first call to Next()
   }
   
   // Move on to the next cluster and return the starting entry
   // of this next cluster
   Long64_t Next() {
      fStartEntry = fNextEntry;
      if ( fTree->GetAutoFlush() <= 0 ) {
         // Case of old files before November 9 2009
         Long64_t clusterEstimate = GetEstimatedClusterSize();
         fNextEntry = fStartEntry + clusterEstimate;
      } else {
         if (fClusterRange == fTree->fNClusterRange) {
            // We are looking at the last range ; its size
            // is defined by AutoFlush itself and goes to the GetEntries.
            fNextEntry += fTree->GetAutoFlush();
//            fprintf(stdout,"end of file next= %lld %d %d \n",fNextEntry,fClusterRange,fTree->fNClusterRange);
         } else {
            if (fStartEntry > fTree->fClusterRangeEnd[fClusterRange]) {
               ++fClusterRange;
            }
            if (fClusterRange == fTree->fNClusterRange) {
               // We are looking at the last range which size
               // is defined by AutoFlush itself and goes to the GetEntries.
               fNextEntry += fTree->GetAutoFlush();
//               fprintf(stdout,"end of file reached next= %lld %d\n",fNextEntry,fClusterRange);
            } else {
               Long64_t clusterSize = fTree->fClusterSize[fClusterRange];
               if (clusterSize == 0) {
                  clusterSize = GetEstimatedClusterSize();
               }
               fNextEntry += clusterSize;
//               fprintf(stdout,"reg next= %lld %d\n",fNextEntry,fClusterRange);
               if (fNextEntry > fTree->fClusterRangeEnd[fClusterRange]) {
                  fNextEntry = fTree->fClusterRangeEnd[fClusterRange] + 1;
//                  fprintf(stdout,"rewind next= %lld %d\n",fNextEntry,fClusterRange);
               }
            }
         }
      }
//      fprintf(stdout,"next= %lld %d %d \n",fNextEntry,fClusterRange,fTree->fNClusterRange);
      return fStartEntry;
   }
   // Return the start entry of the current cluster.
   Long64_t GetStartEntry() {
      return fStartEntry;
   }
   // Return the first entry of the next cluster.
   Long64_t GetNextEntry() {
      return fNextEntry;
   }
   
   Long64_t operator()() { return Next(); }

};

#endif

Long64_t fBufferSizeMin = 40;

class TTreeCacheEmul 
{
   
public:
   
};

Long64_t checkBoundary(TTree *tree, Long64_t entry) 
{
   Long64_t fEntryMax = tree->GetEntries();
   Long64_t fNReadPref = 0;
   Long64_t fEntryCurrent = 0;
   Long64_t fEntryNext = 0;
   Long64_t fEntryMin = 0;
   // Long64_t fZipBytes = tree->GetZipBytes();

   // Triggered by the user, not the learning phase
   if (entry == -1)  entry = 0;
   
   // Estimate number of entries that can fit in the cache compare it
   // to the original value of fBufferSize not to the real one
//   Long64_t autoFlush = tree->GetAutoFlush();
//   if (autoFlush > 0) {
//      //case when the tree autoflush has been set
//      Int_t averageEntrySize = tree->GetZipBytes()/tree->GetEntries();
//      if (averageEntrySize < 1) averageEntrySize = 1;
//      Int_t nauto = fBufferSizeMin/(averageEntrySize*autoFlush);
//      if (nauto < 1) nauto = 1;
//      fEntryCurrent = entry - entry%autoFlush;
//      fEntryNext = entry - entry%autoFlush + nauto*autoFlush;
//   } else {
//      // Below we increment by "autoFlush" events each iteration.
//      // Thus, autoFlush cannot be negative.
//      autoFlush = 0;
//      
//      //case of old files before November 9 2009
//      fEntryCurrent = entry;
//      if (fZipBytes==0) {
//         fEntryNext = entry + tree->GetEntries();
//      } else {
//         Long64_t clusterEstimate = tree->GetEntries()*fBufferSizeMin/fZipBytes;
//         if (clusterEstimate == 0)
//            clusterEstimate = 1;
//         fEntryNext = entry + clusterEstimate;         
//      }
//   }
//   fprintf(stdout,"expects = %lld %lld\n",fEntryCurrent,fEntryNext);

   TTree::TClusterIterator clusterIter = tree->GetClusterIterator(entry);
   fEntryCurrent = clusterIter();
   fEntryNext = clusterIter.GetNextEntry();
   
//   fprintf(stdout,"finds = %lld %lld\n",fEntryCurrent,fEntryNext);
   
   if (fEntryCurrent < fEntryMin) fEntryCurrent = fEntryMin;
   if (fEntryMax <= 0) fEntryMax = tree->GetEntries();
   if (fEntryNext > fEntryMax) fEntryNext = fEntryMax;
   
   // Check if owner has a TEventList set. If yes we optimize for this
   // Special case reading only the baskets containing entries in the
   // list.
//   TEventList *elist = fOwner->GetEventList();
//   Long64_t chainOffset = 0;
//   if (elist) {
//      if (fOwner->IsA() ==TChain::Class()) {
//         TChain *chain = (TChain*)fOwner;
//         Int_t t = chain->GetTreeNumber();
//         chainOffset = chain->GetTreeOffset()[t];
//      }
//   }
   
   Int_t flushIntervals = 0;
   Long64_t minEntry = fEntryCurrent;
   Long64_t prevNtot;
   Long64_t fNtot = 0;
   Int_t minBasket = 0;
   do {
      prevNtot = fNtot;
      TIter next(tree->GetListOfLeaves());
      TLeaf *leaf = 0;
      printf("Getting basket between [%lld,%lld[\n",minEntry,fEntryNext);
      Int_t nextMinBasket = INT_MAX;
      while( (leaf = (TLeaf*)next()) ) {
         TBranch *b = leaf->GetBranch();
         if (b->GetDirectory()==0) continue;
         // if (b->GetDirectory()->GetFile() != fFile) continue;
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
            if (entries[j] >= fEntryNext) break;
            if (entries[j] < minEntry && (j<nb-1 && entries[j+1] <= minEntry)) continue;
//            if (elist) {
//               Long64_t emax = fEntryMax;
//               if (j<nb-1) emax = entries[j+1]-1;
//               if (!elist->ContainsRange(entries[j]+chainOffset,emax+chainOffset)) continue;
//            }
            fNReadPref++;

            // This part is emulate and check we are getting the expected result.
            fNtot += len;
            Long64_t next_entry = (j+1) < nb ? entries[j+1] : b->GetEntries();
            printf("Using a basket with entry: [%lld,%lld[ pos=%lld len=%d\n",entries[j],next_entry,pos,len);
            if (entries[j] < fEntryCurrent) {
               printf("warning: reading some entries before the requested value %lld < %lld\n",entries[j],fEntryCurrent);
            }
            if (next_entry > fEntryNext) {
               printf("warning: reading some entries after the requested value %lld > %lld\n",next_entry,fEntryNext);
            }
            b->GetBasket(j);
         }
         if (j < nextMinBasket) nextMinBasket = j;
         if (gDebug > 0) printf("Entry: %lld, interv: %d, registering baskets branch %s, fEntryNext=%lld, fNtot=%lld\n",minEntry,flushIntervals,b->GetName(),fEntryNext,fNtot);
      }
      flushIntervals++;      
      minEntry = clusterIter.Next();

      if (!(/* (autoFlush > 0) && */ (fBufferSizeMin > (fNtot*(flushIntervals+1))/flushIntervals) && (prevNtot < fNtot) && (minEntry < fEntryMax))) 
      {
//         printf("Breaking out because %d || %d (%lld) || %d || %d\n",
//                !(autoFlush > 0),!(fBufferSizeMin > (fNtot*(flushIntervals+1))/flushIntervals),fNtot,!(prevNtot < fNtot),!(minEntry < fEntryMax));
         break;
      }
      
      minBasket = nextMinBasket;
      fEntryNext = clusterIter.GetNextEntry();
      if (fEntryNext > fEntryMax) fEntryNext = fEntryMax;
   } while (kTRUE);
   return fEntryNext;
}

void testIterator(TTree *tree)
{   
   TTree::TClusterIterator clusterIter = tree->GetClusterIterator(0);
   Long64_t clusterStart;
   while( (clusterStart = clusterIter()) < tree->GetEntries()) {
      printf("The cluster starts at %lld and ends at %lld\n",clusterStart,clusterIter.GetNextEntry()-1);
   }
}

void variableCluster()
{
   TFile *file = TFile::Open("variableCluster.root","RECREATE");
   
   printf("Checking the file with explicit cluster size variation\n");
   TTree *varCluster = createVarClusterTree(12,2);
   varCluster->Print("clusters");
   testIterator(varCluster);
   fBufferSizeMin  = 400;
   Long64_t next = checkBoundary(varCluster,0);
   while (next < varCluster->GetEntries()) {
      printf("1. next is %lld\n",next);
      next = checkBoundary(varCluster,next);
   }
   
   printf("Checking the normal file\n");
   TTree *first = createTree(12,10);
   testIterator(first);
   next = checkBoundary(first,0);
   while (next < first->GetEntries()) {
      printf("2. next is %lld\n",next);
      next = checkBoundary(first,next);
   }
   
   TTree *second = createTree(11,0);
   file->Write();
   
   first->CopyAddresses(second);
   first->CopyEntries(second,-1,"fast");
   
   TTree *third = createTree(23,15);
   file->Write();
   
   first->CopyAddresses(third);
   first->CopyEntries(third,-1,"fast");
   file->Write();

   // first->Scan();
   
   fBufferSizeMin = 40;
   printf("Checking the merged file\n");
   first->Print("clusterRange");
   testIterator(first);
   next = checkBoundary(first,0);
   while (next < first->GetEntries()) {
      printf("3. next is %lld\n",next);
      next = checkBoundary(first,next);
   }
   
   printf("Checking the merged trees both with cluster ranges.\n");

   TFile *dupfile = TFile::Open("variableCluster.root","READ");
   TTree *duptree; dupfile->GetObject(first->GetName(),duptree);
   first->CopyAddresses(duptree);
   first->CopyEntries(duptree,-1,"fast");
   file->Write();
   first->Print("clusters");
   testIterator(first);
   next = checkBoundary(first,0);
   while (next < first->GetEntries()) {
      printf("4. next is %lld\n",next);
      next = checkBoundary(first,next);
   }
   duptree->Print("clusters");
}
