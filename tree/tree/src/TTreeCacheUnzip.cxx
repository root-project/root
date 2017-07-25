// @(#)root/tree:$Id$
// Author: Leandro Franco   10/04/2008

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TTreeCacheUnzip
\ingroup tree

Specialization of TTreeCache for parallel Unzipping.

Fabrizio Furano (CERN) Aug 2009
Core TTree-related code borrowed from the previous version
 by Leandro Franco and Rene Brun

## Parallel Unzipping

TTreeCache has been specialised in order to let additional threads
free to unzip in advance its content. In this implementation we
support up to 10 threads, but right now it makes more sense to
limit their number to 1-2

The application reading data is carefully synchronized, in order to:
 - if the block it wants is not unzipped, it self-unzips it without
   waiting
 - if the block is being unzipped in parallel, it waits only
   for that unzip to finish
 - if the block has already been unzipped, it takes it

This is supposed to cancel a part of the unzipping latency, at the
expenses of cpu time.

The default parameters are the same of the prev version, i.e. 20%
of the TTreeCache cache size. To change it use
TTreeCache::SetUnzipBufferSize(Long64_t bufferSize)
where bufferSize must be passed in bytes.
*/

#include "TTreeCacheUnzip.h"
#include "TChain.h"
#include "TBranch.h"
#include "TFile.h"
#include "TEventList.h"
#include "TMutex.h"
#include "TVirtualMutex.h"
#include "TThread.h"
#include "TCondition.h"
#include "TMath.h"
#include "Bytes.h"
#include "TROOT.h"
#include "TEnv.h"

#ifdef R__USE_IMT
#include "tbb/task.h"
#include <vector>
#endif

extern "C" void R__unzip(Int_t *nin, UChar_t *bufin, Int_t *lout, char *bufout, Int_t *nout);
extern "C" int R__unzip_header(Int_t *nin, UChar_t *bufin, Int_t *lout);

TTreeCacheUnzip::EParUnzipMode TTreeCacheUnzip::fgParallel = TTreeCacheUnzip::kDisable;

// The unzip cache does not consume memory by itself, it just allocates in advance
// mem blocks which are then picked as they are by the baskets.
// Hence there is no good reason to limit it too much
Double_t TTreeCacheUnzip::fgRelBuffSize = .5;

ClassImp(TTreeCacheUnzip);

////////////////////////////////////////////////////////////////////////////////

TTreeCacheUnzip::TTreeCacheUnzip() : TTreeCache(),

   fAsyncReading(kFALSE),
   fCycle(0),
   fBlocksToGo(0),
   fUnzipLen(0),
   fUnzipChunks(0),
   fUnzipStatus(0),
   fNseekMax(0),
   fUnzipBufferSize(0),
   fNUnzip(0),
   fNFound(0),
   fNStalls(0),
   fNMissed(0)

{
   // Default Constructor.

   Init();
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TTreeCacheUnzip::TTreeCacheUnzip(TTree *tree, Int_t buffersize) : TTreeCache(tree,buffersize),
   fAsyncReading(kFALSE),
   fCycle(0),
   fBlocksToGo(0),
   fUnzipLen(0),
   fUnzipChunks(0),
   fUnzipStatus(0),
   fNseekMax(0),
   fUnzipBufferSize(0),
   fNUnzip(0),
   fNFound(0),
   fNStalls(0),
   fNMissed(0)
{
   Init();
}

////////////////////////////////////////////////////////////////////////////////
/// Initialization procedure common to all the constructors.

void TTreeCacheUnzip::Init()
{
   root = nullptr;
   fIOMutex          = new TMutex(kTRUE);

   fCompBuffer = new char[16384];
   fCompBufferSize = 16384;

   if (fgParallel == kDisable) {
      fParallel = kFALSE;
   }
   else if(fgParallel == kEnable || fgParallel == kForce) {
      SysInfo_t info;
      gSystem->GetSysInfo(&info);

      fUnzipBufferSize = Long64_t(fgRelBuffSize * GetBufferSize());

      if(gDebug > 0)
         Info("TTreeCacheUnzip", "Enabling Parallel Unzipping");

      fParallel = kTRUE;

   }
   else {
      Warning("TTreeCacheUnzip", "Parallel Option unknown");
   }

   // Check if asynchronous reading is supported by this TFile specialization
   if (gEnv->GetValue("TFile.AsyncReading", 1)) {
      if (fFile && !(fFile->ReadBufferAsync(0, 0)))
         fAsyncReading = kTRUE;
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Destructor. (in general called by the TFile destructor)

TTreeCacheUnzip::~TTreeCacheUnzip()
{
   ResetCache();

   delete [] fUnzipLen;

   delete fIOMutex;

   delete [] fUnzipStatus;
   delete [] fUnzipChunks;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a branch to the list of branches to be stored in the cache
/// this function is called by TBranch::GetBasket
/// Returns:
///  - 0 branch added or already included
///  - -1 on error

Int_t TTreeCacheUnzip::AddBranch(TBranch *b, Bool_t subbranches /*= kFALSE*/)
{
   return TTreeCache::AddBranch(b, subbranches);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a branch to the list of branches to be stored in the cache
/// this function is called by TBranch::GetBasket
/// Returns:
///  - 0 branch added or already included
///  - -1 on error

Int_t TTreeCacheUnzip::AddBranch(const char *branch, Bool_t subbranches /*= kFALSE*/)
{
   return TTreeCache::AddBranch(branch, subbranches);
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TTreeCacheUnzip::FillBuffer()
{

   if (fNbranches <= 0)
      return kFALSE;

   // Fill the cache buffer with the branches in the cache.
   fIsTransferred = kFALSE;

   TTree *tree = ((TBranch *)fBranches->UncheckedAt(0))->GetTree();
   Long64_t entry = tree->GetReadEntry();

   // If the entry is in the range we previously prefetched, there is
   // no point in retrying.   Note that this will also return false
   // during the training phase (fEntryNext is then set intentional to
   // the end of the training phase).
   if (fEntryCurrent <= entry && entry < fEntryNext)
      return kFALSE;

   // Triggered by the user, not the learning phase
   if (entry == -1)
      entry = 0;

   TTree::TClusterIterator clusterIter = tree->GetClusterIterator(entry);
   fEntryCurrent = clusterIter();
   fEntryNext = clusterIter.GetNextEntry();

   if (fEntryCurrent < fEntryMin)
      fEntryCurrent = fEntryMin;
   if (fEntryMax <= 0)
      fEntryMax = tree->GetEntries();
   if (fEntryNext > fEntryMax)
      fEntryNext = fEntryMax;

   // Check if owner has a TEventList set. If yes we optimize for this
   // Special case reading only the baskets containing entries in the
   // list.
   TEventList *elist = fTree->GetEventList();
   Long64_t chainOffset = 0;
   if (elist) {
      if (fTree->IsA() == TChain::Class()) {
         TChain *chain = (TChain *)fTree;
         Int_t t = chain->GetTreeNumber();
         chainOffset = chain->GetTreeOffset()[t];
      }
   }

   // clear cache buffer
   TFileCacheRead::Prefetch(0, 0);

   // store baskets
   for (Int_t i = 0; i < fNbranches; i++) {
      TBranch *b = (TBranch *)fBranches->UncheckedAt(i);
      if (b->GetDirectory() == 0)
         continue;
      if (b->GetDirectory()->GetFile() != fFile)
         continue;
      Int_t nb = b->GetMaxBaskets();
      Int_t *lbaskets = b->GetBasketBytes();
      Long64_t *entries = b->GetBasketEntry();
      if (!lbaskets || !entries)
         continue;
      // we have found the branch. We now register all its baskets
      // from the requested offset to the basket below fEntrymax
      Int_t blistsize = b->GetListOfBaskets()->GetSize();
      for (Int_t j = 0; j < nb; j++) {
         // This basket has already been read, skip it
         if (j < blistsize && b->GetListOfBaskets()->UncheckedAt(j))
            continue;

         Long64_t pos = b->GetBasketSeek(j);
         Int_t len = lbaskets[j];
         if (pos <= 0 || len <= 0)
            continue;
         // important: do not try to read fEntryNext, otherwise you jump to the next autoflush
         if (entries[j] >= fEntryNext)
            continue;
         if (entries[j] < entry && (j < nb - 1 && entries[j + 1] <= entry))
            continue;
         if (elist) {
            Long64_t emax = fEntryMax;
            if (j < nb - 1)
               emax = entries[j + 1] - 1;
            if (!elist->ContainsRange(entries[j] + chainOffset, emax + chainOffset))
               continue;
         }
         fNReadPref++;

         TFileCacheRead::Prefetch(pos, len);
      }
      if (gDebug > 0)
         printf("Entry: %lld, registering baskets branch %s, fEntryNext=%lld, fNseek=%d, fNtot=%d\n", entry,
                ((TBranch *)fBranches->UncheckedAt(i))->GetName(), fEntryNext, fNseek, fNtot);
   }

   // Now fix the size of the status arrays
   ResetCache();
   fIsLearning = kFALSE;
   CreateTasks();

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Change the underlying buffer size of the cache.
/// Returns:
///  - 0 if the buffer content is still available
///  - 1 if some or all of the buffer content has been made unavailable
///  - -1 on error

Int_t TTreeCacheUnzip::SetBufferSize(Int_t buffersize)
{
   Int_t res = TTreeCache::SetBufferSize(buffersize);
   if (res < 0) {
      return res;
   }
   fUnzipBufferSize = Long64_t(fgRelBuffSize * GetBufferSize());
   ResetCache();
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the minimum and maximum entry number to be processed
/// this information helps to optimize the number of baskets to read
/// when prefetching the branch buffers.

void TTreeCacheUnzip::SetEntryRange(Long64_t emin, Long64_t emax)
{
   TTreeCache::SetEntryRange(emin, emax);
}

////////////////////////////////////////////////////////////////////////////////
/// It's the same as TTreeCache::StopLearningPhase but we guarantee that
/// we start the unzipping just after getting the buffers

void TTreeCacheUnzip::StopLearningPhase()
{
   TTreeCache::StopLearningPhase();
}

////////////////////////////////////////////////////////////////////////////////
///update pointer to current Tree and recompute pointers to the branches in the cache

void TTreeCacheUnzip::UpdateBranches(TTree *tree)
{
   TTreeCache::UpdateBranches(tree);
}

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// From now on we have the methods concerning the threading part of the cache //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Static function that returns the parallel option
/// (to indicate an additional thread)

TTreeCacheUnzip::EParUnzipMode TTreeCacheUnzip::GetParallelUnzip()
{
   return fgParallel;
}

////////////////////////////////////////////////////////////////////////////////
/// Static function that tells wether the multithreading unzipping is activated

Bool_t TTreeCacheUnzip::IsParallelUnzip()
{
   if (fgParallel == kEnable || fgParallel == kForce)
      return kTRUE;

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// It says if the queue is empty... useful to see if we have to process it.

Bool_t TTreeCacheUnzip::IsQueueEmpty()
{
   if ( fIsLearning )
      return kTRUE;

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Static function that (de)activates multithreading unzipping
///
/// The possible options are:
///  - kEnable _Enable_ it, which causes an automatic detection and launches the
///    additional thread if the number of cores in the machine is greater than
///    one
///  - kDisable _Disable_ will not activate the additional thread.
///  - kForce _Force_ will start the additional thread even if there is only one
///    core. the default will be taken as kEnable.
///
/// Returns 0 if there was an error, 1 otherwise.

Int_t TTreeCacheUnzip::SetParallelUnzip(TTreeCacheUnzip::EParUnzipMode option)
{
   if(fgParallel == kEnable || fgParallel == kForce || fgParallel == kDisable) {
      fgParallel = option;
      return 1;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// From now on we have the methods concerning the unzipping part of the cache //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Read the logical record header from the buffer buf.
/// That must be the pointer tho the header part not the object by itself and
/// must contain data of at least maxbytes
/// Returns nread;
///
/// In output arguments:
///
/// -  nbytes : number of bytes in record
///             if negative, this is a deleted record
///             if 0, cannot read record, wrong value of argument first
/// -  objlen : uncompressed object size
/// -  keylen : length of logical record header
///
/// Note that the arguments objlen and keylen are returned only
/// if maxbytes >=16
/// Note: This was adapted from TFile... so some things dont apply

Int_t TTreeCacheUnzip::GetRecordHeader(char *buf, Int_t maxbytes, Int_t &nbytes, Int_t &objlen, Int_t &keylen)
{
   Version_t versionkey;
   Short_t klen;
   UInt_t datime;
   Int_t nb = 0,olen;
   Int_t nread = maxbytes;
   frombuf(buf,&nb);
   nbytes = nb;
   if (nb < 0) return nread;
   //   const Int_t headerSize = Int_t(sizeof(nb) +sizeof(versionkey) +sizeof(olen) +sizeof(datime) +sizeof(klen));
   const Int_t headerSize = 16;
   if (nread < headerSize) return nread;
   frombuf(buf, &versionkey);
   frombuf(buf, &olen);
   frombuf(buf, &datime);
   frombuf(buf, &klen);
   if (!olen) olen = nbytes-klen;
   objlen = olen;
   keylen = klen;
   return nread;
}

////////////////////////////////////////////////////////////////////////////////
/// This will delete the list of buffers that are in the unzipping cache
/// and will reset certain values in the cache.
/// This name is ambiguos because the method doesn't reset the whole cache,
/// only the part related to the unzipping
/// Note: This method is completely different from TTreeCache::ResetCache(),
/// in that method we were cleaning the prefetching buffer while here we
/// delete the information about the unzipped buffers

void TTreeCacheUnzip::ResetCache()
{
   // If root spawn unzipping tasks before, we need to wait here until all basket in current cache are unzipped.
   if (root) {
      root->wait_for_all();
      root->destroy(*root);
   }

   root = nullptr;

   if (gDebug > 0)
      Info("ResetCache", "Thread: %ld -- Resetting the cache. fNseek:%d fNSeekMax:%d", TThread::SelfId(), fNseek,
           fNseekMax);

   // Reset all the lists and wipe all the chunks
   fCycle++;
   for (Int_t i = 0; i < fNseekMax; i++) {
      if (fUnzipLen) fUnzipLen[i] = 0;
      if (fUnzipChunks) {
         if (fUnzipChunks[i]) delete [] fUnzipChunks[i];
         fUnzipChunks[i] = 0;
      }
      if (fUnzipStatus) fUnzipStatus[i] = 0;

   }

   if(fNseekMax < fNseek){
      if (gDebug > 0)
         Info("ResetCache", "Changing fNseekMax from:%d to:%d", fNseekMax, fNseek);

      std::atomic<Byte_t> *aUnzipStatus = new std::atomic<Byte_t>[ fNseek ];
      memset(aUnzipStatus, 0, fNseek * sizeof(std::atomic<Byte_t>));

      Int_t *aUnzipLen = new Int_t[fNseek];
      memset(aUnzipLen, 0, fNseek * sizeof(Int_t));

      char **aUnzipChunks = new char *[fNseek];
      memset(aUnzipChunks, 0, fNseek * sizeof(char *));

      if (fUnzipStatus) delete [] fUnzipStatus;
      if (fUnzipLen) delete [] fUnzipLen;
      if (fUnzipChunks) delete [] fUnzipChunks;

      fUnzipStatus = aUnzipStatus;
      fUnzipLen = aUnzipLen;
      fUnzipChunks = aUnzipChunks;

      fNseekMax = fNseek;
      fBlocksToGo = fNseek;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// This inflates a basket in the cache.. passing the data to a new
/// buffer that will only wait there to be read...
/// This function is responsible to update corresponding elements in
/// fUnzipStatus, fUnzipChunks and fUnzipLen. Since we use atomic variables
/// in fUnzipStatus to exclusively unzip the basket, we must update
/// fUnzipStatus after fUnzipChunks and fUnzipLen and make sure fUnzipChunks
/// and fUnzipLen are ready before main thread fetch the data.

Int_t TTreeCacheUnzip::UnzipCache(Int_t reqi, Int_t &locbuffsz, char *&locbuff)
{
   Int_t myCycle;
   const Int_t hlen = 128;
   Int_t objlen = 0, keylen = 0;
   Int_t nbytes = 0;
   Int_t readbuf = 0;

   Long64_t rdoffs = 0;
   Int_t rdlen = 0;

   // To synchronize with the 'paging'
   myCycle = fCycle;
   rdoffs = fSeek[reqi];
   rdlen = fSeekLen[reqi];

   Int_t loc = -1;
   if (!fNseek || fIsLearning) {
      return 1;
   }
   // Prepare a static tmp buf of adequate size
   if (locbuffsz < rdlen) {
      if (locbuff)
         delete[] locbuff;
      locbuffsz = rdlen;
      locbuff = new char[locbuffsz];
      // memset(locbuff, 0, locbuffsz);
   } else if (locbuffsz > rdlen * 3) {
      if (locbuff)
         delete[] locbuff;
      locbuffsz = rdlen * 2;
      locbuff = new char[locbuffsz];
      // memset(locbuff, 0, locbuffsz);
   }
   readbuf = ReadBufferExt(locbuff, rdoffs, rdlen, loc);

   if ((myCycle != fCycle) || !fIsTransferred) {
      fUnzipChunks[reqi] = 0;
      fUnzipLen[reqi] = 0;
      fUnzipStatus[reqi] = 2; // Set it as not done
      return 1;
   }

   if (readbuf <= 0) {
      fUnzipChunks[reqi] = 0;
      fUnzipLen[reqi] = 0;
      fUnzipStatus[reqi] = 2; // Set it as not done
      return -1;
   }

   GetRecordHeader(locbuff, hlen, nbytes, objlen, keylen);

   Int_t len = (objlen > nbytes - keylen) ? keylen + objlen : nbytes;
   // If the single unzipped chunk is really too big, reset it to not processable
   // I.e. mark it as done but set the pointer to 0
   // This block will be unzipped synchronously in the main thread
   if (len > 4 * fUnzipBufferSize) {

      if (gDebug > 0)
         Info("UnzipCache", "Block %d is too big, skipping.", reqi);

      fUnzipChunks[reqi] = 0;
      fUnzipLen[reqi] = 0;
      fUnzipStatus[reqi] = 2; // Set it as done

      return 0;
   }

   // Unzip it into a new blk

   char *ptr = 0;
   Int_t loclen = UnzipBuffer(&ptr, locbuff);
   if ((loclen > 0) && (loclen == objlen + keylen)) {
      if ((myCycle != fCycle) || !fIsTransferred) {
         if (ptr)
            delete[] ptr; // Previously it deletes ptr without verifying ptr. It causes double free memory. Need more
                          // examination.

         fUnzipChunks[reqi] = 0;
         fUnzipLen[reqi] = 0;
         fUnzipStatus[reqi] = 2; // Set it as not done

         return 1;
      }

      fUnzipChunks[reqi] = ptr;
      fUnzipLen[reqi] = loclen;
      fUnzipStatus[reqi] = 2; // fUnzipStatus[reqi] is updated after fUnzipChunks[reqi] and fUnzipLen[reqi] get updated
                              // because we need to synchronize the main thread in GetUnzipBuffer().

      fNUnzip++;
   } else {
      fUnzipChunks[reqi] = 0;
      fUnzipLen[reqi] = 0;
      fUnzipStatus[reqi] = 2;
   }

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// This tbb task class try unzipping a set of baskets, we use a vector to
/// store the basket indices to be unzipped by this task.

class UnzipTask : public tbb::task {

private:
   TTreeCacheUnzip *cache;
   std::vector<Int_t> indices;

public:
   UnzipTask(TTreeCacheUnzip *c, std::vector<Int_t> &ids)
   {
      cache = c;
      indices = ids;
   }

   tbb::task *execute()
   {
      for (auto ii : indices) {
         Byte_t oldV = 0;
         Byte_t newV = 1;
         if (cache->fUnzipStatus[ii].compare_exchange_weak(oldV, newV, std::memory_order_release,
                                                           std::memory_order_relaxed)) {
            Int_t locbuffsz = 16384;
            char *locbuff = new char[16384];
            Int_t res = cache->UnzipCache(ii, locbuffsz, locbuff);
            if (res)
               if (gDebug > 0)
                  Info("UnzipCache", "Unzipping failed or cache is in learning state");
         }
      }
      return nullptr;
   }
};

////////////////////////////////////////////////////////////////////////////////
/// This tbb task class allocate UnzipTasks to unzip baskets. We spawn a
/// UnzipTask to unzip a set of basksets where their accumulated size goes
/// beyond 100KB.

class MappingTask : public tbb::task {

private:
   TTreeCacheUnzip *cache;
   std::vector<Int_t> indices;

public:
   MappingTask(TTreeCacheUnzip *c)
   {
      cache = c;
      indices.clear();
   }

   tbb::task *execute()
   {
      Int_t accusz = 0;
      Int_t taskcnt = 0;
      tbb::task_list tl;

      UnzipTask *t = nullptr;
      for (Int_t ii = 0; ii < cache->fNseek; ii++) {
         while (accusz < 102400) {
            accusz += cache->fSeekLen[ii];
            indices.push_back(ii);
            ii++;
            if (ii >= cache->fNseek)
               break;
         }
         if (ii < cache->fNseek)
            ii--;
         t = new (this->allocate_child()) UnzipTask(cache, indices);
         tl.push_back(*t);
         taskcnt++;
         indices.clear();
         accusz = 0;
      }
      this->set_ref_count(taskcnt + 1);
      this->spawn_and_wait_for_all(tl);
      return nullptr;
   }
};

////////////////////////////////////////////////////////////////////////////////
/// We create a dummy task and spawn a MappingTask which spawn actual unzipping
/// tasks at background without competing with main thread.

Int_t TTreeCacheUnzip::CreateTasks()
{
   root = new (tbb::task::allocate_root()) tbb::empty_task;

   root->set_ref_count(2);
   MappingTask *t = new (root->allocate_child()) MappingTask(this);
   root->spawn(*t);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// We try to read a buffer that has already been unzipped
/// Returns -1 in case of read failure, 0 in case it's not in the
/// cache and n>0 in case read from cache (number of bytes copied).
/// pos and len are the original values as were passed to ReadBuffer
/// but instead we will return the inflated buffer.
/// Note!! : If *buf == 0 we will allocate the buffer and it will be the
/// responsability of the caller to free it... it is useful for example
/// to pass it to the creator of TBuffer

Int_t TTreeCacheUnzip::GetUnzipBuffer(char **buf, Long64_t pos, Int_t len, Bool_t *free)
{
   Int_t res = 0;
   Int_t loc = -1;

   // We go straight to TTreeCache/TfileCacheRead, in order to get the info we need pointer to the original zipped
   // chunk its index in the original unsorted offsets lists. Actually there are situations in which copying the
   // buffer is not useful. But the choice is among doing once more a small memcpy or a binary search in a large
   // array. I prefer the former. Also, here we prefer not to trigger the (re)population of the chunks in the
   // TFileCacheRead. That is better to be done in the main thread.

   Int_t myCycle = fCycle;

   if (fParallel && !fIsLearning) {

      if (fNseekMax < fNseek) {
         if (gDebug > 0)
            Info("GetUnzipBuffer", "Changing fNseekMax from:%d to:%d", fNseekMax, fNseek);

         std::atomic<Byte_t> *aUnzipStatus = new std::atomic<Byte_t>[ fNseek ];
         memset(aUnzipStatus, 0, fNseek * sizeof(std::atomic<Byte_t>));

         Int_t *aUnzipLen = new Int_t[fNseek];
         memset(aUnzipLen, 0, fNseek * sizeof(Int_t));

         char **aUnzipChunks = new char *[fNseek];
         memset(aUnzipChunks, 0, fNseek * sizeof(char *));

         for (Int_t i = 0; i < fNseekMax; i++) {
            Byte_t tmp = fUnzipStatus[i];
            aUnzipStatus[i] = tmp;
            aUnzipLen[i] = fUnzipLen[i];
            aUnzipChunks[i] = fUnzipChunks[i];
         }

         if (fUnzipStatus)
            delete[] fUnzipStatus;
         if (fUnzipLen)
            delete[] fUnzipLen;
         if (fUnzipChunks)
            delete[] fUnzipChunks;

         fUnzipStatus = aUnzipStatus;
         fUnzipLen = aUnzipLen;
         fUnzipChunks = aUnzipChunks;

         fNseekMax = fNseek;
      }

      loc = (Int_t)TMath::BinarySearch(fNseek, fSeekSort, pos);
      if ((fCycle == myCycle) && (loc >= 0) && (loc < fNseek) && (pos == fSeekSort[loc])) {

         // The buffer is, at minimum, in the file cache. We must know its index in the requests list
         // In order to get its info
         Int_t seekidx = fSeekIndex[loc];

         do {

            // If the block is ready we get it immediately.
            // And also we don't have to alloc the blks. This is supposed to be
            // the main thread of the app.
            if ((fUnzipStatus[seekidx] == 2) && (fUnzipChunks[seekidx]) && (fUnzipLen[seekidx] > 0)) {

               //if (gDebug > 0)
               //   Info("GetUnzipBuffer", "++++++++++++++++++++ CacheHIT Block wanted: %d  len:%d req_len:%d
               //   fNseek:%d", seekidx, fUnzipLen[seekidx], len,  fNseek);

               if(!(*buf)) {
                  *buf = fUnzipChunks[seekidx];
                  fUnzipChunks[seekidx] = 0;
                  *free = kTRUE;
               }
               else {
                  memcpy(*buf, fUnzipChunks[seekidx], fUnzipLen[seekidx]);
                  delete fUnzipChunks[seekidx];
                  fUnzipChunks[seekidx] = 0;
                  *free = kFALSE;
               }

               fNFound++;

               return fUnzipLen[seekidx];
            }

            // If the requested basket is being unzipped by a background task, we try to steal a blk to unzip.
            Int_t reqi = -1;

            if (fUnzipStatus[seekidx] == 1) {
               if (fBlocksToGo > 0) {
                  for (Int_t ii = 0; ii < fNseek; ++ii) {
                     if (!fUnzipStatus[ii]) {
                        Byte_t oldV = 0;
                        Byte_t newV = 1;
                        if (fUnzipStatus[ii].compare_exchange_weak(oldV, newV, std::memory_order_release,
                                                                   std::memory_order_relaxed)) {
                           reqi = ii;
                           break;
                        }
                     }
                  }
                  if (reqi < 0) {
                     fBlocksToGo = 0;
                  } else {
                     Int_t locbuffsz = 16384;
                     char *locbuff = new char[16384];
                     UnzipCache(reqi, locbuffsz, locbuff);
                  }
               }

               if (myCycle != fCycle) {
                  if (gDebug > 0)
                     Info("GetUnzipBuffer", "Sudden paging Break!!! fNseek: %d, fIsLearning:%d", fNseek, fIsLearning);

                  seekidx = -1;
                  break;
               }
            }

         } while (fUnzipStatus[seekidx] == 1);

         // if (gDebug > 0)
         //    Info("GetUnzipBuffer", "------- Block wanted: %d  status: %d len: %d chunk: %llx ", seekidx,
         //         fUnzipStatus[seekidx], fUnzipLen[seekidx], fUnzipChunks[seekidx]);

         // Here the block is not pending. It could be done or aborted or not yet being processed.
         if ((seekidx >= 0) && (fUnzipStatus[seekidx] == 2) && (fUnzipChunks[seekidx]) && (fUnzipLen[seekidx] > 0)) {

            // if (gDebug > 0)
            //    Info("GetUnzipBuffer", "++++++++++++++++++++ CacheLateHIT Block wanted: %d  len:%d fNseek:%d",
            //    seekidx, fUnzipLen[seekidx], fNseek);

            if (!(*buf)) {
               *buf = fUnzipChunks[seekidx];
               fUnzipChunks[seekidx] = 0;
               *free = kTRUE;
            } else {
               memcpy(*buf, fUnzipChunks[seekidx], fUnzipLen[seekidx]);
               delete fUnzipChunks[seekidx];
               fUnzipChunks[seekidx] = 0;
               *free = kFALSE;
            }

            fNStalls++;

            return fUnzipLen[seekidx];
         } else {
            // This is a complete miss. We want to avoid the background tasks
            // to try unzipping this block in the future.
            fUnzipStatus[seekidx] = 2;
            fUnzipChunks[seekidx] = 0;
         }

      } else {
         loc = -1;
         fIsTransferred = kFALSE;
      }

   } else {
      // We need to reset it for new transferences...
      // ResetCache();
      // TFileCacheRead::Prefetch(0,0);
   }

   if (len > fCompBufferSize) {
      delete[] fCompBuffer;
      fCompBuffer = new char[len];
      fCompBufferSize = len;
   } else {
      if (fCompBufferSize > len * 4) {
         delete[] fCompBuffer;
         fCompBuffer = new char[len * 2];
         fCompBufferSize = len * 2;
      }
   }

   {
      R__LOCKGUARD(fIOMutex);

      res = 0;
      if (!ReadBufferExt(fCompBuffer, pos, len, loc)) {
         // Info("GetUnzipBuffer", "++++++++++++++++++++ CacheMISS %d %d", loc, fNseek);
         fFile->Seek(pos);
         res = fFile->ReadBuffer(fCompBuffer, len);
      }

      if (res) res = -1;
   } // end of lock scope

   if (!res) {
      res = UnzipBuffer(buf, fCompBuffer);
      *free = kTRUE;
   }

   if (!fIsLearning) {
      fNMissed++;
   }

   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// static function: Sets the unzip relatibe buffer size

void TTreeCacheUnzip::SetUnzipRelBufferSize(Float_t relbufferSize)
{
   fgRelBuffSize = relbufferSize;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the size for the unzipping cache... by default it should be
/// two times the size of the prefetching cache

void TTreeCacheUnzip::SetUnzipBufferSize(Long64_t bufferSize)
{
   fUnzipBufferSize = bufferSize;
}

Int_t TTreeCacheUnzip::UnzipBuffer(char **dest, char *src)
{
   Int_t  uzlen = 0;
   Bool_t alloc = kFALSE;

   // Here we read the header of the buffer
   const Int_t hlen=128;
   Int_t nbytes=0, objlen=0, keylen=0;
   GetRecordHeader(src, hlen, nbytes, objlen, keylen);

   if (!(*dest)) {
      /* early consistency check */
      UChar_t *bufcur = (UChar_t *) (src + keylen);
      Int_t nin, nbuf;
      if (objlen > nbytes - keylen && R__unzip_header(&nin, bufcur, &nbuf) != 0) {
         Error("UnzipBuffer", "Inconsistency found in header (nin=%d, nbuf=%d)", nin, nbuf);
         uzlen = -1;
         return uzlen;
      }
      Int_t l = keylen + objlen;
      *dest = new char[l];
      alloc = kTRUE;
   }
   // Must unzip the buffer
   // fSeekPos[ind]; adress of zipped buffer
   // fSeekLen[ind]; len of the zipped buffer
   // &fBuffer[fSeekPos[ind]]; memory address

   // This is similar to TBasket::ReadBasketBuffers
   Bool_t oldCase = objlen==nbytes-keylen
      && ((TBranch*)fBranches->UncheckedAt(0))->GetCompressionLevel()!=0
      && fFile->GetVersion()<=30401;

   if (objlen > nbytes-keylen || oldCase) {

      // Copy the key
      memcpy(*dest, src, keylen);
      uzlen += keylen;

      char *objbuf = *dest + keylen;
      UChar_t *bufcur = (UChar_t *) (src + keylen);
      Int_t nin, nbuf;
      Int_t nout = 0;
      Int_t noutot = 0;

      while (1) {
         Int_t hc = R__unzip_header(&nin, bufcur, &nbuf);
         if (hc!=0) break;
         if (gDebug > 2)
            Info("UnzipBuffer", " nin:%d, nbuf:%d, bufcur[3] :%d, bufcur[4] :%d, bufcur[5] :%d ",
                 nin, nbuf, bufcur[3], bufcur[4], bufcur[5]);
         if (oldCase && (nin > objlen || nbuf > objlen)) {
            if (gDebug > 2)
               Info("UnzipBuffer", "oldcase objlen :%d ", objlen);

            //buffer was very likely not compressed in an old version
            memcpy( *dest + keylen, src + keylen, objlen);
            uzlen += objlen;
            return uzlen;
         }

         R__unzip(&nin, bufcur, &nbuf, objbuf, &nout);

         if (gDebug > 2)
            Info("UnzipBuffer", "R__unzip nin:%d, bufcur:%p, nbuf:%d, objbuf:%p, nout:%d",
                 nin, bufcur, nbuf, objbuf, nout);

         if (!nout) break;
         noutot += nout;
         if (noutot >= objlen) break;
         bufcur += nin;
         objbuf += nout;
      }

      if (noutot != objlen) {
         Error("UnzipBuffer", "nbytes = %d, keylen = %d, objlen = %d, noutot = %d, nout=%d, nin=%d, nbuf=%d",
               nbytes,keylen,objlen, noutot,nout,nin,nbuf);
         uzlen = -1;
         if(alloc) delete [] *dest;
         *dest = 0;
         return uzlen;
      }
      uzlen += objlen;
   } else {
      memcpy(*dest, src, keylen);
      uzlen += keylen;
      memcpy(*dest + keylen, src + keylen, objlen);
      uzlen += objlen;
   }
   return uzlen;
}

void  TTreeCacheUnzip::Print(Option_t* option) const {

   printf("******TreeCacheUnzip statistics for file: %s ******\n",fFile->GetName());
   printf("Max allowed mem for pending buffers: %lld\n", fUnzipBufferSize);
   printf("Number of blocks unzipped by threads: %d\n", fNUnzip);
   printf("Number of hits: %d\n", fNFound);
   printf("Number of stalls: %d\n", fNStalls);
   printf("Number of misses: %d\n", fNMissed);

   TTreeCache::Print(option);
}

////////////////////////////////////////////////////////////////////////////////

Int_t TTreeCacheUnzip::ReadBufferExt(char *buf, Long64_t pos, Int_t len, Int_t &loc) {
   R__LOCKGUARD(fIOMutex);
   return TTreeCache::ReadBufferExt(buf, pos, len, loc);
}
