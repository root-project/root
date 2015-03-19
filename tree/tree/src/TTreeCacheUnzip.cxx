// @(#)root/tree:$Id$
// Author: Leandro Franco   10/04/2008

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeCacheUnzip                                                      //
//                                                                      //
// Specialization of TTreeCache for parallel Unzipping                  //
//                                                                      //
// Fabrizio Furano (CERN) Aug 2009                                      //
// Core TTree-related code borrowed from the previous version           //
//  by Leandro Franco and Rene Brun                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Parallel Unzipping                                                   //
//                                                                      //
// TTreeCache has been specialised in order to let additional threads   //
//  free to unzip in advance its content. In this implementation we     //
//  support up to 10 threads, but right now it makes more sense to      //
//  limit their number to 1-2                                           //
//                                                                      //
// The application reading data is carefully synchronized, in order to: //
//  - if the block it wants is not unzipped, it self-unzips it without  //
//     waiting                                                          //
//  - if the block is being unzipped in parallel, it waits only         //
//    for that unzip to finish                                          //
//  - if the block has already been unzipped, it takes it               //
//                                                                      //
// This is supposed to cancel a part of the unzipping latency, at the   //
//  expenses of cpu time.                                               //
//                                                                      //
// The default parameters are the same of the prev version, i.e. 20%    //
//  of the TTreeCache cache size. To change it use                      //
// TTreeCache::SetUnzipBufferSize(Long64_t bufferSize)                  //
// where bufferSize must be passed in bytes.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTreeCacheUnzip.h"
#include "TChain.h"
#include "TBranch.h"
#include "TFile.h"
#include "TEventList.h"
#include "TVirtualMutex.h"
#include "TThread.h"
#include "TCondition.h"
#include "TMath.h"
#include "Bytes.h"

#include "TEnv.h"

#define THREADCNT 2
extern "C" void R__unzip(Int_t *nin, UChar_t *bufin, Int_t *lout, char *bufout, Int_t *nout);
extern "C" int R__unzip_header(Int_t *nin, UChar_t *bufin, Int_t *lout);

TTreeCacheUnzip::EParUnzipMode TTreeCacheUnzip::fgParallel = TTreeCacheUnzip::kDisable;

// The unzip cache does not consume memory by itself, it just allocates in advance
// mem blocks which are then picked as they are by the baskets.
// Hence there is no good reason to limit it too much
Double_t TTreeCacheUnzip::fgRelBuffSize = .5;

ClassImp(TTreeCacheUnzip)

//______________________________________________________________________________
TTreeCacheUnzip::TTreeCacheUnzip() : TTreeCache(),

   fActiveThread(kFALSE),
   fAsyncReading(kFALSE),
   fCycle(0),
   fLastReadPos(0),
   fBlocksToGo(0),
   fUnzipLen(0),
   fUnzipChunks(0),
   fUnzipStatus(0),
   fTotalUnzipBytes(0),
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

//______________________________________________________________________________
TTreeCacheUnzip::TTreeCacheUnzip(TTree *tree, Int_t buffersize) : TTreeCache(tree,buffersize),
   fActiveThread(kFALSE),
   fAsyncReading(kFALSE),
   fCycle(0),
   fLastReadPos(0),
   fBlocksToGo(0),
   fUnzipLen(0),
   fUnzipChunks(0),
   fUnzipStatus(0),
   fTotalUnzipBytes(0),
   fNseekMax(0),
   fUnzipBufferSize(0),
   fNUnzip(0),
   fNFound(0),
   fNStalls(0),
   fNMissed(0)
{
   // Constructor.

   Init();
}

//______________________________________________________________________________
void TTreeCacheUnzip::Init()
{
   // Initialization procedure common to all the constructors

   fMutexList        = new TMutex(kTRUE);
   fIOMutex          = new TMutex(kTRUE);

   fUnzipStartCondition   = new TCondition(fMutexList);
   fUnzipDoneCondition   = new TCondition(fMutexList);

   fTotalUnzipBytes = 0;

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

      for (Int_t i = 0; i < 10; i++) fUnzipThread[i] = 0;

      StartThreadUnzip(THREADCNT);

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

//______________________________________________________________________________
TTreeCacheUnzip::~TTreeCacheUnzip()
{
   // destructor. (in general called by the TFile destructor
   // destructor. (in general called by the TFile destructor)

   ResetCache();

   if (IsActiveThread())
      StopThreadUnzip();


   delete [] fUnzipLen;

   delete fUnzipStartCondition;
   delete fUnzipDoneCondition;


   delete fMutexList;
   delete fIOMutex;

   delete [] fUnzipStatus;
   delete [] fUnzipChunks;
}

//_____________________________________________________________________________
Int_t TTreeCacheUnzip::AddBranch(TBranch *b, Bool_t subbranches /*= kFALSE*/)
{
   //add a branch to the list of branches to be stored in the cache
   //this function is called by TBranch::GetBasket
   // Returns  0 branch added or already included
   //         -1 on error
   R__LOCKGUARD(fMutexList);

   return TTreeCache::AddBranch(b, subbranches);
}

//_____________________________________________________________________________
Int_t TTreeCacheUnzip::AddBranch(const char *branch, Bool_t subbranches /*= kFALSE*/)
{
   //add a branch to the list of branches to be stored in the cache
   //this function is called by TBranch::GetBasket
   // Returns  0 branch added or already included
   //         -1 on error
   R__LOCKGUARD(fMutexList);

   return TTreeCache::AddBranch(branch, subbranches);
}

//_____________________________________________________________________________
Bool_t TTreeCacheUnzip::FillBuffer()
{

   if (fNbranches <= 0) return kFALSE;
   {
      // Fill the cache buffer with the branches in the cache.
      R__LOCKGUARD(fMutexList);
      fIsTransferred = kFALSE;

      TTree *tree = ((TBranch*)fBranches->UncheckedAt(0))->GetTree();
      Long64_t entry = tree->GetReadEntry();

      // If the entry is in the range we previously prefetched, there is
      // no point in retrying.   Note that this will also return false
      // during the training phase (fEntryNext is then set intentional to
      // the end of the training phase).
      if (fEntryCurrent <= entry  && entry < fEntryNext) return kFALSE;

      // Triggered by the user, not the learning phase
      if (entry == -1)  entry=0;

      TTree::TClusterIterator clusterIter = tree->GetClusterIterator(entry);
      fEntryCurrent = clusterIter();
      fEntryNext = clusterIter.GetNextEntry();

      if (fEntryCurrent < fEntryMin) fEntryCurrent = fEntryMin;
      if (fEntryMax <= 0) fEntryMax = tree->GetEntries();
      if (fEntryNext > fEntryMax) fEntryNext = fEntryMax;

      // Check if owner has a TEventList set. If yes we optimize for this
      // Special case reading only the baskets containing entries in the
      // list.
      TEventList *elist = fTree->GetEventList();
      Long64_t chainOffset = 0;
      if (elist) {
         if (fTree->IsA() ==TChain::Class()) {
            TChain *chain = (TChain*)fTree;
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
            if (entries[j] < entry && (j<nb-1 && entries[j+1] <= entry)) continue;
            if (elist) {
               Long64_t emax = fEntryMax;
               if (j<nb-1) emax = entries[j+1]-1;
               if (!elist->ContainsRange(entries[j]+chainOffset,emax+chainOffset)) continue;
            }
            fNReadPref++;

            TFileCacheRead::Prefetch(pos,len);
         }
         if (gDebug > 0) printf("Entry: %lld, registering baskets branch %s, fEntryNext=%lld, fNseek=%d, fNtot=%d\n",entry,((TBranch*)fBranches->UncheckedAt(i))->GetName(),fEntryNext,fNseek,fNtot);
      }


      // Now fix the size of the status arrays
      ResetCache();

      fIsLearning = kFALSE;

   }

   return kTRUE;
}


//_____________________________________________________________________________
Int_t TTreeCacheUnzip::SetBufferSize(Int_t buffersize)
{
   // Change the underlying buffer size of the cache.
   // Returns  0 if the buffer content is still available
   //          1 if some or all of the buffer content has been made unavailable
   //         -1 on error

   R__LOCKGUARD(fMutexList);

   Int_t res = TTreeCache::SetBufferSize(buffersize);
   if (res < 0) {
      return res;
   }
   fUnzipBufferSize = Long64_t(fgRelBuffSize * GetBufferSize());
   ResetCache();
   return 1;
}


//_____________________________________________________________________________
void TTreeCacheUnzip::SetEntryRange(Long64_t emin, Long64_t emax)
{
   // Set the minimum and maximum entry number to be processed
   // this information helps to optimize the number of baskets to read
   // when prefetching the branch buffers.
   R__LOCKGUARD(fMutexList);

   TTreeCache::SetEntryRange(emin, emax);
}

//_____________________________________________________________________________
void TTreeCacheUnzip::StopLearningPhase()
{
   // It's the same as TTreeCache::StopLearningPhase but we guarantee that
   // we start the unzipping just after getting the buffers
   R__LOCKGUARD(fMutexList);


   TTreeCache::StopLearningPhase();

}

//_____________________________________________________________________________
void TTreeCacheUnzip::UpdateBranches(TTree *tree)
{
   //update pointer to current Tree and recompute pointers to the branches in the cache
   R__LOCKGUARD(fMutexList);

   TTreeCache::UpdateBranches(tree);
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// From now on we have the method concerning the threading part of the cache //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________________
TTreeCacheUnzip::EParUnzipMode TTreeCacheUnzip::GetParallelUnzip()
{
   // Static function that returns the parallel option
   // (to indicate an additional thread)

   return fgParallel;
}

//_____________________________________________________________________________
Bool_t TTreeCacheUnzip::IsParallelUnzip()
{
   // Static function that tells wether the multithreading unzipping
   // is activated

   if (fgParallel == kEnable || fgParallel == kForce)
      return kTRUE;

   return kFALSE;
}

//_____________________________________________________________________________
Bool_t TTreeCacheUnzip::IsActiveThread()
{
   // This indicates if the thread is active in this moment...
   // this variable is very important because if we change it from true to
   // false the thread will stop... ( see StopThreadTreeCacheUnzip() )
   R__LOCKGUARD(fMutexList);

   return fActiveThread;
}

//_____________________________________________________________________________
Bool_t TTreeCacheUnzip::IsQueueEmpty()
{
   // It says if the queue is empty... useful to see if we have to process
   // it.
   R__LOCKGUARD(fMutexList);

   if ( fIsLearning )
      return kTRUE;

   return kFALSE;
}

void TTreeCacheUnzip::WaitUnzipStartSignal()
{
   // Here the threads sleep waiting for some blocks to unzip

   fUnzipStartCondition->TimedWaitRelative(2000);

}
//_____________________________________________________________________________
void TTreeCacheUnzip::SendUnzipStartSignal(Bool_t broadcast)
{
   // This will send the signal corresponfing to the queue... normally used
   // when we want to start processing the list of buffers.

   if (gDebug > 0) Info("SendSignal", " fUnzipCondition->Signal()");

   if (broadcast)
      fUnzipStartCondition->Broadcast();
   else
      fUnzipStartCondition->Signal();
}

//_____________________________________________________________________________
Int_t TTreeCacheUnzip::SetParallelUnzip(TTreeCacheUnzip::EParUnzipMode option)
{
   // Static function that(de)activates multithreading unzipping
   // The possible options are:
   // kEnable _Enable_ it, which causes an automatic detection and launches the
   // additional thread if the number of cores in the machine is greater than one
   // kDisable _Disable_ will not activate the additional thread.
   // kForce _Force_ will start the additional thread even if there is only one core.
   // the default will be taken as kEnable.
   // returns 0 if there was an error, 1 otherwise.

   if(fgParallel == kEnable || fgParallel == kForce || fgParallel == kDisable) {
      fgParallel = option;
      return 1;
   }
   return 0;
}


class TTreeCacheUnzipData {
public:
   TTreeCacheUnzip *fInstance;
   Int_t            fCount;
};

//_____________________________________________________________________________
Int_t TTreeCacheUnzip::StartThreadUnzip(Int_t nthreads)
{
   // The Thread is only a part of the TTreeCache but it is the part that
   // waits for info in the queue and process it... unfortunatly, a Thread is
   // not an object an we have to deal with it in the old C-Style way
   // Returns 0 if the thread was initialized or 1 if it was already running
   Int_t nt = nthreads;
   if (nt > 10) nt = 10;

   if (gDebug > 0)
      Info("StartThreadUnzip", "Going to start %d threads.", nt);

   for (Int_t i = 0; i < nt; i++) {
      if (!fUnzipThread[i]) {
         TString nm("UnzipLoop");
         nm += i;

         if (gDebug > 0)
            Info("StartThreadUnzip", "Going to start thread '%s'", nm.Data());

         TTreeCacheUnzipData *d = new TTreeCacheUnzipData;
         d->fInstance = this;
         d->fCount = i;

         fUnzipThread[i] = new TThread(nm.Data(), UnzipLoop, (void*)d);
         if (!fUnzipThread[i])
            Error("TTreeCacheUnzip::StartThreadUnzip", " Unable to create new thread.");

         fUnzipThread[i]->Run();

         // There is at least one active thread
         fActiveThread=kTRUE;

      }
   }

   return (fActiveThread == kTRUE);
}

//_____________________________________________________________________________
Int_t TTreeCacheUnzip::StopThreadUnzip()
{
   // To stop the thread we only need to change the value of the variable
   // fActiveThread to false and the loop will stop (of course, we will have)
   // to do the cleaning after that.
   // Note: The syncronization part is important here or we will try to delete
   //       teh object while it's still processing the queue
   fActiveThread = kFALSE;

   for (Int_t i = 0; i < 1; i++) {
      if(fUnzipThread[i]){

         SendUnzipStartSignal(kTRUE);

         if (fUnzipThread[i]->Exists()) {
            fUnzipThread[i]->Join();
            delete fUnzipThread[i];
         }
      }

   }

   return 1;
}



//_____________________________________________________________________________
void* TTreeCacheUnzip::UnzipLoop(void *arg)
{
   // This is a static function.
   // This is the call that will be executed in the Thread generated by
   // StartThreadTreeCacheUnzip... what we want to do is to inflate the next
   // series of buffers leaving them in the second cache.
   // Returns 0 when it finishes
   TTreeCacheUnzipData *d = (TTreeCacheUnzipData *)arg;
   TTreeCacheUnzip *unzipMng = d->fInstance;

   TThread::SetCancelOn();
   TThread::SetCancelDeferred();

   Int_t thrnum = d->fCount;
   Int_t startindex = thrnum;
   Int_t locbuffsz = 16384;
   char *locbuff = new char[16384];
   Int_t res = 0;
   Int_t myCycle = 0;

   while( unzipMng->IsActiveThread() ) {
      res = 1;

      {
         R__LOCKGUARD(unzipMng->fMutexList);
         if (myCycle != unzipMng->fCycle) startindex = thrnum;
         myCycle = unzipMng->fCycle;
         if (unzipMng->fNseek) startindex = startindex % unzipMng->fNseek;
         else startindex = -1;
      }


      if (startindex >= 0)
         res = unzipMng->UnzipCache(startindex, locbuffsz, locbuff);

      {
         R__LOCKGUARD(unzipMng->fMutexList);

         if(!unzipMng->IsActiveThread()) break;

         if ((res == 1) || (!unzipMng->fIsTransferred)) {
            unzipMng->WaitUnzipStartSignal();
            startindex = unzipMng->fLastReadPos+3+thrnum;
         }
      }


   }

   delete d;
   delete [] locbuff;
   return (void *)0;
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// From now on we have the method concerning the unzipping part of the cache //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________________
Int_t TTreeCacheUnzip::GetRecordHeader(char *buf, Int_t maxbytes, Int_t &nbytes, Int_t &objlen, Int_t &keylen)
{
   // Read the logical record header from the buffer buf.
   // That must be the pointer tho the header part not the object by itself and
   // must contain data of at least maxbytes
   // Returns nread;
   // In output arguments:
   //    nbytes : number of bytes in record
   //             if negative, this is a deleted record
   //             if 0, cannot read record, wrong value of argument first
   //    objlen : uncompressed object size
   //    keylen : length of logical record header
   // Note that the arguments objlen and keylen are returned only
   // if maxbytes >=16
   // Note: This was adapted from TFile... so some things dont apply

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

//_____________________________________________________________________________
void TTreeCacheUnzip::ResetCache()
{
   // This will delete the list of buffers that are in the unzipping cache
   // and will reset certain values in the cache.
   // This name is ambiguos because the method doesn't reset the whole cache,
   // only the part related to the unzipping
   // Note: This method is completely different from TTreeCache::ResetCache(),
   // in that method we were cleaning the prefetching buffer while here we
   // delete the information about the unzipped buffers

   {
   R__LOCKGUARD(fMutexList);

   if (gDebug > 0)
      Info("ResetCache", "Thread: %ld -- Resetting the cache. fNseek:%d fNSeekMax:%d fTotalUnzipBytes:%lld", TThread::SelfId(), fNseek, fNseekMax, fTotalUnzipBytes);

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

   while (fActiveBlks.size()) fActiveBlks.pop();

   if(fNseekMax < fNseek){
      if (gDebug > 0)
         Info("ResetCache", "Changing fNseekMax from:%d to:%d", fNseekMax, fNseek);

      Byte_t *aUnzipStatus = new Byte_t[fNseek];
      memset(aUnzipStatus, 0, fNseek*sizeof(Byte_t));

      Int_t *aUnzipLen = new Int_t[fNseek];
      memset(aUnzipLen, 0, fNseek*sizeof(Int_t));

      char **aUnzipChunks = new char *[fNseek];
      memset(aUnzipChunks, 0, fNseek*sizeof(char *));

      if (fUnzipStatus) delete [] fUnzipStatus;
      if (fUnzipLen) delete [] fUnzipLen;
      if (fUnzipChunks) delete [] fUnzipChunks;

      fUnzipStatus  = aUnzipStatus;
      fUnzipLen  = aUnzipLen;
      fUnzipChunks = aUnzipChunks;

      fNseekMax  = fNseek;
   }



   fLastReadPos = 0;
   fTotalUnzipBytes = 0;
   fBlocksToGo = fNseek;
   }


   SendUnzipStartSignal(kTRUE);


}

//_____________________________________________________________________________
Int_t TTreeCacheUnzip::GetUnzipBuffer(char **buf, Long64_t pos, Int_t len, Bool_t *free)
{
   // We try to read a buffer that has already been unzipped
   // Returns -1 in case of read failure, 0 in case it's not in the
   // cache and n>0 in case read from cache (number of bytes copied).
   // pos and len are the original values as were passed to ReadBuffer
   // but instead we will return the inflated buffer.
   // Note!! : If *buf == 0 we will allocate the buffer and it will be the
   // responsability of the caller to free it... it is useful for example
   // to pass it to the creator of TBuffer
   Int_t res = 0;
   Int_t loc = -1;

   {
      R__LOCKGUARD(fMutexList);


      // We go straight to TTreeCache/TfileCacheRead, in order to get the info we need
      //  pointer to the original zipped chunk
      //  its index in the original unsorted offsets lists
      //
      // Actually there are situations in which copying the buffer is not
      // useful. But the choice is among doing once more a small memcpy or a binary search in a large array. I prefer the former.
      // Also, here we prefer not to trigger the (re)population of the chunks in the TFileCacheRead. That is
      // better to be done in the main thread.

      // And now loc is the position of the chunk in the array of the sorted chunks
      Int_t myCycle = fCycle;


      if (fParallel && !fIsLearning) {


         if(fNseekMax < fNseek){
            if (gDebug > 0)
               Info("GetUnzipBuffer", "Changing fNseekMax from:%d to:%d", fNseekMax, fNseek);

            Byte_t *aUnzipStatus = new Byte_t[fNseek];
            memset(aUnzipStatus, 0, fNseek*sizeof(Byte_t));

            Int_t *aUnzipLen = new Int_t[fNseek];
            memset(aUnzipLen, 0, fNseek*sizeof(Int_t));

            char **aUnzipChunks = new char *[fNseek];
            memset(aUnzipChunks, 0, fNseek*sizeof(char *));

            for (Int_t i = 0; i < fNseekMax; i++) {
               aUnzipStatus[i] = fUnzipStatus[i];
               aUnzipLen[i] = fUnzipLen[i];
               aUnzipChunks[i] = fUnzipChunks[i];
            }

            if (fUnzipStatus) delete [] fUnzipStatus;
            if (fUnzipLen) delete [] fUnzipLen;
            if (fUnzipChunks) delete [] fUnzipChunks;

            fUnzipStatus  = aUnzipStatus;
            fUnzipLen  = aUnzipLen;
            fUnzipChunks = aUnzipChunks;

            fNseekMax  = fNseek;
         }







         loc = (Int_t)TMath::BinarySearch(fNseek,fSeekSort,pos);
         if ( (fCycle == myCycle) && (loc >= 0) && (loc < fNseek) && (pos == fSeekSort[loc]) ) {

            // The buffer is, at minimum, in the file cache. We must know its index in the requests list
            // In order to get its info
            Int_t seekidx = fSeekIndex[loc];

            fLastReadPos = seekidx;

            do {




               // If the block is ready we get it immediately.
               // And also we don't have to alloc the blks. This is supposed to be
               // the main thread of the app.
               if ((fUnzipStatus[seekidx] == 2) && (fUnzipChunks[seekidx]) && (fUnzipLen[seekidx] > 0)) {

                  //if (gDebug > 0)
                  //   Info("GetUnzipBuffer", "++++++++++++++++++++ CacheHIT Block wanted: %d  len:%d req_len:%d fNseek:%d", seekidx, fUnzipLen[seekidx], len,  fNseek);

                  if(!(*buf)) {
                     *buf = fUnzipChunks[seekidx];
                     fUnzipChunks[seekidx] = 0;
                     fTotalUnzipBytes -= fUnzipLen[seekidx];
                     SendUnzipStartSignal(kFALSE);
                     *free = kTRUE;
                  }
                  else {
                     memcpy(*buf, fUnzipChunks[seekidx], fUnzipLen[seekidx]);
                     delete fUnzipChunks[seekidx];
                     fTotalUnzipBytes -= fUnzipLen[seekidx];
                     fUnzipChunks[seekidx] = 0;
                     SendUnzipStartSignal(kFALSE);
                     *free = kFALSE;
                  }

                  fNFound++;

                  return fUnzipLen[seekidx];
               }

               // If the status of the unzipped chunk is pending
               // we wait on the condvar, hoping that the next signal is the good one
               if ( fUnzipStatus[seekidx] == 1 ) {
                  //fMutexList->UnLock();
                  fUnzipDoneCondition->TimedWaitRelative(200);
                  //fMutexList->Lock();

                  if ( myCycle != fCycle ) {
                     if (gDebug > 0)
                        Info("GetUnzipBuffer", "Sudden paging Break!!! IsActiveThread(): %d, fNseek: %d, fIsLearning:%d",
                             IsActiveThread(), fNseek, fIsLearning);

                     fLastReadPos = 0;

                     seekidx = -1;
                     break;
                  }

               }

            } while ( fUnzipStatus[seekidx] == 1 );

            //if (gDebug > 0)
            //   Info("GetUnzipBuffer", "------- Block wanted: %d  status: %d len: %d chunk: %llx ", seekidx, fUnzipStatus[seekidx], fUnzipLen[seekidx], fUnzipChunks[seekidx]);

            // Here the block is not pending. It could be done or aborted or not yet being processed.
            if ( (seekidx >= 0) && (fUnzipStatus[seekidx] == 2) && (fUnzipChunks[seekidx]) && (fUnzipLen[seekidx] > 0) ) {

               //if (gDebug > 0)
               //   Info("GetUnzipBuffer", "++++++++++++++++++++ CacheLateHIT Block wanted: %d  len:%d fNseek:%d", seekidx, fUnzipLen[seekidx], fNseek);

               if(!(*buf)) {
                  *buf = fUnzipChunks[seekidx];
                  fUnzipChunks[seekidx] = 0;
                  fTotalUnzipBytes -= fUnzipLen[seekidx];
                  SendUnzipStartSignal(kFALSE);
                  *free = kTRUE;
               }
               else {
                  memcpy(*buf, fUnzipChunks[seekidx], fUnzipLen[seekidx]);
                  delete fUnzipChunks[seekidx];
                  fTotalUnzipBytes -= fUnzipLen[seekidx];
                  fUnzipChunks[seekidx] = 0;
                  SendUnzipStartSignal(kFALSE);
                  *free = kFALSE;
               }


               fNStalls++;

               return fUnzipLen[seekidx];
            }
            else {
               // This is a complete miss. We want to avoid the threads
               // to try unzipping this block in the future.
               fUnzipStatus[seekidx] = 2;
               fUnzipChunks[seekidx] = 0;

               if ((fTotalUnzipBytes < fUnzipBufferSize) && fBlocksToGo)
                  SendUnzipStartSignal(kFALSE);

               //if (gDebug > 0)
               //   Info("GetUnzipBuffer", "++++++++++++++++++++ CacheMISS Block wanted: %d  len:%d fNseek:%d", seekidx, len, fNseek);
            }



         } else {
            loc = -1;
            //fLastReadPos = 0;
            fIsTransferred = kFALSE;
         }

      } else {
         // We need to reset it for new transferences...
         //ResetCache();
         //TFileCacheRead::Prefetch(0,0);
      }

   } // scope of the lock!

   if (len > fCompBufferSize) {
      delete [] fCompBuffer;
      fCompBuffer = new char[len];
      fCompBufferSize = len;
   } else {
      if (fCompBufferSize > len*4) {
      delete [] fCompBuffer;
      fCompBuffer = new char[len*2];
      fCompBufferSize = len*2;
      }
   }


   {
      R__LOCKGUARD(fIOMutex);
      // Here we know that the async unzip of the wanted chunk
      // was not done for some reason. We continue.

      res = 0;
      if (!ReadBufferExt(fCompBuffer, pos, len, loc)) {
         //Info("GetUnzipBuffer", "++++++++++++++++++++ CacheMISS %d %d", loc, fNseek);
         fFile->Seek(pos);
         res = fFile->ReadBuffer(fCompBuffer, len);
      }


      if (res) res = -1;

   } // scope of the lock!

   if (!res) {
      res = UnzipBuffer(buf, fCompBuffer);
      *free = kTRUE;
   }


   if (!fIsLearning) {
      fNMissed++;
   }

   return res;

}


//_____________________________________________________________________________
void TTreeCacheUnzip::SetUnzipRelBufferSize(Float_t relbufferSize)
{
   // static function: Sets the unzip relatibe buffer size
   // FABRIZIO: PLEASE DOCUMENT and also in TTree::Set...

   fgRelBuffSize = relbufferSize;
}


//_____________________________________________________________________________
void TTreeCacheUnzip::SetUnzipBufferSize(Long64_t bufferSize)
{
   // Sets the size for the unzipping cache... by default it should be
   // two times the size of the prefetching cache
   R__LOCKGUARD(fMutexList);

   fUnzipBufferSize = bufferSize;
}

//_____________________________________________________________________________
Int_t TTreeCacheUnzip::UnzipBuffer(char **dest, char *src)
{
   // UNzips a ROOT specific buffer... by reading the header at the beginning.
   // returns the size of the inflated buffer or -1 if error
   // Note!! : If *dest == 0 we will allocate the buffer and it will be the
   // responsability of the caller to free it... it is useful for example
   // to pass it to the creator of TBuffer
   // src is the original buffer with the record (header+compressed data)
   // *dest is the inflated buffer (including the header)
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
      if(R__unzip_header(&nin, bufcur, &nbuf)!=0) {
         Error("UnzipBuffer", "Inconsistency found in header (nin=%d, nbuf=%d)", nin, nbuf);
         uzlen = -1;
         return uzlen;
      }
      Int_t l = keylen+objlen;
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

//_____________________________________________________________________________
Int_t TTreeCacheUnzip::UnzipCache(Int_t &startindex, Int_t &locbuffsz, char *&locbuff)
{
   // This inflates all the buffers in the cache.. passing the data to a new
   // buffer that will only wait there to be read...
   // We can not inflate all the buffers in the cache so we will try to do
   // it until the cache gets full... there is a member called fUnzipBufferSize which will
   // tell us the max size we can allocate for this cache.
   //
   // note that we will  unzip in the order they were put into the cache not
   // the order of the transference so it has to be read in that order or the
   // pre-unzipping will be useless.
   //
   // startindex is used as start index to check for blks to be unzipped
   //
   // returns 0 in normal conditions or -1 if error, 1 if it would like to sleep
   //
   // This func is supposed to compete among an indefinite number of threads to get a chunk to inflate
   // in order to accommodate multiple unzippers
   // Since everything is so async, we cannot use a fixed buffer, we are forced to keep
   // the individual chunks as separate blocks, whose summed size does not exceed the maximum
   // allowed. The pointers are kept globally in the array fUnzipChunks
   Int_t myCycle;
   const Int_t hlen=128;
   Int_t objlen=0, keylen=0;
   Int_t nbytes=0;
   Int_t readbuf = 0;

   Int_t idxtounzip = -1;
   Long64_t rdoffs = 0;
   Int_t rdlen = 0;
   {
      R__LOCKGUARD(fMutexList);

      if (!IsActiveThread() || !fNseek || fIsLearning || !fIsTransferred) {
         if (gDebug > 0)
            Info("UnzipCache", "Sudden Break!!! IsActiveThread(): %d, fNseek: %d, fIsLearning:%d",
                 IsActiveThread(), fNseek, fIsLearning);
         return 1;
      }

      // To synchronize with the 'paging'
      myCycle = fCycle;

      // Try to look for a blk to unzip
      idxtounzip = -1;
      rdoffs = 0;
      rdlen = 0;
      if (fTotalUnzipBytes < fUnzipBufferSize) {


         if (fBlocksToGo > 0) {
            for (Int_t ii=0; ii < fNseek; ii++) {
               Int_t reqi = (startindex+ii) % fNseek;
               if (!fUnzipStatus[reqi] && (fSeekLen[reqi] > 256)   ) {
                  // We found a chunk which is not unzipped nor pending
                  fUnzipStatus[reqi] = 1; // Set it as pending
                  idxtounzip = reqi;

                  rdoffs = fSeek[idxtounzip];
                  rdlen = fSeekLen[idxtounzip];
                  break;
               }
            }
            if (idxtounzip < 0) fBlocksToGo = 0;
         }
      }

   } // lock scope



   if (idxtounzip < 0) {
      if (gDebug > 0)
         Info("UnzipCache", "Nothing to do... startindex:%d fTotalUnzipBytes:%lld fUnzipBufferSize:%lld fNseek:%d",
              startindex, fTotalUnzipBytes, fUnzipBufferSize, fNseek );
      return 1;
   }


   // And here we have a new blk to unzip
   startindex = idxtounzip+THREADCNT;


   if (!IsActiveThread() || !fNseek || fIsLearning ) {
      if (gDebug > 0)
         Info("UnzipCache", "Sudden Break!!! IsActiveThread(): %d, fNseek: %d, fIsLearning:%d",
              IsActiveThread(), fNseek, fIsLearning);
      return 1;
   }

   Int_t loc = -1;

   // Prepare a static tmp buf of adequate size
   if(locbuffsz < rdlen) {
      if (locbuff) delete [] locbuff;
      locbuffsz = rdlen;
      locbuff = new char[locbuffsz];
      //memset(locbuff, 0, locbuffsz);
   } else
      if(locbuffsz > rdlen*3) {
         if (locbuff) delete [] locbuff;
         locbuffsz = rdlen*2;
         locbuff = new char[locbuffsz];
         //memset(locbuff, 0, locbuffsz);
      }


      if (gDebug > 0)
      Info("UnzipCache", "Going to unzip block %d", idxtounzip);

      readbuf = ReadBufferExt(locbuff, rdoffs, rdlen, loc);

   {
      R__LOCKGUARD(fMutexList);

      if ( (myCycle != fCycle) || !fIsTransferred )  {
         if (gDebug > 0)
            Info("UnzipCache", "Sudden paging Break!!! IsActiveThread(): %d, fNseek: %d, fIsLearning:%d",
                 IsActiveThread(), fNseek, fIsLearning);

         fUnzipStatus[idxtounzip] = 2; // Set it as not done
         fUnzipChunks[idxtounzip] = 0;
         fUnzipLen[idxtounzip] = 0;
         fUnzipDoneCondition->Signal();

         startindex = 0;
         return 1;
      }


      if (readbuf <= 0) {
         fUnzipStatus[idxtounzip] = 2; // Set it as not done
         fUnzipChunks[idxtounzip] = 0;
         fUnzipLen[idxtounzip] = 0;
         if (gDebug > 0)
            Info("UnzipCache", "Block %d not done. rdoffs=%lld rdlen=%d readbuf=%d", idxtounzip, rdoffs, rdlen, readbuf);
         return -1;
      }


      GetRecordHeader(locbuff, hlen, nbytes, objlen, keylen);

      Int_t len = (objlen > nbytes-keylen)? keylen+objlen : nbytes;

      // If the single unzipped chunk is really too big, reset it to not processable
      // I.e. mark it as done but set the pointer to 0
      // This block will be unzipped synchronously in the main thread
      if (len > 4*fUnzipBufferSize) {

         //if (gDebug > 0)
            Info("UnzipCache", "Block %d is too big, skipping.", idxtounzip);

         fUnzipStatus[idxtounzip] = 2; // Set it as done
         fUnzipChunks[idxtounzip] = 0;
         fUnzipLen[idxtounzip] = 0;

         fUnzipDoneCondition->Signal();
         return 0;
      }

   } // Scope of the lock

   // Unzip it into a new blk
   char *ptr = 0;
   Int_t loclen = 0;

   loclen = UnzipBuffer(&ptr, locbuff);

   if ((loclen > 0) && (loclen == objlen+keylen)) {
      R__LOCKGUARD(fMutexList);

      if ( (myCycle != fCycle)  || !fIsTransferred) {
         if (gDebug > 0)
            Info("UnzipCache", "Sudden paging Break!!! IsActiveThread(): %d, fNseek: %d, fIsLearning:%d",
                 IsActiveThread(), fNseek, fIsLearning);
         delete [] ptr;

         fUnzipStatus[idxtounzip] = 2; // Set it as not done
         fUnzipChunks[idxtounzip] = 0;
         fUnzipLen[idxtounzip] = 0;

         startindex = 0;
         fUnzipDoneCondition->Signal();
         return 1;
      }

      fUnzipStatus[idxtounzip] = 2; // Set it as done
      fUnzipChunks[idxtounzip] = ptr;
      fUnzipLen[idxtounzip] = loclen;
      fTotalUnzipBytes += loclen;

      fActiveBlks.push(idxtounzip);

      if (gDebug > 0)
         Info("UnzipCache", "reqi:%d, rdoffs:%lld, rdlen: %d, loclen:%d",
              idxtounzip, rdoffs, rdlen, loclen);

      fNUnzip++;
   }
   else {
      R__LOCKGUARD(fMutexList);
      Info("argh", "loclen:%d objlen:%d loc:%d readbuf:%d", loclen, objlen, loc, readbuf);
      fUnzipStatus[idxtounzip] = 2; // Set it as done
      fUnzipChunks[idxtounzip] = 0;
      fUnzipLen[idxtounzip] = 0;
   }


   fUnzipDoneCondition->Signal();

   delete [] ptr;
   return 0;
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


//_____________________________________________________________________________
Int_t TTreeCacheUnzip::ReadBufferExt(char *buf, Long64_t pos, Int_t len, Int_t &loc) {

   R__LOCKGUARD(fIOMutex);
   return TTreeCache::ReadBufferExt(buf, pos, len, loc);

}
