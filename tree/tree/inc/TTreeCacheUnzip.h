// @(#)root/tree:$Id$
// Author: Rene Brun   04/06/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeCacheUnzip
#define ROOT_TTreeCacheUnzip


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeCacheUnzip                                                      //
//                                                                      //
// Specialization of TTreeCache for parallel Unzipping                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TTreeCache
#include "TTreeCache.h"
#endif

class TTree;
class TBranch;
class TThread;
class TCondition;
class TBasket;
class TMutex;

class TTreeCacheUnzip : public TTreeCache {
public:
   // We have three possibilities for the unzipping mode:
   // enable, disable and force
   enum EParUnzipMode { kEnable, kDisable, kForce };

protected:
   TMutex     *fMutexCache;

   // Members for paral. managing
   TThread    *fUnzipThread;
   Bool_t      fActiveThread;
   TCondition *fUnzipCondition;
   Bool_t      fNewTransfer;      // Used to indicate the second thread taht a new transfer is in progress
   Bool_t      fParallel;         // Indicate if we want to activate the parallelism (for this instance)

   TMutex     *fMutexBuffer;      // Mutex to protect the unzipping buffer 'fUnzipBuffer'
   TMutex     *fMutexList;        // Mutex to protect the list of inflated buffer

   Int_t       fTmpBufferSz;      //!  Size for the fTmpBuffer (default is 10KB... used to unzip a buffer)
   char       *fTmpBuffer;        //! [fTmpBufferSz] buffer of contiguous unzipped blocks

   static TTreeCacheUnzip::EParUnzipMode fgParallel;  // Indicate if we want to activate the parallelism

   // Members to keep track of the unzipping buffer
   Long64_t    fPosWrite;

   // Unzipping related member
   Int_t      *fUnzipLen;         //! [fNseek] Length of buffers to be unzipped
   Int_t      *fUnzipPos;         //! [fNseek] Position of sorted blocks in fUnzipBuffer
   Int_t       fNseekMax;         //!  fNseek can change so we need to know its max size
   Long64_t    fUnzipBufferSize;  //!  Size for the fUnzipBuffer (default is 2*fBufferSize)
   char       *fUnzipBuffer;      //! [fTotBytes] buffer of contiguous unzipped blocks
   Bool_t      fSkipZip;          //  say if we should skip the uncompression of all buffers
   static Double_t fgRelBuffSize; // This is the percentage of the TTreeCacheUnzip that will be used

   //! keep track of the buffer we are currently unzipping
   Int_t       fUnzipStart;       //! This will give uf the start index (fSeekSort)
   Int_t       fUnzipEnd;         //! Unzipped buffers go from fUnzipStart to fUnzipEnd
   Int_t       fUnzipNext;        //! From fUnzipEnd to fUnzipNext we have to buffer that will be unzipped soon

   // Members use to keep statistics
   Int_t       fNUnzip;           //! number of blocks that were unzipped
   Int_t       fNFound;           //! number of blocks that were found in the cache
   Int_t       fNMissed;          //! number of blocks that were not found in the cache and were unzipped


private:
   TTreeCacheUnzip(const TTreeCacheUnzip &);            //this class cannot be copied
   TTreeCacheUnzip& operator=(const TTreeCacheUnzip &);

   // Private methods
   void  Init();
   Int_t StartThreadUnzip();
   Int_t StopThreadUnzip();

public:
   TTreeCacheUnzip();
   TTreeCacheUnzip(TTree *tree, Int_t buffersize=0);
   virtual ~TTreeCacheUnzip();
   virtual void        AddBranch(TBranch *b, Bool_t subbranches = kFALSE);
   virtual void        AddBranch(const char *branch, Bool_t subbranches = kFALSE);
   Bool_t              FillBuffer();
   void                SetEntryRange(Long64_t emin,   Long64_t emax);
   virtual void        StopLearningPhase();
   void                UpdateBranches(TTree *tree, Bool_t owner = kFALSE);

   // Methods related to the thread
   static EParUnzipMode GetParallelUnzip();
   static Bool_t        IsParallelUnzip();
   Bool_t               IsActiveThread();
   Bool_t               IsQueueEmpty();
   Int_t                ProcessQueue();
   void                 SendSignal();
   static Int_t         SetParallelUnzip(TTreeCacheUnzip::EParUnzipMode option = TTreeCacheUnzip::kEnable);
   void                 WaitForSignal();

   // Unzipping related methods
   Int_t          GetRecordHeader(char *buf, Int_t maxbytes, Int_t &nbytes, Int_t &objlen, Int_t &keylen);
   virtual Bool_t GetSkipZip() { return fSkipZip; }
   virtual void   ResetCache();
   Int_t          GetUnzipBuffer(char **buf, Long64_t pos, Int_t len, Bool_t *free);
   void           SetUnzipBufferSize(Long64_t bufferSize);
   virtual void   SetSkipZip(Bool_t skip = kTRUE) { fSkipZip = skip; }
   Int_t          UnzipBuffer(char **dest, char *src);
   Int_t          UnzipCache();

   // Methods to get stats
   Int_t  GetNUnzip() { return fNUnzip; }
   Int_t  GetNFound() { return fNFound; }
   Int_t  GetNMissed(){ return fNMissed; }

   // static members
   static void* UnzipLoop(void *arg);
   ClassDef(TTreeCacheUnzip,0)  //Specialization of TTreeCache for parallel unzipping
};

#endif
