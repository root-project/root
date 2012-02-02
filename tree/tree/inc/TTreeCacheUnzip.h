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
// Fabrizio Furano (CERN) Aug 2009                                      //
// Core TTree-related code borrowed from the previous version           //
//  by Leandro Franco and Rene Brun                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TTreeCache
#include "TTreeCache.h"
#endif

#include <queue>

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

   // Members for paral. managing
   TThread    *fUnzipThread[10];
   Bool_t      fActiveThread;          // Used to terminate gracefully the unzippers
   TCondition *fUnzipStartCondition;   // Used to signal the threads to start.
   TCondition *fUnzipDoneCondition;    // Used to wait for an unzip tour to finish. Gives the Async feel.
   Bool_t      fParallel;              // Indicate if we want to activate the parallelism (for this instance)
   Bool_t      fAsyncReading;
   TMutex     *fMutexList;             // Mutex to protect the various lists. Used by the condvars.
   TMutex     *fIOMutex;

   Int_t       fCycle;
   static TTreeCacheUnzip::EParUnzipMode fgParallel;  // Indicate if we want to activate the parallelism

   Int_t       fLastReadPos;
   Int_t       fBlocksToGo;

   // Unzipping related members
   Int_t      *fUnzipLen;         //! [fNseek] Length of the unzipped buffers
   char      **fUnzipChunks;      //! [fNseek] Individual unzipped chunks. Their summed size is kept under control.
   Byte_t     *fUnzipStatus;      //! [fNSeek] For each blk, tells us if it's unzipped or pending
   Long64_t    fTotalUnzipBytes;  //! The total sum of the currently unzipped blks

   Int_t       fNseekMax;         //!  fNseek can change so we need to know its max size
   Long64_t    fUnzipBufferSize;  //!  Max Size for the ready unzipped blocks (default is 2*fBufferSize)

   static Double_t fgRelBuffSize; // This is the percentage of the TTreeCacheUnzip that will be used

   // Members use to keep statistics
   Int_t       fNUnzip;           //! number of blocks that were unzipped
   Int_t       fNFound;           //! number of blocks that were found in the cache
   Int_t       fNStalls;          //! number of hits which caused a stall
   Int_t       fNMissed;          //! number of blocks that were not found in the cache and were unzipped

   std::queue<Int_t>       fActiveBlks; // The blocks which are active now

private:
   TTreeCacheUnzip(const TTreeCacheUnzip &);            //this class cannot be copied
   TTreeCacheUnzip& operator=(const TTreeCacheUnzip &);

   char *fCompBuffer;
   Int_t fCompBufferSize;

   // Private methods
   void  Init();
   Int_t StartThreadUnzip(Int_t nthreads);
   Int_t StopThreadUnzip();

public:
   TTreeCacheUnzip();
   TTreeCacheUnzip(TTree *tree, Int_t buffersize=0);
   virtual ~TTreeCacheUnzip();
   virtual void        AddBranch(TBranch *b, Bool_t subbranches = kFALSE);
   virtual void        AddBranch(const char *branch, Bool_t subbranches = kFALSE);
   Bool_t              FillBuffer();
   virtual Int_t       ReadBufferExt(char *buf, Long64_t pos, Int_t len, Int_t &loc);
   void                SetEntryRange(Long64_t emin,   Long64_t emax);
   virtual void        StopLearningPhase();
   void                UpdateBranches(TTree *tree);

   // Methods related to the thread
   static EParUnzipMode GetParallelUnzip();
   static Bool_t        IsParallelUnzip();
   static Int_t         SetParallelUnzip(TTreeCacheUnzip::EParUnzipMode option = TTreeCacheUnzip::kEnable);

   Bool_t               IsActiveThread();
   Bool_t               IsQueueEmpty();

   void                 WaitUnzipStartSignal();
   void                 SendUnzipStartSignal(Bool_t broadcast);

   // Unzipping related methods
   Int_t          GetRecordHeader(char *buf, Int_t maxbytes, Int_t &nbytes, Int_t &objlen, Int_t &keylen);
   virtual void   ResetCache();
   virtual Int_t  GetUnzipBuffer(char **buf, Long64_t pos, Int_t len, Bool_t *free);
   void           SetUnzipBufferSize(Long64_t bufferSize);
   static void    SetUnzipRelBufferSize(Float_t relbufferSize);
   Int_t          UnzipBuffer(char **dest, char *src);
   Int_t          UnzipCache(Int_t &startindex, Int_t &locbuffsz, char *&locbuff);

   // Methods to get stats
   Int_t  GetNUnzip() { return fNUnzip; }
   Int_t  GetNFound() { return fNFound; }
   Int_t  GetNMissed(){ return fNMissed; }

   void Print(Option_t* option = "") const;

   // static members
   static void* UnzipLoop(void *arg);
   ClassDef(TTreeCacheUnzip,0)  //Specialization of TTreeCache for parallel unzipping
};

#endif
