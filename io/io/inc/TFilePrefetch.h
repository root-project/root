#ifndef ROOT_TFilePrefetch
#define ROOT_TFilePrefetch

#ifndef ROOT_TFile
#include "TFile.h"
#endif
#ifndef ROOT_TThread
#include "TThread.h"
#endif
#ifndef ROOT_TFPBlock
#include "TFPBlock.h"
#endif
#ifndef ROOT_TCondition
#include "TCondition.h"
#endif
#ifndef ROOT_TSemaphore
#include "TSemaphore.h"
#endif
#ifndef ROOT_TMD5
#include "TMD5.h"
#endif
#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TObjString
#include "TObjString.h"
#endif
#ifndef ROOT_TMutex
#include "TMutex.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif
#ifndef ROOT_TStopwatch
#include "TStopwatch.h"
#endif


class TFilePrefetch {

public:
  
   TFilePrefetch(TFile*);                                    //! constructor
   virtual ~TFilePrefetch();                                         //! destructor

   void ReadAsync(TFPBlock*, Bool_t&);                       //! function that does the actual reading(lowest level)
   void ReadListOfBlocks();                                  //! read all the blocks in the list prefetchBlocks

   void AddPendingBlock(TFPBlock*);                          //! method to add a block to the pending list of blocks
   TFPBlock* GetPendingBlock();                              // retrieve a block from the list of pending blocks

   void AddReadBlock(TFPBlock*);                             // method to add a block to the list of read blocks
   void AddRecycleBlock(TFPBlock*);                          // method to add a block to the list of recycled blocks
   Bool_t ReadBuffer(char*, Long64_t, Int_t);                // test if segment if prefetched and return it
   void ReadBlock(Long64_t*, Int_t*, Int_t);                 // submit a request from the main thread
   TFPBlock* CreateBlockObj(Long64_t*, Int_t*, Int_t);       // create a new block or recycle an old one

   TThread* GetThread();                                     // get reference to consumer thread
   static void ThreadProc(void*);                            // actions executed by the consumer thread
   Int_t ThreadStart();                                      // start thread

   Bool_t SetCache(char*);                                   // set the path to the cache directory
   Bool_t CheckCachePath(char*);                             // used to validate the input pathCache
   Bool_t CheckBlockInCache(char*&, TFPBlock*);              // test if block is in cache
   char* GetBlockFromCache(char*, Int_t);                    // get the block if in cache else return null
   void SaveBlockInCache(TFPBlock*);                         // save the buffer content in cache

   Int_t SumHex(char*);                                      // add the values from a hex rep. to produce an integer 
   Bool_t BinarySearchReadList(TFPBlock*, Long64_t, Int_t, Int_t*);  // search segments in a block corresponding to the current segment request
   Long64_t GetWaitTime() { return Long64_t(fWaitTime.RealTime()*1.e+6); } // return the time spent wating for buffer to be read in microseconds

private:
 
   TFile      *fFile;                                        //! reference to the file
   TList      *fPendingBlocks;                               //! list of pending block to be read
   TList      *fReadBlocks;                                  //! list of block read
   TList      *fRecycleBlocks;                               //! list of recycled blocks 
   TThread    *fConsumer;                                    //! consumer thread
   TMutex     *fMutexPendingList;                            //! mutex for the pending list
   TMutex     *fMutexReadList;                               //! mutex for the list of read blocks
   TMutex     *fMutexRecycleList;                            //! mutex for the list of recycled blocks 
   TCondition *fNewBlockAdded;                               //! condition used to signal the addition of a new pending block
   TCondition *fReadBlockAdded;                              //! condition usd to signal the addition of a new red block
   TSemaphore *fSem;                                         //! semaphore used to kill the consumer thread
   TString    *fPathCache;                                   //! path to the cache directory
   TStopwatch  fWaitTime;                                    //! time wating to prefetch a buffer (in usec)

   ClassDef(TFilePrefetch, 0);
};
#endif
