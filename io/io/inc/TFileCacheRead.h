// @(#)root/io:$Id$
// Author: Rene Brun   19/05/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFileCacheRead
#define ROOT_TFileCacheRead


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileCacheRead                                                       //
//                                                                      //
// TFile cache when reading                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TFile;
class TBranch;
class TFilePrefetch;

class TFileCacheRead : public TObject {

protected:
   TFilePrefetch* fPrefetch;       //!Object that does the asynchronous reading in another thread
   Int_t         fBufferSizeMin;  //Original size of fBuffer
   Int_t         fBufferSize;     //Allocated size of fBuffer (at a given time)
   Int_t         fBufferLen;      //Current buffer length (<= fBufferSize)

   Bool_t        fAsyncReading;
   Bool_t        fEnablePrefetching; //reading by prefetching asynchronously 

   Int_t         fNseek;          //Number of blocks to be prefetched
   Int_t         fNtot;           //Total size of prefetched blocks
   Int_t         fNb;             //Number of long buffers
   Int_t         fSeekSize;       //Allocated size of fSeek
   Long64_t     *fSeek;           //[fNseek] Position on file of buffers to be prefetched
   Long64_t     *fSeekSort;       //[fNseek] Position on file of buffers to be prefetched (sorted)
   Int_t        *fSeekIndex;      //[fNseek] sorted index table of fSeek
   Long64_t     *fPos;            //[fNb] start of long buffers
   Int_t        *fSeekLen;        //[fNseek] Length of buffers to be prefetched
   Int_t        *fSeekSortLen;    //[fNseek] Length of buffers to be prefetched (sorted)
   Int_t        *fSeekPos;        //[fNseek] Position of sorted blocks in fBuffer
   Int_t        *fLen;            //[fNb] Length of long buffers
   TFile        *fFile;           //Pointer to file
   char         *fBuffer;         //[fBufferSize] buffer of contiguous prefetched blocks
   Bool_t        fIsSorted;       //True if fSeek array is sorted
   Bool_t        fIsTransferred;   //True when fBuffer contains something valid
   Long64_t      fPrefetchedBlocks; // Number of blocks prefetched.

   //varibles for the second block prefetched with the same semantics as for the first one
   Int_t         fBNseek;
   Int_t         fBNtot;
   Int_t         fBNb;
   Int_t         fBSeekSize;
   Long64_t     *fBSeek;
   Long64_t     *fBSeekSort;
   Int_t        *fBSeekIndex;
   Long64_t     *fBPos;
   Int_t        *fBSeekLen;
   Int_t        *fBSeekSortLen;
   Int_t        *fBSeekPos;
   Int_t        *fBLen;
   Bool_t        fBIsSorted;
   Bool_t        fBIsTransferred;

private:
   TFileCacheRead(const TFileCacheRead &);            //cannot be copied
   TFileCacheRead& operator=(const TFileCacheRead &);

public:
   TFileCacheRead();
   TFileCacheRead(TFile *file, Int_t buffersize);
   virtual ~TFileCacheRead();
   virtual void        AddBranch(TBranch * /*b*/, Bool_t /*subbranches*/ = kFALSE) {}
   virtual void        AddBranch(const char * /*branch*/, Bool_t /*subbranches*/ = kFALSE) {}
   virtual Int_t       GetBufferSize() const { return fBufferSize; };
   virtual Int_t       GetUnzipBuffer(char ** /*buf*/, Long64_t /*pos*/, Int_t /*len*/, Bool_t * /*free*/) { return -1; }
           Long64_t    GetPrefetchedBlocks() const { return fPrefetchedBlocks; }
   virtual Bool_t      IsAsyncReading() const { return fAsyncReading; };
   virtual void        SetEnablePrefetching(Bool_t setPrefetching = kFALSE) { fEnablePrefetching = setPrefetching; }
   virtual Bool_t      IsEnablePrefetching() const { return fEnablePrefetching; };
   virtual Bool_t      IsLearning() const {return kFALSE;}
   virtual void        Prefetch(Long64_t pos, Int_t len);
   virtual void        Print(Option_t *option="") const;
   virtual Int_t       ReadBufferExt(char *buf, Long64_t pos, Int_t len, Int_t &loc);
   virtual Int_t       ReadBufferExtNormal(char *buf, Long64_t pos, Int_t len, Int_t &loc);
   virtual Int_t       ReadBufferExtPrefetch(char *buf, Long64_t pos, Int_t len, Int_t &loc);
   virtual Int_t       ReadBuffer(char *buf, Long64_t pos, Int_t len);
   virtual void        SetFile(TFile *file);
   virtual void        SetSkipZip(Bool_t /*skip*/ = kTRUE) {} // This function is only used by TTreeCacheUnzip (ignore it)
   virtual void        Sort();
   virtual void        SecondSort();                          //Method used to sort and merge the chunks in the second block
   virtual void        SecondPrefetch(Long64_t, Int_t);       //Used to add chunks to the second block
   virtual TFilePrefetch* GetPrefetchObj();

   ClassDef(TFileCacheRead,1)  //TFile cache when reading
};

#endif
