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

class TFileCacheRead : public TObject {

protected:
   Int_t         fBufferSizeMin;  //Original size of fBuffer
   Int_t         fBufferSize;     //Allocated size of fBuffer (at a given time)
   Int_t         fBufferLen;      //Current buffer length (<= fBufferSize)

   Bool_t        fAsyncReading;

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
   virtual Bool_t      IsAsyncReading() const { return fAsyncReading; };
   virtual Bool_t      IsLearning() const {return kFALSE;}
   virtual void        Prefetch(Long64_t pos, Int_t len);
   virtual void        Print(Option_t *option="") const;
   virtual Int_t       ReadBufferExt(char *buf, Long64_t pos, Int_t len, Int_t &loc);
   virtual Int_t       ReadBuffer(char *buf, Long64_t pos, Int_t len);
   virtual void        SetFile(TFile *file);
   virtual void        SetSkipZip(Bool_t /*skip*/ = kTRUE) {} // This function is only used by TTreeCacheUnzip (ignore it)
   virtual void        Sort();

   ClassDef(TFileCacheRead,1)  //TFile cache when reading
};

#endif
