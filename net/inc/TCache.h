// @(#)root/net:$Name:  $:$Id: TCache.h,v 1.3 2001/01/16 17:23:26 rdm Exp $
// Author: Fons Rademakers   13/01/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCache
#define ROOT_TCache


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCache                                                               //
//                                                                      //
// A caching system to speed up network I/O, i.e. when there is         //
// no operating system caching support (like the buffer cache for       //
// local disk I/O). The cache makes sure that every I/O is done with    //
// a (large) fixed length buffer thereby avoiding many small I/O's.     //
// The default page size is 512KB. The cache size is not very important //
// when writing sequentially a file, since the pages will not be        //
// reused. In that case use a small cache containing 10 to 20 pages.    //
// In case a file is used for random-access the cache size should be    //
// taken much larger to avoid re-reading pages over the network.        //
// Notice that the TTree's have their own caching mechanism (see        //
// TTree::SetMaxVirtualSize()), so when using mainly TTree's with large //
// basket buffers the cache can be kept quite small.                    //
// Currently the TCache system is used by the classes TNetFile,         //
// TRFIOFile and TWebFile.                                              //
//                                                                      //
// Extra improvement would be to run the Free() process in a separate   //
// thread. Possible flush parameters:                                   //
// nfract  25   fraction of dirty buffers above which the flush process //
//              is activated                                            //
// ndirty  500  maximum number of buffer block which may be written     //
//              during a flush                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_THashList
#include "THashList.h"
#endif

class TSortedList;
class TFile;


class TCache : public TObject {

friend class TFile;

private:
   // The TPage class describes a cache page
   class TPage : public TObject {
   friend class TCache;
   private:
      Seek_t    fOffset; // offset of page in file
      char     *fData;   // pointer to page data
      Int_t     fSize;   // size of page
   public:
      enum { kDirty = BIT(14), kLocked = BIT(15) };
      TPage(Seek_t offset, char *page, Int_t size)
         { fOffset = offset; fData = page; fSize = size; }
      ~TPage() { delete [] fData; }
      ULong_t Hash() const { return fOffset; }
      Bool_t  IsEqual(const TObject *obj) const
         { return fOffset == ((const TPage*)obj)->fOffset; }
      Bool_t  IsSortable() const { return kTRUE; }
      Int_t   Compare(const TObject *obj) const
         { return fOffset > ((const TPage*)obj)->fOffset ? 1 :
                  fOffset < ((const TPage*)obj)->fOffset ? -1 : 0; }
      Seek_t  Offset() const { return fOffset; }
      char   *Data() const { return fData; }
      Int_t   Size() const { return fSize; }
   };

   class TCacheList : public THashList {
   public:
      TCacheList(Int_t capacity = 1000) : THashList(capacity, 3) { }
      void PageUsed(TObject *page) { TList::Remove(page); TList::AddLast(page); }
   };

   TCacheList  *fCache;         // hash list containing cached pages
   TSortedList *fNew;           // list constaining new pages that have to be written to disk
   TList       *fFree;          // list containing unused pages
   TFile       *fFile;          // file for which pages are being cached
   Seek_t       fEOF;           // end of file
   ULong_t      fHighWater;     // high water mark (i.e. maximum cache size in bytes)
   ULong_t      fLowWater;      // low water mark (free pages till low water mark is reached)
   Int_t        fPageSize;      // size of cached pages
   Int_t        fLowLevel;      // low water mark is at low level percent of high
   Int_t        fDiv;           // page size divider
   Bool_t       fRecursive;     // true to prevent recusively calling ReadBuffer()

   void   SetPageSize(Int_t size);
   TPage *ReadPage(Seek_t offset);
   Int_t  WritePage(TPage *page);
   Int_t  FlushList(TList *list);
   Int_t  FlushNew();
   Int_t  Free(ULong_t upto);

public:
   enum {
      kDfltPageSize = 0x80000,    // 512KB
      kDfltLowLevel = 70          // 70% of fHighWater
   };

   TCache(Int_t maxCacheSize, TFile *file, Int_t pageSize = kDfltPageSize);
   virtual ~TCache();

   Int_t GetMaxCacheSize() const { return Int_t(fHighWater / 1024 / 1024); }
   Int_t GetActiveCacheSize() const;
   Int_t GetPageSize() const { return fPageSize; }
   Int_t GetLowLevel() const { return fLowLevel; }
   Int_t Resize(Int_t maxCacheSize);
   void  SetLowLevel(Int_t percentOfHigh);

   Int_t ReadBuffer(Seek_t offset, char *buf, Int_t len);
   Int_t WriteBuffer(Seek_t offset, const char *buf, Int_t len);
   Int_t Flush();

   ClassDef(TCache,0)  // Page cache used for remote I/O
};

#endif
