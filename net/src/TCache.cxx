// @(#)root/net:$Name:$:$Id:$
// Author: Fons Rademakers   13/01/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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


#include "TCache.h"
#include "TSortedList.h"
#include "TFile.h"


ClassImp(TCache)

//______________________________________________________________________________
TCache::TCache(Int_t maxCacheSize, TFile *file, Int_t pageSize)
{
   // Create a file cache. Mainly useful for remote files like TNetFile,
   // TWebFile and TRFIOFile where the local OS is not doing any caching.
   // The maxCacheSize is in MBytes and the pageSize is in bytes (default
   // being kDfltPageSize).

   fFile      = file;
   fHighWater = maxCacheSize * 1024 * 1024;
   fLowWater  = ULong_t(fHighWater * kDfltLowWater / 100);
   fRecursive = kFALSE;
   SetPageSize(pageSize);

   fCache = new TCacheList(Int_t(fHighWater/fPageSize/3));
   fNew   = new TSortedList;
   fFree  = new TList;
   fCache->SetOwner();
   fFree->SetOwner();
}

//______________________________________________________________________________
TCache::~TCache()
{
   // Clean up cache.

   delete fCache;
   delete fNew;
   delete fFree;
}

//______________________________________________________________________________
void TCache::SetPageSize(Int_t size)
{
   // Make sure the page size is a power of two.

   for (int i = 0; i < int(sizeof(size)*8); i++)
      if ((size >> i) == 1) {
         fDiv = i;
         break;
      }

   fPageSize = 1 << fDiv;
}

//______________________________________________________________________________
TCache::TPage *TCache::ReadPage(Seek_t offset)
{
   // Read the page starting at offset in the cache and return the
   // page object. If there are no more free pages free pages up
   // to the low water mark. The freed pages are the least recently
   // used pages (lru) which are at the head of the fCache hash list.
   // Returns 0 in case of error.

   fRecursive = kTRUE;

   // check if there is a free page object in the free list and use that
   if (fFree->GetSize() > 0) {
      TPage *p = (TPage*) fFree->First();
      fFile->Seek(offset);
      Int_t len = offset + fPageSize > fFile->GetEND() ?
                  Int_t(fFile->GetEND() - offset) : fPageSize;
      if (len < 0) len = 0;
      if (len && fFile->ReadBuffer(p->Data(), len)) {
         fRecursive = kFALSE;
         return 0;
      }
      p->fOffset = offset;
      p->fSize   = len;
      fFree->Remove(p);
      fCache->Add(p);
      fRecursive = kFALSE;
      return p;
   }

   // if cache is not full, create new page and use it
   if (ULong_t(fCache->GetSize() * fPageSize) < fHighWater) {
      char *data = new char[fPageSize];
      fFile->Seek(offset);
      Int_t len = offset + fPageSize > fFile->GetEND() ?
                  Int_t(fFile->GetEND() - offset) : fPageSize;
      if (len < 0) len = 0;
      if (len && fFile->ReadBuffer(data, len)) {
         fRecursive = kFALSE;
         return 0;
      }
      TPage *p = new TPage(offset, data, len);
      fCache->Add(p);
      fRecursive = kFALSE;
      return p;
   }

   // if we come here there are no free pages and the cache is full, free
   // pages up to low water mark
   if (Free(fLowWater) < 0) {
      fRecursive = kFALSE;
      return 0;
   }

   return ReadPage(offset);
}

//______________________________________________________________________________
Int_t TCache::ReadBuffer(Seek_t offset, char *buf, Int_t len)
{
   // Return in buf len bytes starting at offset. Returns < 0 in
   // case of error, 0 in case ReadBuffer() was recursively called
   // via ReadPage() and 1 in case of success.

   if (fRecursive) return 0;

   // Find in which page offset is located
   Seek_t pageoffset = (offset >> fDiv) << fDiv;  // offset & ~(fPageSize-1)
   Int_t  begin = Int_t(offset & (fPageSize-1));
   Seek_t boff  = 0;

   do {
      Int_t blen = begin+len>fPageSize ? fPageSize-begin : len;
      TPage t(pageoffset, 0, 0);
      TPage *p = (TPage*) fCache->FindObject(&t);
      if (p) {
         // found page in cache, copy to buf
         memcpy(buf+boff, p->Data()+begin, blen);
         fCache->PageUsed(p);
      } else {
         // read page in cache and copy to buf
         p = ReadPage(pageoffset);
         if (p)
            memcpy(buf+boff, p->Data()+begin, blen);
         else
            return -1;
      }
      boff       += blen;
      len        -= blen;
      pageoffset += fPageSize;
      begin       = 0;
   } while (len > 0);

   return 1;
}

//______________________________________________________________________________
Int_t TCache::WritePage(TPage *page)
{
   // Write dirty page to file. Returns -1 in case of error.

   fRecursive = kTRUE;

   page->SetBit(TPage::kLocked);

   fFile->Seek(page->Offset());
   if (fFile->WriteBuffer(page->Data(), page->Size())) {
      page->ResetBit(TPage::kLocked);
      fRecursive = kFALSE;
      return -1;
   }

   page->ResetBit(TPage::kDirty);
   page->ResetBit(TPage::kLocked);

   fRecursive = kFALSE;

   return 0;
}

//______________________________________________________________________________
Int_t TCache::WriteBuffer(Seek_t offset, const char *buf, Int_t len)
{
   // Write a buffer to the cache. Returns < 0 in case of error, 0 in
   // case WriteBuffer() was recursively called via WritePage() and 1
   // in case of success.

   if (fRecursive) return 0;

   // Find in which page offset is located
   Seek_t pageoffset = (offset >> fDiv) << fDiv;  // offset & ~(fPageSize-1)
   Int_t  begin = Int_t(offset & (fPageSize-1));
   Seek_t boff  = 0;

   do {
      Int_t blen = begin+len>fPageSize ? fPageSize-begin : len;
      TPage t(pageoffset, 0, 0);
      TPage *p = (TPage*) fCache->FindObject(&t);
      if (p) {
         // found page in cache, copy buf to it
         memcpy(p->Data()+begin, buf+boff, blen);
         if (p->fSize < begin+blen) {
            p->fSize = begin+blen;
            fNew->Add(p);
         }
         p->SetBit(TPage::kDirty);
         fCache->PageUsed(p);
      } else {
         // read page in cache and copy buf to it
         p = ReadPage(pageoffset);
         if (p) {
            memcpy(p->Data()+begin, buf+boff, blen);
            if (p->fSize < begin+blen) {
               p->fSize = begin+blen;
               fNew->Add(p);
            }
            p->SetBit(TPage::kDirty);
         } else
            return -1;
      }
      boff       += blen;
      len        -= blen;
      pageoffset += fPageSize;
      begin       = 0;
   } while (len > 0);

   return 1;
}

//______________________________________________________________________________
Int_t TCache::Free(ULong_t upto)
{
   // Free pages so that specified number of bytes remains in the cache.
   // Dirty pages are written to file. Returns < 0 in case of error
   // (typically when there was an error writing a dirty page).

   Int_t err;
   if ((err = FlushNew()) < 0)
      return err;

   while (ULong_t(fCache->GetSize() * fPageSize) > upto) {
      TPage *p = (TPage*) fCache->First();
      if (p->TestBit(TPage::kDirty)) {
         if ((err = WritePage(p)) < 0)
            return err;
      }
      fCache->Remove(p);
      fFree->Add(p);
   }
   return 0;
}

//______________________________________________________________________________
Int_t TCache::FlushList(TList *list)
{
   // Flush all dirty pages in the specified list to file. Return < 0 in
   // case of error (typically when there was an error writing a dirty page).

   TIter next(list);
   TPage *p;
   while ((p = (TPage*) next())) {
      if (p->TestBit(TPage::kDirty)) {
         Int_t err;
         if ((err = WritePage(p)) < 0)
            return err;
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t TCache::Flush()
{
   // Flush all dirty pages to file. Return < 0 in case of error
   // (typically when there was an error writing a dirty page).

   Int_t err;
   if ((err = FlushNew()) < 0)
      return err;
   if ((err = FlushList(fCache)) < 0)
      return err;

   return 0;
}

//______________________________________________________________________________
Int_t TCache::FlushNew()
{
   // Flush all new pages to file. New pages are pages in the fNew list.
   // When a page is in this list it means that this page extends the file
   // (i.e. is adds new pages to the file). The issue with new pages is that
   // they need to be flushed in the right order. One can not write at an
   // offset past the EOF. Therefore the new pages are put in a sorted
   // list and then written in ascending fOffset order.

   Int_t err;
   if ((err = FlushList(fNew)) < 0)
      return err;

   fNew->Clear();

   return 0;
}
