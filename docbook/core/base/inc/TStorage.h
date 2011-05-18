// @(#)root/base:$Id$
// Author: Fons Rademakers   29/07/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TStorage
#define ROOT_TStorage


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStorage                                                             //
//                                                                      //
// Storage manager.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

typedef void (*FreeHookFun_t)(void*, void *addr, size_t);
typedef void *(*ReAllocFun_t)(void*, size_t);
typedef void *(*ReAllocCFun_t)(void*, size_t, size_t);
typedef char *(*ReAllocCharFun_t)(char*, size_t, size_t);


class TStorage {

private:
   static ULong_t        fgHeapBegin;          // begin address of heap
   static ULong_t        fgHeapEnd;            // end address of heap
   static size_t         fgMaxBlockSize;       // largest block allocated
   static FreeHookFun_t  fgFreeHook;           // function called on free
   static void          *fgFreeHookData;       // data used by this function
   static ReAllocFun_t   fgReAllocHook;        // custom ReAlloc
   static ReAllocCFun_t  fgReAllocCHook;       // custom ReAlloc with length check
   static Bool_t         fgHasCustomNewDelete; // true if using ROOT's new/delete

public:
   virtual ~TStorage() { }

   static ULong_t        GetHeapBegin();
   static ULong_t        GetHeapEnd();
   static FreeHookFun_t  GetFreeHook();
   static void          *GetFreeHookData();
   static size_t         GetMaxBlockSize();
   static void          *Alloc(size_t size);
   static void           Dealloc(void *ptr);
   static void          *ReAlloc(void *vp, size_t size);
   static void          *ReAlloc(void *vp, size_t size, size_t oldsize);
   static char          *ReAllocChar(char *vp, size_t size, size_t oldsize);
   static Int_t         *ReAllocInt(Int_t *vp, size_t size, size_t oldsize);
   static void          *ObjectAlloc(size_t size);
   static void          *ObjectAlloc(size_t size, void *vp);
   static void           ObjectDealloc(void *vp);
   static void           ObjectDealloc(void *vp, void *ptr);

   static void EnterStat(size_t size, void *p);
   static void RemoveStat(void *p);
   static void PrintStatistics();
   static void SetMaxBlockSize(size_t size);
   static void SetFreeHook(FreeHookFun_t func, void *data);
   static void SetReAllocHooks(ReAllocFun_t func1, ReAllocCFun_t func2);
   static void SetCustomNewDelete();
   static void EnableStatistics(int size= -1, int ix= -1);

   static Bool_t HasCustomNewDelete();

   // only valid after call to a TStorage allocating method
   static void   AddToHeap(ULong_t begin, ULong_t end);
   static Bool_t IsOnHeap(void *p);

   ClassDef(TStorage,0)  //Storage manager class
};

#ifndef WIN32
inline void TStorage::AddToHeap(ULong_t begin, ULong_t end)
   { if (begin < fgHeapBegin) fgHeapBegin = begin;
     if (end   > fgHeapEnd)   fgHeapEnd   = end; }

inline Bool_t TStorage::IsOnHeap(void *p)
   { return (ULong_t)p >= fgHeapBegin && (ULong_t)p < fgHeapEnd; }

inline size_t TStorage::GetMaxBlockSize() { return fgMaxBlockSize; }

inline void TStorage::SetMaxBlockSize(size_t size) { fgMaxBlockSize = size; }

inline FreeHookFun_t TStorage::GetFreeHook() { return fgFreeHook; }
#endif

#endif
