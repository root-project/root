// @(#)root/base:$Name:  $:$Id: TStorage.cxx,v 1.10 2002/02/23 16:01:44 rdm Exp $
// Author: Fons Rademakers   29/07/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStorage                                                             //
//                                                                      //
// Storage manager. The storage manager works best in conjunction with  //
// the custom ROOT new and delete operators defined in the file         //
// NewDelete.cxx (libNew.so). Only when using the custom allocation     //
// operators will memory usage statistics be gathered using the         //
// TStorage EnterStat(), RemoveStat(), etc. functions.                  //
// Memory checking is by default enabled (when using libNew.so) and     //
// usage statistics is gathered. Using the resource (in .rootrc):       //
// Root.MemStat one can toggle statistics gathering on or off. More     //
// specifically on can trap the allocation of a block of memory of a    //
// certain size. This can be specified using the resource:              //
// Root.MemStat.size, using the resource Root.MemStat.cnt one can       //
// specify after how many allocations of this size the trap should      //
// occur.                                                               //
// Set the compile option R__NOSTATS to de-activate all memory checking //
// and statistics gathering in the system.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>

#include "TROOT.h"
#include "TObjectTable.h"
#include "TError.h"
#include "TMath.h"
#include "TString.h"
#include "TVirtualMutex.h"

#if !defined(R__NOSTATS)
#   define MEM_DEBUG
#   define MEM_STAT
#   define MEM_CHECKOBJECTPOINTERS
#endif

#if defined(MEM_STAT) && !defined(MEM_DEBUG)
#   define MEM_DEBUG
#endif

#ifdef MEM_DEBUG
#   ifdef R__B64
#      define storage_size(p) ((size_t)(((size_t*)p)[-1]))
#   else
#      define storage_size(p) ((size_t)(((int*)p)[-2]))
#   endif
#else
#   define storage_size(p) ((size_t)0)
#endif

#ifndef NOCINT
#define G__PVOID (-1)
#ifndef WIN32
extern long G__globalvarpointer;
#else
#include "G__ci.h"
#endif
#endif

ULong_t       TStorage::fgHeapBegin = (ULong_t)-1L;
ULong_t       TStorage::fgHeapEnd;
size_t        TStorage::fgMaxBlockSize;
FreeHookFun_t TStorage::fgFreeHook;
void         *TStorage::fgFreeHookData;
ReAllocFun_t  TStorage::fgReAllocHook;
ReAllocCFun_t TStorage::fgReAllocCHook;
Bool_t        TStorage::fgHasCustomNewDelete;


ClassImp(TStorage)

//------------------------------------------------------------------------------

static const char *spaceErr = "storage exhausted";

const size_t kObjMaxSize = 10024;

static Bool_t   memStatistics;
static Int_t    allocated[kObjMaxSize], freed[kObjMaxSize];
static Int_t    allocatedTotal, freedTotal;
static void   **traceArray = 0;
static Int_t    traceCapacity = 10, traceIndex = 0, memSize = -1, memIndex = -1;


//______________________________________________________________________________
void TStorage::EnterStat(size_t size, void *p)
{
   // Register a memory allocation operation. If desired one can trap an
   // allocation of a certain size in case one tries to find a memory
   // leak of that particular size. This function is only called via
   // the ROOT custom new operators.

   TStorage::SetMaxBlockSize(TMath::Max(TStorage::GetMaxBlockSize(), size));

   if (!memStatistics) return;

   if ((Int_t)size == memSize) {
      if (traceIndex == memIndex)
         Fatal("EnterStat", "trapped allocation %d", memIndex);

      if (!traceArray) traceArray = (void**) malloc(sizeof(void*)*traceCapacity);

      if (traceIndex >= traceCapacity) {
         traceCapacity = traceCapacity*2;
         traceArray = (void**) realloc(traceArray, sizeof(void*)*traceCapacity);
      }
      traceArray[traceIndex++] = p;
   }
   if (size >= kObjMaxSize)
      allocated[kObjMaxSize-1]++;
   else
      allocated[size]++;
   allocatedTotal += size;
}

//______________________________________________________________________________
void TStorage::RemoveStat(void *vp)
{
   // Register a memory free operation. This function is only called via
   // the custom ROOT delete operator.

   if (!memStatistics) return;

   size_t size = storage_size(vp);
   if ((Int_t)size == memSize) {
      for (int i = 0; i < traceIndex; i++)
         if (traceArray[i] == vp) {
            traceArray[i] = 0;
            break;
         }
   }
   if (size >= kObjMaxSize)
      freed[kObjMaxSize-1]++;
   else
      freed[size]++;
   freedTotal += size;
}

//______________________________________________________________________________
void *TStorage::ReAlloc(void *ovp, size_t size)
{
   // Reallocate (i.e. resize) block of memory.

   R__LOCKGUARD(gCINTMutex);

   if (fgReAllocHook && fgHasCustomNewDelete && !TROOT::MemCheck())
      return (*fgReAllocHook)(ovp, size);

   static const char *where = "TStorage::ReAlloc";

   void *vp;
   if (ovp == 0) {
      vp = ::operator new[](size);
      if (vp == 0)
         Fatal(where, spaceErr);
      return vp;
   }

   vp = ::operator new[](size);
   if (vp == 0)
      Fatal(where, spaceErr);
   memmove(vp, ovp, size);
   ::operator delete[](ovp);
   return vp;
}

//______________________________________________________________________________
void *TStorage::ReAlloc(void *ovp, size_t size, size_t oldsize)
{
   // Reallocate (i.e. resize) block of memory. Checks if current size is
   // equal to oldsize. If not memory was overwritten.

   R__LOCKGUARD(gCINTMutex);

   if (fgReAllocCHook && fgHasCustomNewDelete && !TROOT::MemCheck())
      return (*fgReAllocCHook)(ovp, size, oldsize);

   static const char *where = "TStorage::ReAlloc";

   void *vp;
   if (ovp == 0) {
     vp = ::operator new[](size);
     if (vp == 0)
        Fatal(where, spaceErr);
     return vp;
   }
   if (oldsize == size)
      return ovp;

   vp = ::operator new[](size);
   if (vp == 0)
      Fatal(where, spaceErr);
   if (size > oldsize) {
      memcpy(vp, ovp, oldsize);
      memset((char*)vp+oldsize, 0, size-oldsize);
   } else
      memcpy(vp, ovp, size);
   ::operator delete[](ovp);
   return vp;
}

//______________________________________________________________________________
char *TStorage::ReAllocChar(char *ovp, size_t size, size_t oldsize)
{
   // Reallocate (i.e. resize) array of chars. Size and oldsize are
   // in number of chars.

   R__LOCKGUARD(gCINTMutex);

   static const char *where = "TStorage::ReAllocChar";

   char *vp;
   if (ovp == 0) {
     vp = new char[size];
     if (vp == 0)
        Fatal(where, spaceErr);
     return vp;
   }
   if (oldsize == size)
      return ovp;

   vp = new char[size];
   if (vp == 0)
      Fatal(where, spaceErr);
   if (size > oldsize) {
      memcpy(vp, ovp, oldsize);
      memset((char*)vp+oldsize, 0, size-oldsize);
   } else
      memcpy(vp, ovp, size);
   delete [] ovp;
   return vp;
}

//______________________________________________________________________________
Int_t *TStorage::ReAllocInt(Int_t *ovp, size_t size, size_t oldsize)
{
   // Reallocate (i.e. resize) array of integers. Size and oldsize are
   // number of integers (not number of bytes).

   R__LOCKGUARD(gCINTMutex);

   static const char *where = "TStorage::ReAllocInt";

   Int_t *vp;
   if (ovp == 0) {
     vp = new Int_t[size];
     if (vp == 0)
        Fatal(where, spaceErr);
     return vp;
   }
   if (oldsize == size)
      return ovp;

   vp = new Int_t[size];
   if (vp == 0)
      Fatal(where, spaceErr);
   if (size > oldsize) {
      memcpy(vp, ovp, oldsize*sizeof(Int_t));
      memset((Int_t*)vp+oldsize, 0, (size-oldsize)*sizeof(Int_t));
   } else
      memcpy(vp, ovp, size*sizeof(Int_t));
   delete [] ovp;
   return vp;
}

//______________________________________________________________________________
void *TStorage::ObjectAlloc(size_t sz)
{
   // Used to allocate a TObject on the heap (via TObject::operator new()).
   // Directly after this routine one can call (in the TObject ctor)
   // TStorage::IsOnHeap() to find out if the just created object is on
   // the heap.

   R__LOCKGUARD(gCINTMutex);

   ULong_t space;

#ifndef NOCINT
   // to handle new with placement called via CINT
#ifndef WIN32
   if (G__globalvarpointer != G__PVOID) {
      space = G__globalvarpointer;
      G__globalvarpointer = G__PVOID;
   } else
#else
   space = G__getgvp();
   if ((long)space != G__PVOID) {
      G__setgvp(G__PVOID);
   } else
#endif
#endif
   space = (ULong_t) ::operator new(sz);
   AddToHeap(space, space+sz);
   return (void*) space;
}

//______________________________________________________________________________
void *TStorage::ObjectAlloc(size_t , void *vp)
{
   // Used to allocate a TObject on the heap (via TObject::operator new(size_t,void*))
   // in position vp. vp is already allocated (maybe on heap, maybe on
   // stack) so just return.

   return vp;
}

//______________________________________________________________________________
void TStorage::ObjectDealloc(void *vp)
{
   // Used to deallocate a TObject on the heap (via TObject::operator delete()).

   R__LOCKGUARD(gCINTMutex);

#ifndef NOCINT
   // to handle delete with placement called via CINT
#ifndef WIN32
   if ((long)vp == G__globalvarpointer && G__globalvarpointer != G__PVOID)
      return;
#else
   long gvp = G__getgvp();
   if ((long)vp == gvp && gvp != G__PVOID)
      return;
#endif
#endif
   ::operator delete(vp);
}

//______________________________________________________________________________
void TStorage::ObjectDealloc(void *vp, void *ptr)
{
   // Used to deallocate a TObject on the heap (via TObject::operator delete(void*,void*)).

   if (vp && ptr) { }
}

//______________________________________________________________________________
void TStorage::SetFreeHook(FreeHookFun_t fh, void *data)
{
   // Set a free handler.

   fgFreeHook     = fh;
   fgFreeHookData = data;
}

//______________________________________________________________________________
void TStorage::SetReAllocHooks(ReAllocFun_t rh1, ReAllocCFun_t rh2)
{
   // Set a custom ReAlloc handlers. This function is typically
   // called via a static object in the ROOT libNew.so shared library.

   fgReAllocHook  = rh1;
   fgReAllocCHook = rh2;
}

//______________________________________________________________________________
void TStorage::PrintStatistics()
{
   // Print memory usage statistics.

   R__LOCKGUARD(gCINTMutex);

#if defined(MEM_DEBUG) && defined(MEM_STAT)

   if (!memStatistics || !HasCustomNewDelete())
      return;

   //Printf("");
   Printf("Heap statistics");
   Printf("%12s%12s%12s%12s", "size", "alloc", "free", "diff");
   Printf("================================================");

   int i;
   for (i = 0; i < (int)kObjMaxSize; i++)
      if (allocated[i] != freed[i])
      //if (allocated[i])
         Printf("%12d%12d%12d%12d", i, allocated[i], freed[i],
                allocated[i]-freed[i]);

   if (allocatedTotal != freedTotal) {
      Printf("------------------------------------------------");
      Printf("Total:      %12d%12d%12d", allocatedTotal, freedTotal,
              allocatedTotal-freedTotal);
   }

   if (memSize != -1) {
      Printf("------------------------------------------------");
      for (i= 0; i < traceIndex; i++)
         if (traceArray[i])
            Printf("block %d of size %d not freed", i, memSize);
   }
   Printf("================================================");
   Printf("");
#endif
}

//______________________________________________________________________________
void TStorage::EnableStatistics(int size, int ix)
{
   // Enable memory usage statistics gathering. Size is the size of the memory
   // block that should be trapped and ix is after how many such allocations
   // the trap should happen.

#ifdef MEM_STAT
   memSize       = size;
   memIndex      = ix;
   memStatistics = kTRUE;
#else
   int idum = size; int iidum = ix;
#endif
}

//______________________________________________________________________________
ULong_t TStorage::GetHeapBegin()
{
   return fgHeapBegin;
}

//______________________________________________________________________________
ULong_t TStorage::GetHeapEnd()
{
   return fgHeapEnd;
}

//______________________________________________________________________________
void *TStorage::GetFreeHookData()
{
   return fgFreeHookData;
}

//______________________________________________________________________________
Bool_t TStorage::HasCustomNewDelete()
{
   return fgHasCustomNewDelete;
}

//______________________________________________________________________________
void TStorage::SetCustomNewDelete()
{
   fgHasCustomNewDelete = kTRUE;
}

#ifdef WIN32

//______________________________________________________________________________
void TStorage::AddToHeap(ULong_t begin, ULong_t end)
{
   if (begin < fgHeapBegin) fgHeapBegin = begin;
   if (end   > fgHeapEnd)   fgHeapEnd   = end;
}

//______________________________________________________________________________
Bool_t TStorage::IsOnHeap(void *p)
{
   return (ULong_t)p >= fgHeapBegin && (ULong_t)p < fgHeapEnd;
}

//______________________________________________________________________________
size_t TStorage::GetMaxBlockSize()
{
   return fgMaxBlockSize;
}

//______________________________________________________________________________
void TStorage::SetMaxBlockSize(size_t size)
{
   fgMaxBlockSize = size;
}

//______________________________________________________________________________
FreeHookFun_t TStorage::GetFreeHook()
{
   return fgFreeHook;
}

#endif
