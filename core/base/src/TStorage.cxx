// @(#)root/base:$Id$
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
#include "TString.h"
#include "TVirtualMutex.h"
#include "TInterpreter.h"

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

#define PVOID (-1)

size_t        TStorage::fgMaxBlockSize;
FreeHookFun_t TStorage::fgFreeHook;
void         *TStorage::fgFreeHookData;
ReAllocFun_t  TStorage::fgReAllocHook;
ReAllocCFun_t TStorage::fgReAllocCHook;
Bool_t        TStorage::fgHasCustomNewDelete;


ClassImp(TStorage)

//------------------------------------------------------------------------------

static const char *gSpaceErr = "storage exhausted";

const size_t kObjMaxSize = 10024;

static Bool_t   gMemStatistics;
static Int_t    gAllocated[kObjMaxSize], gFreed[kObjMaxSize];
static Int_t    gAllocatedTotal, gFreedTotal;
static void   **gTraceArray = 0;
static Int_t    gTraceCapacity = 10, gTraceIndex = 0,
                gMemSize = -1, gMemIndex = -1;


//______________________________________________________________________________
void TStorage::EnterStat(size_t size, void *p)
{
   // Register a memory allocation operation. If desired one can trap an
   // allocation of a certain size in case one tries to find a memory
   // leak of that particular size. This function is only called via
   // the ROOT custom new operators.

   TStorage::SetMaxBlockSize(TMath::Max(TStorage::GetMaxBlockSize(), size));

   if (!gMemStatistics) return;

   if ((Int_t)size == gMemSize) {
      if (gTraceIndex == gMemIndex)
         Fatal("EnterStat", "trapped allocation %d", gMemIndex);

      if (!gTraceArray)
         gTraceArray = (void**) malloc(sizeof(void*)*gTraceCapacity);

      if (gTraceIndex >= gTraceCapacity) {
         gTraceCapacity = gTraceCapacity*2;
         gTraceArray = (void**) realloc(gTraceArray, sizeof(void*)*gTraceCapacity);
      }
      gTraceArray[gTraceIndex++] = p;
   }
   if (size >= kObjMaxSize)
      gAllocated[kObjMaxSize-1]++;
   else
      gAllocated[size]++;
   gAllocatedTotal += size;
}

//______________________________________________________________________________
void TStorage::RemoveStat(void *vp)
{
   // Register a memory free operation. This function is only called via
   // the custom ROOT delete operator.

   if (!gMemStatistics) return;

   size_t size = storage_size(vp);
   if ((Int_t)size == gMemSize) {
      for (int i = 0; i < gTraceIndex; i++)
         if (gTraceArray[i] == vp) {
            gTraceArray[i] = 0;
            break;
         }
   }
   if (size >= kObjMaxSize)
      gFreed[kObjMaxSize-1]++;
   else
      gFreed[size]++;
   gFreedTotal += size;
}

//______________________________________________________________________________
void *TStorage::Alloc(size_t size)
{
   // Allocate a block of memory, that later can be resized using
   // TStorage::ReAlloc().

   static const char *where = "TStorage::Alloc";

#ifndef WIN32
   void *vp = ::operator new[](size);
#else
   void *vp = ::operator new(size);
#endif
   if (vp == 0)
      Fatal(where, "%s", gSpaceErr);

   return vp;
}

//______________________________________________________________________________
void TStorage::Dealloc(void *ptr)
{
   // De-allocate block of memory, that was allocated via TStorage::Alloc().

#ifndef WIN32
   ::operator delete[](ptr);
#else
   ::operator delete(ptr);
#endif
}

//______________________________________________________________________________
void *TStorage::ReAlloc(void *ovp, size_t size)
{
   // Reallocate (i.e. resize) block of memory. Don't use if size is larger
   // than old size, use ReAlloc(void *, size_t, size_t) instead.

   ::Obsolete("ReAlloc(void*,size_t)", "v5-34-00", "v6-02-00");
   ::Info("ReAlloc(void*,size_t)", "please use ReAlloc(void*,size_t,size_t)");

   {
      // Needs to be protected by global mutex
      R__LOCKGUARD(gGlobalMutex);

      if (fgReAllocHook && fgHasCustomNewDelete && !TROOT::MemCheck())
         return (*fgReAllocHook)(ovp, size);
   }

   static const char *where = "TStorage::ReAlloc";

#ifndef WIN32
   void *vp = ::operator new[](size);
#else
   void *vp = ::operator new(size);
#endif
   if (vp == 0)
      Fatal(where, "%s", gSpaceErr);

   if (ovp == 0)
      return vp;

   memmove(vp, ovp, size);
#ifndef WIN32
   ::operator delete[](ovp);
#else
   ::operator delete(ovp);
#endif
   return vp;
}

//______________________________________________________________________________
void *TStorage::ReAlloc(void *ovp, size_t size, size_t oldsize)
{
   // Reallocate (i.e. resize) block of memory. Checks if current size is
   // equal to oldsize. If not memory was overwritten.

   // Needs to be protected by global mutex
   {
      R__LOCKGUARD(gGlobalMutex);

      if (fgReAllocCHook && fgHasCustomNewDelete && !TROOT::MemCheck())
         return (*fgReAllocCHook)(ovp, size, oldsize);
   }

   static const char *where = "TStorage::ReAlloc";

   if (oldsize == size)
      return ovp;

#ifndef WIN32
   void *vp = ::operator new[](size);
#else
   void *vp = ::operator new(size);
#endif
   if (vp == 0)
      Fatal(where, "%s", gSpaceErr);

   if (ovp == 0)
      return vp;

   if (size > oldsize) {
      memcpy(vp, ovp, oldsize);
      memset((char*)vp+oldsize, 0, size-oldsize);
   } else
      memcpy(vp, ovp, size);
#ifndef WIN32
   ::operator delete[](ovp);
#else
   ::operator delete(ovp);
#endif
   return vp;
}

//______________________________________________________________________________
char *TStorage::ReAllocChar(char *ovp, size_t size, size_t oldsize)
{
   // Reallocate (i.e. resize) array of chars. Size and oldsize are
   // in number of chars.

   static const char *where = "TStorage::ReAllocChar";

   char *vp;
   if (ovp == 0) {
      vp = new char[size];
      if (vp == 0)
         Fatal(where, "%s", gSpaceErr);
      return vp;
   }
   if (oldsize == size)
      return ovp;

   vp = new char[size];
   if (vp == 0)
      Fatal(where, "%s", gSpaceErr);
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

   static const char *where = "TStorage::ReAllocInt";

   Int_t *vp;
   if (ovp == 0) {
      vp = new Int_t[size];
      if (vp == 0)
         Fatal(where, "%s", gSpaceErr);
      return vp;
   }
   if (oldsize == size)
      return ovp;

   vp = new Int_t[size];
   if (vp == 0)
      Fatal(where, "%s", gSpaceErr);
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
   // TStorage::FilledByObjectAlloc() to find out if the just created object is on
   // the heap.

   void* space =  ::operator new(sz);
   memset(space, kObjectAllocMemValue, sz);
   return space;
}

//______________________________________________________________________________
void *TStorage::ObjectAllocArray(size_t sz)
{
   // Used to allocate array of TObject on the heap (via TObject::operator new[]()).
   // Unlike the 'singular' ObjectAlloc, we do not mark those object has being
   // allocated on the heap as they can not be individually deleted.
   void* space =  ::operator new(sz);
   return space;
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

#ifndef NOCINT
   // to handle delete with placement called via CINT
   Long_t gvp = 0;
   if (gCint) gvp = gCint->Getgvp();
   if ((Long_t)vp == gvp && gvp != (Long_t)PVOID)
      return;
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

   // Needs to be protected by global mutex
   R__LOCKGUARD(gGlobalMutex);

#if defined(MEM_DEBUG) && defined(MEM_STAT)

   if (!gMemStatistics || !HasCustomNewDelete())
      return;

   //Printf("");
   Printf("Heap statistics");
   Printf("%12s%12s%12s%12s", "size", "alloc", "free", "diff");
   Printf("================================================");

   int i;
   for (i = 0; i < (int)kObjMaxSize; i++)
      if (gAllocated[i] != gFreed[i])
      //if (gAllocated[i])
         Printf("%12d%12d%12d%12d", i, gAllocated[i], gFreed[i],
                gAllocated[i]-gFreed[i]);

   if (gAllocatedTotal != gFreedTotal) {
      Printf("------------------------------------------------");
      Printf("Total:      %12d%12d%12d", gAllocatedTotal, gFreedTotal,
              gAllocatedTotal-gFreedTotal);
   }

   if (gMemSize != -1) {
      Printf("------------------------------------------------");
      for (i= 0; i < gTraceIndex; i++)
         if (gTraceArray[i])
            Printf("block %d of size %d not freed", i, gMemSize);
   }
   Printf("================================================");
   Printf(" ");
#endif
}

//______________________________________________________________________________
void TStorage::EnableStatistics(int size, int ix)
{
   // Enable memory usage statistics gathering. Size is the size of the memory
   // block that should be trapped and ix is after how many such allocations
   // the trap should happen.

#ifdef MEM_STAT
   gMemSize       = size;
   gMemIndex      = ix;
   gMemStatistics = kTRUE;
#else
   int idum = size; int iidum = ix;
#endif
}

//______________________________________________________________________________
ULong_t TStorage::GetHeapBegin()
{
   ::Obsolete("GetHeapBegin()", "v5-34-00", "v6-02-00");
   //return begin of heap
   return 0;
}

//______________________________________________________________________________
ULong_t TStorage::GetHeapEnd()
{
   ::Obsolete("GetHeapBegin()", "v5-34-00", "v6-02-00");
   //return end of heap
   return 0;
}

//______________________________________________________________________________
void *TStorage::GetFreeHookData()
{
   //return static free hook data
   return fgFreeHookData;
}

//______________________________________________________________________________
Bool_t TStorage::HasCustomNewDelete()
{
   //return the has custom delete flag
   return fgHasCustomNewDelete;
}

//______________________________________________________________________________
void TStorage::SetCustomNewDelete()
{
   //set the has custom delete flag
   fgHasCustomNewDelete = kTRUE;
}


//______________________________________________________________________________
void TStorage::AddToHeap(ULong_t, ULong_t)
{
   //add a range to the heap
   ::Obsolete("AddToHeap(ULong_t,ULong_t)", "v5-34-00", "v6-02-00");
}

//______________________________________________________________________________
Bool_t TStorage::IsOnHeap(void *)
{
   //is object at p in the heap?
   ::Obsolete("IsOnHeap(void*)", "v5-34-00", "v6-02-00");
   return false;
}

#ifdef WIN32
//______________________________________________________________________________
Bool_t TStorage::FilledByObjectAlloc(UInt_t *member)
{
   //called by TObject's constructor to determine if object was created by call to new
   return *member == kObjectAllocMemValue;
}

//______________________________________________________________________________
size_t TStorage::GetMaxBlockSize()
{
   //return max block size
   return fgMaxBlockSize;
}

//______________________________________________________________________________
void TStorage::SetMaxBlockSize(size_t size)
{
   //set max block size
   fgMaxBlockSize = size;
}

//______________________________________________________________________________
FreeHookFun_t TStorage::GetFreeHook()
{
   //return free hook
   return fgFreeHook;
}

#endif
