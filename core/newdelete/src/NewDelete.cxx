// @(#)root/new:$Id$
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
// Custom operators new and delete and ReAlloc functions.               //
//                                                                      //
// All new and delete operations in the ROOT system pass                //
// via the custom new and delete operators defined in this file.        //
// This scheme allows extensive memory checking and usage statistics    //
// gathering and an easy way to access shared memory segments.          //
// Memory checking is by default enabled and usage statistics is        //
// gathered. Using the resource (in .rootrc): Root.MemStat one can      //
// toggle statistics gathering on or off. More specifically on can trap //
// the allocation of a block of memory of a certain size. This can be   //
// specified using the resource: Root.MemStat.size, using the resource  //
// Root.MemStat.cnt one can specify after how many allocations of       //
// this size the trap should occur.                                     //
// Set the compile option R__NOSTATS to de-activate all memory checking //
// statistics gathering in the system.                                  //
//                                                                      //
// When memory checking is enabled the following happens during         //
// allocation:                                                          //
//  - each allocation results in the allocation of 9 extra bytes:       //
//    2 words in front and 1 byte at the end of the memory chunck       //
//    returned to the caller.                                           //
//  - the allocated memory is set to 0.                                 //
//  - the size of the chunck is stored in the first word. The second    //
//    word is left empty (for alignment).                               //
//  - the last byte is initialized to MEM_MAGIC.                        //
//                                                                      //
// And during de-allocation this happens:                               //
//  - first the size if the block is checked. It should be >0 and       //
//    <= than any block allocated up to that moment. If not a Fatal     //
//    error is generated.                                               //
//  - the MEM_MAGIC byte at the end of the block is checked. When not   //
//    there, the memory has been overwritten and a Fatal error is       //
//    generated.                                                        //
//  - memory block is reset to 0.                                       //
//                                                                      //
// Although this does not replace powerful tools like Purify, it is a   //
// good first line of protection.                                       //
//                                                                      //
// Independent of any compile option settings the new, and ReAlloc      //
// functions always set the memory to 0.                                //
//                                                                      //
// The powerful MEM_DEBUG and MEM_STAT macros were originally borrowed  //
// from the ET++ framework.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <errno.h>

#include "TObjectTable.h"
#include "TError.h"
#include "TStorage.h" // for ROOT::Internal::gFreeIfTMapFile
#include "TSystem.h"
#include "mmalloc.h"

void *CustomReAlloc1(void *ovp, size_t size);
void *CustomReAlloc2(void *ovp, size_t size, size_t oldsize);

class TReAllocInit {
public:
   TReAllocInit() { TStorage::SetReAllocHooks(&CustomReAlloc1, &CustomReAlloc2); }
};
static TReAllocInit gReallocInit;

//---- memory checking macros --------------------------------------------------

#if !defined(R__NOSTATS)
#   define MEM_DEBUG
#   define MEM_STAT
#   define MEM_CHECKOBJECTPOINTERS
#endif

#if defined(MEM_STAT) && !defined(MEM_DEBUG)
#   define MEM_DEBUG
#endif

////////////////////////////////////////////////////////////////////////////////

namespace {

#ifdef MEM_STAT

auto EnterStat(size_t s, void *p)
{
   TStorage::EnterStat(s, p);
}
auto RemoveStat(void *p)
{
   TStorage::RemoveStat(p);
}

#else

auto EnterStat(size_t s, void *p)
{
   TStorage::SetMaxBlockSize(TMath::Max(TStorage::GetMaxBlockSize(), s));
}
auto RemoveStat(void *) {}

#endif

#ifdef MEM_DEBUG
#   define MEM_MAGIC ((unsigned char)0xAB)
auto RealStart(void *p)
{
   return ((char *)(p) - sizeof(std::max_align_t));
}
#ifdef R__B64
auto storage_size(void *p)
{
   return (*(size_t *)RealStart(p));
}
auto StoreSize(void *p, size_t sz)
{
   return (*((size_t *)(p)) = (sz));
}
#else
auto StoreSize(void *p, int sz)
{
   return (*((int *)(p)) = (sz));
}
auto storage_size(p)
{
   return ((size_t) * (int *)RealStart(p));
}
#endif
auto ExtStart(void *p)
{
   return ((char *)(p) + sizeof(std::max_align_t));
}
auto RealSize(size_t sz)
{
   return ((sz) + sizeof(std::max_align_t) + sizeof(char));
}
auto StoreMagic(void *p, size_t sz)
{
   return *((unsigned char *)(p) + sz + sizeof(std::max_align_t)) = MEM_MAGIC;
}
auto MemClear(void *p, size_t start, size_t len)
{
   if ((len) > 0)
      memset(&((char *)(p))[(start)], 0, (len));
}
auto TestMagic(void *p, size_t sz)
{
   return (*((unsigned char *)(p) + sz) != MEM_MAGIC);
}
auto CheckMagic(void *p, size_t s, const char *where)
{
   if (TestMagic(p, s))
      Fatal(where, "%s", "storage area overwritten");
}
auto CheckFreeSize(void *p, const char *where)
{
   if (storage_size((p)) > TStorage::GetMaxBlockSize())
      Fatal(where, "unreasonable size (%ld)", (Long_t)storage_size(p));
}
auto RemoveStatMagic(void *p, const char *where)
{
   CheckFreeSize(p, where);
   RemoveStat(p);
   CheckMagic(p, storage_size(p), where);
}
auto StoreSizeMagic(void *p, size_t size, const char * /* where */)
{
   StoreSize(p, size);
   StoreMagic(p, size);
   EnterStat(size, ExtStart(p));
}
#else
auto storage_size(void *)
{
   return ((size_t)0);
}
auto RealSize(size_t sz)
{
   return sz;
}
auto RealStart(p)
{
   return p;
}
auto ExtStart(void *p)
{
   return p;
}
auto MemClear(void *, size_t /* start */, size_t /* len */) {}
auto StoreSizeMagic(void *p, size_t size, const char * /* where */)
{
   EnterStat(size, ExtStart(p));
}
auto RemoveStatMagic(void *p, const char * /* where */)
{
   RemoveStat(p);
}
#endif

auto MemClearRe(void *p, size_t start, size_t len)
{
   if ((len) > 0)
      memset(&((char *)(p))[(start)], 0, (len));
}

auto CallFreeHook(void *p, size_t size)
{
   if (TStorage::GetFreeHook())
      TStorage::GetFreeHook()(TStorage::GetFreeHookData(), (p), (size));
}

} // anonymous namespace

//------------------------------------------------------------------------------
static const char *gSpaceErr = "storage exhausted (failed to allocate %ld bytes)";
static int gNewInit = 0;

////////////////////////////////////////////////////////////////////////////////
/// Custom new() operator.

void *operator new(size_t size)
{
   static const char *where = "operator new";

   if (!gNewInit) {
      TStorage::SetCustomNewDelete();
      gNewInit++;
   }

   // Notes:
   // The return of calloc is aligned with std::max_align_t.
   // If/whe al > std::max_align_t we need to adjust.
   // The layout for the Magic, Stat and Size is:
   //  [0 : sizeof(std::max_align_t) [ -> Record `size`
   //  [sizeof(std::max_align_t) : same + size ] -> Real data; lower bound id return value
   //  [sizeof(std::max_align_t) + size : same + 1 [   -> MEM_MAGIC / Integrity marker
   // We need sizeof(size_t) <= sizeof(std::max_align_t)
   //
   assert(sizeof(size_t) <= sizeof(std::max_align_t));

   void *vp;
   if (ROOT::Internal::gMmallocDesc)
      vp = ::mcalloc(ROOT::Internal::gMmallocDesc, RealSize(size), sizeof(char));
   else
      vp = ::calloc(RealSize(size), sizeof(char));
   if (vp == 0)
      Fatal(where, gSpaceErr, RealSize(size));
   StoreSizeMagic(vp, size, where); // NOLINT
   return ExtStart(vp);
}

void *operator new(size_t size, const std::nothrow_t &) noexcept
{
   return ::operator new(size);
}

#if __cplusplus >= 201700L

void *operator new(size_t size, std::align_val_t)
{
   Fatal("operator new", "with std::align_val_t is not implemented yet");
   return ::operator new(size);
}

void *operator new(size_t size, std::align_val_t, const std::nothrow_t &nt) noexcept
{
   Fatal("operator new", "with std::align_val_t is not implemented yet");
   return ::operator new(size, nt);
}

#endif

#ifndef R__PLACEMENTINLINE
////////////////////////////////////////////////////////////////////////////////
/// Custom new() operator with placement argument.

void *operator new(size_t size, void *vp)
{
   static const char *where = "operator new(void *at)";

   if (!gNewInit) {
      TStorage::SetCustomNewDelete();
      gNewInit++;
   }

   if (vp == 0) {
      void *vp;
      if (ROOT::Internal::gMmallocDesc)
         vp = ::mcalloc(ROOT::Internal::gMmallocDesc, RealSize(size), sizeof(char));
      else
         vp = ::calloc(RealSize(size), sizeof(char));
      if (vp == 0)
         Fatal(where, gSpaceErr, RealSize(size));
      StoreSizeMagic(vp, size, where);
      return ExtStart(vp);
   }
   return vp;
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Custom delete() operator.

void operator delete(void *ptr) noexcept
{
   static const char *where = "operator delete";

   if (!gNewInit)
      Fatal(where, "space was not allocated via custom new");

   if (ptr) {
      CallFreeHook(ptr, storage_size(ptr));
      RemoveStatMagic(ptr, where);
      MemClear(RealStart(ptr), 0, RealSize(storage_size(ptr)));
      TSystem::ResetErrno();
      if (!ROOT::Internal::gFreeIfTMapFile
          || !ROOT::Internal::gFreeIfTMapFile(RealStart(ptr))) {
         do {
            TSystem::ResetErrno();
            ::free(RealStart(ptr)); // NOLINT
         } while (TSystem::GetErrno() == EINTR);
      }
      if (TSystem::GetErrno() != 0)
         SysError(where, "free");
   }
}

void operator delete(void *ptr, const std::nothrow_t &) noexcept
{
   operator delete(ptr);
}

#if __cplusplus >= 201700L
void operator delete(void * /*ptr*/, std::align_val_t /*al*/) noexcept
{
   Fatal("operator delete", "with std::align_val_t is not implemented yet");
}
void operator delete(void * /*ptr*/, std::align_val_t /*al*/, const std::nothrow_t &) noexcept
{
   Fatal("operator delete", "with std::align_val_t is not implemented yet");
}
#endif

#ifdef R__SIZEDDELETE
////////////////////////////////////////////////////////////////////////////////
/// Sized-delete calling non-sized one.
void operator delete(void *ptr, std::size_t) noexcept
{
   operator delete(ptr);
}
#if __cplusplus >= 201700L
void operator delete(void * /*ptr*/, std::size_t, std::align_val_t /*al*/) noexcept
{
   Fatal("operator delete", "with std::align_val_t is not implemented yet");
}
#endif
#endif

#ifdef R__VECNEWDELETE
////////////////////////////////////////////////////////////////////////////////
/// Custom vector new operator.

void *operator new[](size_t size)
{
   return ::operator new(size);
}

void *operator new[](size_t size, const std::nothrow_t &) noexcept
{
   return ::operator new(size);
}

#if __cplusplus >= 201700L

void *operator new[](size_t size, std::align_val_t al)
{
   Fatal("operator new[]", "with std::align_val_t is not implemented yet");
   return ::operator new(size, al);
}

void *operator new[](size_t size, std::align_val_t al, const std::nothrow_t &nt) noexcept
{
   Fatal("operator new[]", "with std::align_val_t is not implemented yet");
   return ::operator new(size, al, nt);
}

#endif

#ifndef R__PLACEMENTINLINE
////////////////////////////////////////////////////////////////////////////////
/// Custom vector new() operator with placement argument.

void *operator new[](size_t size, void *vp)
{
   return ::operator new(size, vp);
}
#endif

////////////////////////////////////////////////////////////////////////////////

void operator delete[](void *ptr) noexcept
{
   ::operator delete(ptr);
}

#if __cplusplus >= 201700L
void operator delete[](void * /*ptr*/, std::align_val_t /*al*/) noexcept
{
   Fatal("operator delete[]", "with std::align_val_t is not implemented yet");
}
#endif

#ifdef R__SIZEDDELETE
////////////////////////////////////////////////////////////////////////////////
/// Sized-delete calling non-sized one.
void operator delete[](void *ptr, std::size_t) noexcept
{
   operator delete[](ptr);
}
#if __cplusplus >= 201700L
void operator delete[](void * /* ptr */, std::size_t, std::align_val_t /* al */) noexcept
{
   Fatal("operator delete[]", "with size_t and std::align_val_t is not implemented yet");
}
#endif
#endif

#endif

////////////////////////////////////////////////////////////////////////////////
/// Reallocate (i.e. resize) block of memory.

void *CustomReAlloc1(void *, size_t)
{
   Fatal("NewDelete::CustomRealloc1", "This should not be used. The TStorage interface using this has been removed.");
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Reallocate (i.e. resize) block of memory. Checks if current size is
/// equal to oldsize. If not memory was overwritten.

void *CustomReAlloc2(void *ovp, size_t size, size_t oldsize)
{
   static const char *where = "CustomReAlloc2";

   if (ovp == 0)
      return ::operator new(size);

   if (!gNewInit)
      Fatal(where, "space was not allocated via custom new");

#if defined(MEM_DEBUG)
   if (oldsize != storage_size(ovp))
      fprintf(stderr, "<%s>: passed oldsize %u, should be %u\n", where, (unsigned int)oldsize,
              (unsigned int)storage_size(ovp));
#endif
   if (oldsize == size)
      return ovp;
   RemoveStatMagic(ovp, where);
   void *vp;
   if (ROOT::Internal::gMmallocDesc)
      vp = ::mrealloc(ROOT::Internal::gMmallocDesc, RealStart(ovp), RealSize(size));
   else
      vp = ::realloc((char *)RealStart(ovp), RealSize(size));
   if (vp == 0)
      Fatal(where, gSpaceErr, RealSize(size));
   if (size > oldsize)
      MemClearRe(ExtStart(vp), oldsize, size - oldsize); // NOLINT

   StoreSizeMagic(vp, size, where);                      // NOLINT
   return ExtStart(vp);
}
