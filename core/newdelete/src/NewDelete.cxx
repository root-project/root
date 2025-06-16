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

auto EnterStat(size_t s, void *)
{
   TStorage::SetMaxBlockSize(TMath::Max(TStorage::GetMaxBlockSize(), s));
}
auto RemoveStat(void *) {}

#endif

constexpr unsigned char kOffsetSlot = 1;
constexpr unsigned char kAlignmentSlot = 2;
constexpr unsigned char kSizeSlot = 3;


auto align_address_up(void *p, std::align_val_t al)
{
   size_t alignment = static_cast<size_t>(al);
   auto aligned_start = (((size_t)p) + (alignment - 1)) & ~(alignment - 1);
   return reinterpret_cast<void*>(aligned_start);
}

auto Offset(void *data_start)
{
   size_t *start = reinterpret_cast<size_t*>(data_start);
   return *(start - kOffsetSlot);
}

auto RealStart(void *data_start)
{
   return ((char *)(data_start) - Offset(data_start));
}

auto RequestedAlignment(void *data_start)
{
   size_t *start = reinterpret_cast<size_t*>(data_start);
   return static_cast<std::align_val_t>(*(start - kAlignmentSlot));
}

#ifdef MEM_DEBUG

constexpr unsigned char kMetaDataSize = 3;

#   define MEM_MAGIC ((unsigned char)0xAB)

auto ExtStart(void *p, std::align_val_t al)
{
   size_t *start = reinterpret_cast<size_t*>(p);
   // start is of type size_t so that the addition of kMetaDataSize is correct
   return reinterpret_cast<size_t*>(align_address_up(start + kMetaDataSize, al));
}

auto storage_size(void *data_start)
{
   size_t *start = reinterpret_cast<size_t*>(data_start);
   return *(start - kSizeSlot);
}

auto StoreSize(void *p, size_t sz, std::align_val_t al)
{
   size_t *start = ExtStart(p, al);
   *(start - kOffsetSlot) = (char*)start - (char*)p;
   *(start - kAlignmentSlot) = static_cast<size_t>(al);
   return (*(start - kSizeSlot) = sz);
}

auto RealSize(void *ptr)
{
   auto requested_size = storage_size(ptr);
   auto offset = Offset(ptr);
   return requested_size + offset + 1;
}

auto RealSize(size_t sz, std::align_val_t ale)
{
   size_t al = static_cast<size_t>(ale);
   // The meta data is the offset and alignment
   auto real_size = sz + sizeof(size_t) * 2; // Always store the offset
   real_size += sizeof(size_t) + 1; // size and MEM_MAGIC
   if ( al > sizeof(size_t) )
      // Maximum possible 'wastage/padding due to alignment'
      // i.e. 'at worst' the allocation was one 'calloc' aligment away from
      // the proper `al` alignment.
      real_size += al - sizeof(size_t);
   return real_size;
}

auto StoreMagic(void *p, size_t sz, std::align_val_t al)
{
   auto where = reinterpret_cast<char*>(ExtStart(p, al)) + sz;
   return *reinterpret_cast<unsigned char*>(where) = MEM_MAGIC;
}

auto MemClear(void *p, size_t start, size_t len)
{
   if ((len) > 0)
      memset(&((char *)(p))[(start)], 0, (len));
}

auto TestMagic(void *data_start, size_t sz)
{
   return (*((unsigned char *)(data_start) + sz) != MEM_MAGIC);
}

auto CheckMagic(void *data_start, size_t s, const char *where)
{
   if (TestMagic(data_start, s))
      Fatal(where, "%s", "storage area overwritten");
}

auto CheckFreeSize(void *data_start, const char *where)
{
   if (storage_size((data_start)) > TStorage::GetMaxBlockSize())
      Fatal(where, "unreasonable size (%ld)", (Long_t)storage_size(data_start));
}

auto RemoveStatMagic(void *data_start, const char *where)
{
   CheckFreeSize(data_start, where);
   RemoveStat(data_start);
   CheckMagic(data_start, storage_size(data_start), where);
}

auto StoreSizeMagic(void *p, size_t size, std::align_val_t al)
{
   StoreSize(p, size, al);
   StoreMagic(p, size, al);
   EnterStat(size, ExtStart(p, al));
}

#else

constexpr unsigned char kMetaDataSize = 2;
auto storage_size(void *)
{
   return ((size_t)0);
}
auto RealSize(void *ptr)
{
   auto requested_size = storage_size(ptr);
   auto offset = Offset(ptr);
   return requested_size + offset;
}
auto RealSize(size_t sz, std::align_val_t  ale)
{
   size_t al = static_cast<size_t>(ale);
   // The meta data is the offset and alignment
   auto real_size = sz + sizeof(size_t) * kMetaDataSize;
   if ( al > sizeof(size_t) )
      // Maximum possible 'wastage/padding due to alignment'
      // i.e. 'at worst' the allocation was one 'calloc' aligment away from
      // the proper `al` alignment.
      real_size += al - sizeof(size_t));
   return real_size;
}
auto ExtStart(void *p, std::align_val_t al)
{
   size_t *start = reinterpret_cast<size_t*>(p);
   return reinterpret_cast<size_t*>(align_address_up(start + kMetaDataSize, al));
}
auto MemClear(void *, size_t /* start */, size_t /* len */) {}
auto StoreSizeMagic(void *p, size_t size, std::align_val_t al)
{
   size_t *start = ExtStart(p, al);
   *(start - kOffsetSlot) = (char*)start - (char*)p;
   *(start - kAlignmentSlot) = static_cast<size_t>(al);
   (void)kSizeSlot;
   EnterStat(size, ExtStart(p, al));
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
   return ::operator new(size, (std::align_val_t)__STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

void *operator new(size_t size, const std::nothrow_t&) noexcept
{
   return ::operator new(size);
}

void *operator new(size_t size, std::align_val_t al)
{
   static const char *where = "operator new";

   if (!gNewInit) {
      TStorage::SetCustomNewDelete();
      gNewInit++;
   }

   // Old Notes:
   // The return of calloc is aligned with std::max_align_t.
   // If/whe al > std::max_align_t we need to adjust.
   // The layout for the Magic, Stat and Size is:
   //  [0 : sizeof(std::max_align_t) [ -> Record `size`
   //  [sizeof(std::max_align_t) : same + size ] -> Real data; lower bound id return value
   //  [sizeof(std::max_align_t) + size : same + 1 [   -> MEM_MAGIC / Integrity marker
   // We need sizeof(size_t) <= sizeof(std::max_align_t)
   //
   // New Notes:
   // The return of calloc is aligned with std::max_align_t.
   // If/whe al > std::max_align_t we need to adjust.
   // The layout for the Offset, Magic, Stat and Size is:
   //  [0 : returned ptr value - sizeof(size_t) * 3 [ -> Unused; Usually of size 0.
#ifdef MEM_DEBUG
   //  [returned ptr value - sizeof(size_t) * 3 : same + sizeof(size_t) [  -> Requested allocation size
#endif
   //  [returned ptr value - sizeof(size_t) * 2 : same + sizeof(size_t) [  -> Requested alignment
   //  [returned ptr value - sizeof(size_t) : same + sizeof(size_t) [  -> Offset between the return ptr value and the allocated start
   //  [returned ptr value : same + size [ -> Real data start
   //  [returned ptr value + size : same + 1 [   -> MEM_MAGIC / Integrity marker
   //
   // Per C++ standard 3.11 Alignment [basic.align]:
   //   Every alignment value shall be a non-negative integral power of two.
#if 0
   assert( sizeof(size_t) <= sizeof(std::max_align_t) );
   if (reinterpret_cast<size_t>(al) > sizeof(std::max_align_t)) {
      // This actually the usual case since __STDCPP_DEFAULT_NEW_ALIGNMENT__ > sizeof(std::max_align_t)

      size_t alignment = reinterpret_cast<size_t>(al);
      char *p = ...;
      // size_t((p + (alignment - 1)) / alignment) * alignment
      char *aligned_start = ((p + (alignment - 1)) & ~(alignment - 1));
   }
#endif

   void *vp;
   auto real_size = RealSize(size, al);
   if (ROOT::Internal::gMmallocDesc)
      vp = ::mcalloc(ROOT::Internal::gMmallocDesc, real_size, sizeof(char));
   else
      vp = ::calloc(real_size, sizeof(char));
   if (vp == nullptr)
      Fatal(where, gSpaceErr, real_size);
   StoreSizeMagic(vp, size, al); // NOLINT
   return ExtStart(vp, al);
}

void *operator new(size_t size, std::align_val_t al, const std::nothrow_t&) noexcept
{
   return ::operator new(size, al);
}

////////////////////////////////////////////////////////////////////////////////
/// Custom delete() operator.

void operator delete(void *ptr) noexcept
{
   static const char *where = "operator delete";

   if (!gNewInit)
      Fatal(where, "space was not allocated via custom new");

   if (ptr) {
      auto requested_size = storage_size(ptr);
      auto start = RealStart(ptr);
      auto real_size = RealSize(ptr);
      CallFreeHook(ptr, requested_size);
      RemoveStatMagic(ptr, where);
      // After this the meta-data is also cleared.
      MemClear(start, 0, real_size);
      TSystem::ResetErrno();
      if (!ROOT::Internal::gFreeIfTMapFile
          || !ROOT::Internal::gFreeIfTMapFile(start)) {
         do {
            TSystem::ResetErrno();
            ::free(start); // NOLINT
         } while (TSystem::GetErrno() == EINTR);
      }
      if (TSystem::GetErrno() != 0)
         SysError(where, "free");
   }
}

void operator delete(void *ptr, const std::nothrow_t &) noexcept
{
   ::operator delete(ptr);
}

void operator delete(void *ptr, std::align_val_t /*al*/) noexcept
{
   ::operator delete(ptr);
}
void operator delete(void *ptr, std::align_val_t al, const std::nothrow_t&) noexcept
{
   ::operator delete(ptr, al);
}

////////////////////////////////////////////////////////////////////////////////
/// Sized-delete calling non-sized one.
void operator delete(void *ptr, std::size_t) noexcept
{
   ::operator delete(ptr);
}
void operator delete(void *ptr, std::size_t, std::align_val_t al) noexcept
{
   ::operator delete(ptr, al);
}

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

void *operator new[](size_t size, std::align_val_t al)
{
   return ::operator new(size, al);
}

void *operator new[](size_t size, std::align_val_t al, const std::nothrow_t &nt) noexcept
{
   return ::operator new(size, al, nt);
}

////////////////////////////////////////////////////////////////////////////////

void operator delete[](void *ptr) noexcept
{
   ::operator delete(ptr);
}

void operator delete[](void *ptr, std::align_val_t al) noexcept
{
   ::operator delete(ptr, al);
}

////////////////////////////////////////////////////////////////////////////////
/// Sized-delete calling non-sized one.
void operator delete[](void *ptr, std::size_t) noexcept
{
   ::operator delete[](ptr);
}
void operator delete[](void *ptr, std::size_t, std::align_val_t al) noexcept
{
   ::operator delete(ptr, al);
}

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

   if (ovp == nullptr)
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
   std::align_val_t al = RequestedAlignment(ovp);
   void *realstart = RealStart(ovp);
   void *localMallocDesc = nullptr;
   if (ROOT::Internal::gGetMapFileMallocDesc &&
       (localMallocDesc = ROOT::Internal::gGetMapFileMallocDesc(realstart))) {
      vp = ::mrealloc(localMallocDesc, realstart, RealSize(size, al));
   } else {
      vp = ::realloc((char *)realstart, RealSize(size, al));
   }
   if (vp == nullptr)
      Fatal(where, gSpaceErr, RealSize(size, al));
   if (size > oldsize)
      MemClearRe(ExtStart(vp, al), oldsize, size - oldsize); // NOLINT

   StoreSizeMagic(vp, size, al);                      // NOLINT
   return ExtStart(vp, al);
}
