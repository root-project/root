// @(#)root/new:$Name:  $:$Id: NewDelete.cxx,v 1.2 2000/06/09 14:56:44 rdm Exp $
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
// The powerful MEM_DEBUG and MEM_STAT macros were borrowed from        //
// the ET++ framework.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>

#include "G__ci.h"
#include "TObjectTable.h"
#include "TError.h"
#include "TMath.h"
#include "TMapFile.h"
#include "TSystem.h"
#include "mmalloc.h"


void *CustomReAlloc1(void *ovp, size_t size);
void *CustomReAlloc2(void *ovp, size_t size, size_t oldsize);

class ReAllocInit {
public:
   ReAllocInit() { TStorage::SetReAllocHooks(&CustomReAlloc1, &CustomReAlloc2); }
};
static ReAllocInit realloc_init;


//---- memory checking macros --------------------------------------------------

#if !defined(R__NOSTATS)
#   define MEM_DEBUG
#   define MEM_STAT
#   define MEM_CHECKOBJECTPOINTERS
#endif

#if defined(MEM_STAT) && !defined(MEM_DEBUG)
#   define MEM_DEBUG
#endif

#ifdef MEM_DEBUG
#   define MEM_MAGIC ((unsigned char)0xAB)
#ifdef R__B64
#   define storage_size(p) ((size_t)(((size_t*)p)[-1]))
#   define RealStart(p) ((char*)(p) - sizeof(size_t))
#   define StoreSize(p, sz) (*((size_t*)(p)) = (sz))
#   define ExtStart(p) ((char*)(p) + sizeof(size_t))
#   define RealSize(sz) ((sz) + sizeof(size_t) + sizeof(char))
#   define StoreMagic(p, sz) *((unsigned char*)(p)+sz+sizeof(size_t)) = MEM_MAGIC
#else
#   define storage_size(p) ((size_t)(((int*)p)[-2]))
#   define RealStart(p) ((char*)(p) - 2*sizeof(int))
#   define StoreSize(p, sz) (*((int*)(p)) = (sz))
#   define ExtStart(p) ((char*)(p) + 2*sizeof(int))
#   define RealSize(sz) ((sz) + 2*sizeof(int) + sizeof(char))
#   define StoreMagic(p, sz) *((unsigned char*)(p)+sz+2*sizeof(int)) = MEM_MAGIC
#endif
#   define MemClear(p, start, len) \
      if ((len) > 0) memset(&((char*)(p))[(start)], 0, (len))
#   define TestMagic(p, sz) (*((unsigned char*)(p)+sz) != MEM_MAGIC)
#   define CheckMagic(p, s, where) \
      if (TestMagic(p, s))    \
         Fatal(where, "storage area overwritten");
#   define CheckFreeSize(p, where) \
      if (storage_size((p)) > TStorage::GetMaxBlockSize()) \
         Fatal(where, "unreasonable size (%ld)", storage_size(p));
#   define RemoveStatMagic(p, where) \
      CheckFreeSize(p, where); \
      RemoveStat(p); \
      CheckMagic(p, storage_size(p), where)
#   define StoreSizeMagic(p, size, where) \
      StoreSize(p, size); \
      StoreMagic(p, size); \
      EnterStat(size, ExtStart(p)); \
      CheckObjPtr(ExtStart(p), where);
#else
#   define storage_size(p) ((size_t)0)
#   define RealSize(sz) (sz)
#   define RealStart(p) (p)
#   define ExtStart(p) (p)
#   define MemClear(p, start, len)
#   define StoreSizeMagic(p, size, where) \
      EnterStat(size, ExtStart(p)); \
      CheckObjPtr(ExtStart(p), where);
#   define RemoveStatMagic(p, where) \
      RemoveStat(p);
#endif

#define MemClearRe(p, start, len) \
   if ((len) > 0) memset(&((char*)(p))[(start)], 0, (len))

#define CallFreeHook(p, size) \
   if (TStorage::GetFreeHook()) TStorage::GetFreeHook()(TStorage::GetFreeHookData(), (p), (size))

#ifdef MEM_CHECKOBJECTPOINTERS
//#   define CheckObjPtr(p, name) gObjectTable->CheckPtrAndWarn((name), (p));
#   define CheckObjPtr(p, name)
#else
#   define CheckObjPtr(p, name)
#endif

//------------------------------------------------------------------------------
#ifdef MEM_STAT

#define EnterStat(s, p) \
   TStorage::EnterStat(s, p)
#define RemoveStat(p) \
   TStorage::RemoveStat(p)

#else

#define EnterStat(s, p) \
   TStorage::SetMaxBlockSize(TMath::Max(TStorage::GetMaxBlockSize(), s))
#define RemoveStat(p)

#endif

//------------------------------------------------------------------------------

#ifndef NOCINT
#define G__PVOID (-1)
#ifndef WIN32
extern long G__globalvarpointer;
#endif
#endif

static const char *spaceErr = "storage exhausted";
static int newInit = 0;

//______________________________________________________________________________
void *operator new(size_t size)
{
   // Custom new() operator.

   static const char *where = "operator new";

   if (!newInit) {
      TStorage::SetCustomNewDelete();
      newInit++;
   }

#ifndef NOCINT
#ifndef WIN32
   if (G__globalvarpointer != G__PVOID) {
      long temp = G__globalvarpointer;
      G__globalvarpointer = G__PVOID;
      return (void*)temp;
   }
#else
   long gvp = G__getgvp();
   if (gvp != G__PVOID) {
      G__setgvp(G__PVOID);
      return (void*)gvp;
   }
#endif
#endif
   register void *vp;
   if (gMmallocDesc)
      vp = ::mcalloc(gMmallocDesc, RealSize(size), sizeof(char));
   else
      vp = ::calloc(RealSize(size), sizeof(char));
   if (vp == 0)
      Fatal(where, spaceErr);
   StoreSizeMagic(vp, size, where);
   return ExtStart(vp);
}

#ifndef R__KCC
//______________________________________________________________________________
void *operator new(size_t size, void *vp)
{
   // Custom new() operator with placement argument.

   static const char *where = "operator new(void *at)";

   if (!newInit) {
      TStorage::SetCustomNewDelete();
      newInit++;
   }

#ifndef NOCINT
#ifndef WIN32
   if ((long)vp == G__globalvarpointer && G__globalvarpointer != G__PVOID)
      return(vp);
#else
   long gvp = G__getgvp();
   if ((long)vp == gvp && gvp != G__PVOID)
      return(vp);
#endif
#endif
   if (vp == 0) {
      register void *vp;
      if (gMmallocDesc)
         vp = ::mcalloc(gMmallocDesc, RealSize(size), sizeof(char));
      else
         vp = ::calloc(RealSize(size), sizeof(char));
      if (vp == 0)
         Fatal(where, spaceErr);
      StoreSizeMagic(vp, size, where);
      return ExtStart(vp);
   }
   return vp;
}
#endif

//______________________________________________________________________________
void operator delete(void *ptr)
{
   // Custom delete() operator.

   static const char *where = "operator delete";

   if (!newInit)
      Fatal(where, "space was not allocated via custom new");

#ifndef NOCINT
#ifndef WIN32
   if ((long)ptr == G__globalvarpointer && G__globalvarpointer!=G__PVOID)
      return;
#else
   long gvp = G__getgvp();
   if ((long)ptr == gvp && gvp != G__PVOID)
      return;
#endif
#endif
   if (ptr) {
      CheckObjPtr(ptr, where);
      CallFreeHook(ptr, storage_size(ptr));
      RemoveStatMagic(ptr, where);
      MemClear(RealStart(ptr), 0, RealSize(storage_size(ptr)));
      TSystem::ResetErrno();
      TMapFile *mf = TMapFile::WhichMapFile(RealStart(ptr));
      if (mf) {
         if (mf->IsWritable()) ::mfree(mf->GetMmallocDesc(), RealStart(ptr));
      } else
         ::free(RealStart(ptr));
      if (TSystem::GetErrno() != 0)
         SysError(where, "free");
   }
}

#if defined(R__VECNEWDELETE)
//______________________________________________________________________________
void *operator new[](size_t size)
{
   return ::operator new(size);
}

#ifndef R__KCC
//______________________________________________________________________________
void *operator new[](size_t size, void *vp)
{
   return ::operator new(size, vp);
}
#endif

//______________________________________________________________________________
void operator delete[](void *ptr)
{
   ::operator delete(ptr);
}
#endif

//______________________________________________________________________________
void *CustomReAlloc1(void *ovp, size_t size)
{
   // Reallocate (i.e. resize) block of memory.

   static const char *where = "CustomReAlloc1";

   if (ovp == 0)
      return new char[size];

   if (!newInit)
      Fatal(where, "space was not allocated via custom new");

   size_t oldsize = storage_size(ovp);
   if (oldsize == size)
      return ovp;
   RemoveStatMagic(ovp, where);
   void *vp;
   if (gMmallocDesc)
      vp = ::mrealloc(gMmallocDesc, RealStart(ovp), RealSize(size));
   else
      vp = ::realloc((char*)RealStart(ovp), RealSize(size));
   if (vp == 0)
      Fatal(where, spaceErr);
   if (size > oldsize)
      MemClearRe(ExtStart(vp), oldsize, size-oldsize);

   StoreSizeMagic(vp, size, where);
   return ExtStart(vp);
}

//______________________________________________________________________________
void *CustomReAlloc2(void *ovp, size_t size, size_t oldsize)
{
   // Reallocate (i.e. resize) block of memory. Checks if current size is
   // equal to oldsize. If not memory was overwritten.

   static const char *where = "CustomReAlloc2";

   if (ovp == 0)
      return new char[size];

   if (!newInit)
      Fatal(where, "space was not allocated via custom new");

#if defined(MEM_DEBUG)
   if (oldsize != storage_size(ovp))
      fprintf(stderr, "<%s>: passed oldsize %d, should be %d\n", where,
              oldsize, storage_size(ovp));
#endif
   if (oldsize == size)
      return ovp;
   RemoveStatMagic(ovp, where);
   void *vp;
   if (gMmallocDesc)
      vp = ::mrealloc(gMmallocDesc, RealStart(ovp), RealSize(size));
   else
      vp = ::realloc((char*)RealStart(ovp), RealSize(size));
   if (vp == 0)
      Fatal(where, spaceErr);
   if (size > oldsize)
      MemClearRe(ExtStart(vp), oldsize, size-oldsize);

   StoreSizeMagic(vp, size, where);
   return ExtStart(vp);
}
