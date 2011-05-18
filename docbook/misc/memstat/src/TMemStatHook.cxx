// @(#)root/memstat:$Id$
// Author: Anar Manafov (A.Manafov@gsi.de) 2008-03-02

/*************************************************************************
* Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/
//STD
#include <iostream>
// MemStat
#include "TMemStatHook.h"
#include "RConfig.h"

// TODO: move it to a separate file
#if defined(__APPLE__)
static malloc_zone_t original_zone;
#ifndef __CINT__
static void* profile_malloc(malloc_zone_t *zone, size_t size);
static void* profile_calloc(malloc_zone_t *zone, size_t num_items, size_t size);
static void* profile_valloc(malloc_zone_t *zone, size_t size);
static void profile_free(malloc_zone_t *zone, void *ptr);
#if defined(MAC_OS_X_VERSION_10_6)
static void profile_free_definite_size(malloc_zone_t *zone, void *ptr, size_t size);
#endif
#endif

static zoneMallocHookFunc_t m_pm;
static zoneFreeHookFunc_t m_pf;
#else
#include <malloc.h>
#endif

#if defined(R__GNU) && (defined(R__LINUX) || defined(__APPLE__))
#define SUPPORTS_MEMSTAT
#endif


using namespace std;

#if !defined(__APPLE__)
//______________________________________________________________________________
TMemStatHook::MallocHookFunc_t TMemStatHook::GetMallocHook()
{
   // GetMallocHook - a static function
   // malloc function getter

#if defined(SUPPORTS_MEMSTAT)
   return __malloc_hook;
#else
   return 0;
#endif
}

//______________________________________________________________________________
TMemStatHook::FreeHookFunc_t TMemStatHook::GetFreeHook()
{
   // GetFreeHook - a static function
   // free function getter

#if defined(SUPPORTS_MEMSTAT)
   return __free_hook;
#else
   return 0;
#endif
}

//______________________________________________________________________________
void TMemStatHook::SetMallocHook(MallocHookFunc_t p)
{
   // SetMallocHook - a static function
   // Set pointer to function replacing alloc function

#if defined(SUPPORTS_MEMSTAT)
   __malloc_hook = p;
#endif
}

//______________________________________________________________________________
void TMemStatHook::SetFreeHook(FreeHookFunc_t p)
{
   // SetFreeHook - a static function
   // Set pointer to function replacing free function

#if defined(SUPPORTS_MEMSTAT)
   __free_hook = p;
#endif
}
#endif // !defined(__APPLE__)

//______________________________________________________________________________
#if defined (__APPLE__)
void TMemStatHook::trackZoneMalloc(zoneMallocHookFunc_t pm,
                                   zoneFreeHookFunc_t pf)
{
   // tracZoneMalloc - a static function
   // override the defualt Mac OS X memory zone

   malloc_zone_t* zone = malloc_default_zone();
   if (!zone) {
      cerr << "Error: Can't get malloc_default_zone" << endl;
      return;
   }
   m_pm = pm;
   m_pf = pf;

   original_zone = *zone;
   zone->malloc = &profile_malloc;
   zone->calloc = &profile_calloc;
   zone->valloc = &profile_valloc;
   zone->free = &profile_free;
#if defined(MAC_OS_X_VERSION_10_6)
   if (zone->version >= 6 && zone->free_definite_size)
      zone->free_definite_size = &profile_free_definite_size;
#endif
}
//______________________________________________________________________________
void TMemStatHook::untrackZoneMalloc()
{
   // untrackZoneMalloc - a static function
   // set the defualt Mac OS X memory zone to original

   malloc_zone_t* zone = malloc_default_zone();
   if (!zone) {
      cerr << "Error: Can't get malloc_default_zone" << endl;
      return;
   }
   *zone = original_zone;
}
//______________________________________________________________________________
void* profile_malloc(malloc_zone_t *zone, size_t size)
{
   // Mac OS X profiler of malloc calls

   void* ptr = (*original_zone.malloc)(zone, size);
   m_pm(ptr, size);
   return ptr;
}

//______________________________________________________________________________
void* profile_calloc(malloc_zone_t *zone, size_t num_items, size_t size)
{
   // Mac OS X profiler of calloc calls

   void* ptr = (*original_zone.calloc)(zone, num_items, size);
   m_pm(ptr, size);
   return ptr;
}

//______________________________________________________________________________
void* profile_valloc(malloc_zone_t *zone, size_t size)
{
   // Mac OS X profiler of valloc calls

   void* ptr = (*original_zone.valloc)(zone, size);
   m_pm(ptr, size);
   return ptr;
}

//______________________________________________________________________________
void profile_free(malloc_zone_t *zone, void *ptr)
{
   // Mac OS X profiler of free calls

   (*original_zone.free)(zone, ptr);
   m_pf(ptr);
}

//______________________________________________________________________________
#if defined(MAC_OS_X_VERSION_10_6)
void profile_free_definite_size(malloc_zone_t *zone, void *ptr, size_t size)
{
   // Mac OS X profiler of free_definite_size calls

   (*original_zone.free_definite_size)(zone, ptr, size);
   m_pf(ptr);
}
#endif // defined(MAC_OS_X_VERSION_10_6)
#endif // defined(__APPLE__)
