// @(#)root/memstat:$Id$
// Author: Anar Manafov (A.Manafov@gsi.de) 2008-03-02

/*************************************************************************
* Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/
//
//  TYamsMemHook
//  Non standard C/C++ functions
//  Needed for memory statistic

#ifndef ROOT_TMemStatHook
#define ROOT_TMemStatHook

#if defined(__APPLE__)
#ifndef __CINT__
#include <malloc/malloc.h>
#endif
typedef void (*zoneMallocHookFunc_t)(void *ptr, size_t size);
typedef void (*zoneFreeHookFunc_t)(void *ptr);
#endif

class TMemStatHook {
public:
#if !defined(__APPLE__)
   //
   // Memory management HOOK functions
   //
   typedef void*(*MallocHookFunc_t)(size_t size, const void *caller);
   typedef void (*FreeHookFunc_t)(void *ptr, const void *caller);

   static MallocHookFunc_t GetMallocHook();         // malloc function getter
   static FreeHookFunc_t   GetFreeHook();           // free function getter
   static void SetMallocHook(MallocHookFunc_t p);   // malloc function setter
   static void SetFreeHook(FreeHookFunc_t p);       // free function setter
#else
   //
   // Public methods for Mac OS X
   //
   static void trackZoneMalloc(zoneMallocHookFunc_t pm, zoneFreeHookFunc_t pf);
   static void untrackZoneMalloc();
#endif
};

#endif

