// @(#)root/memstat:$Name$:$Id$
// Author: M.Ivanov   18/06/2007 -- Anar Manafov (A.Manafov@gsi.de) 28/04/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMemStatDepend
#define ROOT_TMemStatDepend

#if !defined(__CINT__)
#include <sys/types.h>
#endif
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

//
//  TMemStatDepend
//  Non standard C++ functions
//  Needed for memory statistic

#define _INIT_TOP_STECK extern void *g_global_stack_end;
#define _GET_TO_STECK g_global_stack_end = __builtin_frame_address(1);

class TString;

class TMemStatDepend
{
public:
   //
   // Memory management HOOK functions
   //
   typedef void* (*MallocHookFunc_t)(size_t size, const void *caller);
   typedef void (*FreeHookFunc_t)(void* ptr, const void *caller);

   static MallocHookFunc_t GetMallocHook();         // malloc function getter
   static FreeHookFunc_t   GetFreeHook();           // free function getter
   static void SetMallocHook(MallocHookFunc_t p);   // malloc function setter
   static void SetFreeHook(FreeHookFunc_t p);       // free function setter
   //
   // Backtrace functions
   //
   static size_t Backtrace(void **trace, size_t size, Bool_t _bUseGNUBuildinBacktrace = kFALSE);
   static char** BacktraceSymbols(void **trace, size_t size);
   static void GetSymbols(void *pFunction, TString &strInfo,  TString &strLib, TString &strFun, TString &strLine);
   static void Demangle(char *codeInfo, TString &str);
};

#endif
