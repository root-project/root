// @(#)root/new:$Name$:$Id$
// Author: M.Ivanov   18/06/2007  -- Anar Manafov (A.Manafov@gsi.de) 28/04/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
//
//  TMemStatDepend - non standard C++ functions - Needed to make
//  memory statistic (Used by TMemStatManager)
//
//  To be implemented for differnt platforms.
//______________________________________________________________________________


//STD
#include <string>
#include <vector>
// ROOT
#include "TString.h"
// API
#if defined(R__MACOSX)
#if defined(MAC_OS_X_VERSION_10_5)
#include <malloc/malloc.h>
#include <execinfo.h>
#endif
#else
#include <malloc.h>
#include <execinfo.h>
#endif
#include <cxxabi.h>
// MemStat
#include "TMemStatDepend.h"

#if defined(R__GNU) && (defined(R__LINUX) || defined(R__HURD)) && !defined(__alpha__)
#define SUPPORTS_MEMSTAT
#endif

// This is a global variable set at MSManager init time.
// It marks the highest used stack address.
void *g_global_stack_end = NULL;

#if defined(SUPPORTS_MEMSTAT)
// Comment from Anar:
// HACK: there is an ugly bug in gcc (Bug#8743): http://gcc.gnu.org/bugzilla/show_bug.cgi?id=8743
// "receiving result from __builtin_return_address() beyond stack top causes segfault"
// NOTE that __builtin_return_address should only be used with a non-zero argument for
// debugging purposes. So we use it on our risk.
// A workaround:
// This means the address is out of range.  Note that for the
// toplevel we see a frame pointer with value NULL which clearly is out of range.
// NOTE 2: With gcc 4.1, some optimization levels (e.g., -O, -Os, -O2) have started to imply -fomit-frame-pointer.
// One should use GCC builtin function with -fno-omit-frame-pointer option.
#define G__builtin_return_address(N) \
 ((__builtin_frame_address(N) == NULL)  || \
  (__builtin_frame_address(N) >= g_global_stack_end) || \
  (__builtin_frame_address(N) < __builtin_frame_address(0))) ? \
  NULL : __builtin_return_address(N)
// __builtin_return_address(0) yields the address to which the current
// function will return.  __builtin_return_address(1) yields the address to
// which the caller will return, and so on up the stack.
#define _RET_ADDR(x)   case x: return G__builtin_return_address(x);

#endif

using namespace std;

ClassImp(TMemStatDepend)


//______________________________________________________________________________
static void *return_address(int _frame)
{
   // we have a limit on the depth = 35

#if defined(SUPPORTS_MEMSTAT)
   switch (_frame) {
      _RET_ADDR(0);_RET_ADDR(1);_RET_ADDR(2);_RET_ADDR(3);_RET_ADDR(4);_RET_ADDR(5);_RET_ADDR(6);_RET_ADDR(7);
      _RET_ADDR(8);_RET_ADDR(9);_RET_ADDR(10);_RET_ADDR(11);_RET_ADDR(12);_RET_ADDR(13);_RET_ADDR(14);
      _RET_ADDR(15);_RET_ADDR(16);_RET_ADDR(17);_RET_ADDR(18);_RET_ADDR(19);_RET_ADDR(20);_RET_ADDR(21);
      _RET_ADDR(22);_RET_ADDR(23);_RET_ADDR(24);_RET_ADDR(25);_RET_ADDR(26);_RET_ADDR(27);_RET_ADDR(28);
      _RET_ADDR(29);_RET_ADDR(30);_RET_ADDR(31);_RET_ADDR(32);_RET_ADDR(33);_RET_ADDR(34);_RET_ADDR(35);
   default:
      return 0;
   }
#else
   if (_frame) { }
   return 0;
#endif
}

//______________________________________________________________________________
size_t builtin_return_address(void **_Container, size_t _limit)
{
   size_t i(0);
   void *addr;
   for (i = 0; (i < _limit) && (addr = return_address(i)); ++i)
      _Container[i] = addr;
   return i;
}

//______________________________________________________________________________
TMemStatDepend::MallocHookFunc_t TMemStatDepend::GetMallocHook()
{
   //malloc function getter

#if defined(SUPPORTS_MEMSTAT)
   return __malloc_hook;
#else
   return 0;
#endif
}

//______________________________________________________________________________
TMemStatDepend::FreeHookFunc_t TMemStatDepend::GetFreeHook()
{
   //free function   getter

#if defined(SUPPORTS_MEMSTAT)
   return __free_hook;
#else
   return 0;
#endif
}

//______________________________________________________________________________
void TMemStatDepend::SetMallocHook(MallocHookFunc_t p)
{
   // Set pointer to function replacing alloc function

#if defined(SUPPORTS_MEMSTAT)
   __malloc_hook = p;
#else
   if (p) { }
#endif
}

//______________________________________________________________________________
void TMemStatDepend::SetFreeHook(FreeHookFunc_t p)
{
   // Set pointer to function replacing free function

#if defined(SUPPORTS_MEMSTAT)
   __free_hook = p;
#else
   if (p) { }
#endif
}

//______________________________________________________________________________
size_t TMemStatDepend::Backtrace(void **trace, size_t dsize, Bool_t _bUseGNUBuildinBacktrace)
{
   // Get the backtrace
   // dsize - maximal deepness of stack information
   // trace - array of pointers
   // return value =  min(stack deepness, dsize)

   if ( _bUseGNUBuildinBacktrace )
   {
#if defined(SUPPORTS_MEMSTAT)
      // Initialize the stack end variable.
      return builtin_return_address(trace, dsize);
#else
      if (trace || dsize) { }
      return 0;
#endif
   }
#if defined(R__MACOSX)
#if defined(MAC_OS_X_VERSION_10_5)
   return backtrace(trace, dsize);
#else
   if (trace || dsize) { }
   return 0;
#endif
#else
   return backtrace(trace, dsize);
#endif
}

//______________________________________________________________________________
char** TMemStatDepend::BacktraceSymbols(void **trace, size_t size)
{
   // TODO: Comment me

#if defined(SUPPORTS_MEMSTAT)
   return backtrace_symbols(trace, size);
#else
   if (trace || size) { }
#endif
   return 0;
}

//______________________________________________________________________________
void TMemStatDepend::GetSymbols(void *_pFunction,
                                TString &_strInfo, TString &_strLib, TString &_strFun, TString &/*_strLine*/)
{
   // get the name of the function and library

#if defined(SUPPORTS_MEMSTAT)
   char ** code = backtrace_symbols(&_pFunction, 1);
   if (!code || !code[0])
      return;
   const string codeInfo(code[0]);
   // it is the responsibility of the caller to free that pointer
   free(code);

   // information about the call
   _strInfo = codeInfo.c_str();

   // Resolving a library name
   const string::size_type pos_begin = codeInfo.find_first_of("( [");
   if (string::npos == pos_begin) {
      _strLib = codeInfo;
      return;
   }
   _strLib = codeInfo.substr(0, pos_begin);

   // Resolving a function name
   string::size_type pos_end = codeInfo.find('+', pos_begin);
   if (string::npos == pos_end) {
      pos_end = codeInfo.find(')', pos_begin);
      if (string::npos == pos_end)
         return; // TODO: log me!
   }
   const string func(codeInfo.substr(pos_begin + 1, pos_end - pos_begin - 1));

   // Demangling the function name
   int status(0);
   char *ch = abi::__cxa_demangle(func.c_str(), 0, 0, &status);
   if (!ch)
      return;
   _strFun = (!status) ? ch : func.c_str();
   // it is the responsibility of the caller to free that pointer
   free(ch);
#else
   if (!_pFunction) { _strInfo = ""; _strLib = ""; _strFun = ""; }
#endif
}

//______________________________________________________________________________
void TMemStatDepend::Demangle(char *codeInfo, TString &str)
{
   //    get the name of the function and library

#if defined(SUPPORTS_MEMSTAT)
   int status = 0;
   char *ch = abi::__cxa_demangle(codeInfo, 0, 0, &status);
   if (ch) {
      str = ch;
      free(ch);
   } else {
      str = "unknown";
   }
#else
   if (!codeInfo) { str = ""; }
#endif
}
