// @(#)root/memstat:$Id$
// Author: Anar Manafov (A.Manafov@gsi.de) 2010-03-02

/*************************************************************************
* Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/
#ifndef ROOT_TMemStatBacktrace
#define ROOT_TMemStatBacktrace

#define _INIT_TOP_STACK extern void *g_global_stack_end;
#define _GET_CALLER_FRAME_ADDR g_global_stack_end = __builtin_frame_address(1);

// ROOT
#include "Rtypes.h"

class TString;

namespace memstat {
   //
   // Backtrace functions
   //
   size_t getBacktrace(void **_trace, size_t _size, Bool_t _bUseGNUBuiltinBacktrace = kFALSE);
   int getSymbols(void *_pAddr,
                  TString &_strInfo, TString &_strLib, TString &_strSymbol);
   void getSymbolFullInfo(void *_pAddr, TString *_retInfo, const char *const _seporator = " | ");
   void demangle(char *_codeInfo, TString &_str);
}


#endif
