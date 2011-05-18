// @(#)root/win32gdk:$Id$
// Author: Valeriy Onuchin  08/08/2003


/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGWin32ProxyDefs
#define ROOT_TGWin32ProxyDefs

#include "Rtypes.h"  // CINT headers


#define _NAME4_(name1,name2,name3,name4) _NAME3_(name1,name2,name3)name4



///////////////////// debug & profile interface /////////////////////////////
//
// - recompile with gDebugProxy = 1
//
// root [0] gDebug = -123 //  start profiling
//or
// root [0] gDebug = -1234 //  start profiling and debugging(call trace)
//
// root [1] .x macro.C    //  profile macro.C
// root [2] gDebug = 0    //  stop profiling
// root [3] .x macro.C    //  print results
//

static int gDebugProxy = 0; // if kTRUE - use debug & profile interface

static enum { kDebugProfile = -123, kDebugTrace = -1234 };

static unsigned int total = 0;
static double total_time = 0;

#define DEBUG_PROFILE_PROXY_START(method)\
   static int i = 0;\
   static double t = 0;\
   double start = 0;\
   int gDebugValue = 0;\
   int debug = 0;\
   if (gDebugProxy) {\
      gDebugValue = gDebug;\
      debug = (gDebugValue==kDebugProfile) || (gDebugValue==kDebugTrace);\
      if (debug) {\
         start = GetMilliSeconds();\
      } else {\
         if (total) {\
            printf("  method name                       hits     time/hits(ms)   time(ms) | Total = %d hits %6.2f ms\n",total,total_time );\
            printf("------------------------------------------------------------------------------------------------------------\n");\
         }\
         if (i && !total) {\
            printf("  %-30s    %-6d       %-3.2f        %-4.2f\n",#method,i,t/i,t);\
         }\
         total_time = t = total = i = 0;\
      }\
   }\

#define DEBUG_PROFILE_PROXY_STOP(method)\
   if (gDebugProxy) {\
      if (debug) {\
         double dt = GetMilliSeconds() - start;\
         i++; total++;\
         t += dt;\
         total_time += dt;\
         if (gDebugValue==kDebugTrace) printf(#method " %d\n",i);\
      }\
   }\


//______________________________________________________________________________
#define RETURN_PROXY_OBJECT(klass)\
_NAME2_(T,klass)* _NAME3_(TGWin32,klass,Proxy)::ProxyObject()\
{\
   static TList *gListOfProxies = new TList();\
   static _NAME3_(TGWin32,klass,Proxy) *proxy = 0;\
   ULong_t id = ::GetCurrentThreadId();\
   if (proxy && (proxy->GetId()==id)) return proxy;\
   if (id==fgMainThreadId) return _NAME3_(TGWin32,klass,Proxy)::RealObject();\
   TIter next(gListOfProxies);\
   while ((proxy=(_NAME3_(TGWin32,klass,Proxy)*)next())) {\
      if (proxy->GetId()==id) {\
         return proxy;\
      }\
   }\
   proxy = new _NAME3_(TGWin32,klass,Proxy)();\
   gListOfProxies->Add(proxy);\
   return proxy;\
}

// ***_LOCK macros for setter methods which do nothing only set data members
//______________________________________________________________________________
#define VOID_METHOD_ARG0_LOCK(klass,method)\
void _NAME3_(TGWin32,klass,Proxy)::method()\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   TGWin32::Lock();\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method();\
   TGWin32::Unlock();\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG1_LOCK(klass,method,type1,par1)\
void _NAME3_(TGWin32,klass,Proxy)::method(type1 par1)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   TGWin32::Lock();\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(par1);\
   TGWin32::Unlock();\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG2_LOCK(klass,method,type1,par1,type2,par2)\
void _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   TGWin32::Lock();\
    _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(par1,par2);\
   TGWin32::Unlock();\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG3_LOCK(klass,method,type1,par1,type2,par2,type3,par3)\
void _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   TGWin32::Lock();\
    _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(par1,par2,par3);\
   TGWin32::Unlock();\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG4_LOCK(klass,method,type1,par1,type2,par2,type3,par3,type4,par4)\
void  _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   TGWin32::Lock();\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(par1,par2,par3,par4);\
   TGWin32::Unlock();\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG5_LOCK(klass,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5)\
void  _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   TGWin32::Lock();\
    _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(par1,par2,par3,par4,par5);\
   TGWin32::Unlock();\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG0(klass,method,sync)\
void _NAME3_(p2,klass,method)(void *in)\
{\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method();\
}\
\
void _NAME3_(TGWin32,klass,Proxy)::method()\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   fCallBack = &_NAME3_(p2,klass,method);\
   ForwardCallBack(sync);\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG1(klass,method,type1,par1,sync)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1;\
   };\
   tmp *p = (tmp*)in;\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1);\
}\
\
void _NAME3_(TGWin32,klass,Proxy)::method(type1 par1)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   struct tmp {\
      type1 par1;\
      tmp(type1 par1):par1(par1) {}\
   };\
   fParam = new tmp(par1);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(sync);\
   par1 = ((tmp*)fParam)->par1;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG2(klass,method,type1,par1,type2,par2,sync)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2;\
   };\
   tmp *p = (tmp*)in;\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2);\
}\
\
void _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   struct tmp {\
      type1 par1; type2 par2;\
      tmp(type1 par1,type2 par2):par1(par1),par2(par2) {}\
   };\
   fParam = new tmp(par1,par2);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(sync);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG3(klass,method,type1,par1,type2,par2,type3,par3,sync)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3;\
   };\
   tmp *p = (tmp*)in;\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3);\
}\
\
void _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   struct tmp {\
      type1 par1; type2 par2; type3 par3;\
      tmp(type1 par1,type2 par2,type3 par3):par1(par1),par2(par2),par3(par3) {}\
   };\
   fParam = new tmp(par1,par2,par3);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(sync);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG4(klass,method,type1,par1,type2,par2,type3,par3,type4,par4,sync)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4;\
   };\
   tmp *p = (tmp*)in;\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4);\
}\
\
void _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4):par1(par1),par2(par2),par3(par3),par4(par4) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(sync);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG5(klass,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,sync)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5;\
   };\
   tmp *p = (tmp*)in;\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4,p->par5);\
}\
\
void _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5):par1(par1),par2(par2),par3(par3),par4(par4),par5(par5) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4,par5);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(sync);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   par5 = ((tmp*)fParam)->par5;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG6(klass,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,sync)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6;\
   };\
   tmp *p = (tmp*)in;\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4,p->par5,p->par6);\
}\
\
void _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6):par1(par1),par2(par2),par3(par3),par4(par4),par5(par5),par6(par6) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4,par5,par6);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(sync);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   par5 = ((tmp*)fParam)->par5;\
   par6 = ((tmp*)fParam)->par6;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG7(klass,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7,sync)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7;\
   };\
   tmp *p = (tmp*)in;\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4,p->par5,p->par6,p->par7);\
}\
\
void _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7):par1(par1),par2(par2),par3(par3),par4(par4),par5(par5),par6(par6),par7(par7) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4,par5,par6,par7);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(sync);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   par5 = ((tmp*)fParam)->par5;\
   par6 = ((tmp*)fParam)->par6;\
   par7 = ((tmp*)fParam)->par7;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG8(klass,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7,type8,par8,sync)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8;\
   };\
   tmp *p = (tmp*)in;\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4,p->par5,p->par6,p->par7,p->par8);\
}\
\
void _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8):par1(par1),par2(par2),par3(par3),par4(par4),par5(par5),par6(par6),par7(par7),par8(par8) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4,par5,par6,par7,par8);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(sync);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   par5 = ((tmp*)fParam)->par5;\
   par6 = ((tmp*)fParam)->par6;\
   par7 = ((tmp*)fParam)->par7;\
   par8 = ((tmp*)fParam)->par8;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG9(klass,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7,type8,par8,type9,par9,sync)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8; type9 par9;\
   };\
   tmp *p = (tmp*)in;\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4,p->par5,p->par6,p->par7,p->par8,p->par9);\
}\
\
void _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8; type9 par9;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9):par1(par1),par2(par2),par3(par3),par4(par4),par5(par5),par6(par6),par7(par7),par8(par8),par9(par9) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4,par5,par6,par7,par8,par9);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(sync);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   par5 = ((tmp*)fParam)->par5;\
   par6 = ((tmp*)fParam)->par6;\
   par7 = ((tmp*)fParam)->par7;\
   par8 = ((tmp*)fParam)->par8;\
   par9 = ((tmp*)fParam)->par9;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG10(klass,type,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7,type8,par8,type9,par9,type10,par10)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8; type9 par9; type10 par10;\
   };\
   tmp *p = (tmp*)in;\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4,p->par5,p->par6,p->par7,p->par8,p->par9,p->par10);\
}\
\
type _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9,type10 par10)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8; type9 par9; type10 par10;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9,type10 par10):par1(par1),par2(par2),par3(par3),par4(par4),par5(par5),par6(par6),par7(par7),par8(par8),par9(par9),par10(par10) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4,par5,par6,par7,par8,par9,par10);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(1);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   par5 = ((tmp*)fParam)->par5;\
   par6 = ((tmp*)fParam)->par6;\
   par7 = ((tmp*)fParam)->par7;\
   par8 = ((tmp*)fParam)->par8;\
   par9 = ((tmp*)fParam)->par9;\
   par10 = ((tmp*)fParam)->par10;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define VOID_METHOD_ARG11(klass,type,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7,type8,par8,type9,par9,type10,par10,type11,par11)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8; type9 par9; type10 par10; type11 par11;\
   };\
   tmp *p = (tmp*)in;\
   _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4,p->par5,p->par6,p->par7,p->par8,p->par9,p->par10,p->par11);\
}\
\
type _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9,type10 par10,type11 par11)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8; type9 par9; type10 par10; type11 par11;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9,type10 par10,type11 par11):par1(par1),par2(par2),par3(par3),par4(par4),par5(par5),par6(par6),par7(par7),par8(par8),par9(par9),par10(par10),par11(par11) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4,par5,par6,par7,par8,par9,par10,par11);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(1);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   par5 = ((tmp*)fParam)->par5;\
   par6 = ((tmp*)fParam)->par6;\
   par7 = ((tmp*)fParam)->par7;\
   par8 = ((tmp*)fParam)->par8;\
   par9 = ((tmp*)fParam)->par9;\
   par10 = ((tmp*)fParam)->par10;\
   par11 = ((tmp*)fParam)->par11;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
}

//______________________________________________________________________________
#define RETURN_METHOD_ARG0_CONST(klass,type,method)\
type _NAME3_(TGWin32,klass,Proxy)::method() const\
{\
   type ret;\
   TGWin32::Lock();\
   ret = _NAME3_(TGWin32,klass,Proxy)::RealObject()->method();\
   TGWin32::Unlock();\
   return ret;\
}

//______________________________________________________________________________
#define RETURN_METHOD_ARG0(klass,type,method)\
void _NAME3_(p2,klass,method)(void *in)\
{\
   struct tmp {\
      type ret;\
   };\
   tmp *p = (tmp*)in;\
   p->ret = _NAME3_(TGWin32,klass,Proxy)::RealObject()->method();\
}\
\
type _NAME3_(TGWin32,klass,Proxy)::method()\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   type ret;\
   struct tmp {\
      type ret;\
   };\
   fParam = new tmp;\
   fCallBack = &_NAME3_(p2,klass,method);\
   Bool_t batch = ForwardCallBack(1);\
   ret  = ((tmp*)fParam)->ret;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
   return ret;\
}

//______________________________________________________________________________
#define RETURN_METHOD_ARG1(klass,type,method,type1,par1)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type ret;\
   };\
   tmp *p = (tmp*)in;\
   p->ret = _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1);\
}\
\
type _NAME3_(TGWin32,klass,Proxy)::method(type1 par1)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   type ret;\
   struct tmp {\
      type1 par1; type ret;\
      tmp(type1 par1):par1(par1) {}\
   };\
   fParam = new tmp(par1);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(1);\
   par1 = ((tmp*)fParam)->par1;\
   ret  = ((tmp*)fParam)->ret;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
   return ret;\
}

//______________________________________________________________________________
#define RETURN_METHOD_ARG2(klass,type,method,type1,par1,type2,par2)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type ret;\
   };\
   tmp *p = (tmp*)in;\
   p->ret = _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2);\
}\
\
type _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   type ret;\
   struct tmp {\
      type1 par1; type2 par2; type ret;\
      tmp(type1 par1,type2 par2):par1(par1),par2(par2) {}\
   };\
   fParam = new tmp(par1,par2);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(1);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   ret  = ((tmp*)fParam)->ret;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
   return ret;\
}

//______________________________________________________________________________
#define RETURN_METHOD_ARG3(klass,type,method,type1,par1,type2,par2,type3,par3)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type ret;\
   };\
   tmp *p = (tmp*)in;\
   p->ret = _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3);\
}\
\
type _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   type ret;\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type ret;\
      tmp(type1 par1,type2 par2,type3 par3):par1(par1),par2(par2),par3(par3) {}\
   };\
   fParam = new tmp(par1,par2,par3);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(1);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   ret  = ((tmp*)fParam)->ret;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
   return ret;\
}

//______________________________________________________________________________
#define RETURN_METHOD_ARG4(klass,type,method,type1,par1,type2,par2,type3,par3,type4,par4)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type ret;\
   };\
   tmp *p = (tmp*)in;\
   p->ret = _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4);\
}\
\
type _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   type ret;\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type ret;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4):par1(par1),par2(par2),par3(par3),par4(par4) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(1);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   ret  = ((tmp*)fParam)->ret;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
   return ret;\
}

//______________________________________________________________________________
#define RETURN_METHOD_ARG5(klass,type,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type ret;\
   };\
   tmp *p = (tmp*)in;\
   p->ret = _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4,p->par5);\
}\
\
type _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   type ret;\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type ret;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5):par1(par1),par2(par2),par3(par3),par4(par4),par5(par5) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4,par5);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(1);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   par5 = ((tmp*)fParam)->par5;\
   ret  = ((tmp*)fParam)->ret;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
   return ret;\
}

//______________________________________________________________________________
#define RETURN_METHOD_ARG6(klass,type,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type ret;\
   };\
   tmp *p = (tmp*)in;\
   p->ret = _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4,p->par5,p->par6);\
}\
\
type _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   type ret;\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type ret;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6):par1(par1),par2(par2),par3(par3),par4(par4),par5(par5),par6(par6) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4,par5,par6);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(1);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   par5 = ((tmp*)fParam)->par5;\
   par6 = ((tmp*)fParam)->par6;\
   ret  = ((tmp*)fParam)->ret;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
   return ret;\
}

//______________________________________________________________________________
#define RETURN_METHOD_ARG7(klass,type,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type ret;\
   };\
   tmp *p = (tmp*)in;\
   p->ret = _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4,p->par5,p->par6,p->par7);\
}\
\
type _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   type ret;\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type ret;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7):par1(par1),par2(par2),par3(par3),par4(par4),par5(par5),par6(par6),par7(par7) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4,par5,par6,par7);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(1);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   par5 = ((tmp*)fParam)->par5;\
   par6 = ((tmp*)fParam)->par6;\
   par7 = ((tmp*)fParam)->par7;\
   ret  = ((tmp*)fParam)->ret;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
   return ret;\
}

//______________________________________________________________________________
#define RETURN_METHOD_ARG8(klass,type,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7,type8,par8)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8; type ret;\
   };\
   tmp *p = (tmp*)in;\
   p->ret = _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4,p->par5,p->par6,p->par7,p->par8);\
}\
\
type _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   type ret;\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8; type ret;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8):par1(par1),par2(par2),par3(par3),par4(par4),par5(par5),par6(par6),par7(par7),par8(par8) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4,par5,par6,par7,par8);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(1);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   par5 = ((tmp*)fParam)->par5;\
   par6 = ((tmp*)fParam)->par6;\
   par7 = ((tmp*)fParam)->par7;\
   par8 = ((tmp*)fParam)->par8;\
   ret  = ((tmp*)fParam)->ret;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
   return ret;\
}

//______________________________________________________________________________
#define RETURN_METHOD_ARG9(klass,type,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7,type8,par8,type9,par9)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8; type9 par9; type ret;\
   };\
   tmp *p = (tmp*)in;\
   p->ret = _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4,p->par5,p->par6,p->par7,p->par8,p->par9);\
}\
\
type _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   type ret;\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8; type9 par9; type ret;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9):par1(par1),par2(par2),par3(par3),par4(par4),par5(par5),par6(par6),par7(par7),par8(par8),par9(par9) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4,par5,par6,par7,par8,par9);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(1);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   par5 = ((tmp*)fParam)->par5;\
   par6 = ((tmp*)fParam)->par6;\
   par7 = ((tmp*)fParam)->par7;\
   par8 = ((tmp*)fParam)->par8;\
   par9 = ((tmp*)fParam)->par9;\
   ret  = ((tmp*)fParam)->ret;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
   return ret;\
}

//______________________________________________________________________________
#define RETURN_METHOD_ARG10(klass,type,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7,type8,par8,type9,par9,type10,par10)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8; type9 par9; type10 par10; type ret;\
   };\
   tmp *p = (tmp*)in;\
   p->ret = _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4,p->par5,p->par6,p->par7,p->par8,p->par9,p->par10);\
}\
\
type _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9,type10 par10)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   type ret;\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8; type9 par9; type10 par10; type ret;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9,type10 par10):par1(par1),par2(par2),par3(par3),par4(par4),par5(par5),par6(par6),par7(par7),par8(par8),par9(par9),par10(par10) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4,par5,par6,par7,par8,par9,par10);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(1);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   par5 = ((tmp*)fParam)->par5;\
   par6 = ((tmp*)fParam)->par6;\
   par7 = ((tmp*)fParam)->par7;\
   par8 = ((tmp*)fParam)->par8;\
   par9 = ((tmp*)fParam)->par9;\
   par10 = ((tmp*)fParam)->par10;\
   ret  = ((tmp*)fParam)->ret;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
   return ret;\
}

//______________________________________________________________________________
#define RETURN_METHOD_ARG11(klass,type,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7,type8,par8,type9,par9,type10,par10,type11,par11)\
void _NAME4_(p2,klass,method,par1)(void *in)\
{\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8; type9 par9; type10 par10; type11 par11; type ret;\
   };\
   tmp *p = (tmp*)in;\
   p->ret = _NAME3_(TGWin32,klass,Proxy)::RealObject()->method(p->par1,p->par2,p->par3,p->par4,p->par5,p->par6,p->par7,p->par8,p->par9,p->par10,p->par11);\
}\
\
type _NAME3_(TGWin32,klass,Proxy)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9,type10 par10,type11 par11)\
{\
   DEBUG_PROFILE_PROXY_START(method)\
   type ret;\
   struct tmp {\
      type1 par1; type2 par2; type3 par3; type4 par4; type5 par5; type6 par6; type7 par7; type8 par8; type9 par9; type10 par10; type11 par11; type ret;\
      tmp(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9,type10 par10,type11 par11):par1(par1),par2(par2),par3(par3),par4(par4),par5(par5),par6(par6),par7(par7),par8(par8),par9(par9),par10(par10),par11(par11) {}\
   };\
   fParam = new tmp(par1,par2,par3,par4,par5,par6,par7,par8,par9,par10,par11);\
   fCallBack = &_NAME4_(p2,klass,method,par1);\
   Bool_t batch = ForwardCallBack(1);\
   par1 = ((tmp*)fParam)->par1;\
   par2 = ((tmp*)fParam)->par2;\
   par3 = ((tmp*)fParam)->par3;\
   par4 = ((tmp*)fParam)->par4;\
   par5 = ((tmp*)fParam)->par5;\
   par6 = ((tmp*)fParam)->par6;\
   par7 = ((tmp*)fParam)->par7;\
   par8 = ((tmp*)fParam)->par8;\
   par9 = ((tmp*)fParam)->par9;\
   par10 = ((tmp*)fParam)->par10;\
   par11 = ((tmp*)fParam)->par11;\
   ret  = ((tmp*)fParam)->ret;\
   if (!batch) delete fParam;\
   DEBUG_PROFILE_PROXY_STOP(method)\
   return ret;\
}

#endif
