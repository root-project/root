// @(#)root/base:$Name:  $:$Id: TError.h,v 1.5 2005/06/23 00:29:37 rdm Exp $
// Author: Fons Rademakers   29/07/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TError
#define ROOT_TError


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Error handling routines.                                             //
//                                                                      //
// This file defines a number of global error handling routines:        //
// Warning(), Error(), SysError() and Fatal(). They all take a          //
// location string (where the error happened) and a printf style format //
// string plus vararg's. In the end these functions call an             //
// errorhanlder function. By default DefaultErrorHandler() is used.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_Varargs
#include "Varargs.h"
#endif

class TVirtualMutex;

const Int_t kUnset    =  -1;
const Int_t kInfo     =   0;
const Int_t kWarning  =   1000;
const Int_t kError    =   2000;
const Int_t kBreak    =   3000;
const Int_t kSysError =   4000;
const Int_t kFatal    =   5000;

R__EXTERN TVirtualMutex *gErrorMutex;

typedef void (*ErrorHandlerFunc_t)(int level, Bool_t abort, const char *location,
              const char *msg);

extern "C" void ErrorHandler(int level, const char *location, const char *fmt,
                             va_list va);

extern void DefaultErrorHandler(int level, Bool_t abort, const char *location,
                                const char *msg);

extern ErrorHandlerFunc_t SetErrorHandler(ErrorHandlerFunc_t newhandler);
extern ErrorHandlerFunc_t GetErrorHandler();

extern void Info(const char *location, const char *msgfmt, ...);
extern void Warning(const char *location, const char *msgfmt, ...);
extern void Error(const char *location, const char *msgfmt, ...);
extern void Break(const char *location, const char *msgfmt, ...);
extern void SysError(const char *location, const char *msgfmt, ...);
extern void Fatal(const char *location, const char *msgfmt, ...);

extern void AbstractMethod(const char *method);
extern void MayNotUse(const char *method);

R__EXTERN const char *kAssertMsg;
R__EXTERN const char *kCheckMsg;

#define R__ASSERT(e) \
   if (!(e)) Fatal("", kAssertMsg, _QUOTE_(e), __LINE__, __FILE__)
#define R__CHECK(e) \
   if (!(e)) Warning("", kCheckMsg, _QUOTE_(e), __LINE__, __FILE__)

// deprecated macros (will be removed in next release)
#define Assert(e) \
   { if (!(e)) Fatal("", kAssertMsg, _QUOTE_(e), __LINE__, __FILE__); \
   Warning("", "please change Assert to R__ASSERT in %s at line %d", __FILE__, __LINE__); }
#define Check(e) \
   { if (!(e)) Warning("", kCheckMsg, _QUOTE_(e), __LINE__, __FILE__); \
   Warning("", "please change Check to R__CHECK in %s at line %d", __FILE__, __LINE__); }

R__EXTERN Int_t gErrorIgnoreLevel;
R__EXTERN Int_t gErrorAbortLevel;

#endif
