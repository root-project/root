// @(#)root/base:$Id$
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


#include "Rtypes.h"
#include <stdarg.h>


class TVirtualMutex;

const Int_t kUnset    =  -1;
const Int_t kPrint    =   0;
const Int_t kInfo     =   1000;
const Int_t kWarning  =   2000;
const Int_t kError    =   3000;
const Int_t kBreak    =   4000;
const Int_t kSysError =   5000;
const Int_t kFatal    =   6000;

R__EXTERN TVirtualMutex *gErrorMutex;

typedef void (*ErrorHandlerFunc_t)(int level, Bool_t abort, const char *location,
              const char *msg);

extern "C" void ErrorHandler(int level, const char *location, const char *fmt,
                             va_list va);

extern void DefaultErrorHandler(int level, Bool_t abort, const char *location,
                                const char *msg);

extern ErrorHandlerFunc_t SetErrorHandler(ErrorHandlerFunc_t newhandler);
extern ErrorHandlerFunc_t GetErrorHandler();

extern void Info(const char *location, const char *msgfmt, ...)
#if defined(__GNUC__) && !defined(__CINT__)
__attribute__((format(printf, 2, 3)))
#endif
;
extern void Warning(const char *location, const char *msgfmt, ...)
#if defined(__GNUC__) && !defined(__CINT__)
__attribute__((format(printf, 2, 3)))
#endif
;
extern void Error(const char *location, const char *msgfmt, ...)
#if defined(__GNUC__) && !defined(__CINT__)
__attribute__((format(printf, 2, 3)))
#endif
;
extern void Break(const char *location, const char *msgfmt, ...)
#if defined(__GNUC__) && !defined(__CINT__)
__attribute__((format(printf, 2, 3)))
#endif
;
extern void SysError(const char *location, const char *msgfmt, ...)
#if defined(__GNUC__) && !defined(__CINT__)
__attribute__((format(printf, 2, 3)))
#endif
;
extern void Fatal(const char *location, const char *msgfmt, ...)
#if defined(__GNUC__) && !defined(__CINT__)
__attribute__((format(printf, 2, 3)))
#endif
;

extern void AbstractMethod(const char *method);
extern void MayNotUse(const char *method);
extern void Obsolete(const char *function, const char *asOfVers, const char *removedFromVers);

R__EXTERN const char *kAssertMsg;
R__EXTERN const char *kCheckMsg;

#define R__ASSERT(e)                                                     \
   do {                                                                  \
      if (!(e)) ::Fatal("", kAssertMsg, _QUOTE_(e), __LINE__, __FILE__); \
   } while (false)
#define R__CHECK(e)                                                       \
   do {                                                                   \
      if (!(e)) ::Warning("", kCheckMsg, _QUOTE_(e), __LINE__, __FILE__); \
   } while (false)

R__EXTERN Int_t  gErrorIgnoreLevel;
R__EXTERN Int_t  gErrorAbortLevel;
R__EXTERN Bool_t gPrintViaErrorHandler;

#endif
