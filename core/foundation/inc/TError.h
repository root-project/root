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
// errorhanlder function. Initially, a minimal, non thread-safe handler //
// is installed that is supposed to be replaced by the                  //
// DefaultErrorHandler(), which needs to be implemented and installed   //
// by the user of TError.  Normally, the default error handler is set   //
// during gROOT initialization.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include <DllImport.h> // for R__EXTERN
#include "RtypesCore.h"

#include <cstdarg>
#include <functional>


class TVirtualMutex;

R__EXTERN const Int_t kUnset;
R__EXTERN const Int_t kPrint;
R__EXTERN const Int_t kInfo;
R__EXTERN const Int_t kWarning;
R__EXTERN const Int_t kError;
R__EXTERN const Int_t kBreak;
R__EXTERN const Int_t kSysError;
R__EXTERN const Int_t kFatal;

// TROOT sets the error ignore level handler, the system error message handler, and the error abort handler on
// construction such that the "Root.ErrorIgnoreLevel" environment variable is used for the ignore level
// and gSystem is used to generate a stack trace on abort.
namespace ROOT {
namespace Internal {

/// Retrieves the error string associated with the last system error.
using ErrorSystemMsgHandlerFunc_t = std::function<const char *()>;

ErrorSystemMsgHandlerFunc_t GetErrorSystemMsgHandler();
/// Returns the previous system error message handler
ErrorSystemMsgHandlerFunc_t SetErrorSystemMsgHandler(ErrorSystemMsgHandlerFunc_t h);

void MinimalErrorHandler(int level, Bool_t abort, const char *location, const char *msg);

} // namespace Internal
} // namespace ROOT

typedef void (*ErrorHandlerFunc_t)(int level, Bool_t abort, const char *location,
              const char *msg);

extern "C" void ErrorHandler(int level, const char *location, const char *fmt, std::va_list va);

extern void DefaultErrorHandler(int level, Bool_t abort, const char *location, const char *msg);

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
