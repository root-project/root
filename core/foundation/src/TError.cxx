// @(#)root/base:$Id$
// Author: Fons Rademakers   29/07/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
Error handling routines.

This file defines a number of global error handling routines:
Warning(), Error(), SysError() and Fatal(). They all take a
location string (where the error happened) and a printf style format
string plus vararg's. In the end these functions call an
errorhandler function. Initially the MinimalErrorHandler, which is supposed
to be replaced by the proper DefaultErrorHandler()
*/

#include "TError.h"

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>

Int_t  gErrorIgnoreLevel     = kUnset;
Int_t  gErrorAbortLevel      = kSysError+1;
Bool_t gPrintViaErrorHandler = kFALSE;

const char *kAssertMsg = "%s violated at line %d of `%s'";
const char *kCheckMsg  = "%s not true at line %d of `%s'";

static ErrorHandlerFunc_t gErrorHandler = ROOT::Internal::MinimalErrorHandler;


static ROOT::Internal::ErrorSystemMsgHandlerFunc_t &GetErrorSystemMsgHandlerRef()
{
   static ROOT::Internal::ErrorSystemMsgHandlerFunc_t h;
   return h;
}


namespace ROOT {
namespace Internal {

ErrorSystemMsgHandlerFunc_t GetErrorSystemMsgHandler()
{
   return GetErrorSystemMsgHandlerRef();
}

ErrorSystemMsgHandlerFunc_t SetErrorSystemMsgHandler(ErrorSystemMsgHandlerFunc_t h)
{
   auto oldHandler = GetErrorSystemMsgHandlerRef();
   GetErrorSystemMsgHandlerRef() = h;
   return oldHandler;
}

/// A very simple error handler that is usually replaced by the TROOT default error handler.
/// The minimal error handler is not serialized across threads, so that output of multi-threaded programs
/// can get scrambled
/// @note `abort()` is only called if `abort_bool` is `true` and `level >= gErrorIgnoreLevel`
void MinimalErrorHandler(Int_t level, Bool_t abort_bool, const char *location, const char *msg)
{
   if (level < gErrorIgnoreLevel)
      return;

   if (level >= kBreak)
      fprintf(stderr, "\n *** Break *** ");
   fprintf(stderr, "<%s>: %s\n", location ? location : "unspecified location", msg);
   fflush(stderr);
   if (abort_bool) {
      fprintf(stderr, "aborting\n");
      fflush(stderr);
      abort();
   }
}

} // namespace Internal
} // namespace ROOT


////////////////////////////////////////////////////////////////////////////////
/// Set an errorhandler function. Returns the old handler.

ErrorHandlerFunc_t SetErrorHandler(ErrorHandlerFunc_t newhandler)
{
   ErrorHandlerFunc_t oldhandler = gErrorHandler;
   gErrorHandler = newhandler;
   return oldhandler;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the current error handler function.

ErrorHandlerFunc_t GetErrorHandler()
{
   return gErrorHandler;
}


////////////////////////////////////////////////////////////////////////////////
/// General error handler function. It calls the user set error handler.

void ErrorHandler(Int_t level, const char *location, const char *fmt, std::va_list ap)
{
   if (!fmt)
      fmt = "no error message provided";

   char smallBuf[256];

   thread_local int bufSize = sizeof(smallBuf);
   thread_local char *bufDynStorage = nullptr;

   char *buf = bufDynStorage ? bufDynStorage : smallBuf;

   std::va_list apCopy;
   va_copy(apCopy, ap);

   // Figure out exactly how many bytes we need
   int nRequired = vsnprintf(nullptr, 0, fmt, ap) + 1;
   if (level >= kSysError && level < kFatal) {
      auto sysHandler = GetErrorSystemMsgHandlerRef();
      if (sysHandler)
         nRequired += strlen(sysHandler()) + 1; // +1 for the whitespace
      else {
         nRequired += snprintf(nullptr, 0, " (errno: %d)", errno) + 1;
      }
   }

   if (nRequired >= bufSize) {
      // Not enough space: allocate more space on the heap to fit the string
      if (buf != smallBuf)
         delete[] buf;

      bufSize = std::max(bufSize * 2, nRequired + 1);
      buf = bufDynStorage = new char[bufSize];
   }

   // Actually write the string
   int nWrittenPre = vsnprintf(buf, bufSize, fmt, apCopy);
   int nWrittenPost = 0;
   if (level >= kSysError && level < kFatal) {
      auto sysHandler = GetErrorSystemMsgHandlerRef();
      if (sysHandler) {
         // NOTE: overwriting the null byte written by the previous vsnprintf
         nWrittenPost = snprintf(buf + nWrittenPre, bufSize - nWrittenPre, " %s", sysHandler());
      } else {
         // NOTE: overwriting the null byte written by the previous vsnprintf
         nWrittenPost = snprintf(buf + nWrittenPre, bufSize - nWrittenPre, " (errno: %d)", errno);
      }
   }
   assert(nWrittenPre + nWrittenPost + 1 <= nRequired);
   va_end(apCopy);

   if (level != kFatal)
      gErrorHandler(level, level >= gErrorAbortLevel, location, buf);
   else
      gErrorHandler(level, kTRUE, location, buf);
}

////////////////////////////////////////////////////////////////////////////////
/// This function can be used in abstract base classes in case one does
/// not want to make the class a "real" (in C++ sense) ABC. If this
/// function is called it will warn the user that the function should
/// have been overridden.

void AbstractMethod(const char *method)
{
   Warning(method, "this method must be overridden!");
}

////////////////////////////////////////////////////////////////////////////////
/// This function can be used in classes that should override a certain
/// function, but in the inherited class the function makes no sense.

void MayNotUse(const char *method)
{
   Warning(method, "may not use this method");
}

////////////////////////////////////////////////////////////////////////////////
/// Use this function to declare a function obsolete. Specify as of which version
/// the method is obsolete and as from which version it will be removed.

void Obsolete(const char *function, const char *asOfVers, const char *removedFromVers)
{
   Warning(function, "obsolete as of %s and will be removed from %s", asOfVers, removedFromVers);
}

////////////////////////////////////////////////////////////////////////////////
/// Use this function in case an error occurred.

void Error(const char *location, const char *fmt, ...)
{
   std::va_list ap;
   va_start(ap, fmt);
   ErrorHandler(kError, location, fmt, ap);
   va_end(ap);
}

////////////////////////////////////////////////////////////////////////////////
/// Use this function in case a system (OS or GUI) related error occurred.

void SysError(const char *location, const char *fmt, ...)
{
   std::va_list ap;
   va_start(ap, fmt);
   ErrorHandler(kSysError, location, fmt, ap);
   va_end(ap);
}

////////////////////////////////////////////////////////////////////////////////
/// Use this function in case an error occurred.

void Break(const char *location, const char *fmt, ...)
{
   std::va_list ap;
   va_start(ap, fmt);
   ErrorHandler(kBreak, location, fmt, ap);
   va_end(ap);
}

////////////////////////////////////////////////////////////////////////////////
/// Use this function for informational messages.

void Info(const char *location, const char *fmt, ...)
{
   std::va_list ap;
   va_start(ap, fmt);
   ErrorHandler(kInfo, location, fmt, ap);
   va_end(ap);
}

////////////////////////////////////////////////////////////////////////////////
/// Use this function in warning situations.

void Warning(const char *location, const char *fmt, ...)
{
   std::va_list ap;
   va_start(ap, fmt);
   ErrorHandler(kWarning, location, fmt, ap);
   va_end(ap);
}

////////////////////////////////////////////////////////////////////////////////
/// Use this function in case of a fatal error. It will abort the program.

/// @warning Fatal() *will* not abort the program if `gErrorIgnoreLevel > kFatal`
/// - but for all reasonable settings it *will* abort.
// So let's be reasonable wrt Coverity:
// coverity[+kill]
void Fatal(const char *location, const char *fmt, ...)
{
   std::va_list ap;
   va_start(ap, fmt);
   ErrorHandler(kFatal, location, fmt, ap);
   va_end(ap);
}
