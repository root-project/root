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
errorhandler function. By default DefaultErrorHandler() is used.
*/

#ifdef WIN32
#include <windows.h>
#endif

#include <cstdio>
#include <cstdlib>
#include "snprintf.h"
#include "Varargs.h"
#include "TError.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TVirtualMutex.h"
#include "ThreadLocalStorage.h"

#include <cctype> // for tolower
#include <cstring>
#include <string>

// Mutex for error and error format protection
// (exported to be used for similar cases in other classes)

TVirtualMutex *gErrorMutex = 0;

Int_t  gErrorIgnoreLevel     = kUnset;
Int_t  gErrorAbortLevel      = kSysError+1;
Bool_t gPrintViaErrorHandler = kFALSE;

const char *kAssertMsg = "%s violated at line %d of `%s'";
const char *kCheckMsg  = "%s not true at line %d of `%s'";

// Integrate with crash reporter.
#ifdef __APPLE__
extern "C" {
static const char *__crashreporter_info__ = 0;
asm(".desc ___crashreporter_info__, 0x10");
}
#endif

static ErrorHandlerFunc_t gErrorHandler = DefaultErrorHandler;


////////////////////////////////////////////////////////////////////////////////
/// Print debugging message to stderr and, on Windows, to the system debugger.

static void DebugPrint(const char *fmt, ...)
{
   TTHREAD_TLS(Int_t) buf_size = 2048;
   TTHREAD_TLS(char*) buf = 0;

   va_list ap;
   va_start(ap, fmt);

again:
   if (!buf)
      buf = new char[buf_size];

   Int_t n = vsnprintf(buf, buf_size, fmt, ap);
   // old vsnprintf's return -1 if string is truncated new ones return
   // total number of characters that would have been written
   if (n == -1 || n >= buf_size) {
      if (n == -1)
         buf_size *= 2;
      else
         buf_size = n+1;
      delete [] buf;
      buf = 0;
      va_end(ap);
      va_start(ap, fmt);
      goto again;
   }
   va_end(ap);

   // Serialize the actual printing.
   R__LOCKGUARD2(gErrorMutex);

   const char *toprint = buf; // Work around for older platform where we use TThreadTLSWrapper
   fprintf(stderr, "%s", toprint);

#ifdef WIN32
   ::OutputDebugString(buf);
#endif
}

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
/// The default error handler function. It prints the message on stderr and
/// if abort is set it aborts the application.

void DefaultErrorHandler(Int_t level, Bool_t abort_bool, const char *location, const char *msg)
{
   if (gErrorIgnoreLevel == kUnset) {
      R__LOCKGUARD2(gErrorMutex);

      gErrorIgnoreLevel = 0;
      if (gEnv) {
         std::string slevel;
         auto cstrlevel = gEnv->GetValue("Root.ErrorIgnoreLevel", "Print");
         while (cstrlevel && *cstrlevel) {
            slevel.push_back(tolower(*cstrlevel));
            cstrlevel++;
         }

         if (slevel == "print")
            gErrorIgnoreLevel = kPrint;
         else if (slevel == "info")
            gErrorIgnoreLevel = kInfo;
         else if (slevel == "warning")
            gErrorIgnoreLevel = kWarning;
         else if (slevel == "error")
            gErrorIgnoreLevel = kError;
         else if (slevel == "break")
            gErrorIgnoreLevel = kBreak;
         else if (slevel == "syserror")
            gErrorIgnoreLevel = kSysError;
         else if (slevel == "fatal")
            gErrorIgnoreLevel = kFatal;
      }
   }

   if (level < gErrorIgnoreLevel)
      return;

   const char *type = 0;

   if (level >= kInfo)
      type = "Info";
   if (level >= kWarning)
      type = "Warning";
   if (level >= kError)
      type = "Error";
   if (level >= kBreak)
      type = "\n *** Break ***";
   if (level >= kSysError)
      type = "SysError";
   if (level >= kFatal)
      type = "Fatal";

   std::string smsg;
   if (level >= kPrint && level < kInfo)
      smsg = msg;
   else if (level >= kBreak && level < kSysError)
      smsg = std::string(type) + " " + msg;
   else if (!location || !location[0])
      smsg = std::string(type) + ": " + msg;
   else
      smsg = std::string(type) + " in <" + location + ">: " + msg;

   DebugPrint("%s\n", smsg.c_str());

   fflush(stderr);
   if (abort_bool) {

#ifdef __APPLE__
      if (__crashreporter_info__)
         delete [] __crashreporter_info__;
      __crashreporter_info__ = strdup(smsg.c_str());
#endif

      DebugPrint("aborting\n");
      fflush(stderr);
      if (gSystem) {
         gSystem->StackTrace();
         gSystem->Abort();
      } else
         abort();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// General error handler function. It calls the user set error handler.

void ErrorHandler(Int_t level, const char *location, const char *fmt, va_list ap)
{
   TTHREAD_TLS(Int_t) buf_size(256);
   TTHREAD_TLS(char*) buf_storage(0);

   char small_buf[256];
   char *buf = buf_storage ? buf_storage : small_buf;

   int vc = 0;
   va_list sap;
   R__VA_COPY(sap, ap);

again:
   if (!buf) {
      buf_storage = buf = new char[buf_size];
   }

   if (!fmt)
      fmt = "no error message provided";

   Int_t n = vsnprintf(buf, buf_size, fmt, ap);
   // old vsnprintf's return -1 if string is truncated new ones return
   // total number of characters that would have been written
   if (n == -1 || n >= buf_size) {
      if (n == -1)
         buf_size *= 2;
      else
         buf_size = n+1;
      if (buf != &(small_buf[0])) delete [] buf;
      buf = 0;
      va_end(ap);
      R__VA_COPY(ap, sap);
      vc = 1;
      goto again;
   }
   va_end(sap);
   if (vc)
      va_end(ap);

   char *bp;
   if (level >= kSysError && level < kFatal) {
      const char *toprint = buf; // Work around for older platform where we use TThreadTLSWrapper
      bp = Form("%s (%s)", toprint, gSystem->GetError());
   } else
      bp = buf;

   if (level != kFatal)
      gErrorHandler(level, level >= gErrorAbortLevel, location, bp);
   else
      gErrorHandler(level, kTRUE, location, bp);
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

void Error(const char *location, const char *va_(fmt), ...)
{
   va_list ap;
   va_start(ap,va_(fmt));
   ErrorHandler(kError, location, va_(fmt), ap);
   va_end(ap);
}

////////////////////////////////////////////////////////////////////////////////
/// Use this function in case a system (OS or GUI) related error occurred.

void SysError(const char *location, const char *va_(fmt), ...)
{
   va_list ap;
   va_start(ap, va_(fmt));
   ErrorHandler(kSysError, location, va_(fmt), ap);
   va_end(ap);
}

////////////////////////////////////////////////////////////////////////////////
/// Use this function in case an error occurred.

void Break(const char *location, const char *va_(fmt), ...)
{
   va_list ap;
   va_start(ap,va_(fmt));
   ErrorHandler(kBreak, location, va_(fmt), ap);
   va_end(ap);
}

////////////////////////////////////////////////////////////////////////////////
/// Use this function for informational messages.

void Info(const char *location, const char *va_(fmt), ...)
{
   va_list ap;
   va_start(ap,va_(fmt));
   ErrorHandler(kInfo, location, va_(fmt), ap);
   va_end(ap);
}

////////////////////////////////////////////////////////////////////////////////
/// Use this function in warning situations.

void Warning(const char *location, const char *va_(fmt), ...)
{
   va_list ap;
   va_start(ap,va_(fmt));
   ErrorHandler(kWarning, location, va_(fmt), ap);
   va_end(ap);
}

////////////////////////////////////////////////////////////////////////////////
/// Use this function in case of a fatal error. It will abort the program.

// Fatal() *might* not abort the program (if gAbortLevel > kFatal) - but for all
// reasonable settings it *will* abort. So let's be reasonable wrt Coverity:
// coverity[+kill]
void Fatal(const char *location, const char *va_(fmt), ...)
{
   va_list ap;
   va_start(ap,va_(fmt));
   ErrorHandler(kFatal, location, va_(fmt), ap);
   va_end(ap);
}
