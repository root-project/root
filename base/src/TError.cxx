// @(#)root/base:$Name$:$Id$
// Author: Fons Rademakers   29/07/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include <stdio.h>

#include "TError.h"
#include "TSystem.h"
#include "TString.h"

int gErrorIgnoreLevel = 0;
int gErrorAbortLevel  = kSysError+1;

const char *kAssertMsg = "%s violated at line %d of `%s'";
const char *kCheckMsg  = "%s not true at line %d of `%s'";

static ErrorHandlerFunc_t errorhandler = DefaultErrorHandler;

//______________________________________________________________________________
ErrorHandlerFunc_t SetErrorHandler(ErrorHandlerFunc_t newhandler)
{
   // Set an errorhandler function. Returns the old handler.

   ErrorHandlerFunc_t oldhandler = errorhandler;
   errorhandler = newhandler;
   return oldhandler;
}

//______________________________________________________________________________
ErrorHandlerFunc_t GetErrorHandler()
{
   // Returns the current error handler function.

   return errorhandler;
}

//______________________________________________________________________________
void DefaultErrorHandler(int level, Bool_t abort, const char *location, const char *msg)
{
   // The default error handler function. It prints the message on stderr and
   // if abort is set it aborts the application.

   if (level < gErrorIgnoreLevel)
      return;

   const char *type = 0;

   if (level >= kWarning)
      type = "Warning";
   if (level >= kError)
      type = "Error";
   if (level >= kSysError)
      type = "SysError";
   if (level >= kFatal)
      type = "Fatal";

   if (!location || strlen(location) == 0)
      fprintf(stderr, "%s: %s\n", type, msg);
   else
      fprintf(stderr, "%s in <%s>: %s\n", type, location, msg);
   fflush(stderr);
   if (abort) {
      fprintf(stderr, "aborting\n");
      fflush(stderr);
      if (gSystem) {
         gSystem->StackTrace();
         gSystem->Abort();
      } else
         ::abort();
   }
}

//______________________________________________________________________________
void ErrorHandler(int level, const char *location, const char *fmt, va_list ap)
{
   // General error handler function. It calls the user set error handler
   // unless the error is of type kFatal, in which case the
   // DefaultErrorHandler() is called which will abort the application.

   char buf[1024], *bp;

   vsprintf(buf, fmt, ap);
   if (level >= kSysError && level < kFatal)
      bp = Form("%s (%s)", buf, gSystem->GetError());
   else
      bp = buf;

   if (level != kFatal)
      errorhandler(level, level >= gErrorAbortLevel, location, bp);
   else
      errorhandler(level, kTRUE, location, bp);
}

//______________________________________________________________________________
void AbstractMethod(const char *method)
{
   // This function can be used in abstract base classes in case one does
   // not want to make the class a "real" (in C++ sense) ABC. If this
   // function is called it will warn the user that the function should
   // have been overridden.

   Warning(method, "this method must be overridden!");
}

//______________________________________________________________________________
void MayNotUse(const char *method)
{
   // This function can be used in classes that should override a certain
   // function, but in the inherited class the function makes no sense.

   Warning(method, "may not use this method");
}

//______________________________________________________________________________
void Error(const char *location, const char *va_(fmt), ...)
{
   // Use this function in case an error occured.

   va_list ap;
   va_start(ap,va_(fmt));
   ErrorHandler(kError, location, va_(fmt), ap);
   va_end(ap);
}

//______________________________________________________________________________
void SysError(const char *location, const char *va_(fmt), ...)
{
   // Use this function in case a system (OS or GUI) related error occured.

   va_list ap;
   va_start(ap, va_(fmt));
   ErrorHandler(kSysError, location, va_(fmt), ap);
   va_end(ap);
}

//______________________________________________________________________________
void Warning(const char *location, const char *va_(fmt), ...)
{
   // Use this function in warning situations.

   va_list ap;
   va_start(ap,va_(fmt));
   ErrorHandler(kWarning, location, va_(fmt), ap);
   va_end(ap);
}

//______________________________________________________________________________
void Fatal(const char *location, const char *va_(fmt), ...)
{
   // Use this function in case of a fatal error. It will abort the program.

   va_list ap;
   va_start(ap,va_(fmt));
   ErrorHandler(kFatal, location, va_(fmt), ap);
   va_end(ap);
}
