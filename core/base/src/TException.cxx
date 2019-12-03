// @(#)root/base:$Id$
// Author: Fons Rademakers   21/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
Exception Handling

Provide some macro's to simulate the coming C++ try, catch and throw
exception handling functionality.
*/

#include "TException.h"

#ifdef WIN32
#define R__DLLEXPORT __declspec(dllexport)
#else
#define R__DLLEXPORT
#endif

R__DLLEXPORT ExceptionContext_t *gException;

////////////////////////////////////////////////////////////////////////////////
/// If an exception context has been set (using the TRY and RETRY macros)
/// jump back to where it was set.

R__DLLEXPORT void Throw(int code)
{
   if (gException)
#ifdef NEED_SIGJMP
      siglongjmp(gException->fBuf, code);
#else
      longjmp(gException->fBuf, code);
#endif
}

R__DLLEXPORT TExceptionHandler* gExceptionHandler = nullptr;
