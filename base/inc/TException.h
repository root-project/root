// @(#)root/base:$Name$:$Id$
// Author: Fons Rademakers   21/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TException
#define ROOT_TException


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Exception Handling                                                   //
//                                                                      //
// Provide some macro's to simulate the coming C++ try, catch and throw //
// exception handling functionality.                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include <setjmp.h>
#else
struct sigjmp_buf;
struct jmp_buf;
#endif

#include "RConfig.h"
#include "DllImport.h"

struct ExceptionContext_t {
#ifdef NEED_SIGJMP
   sigjmp_buf buf;
#else
   jmp_buf buf;
#endif
};

#ifdef NEED_SIGJMP
#define SETJMP(buf) sigsetjmp(buf,1)
#else
#define SETJMP(buf) setjmp(buf)
#endif

#define RETRY \
   { \
      ExceptionContext_t __curr, *__old = gException; \
      int __code; \
      gException = &__curr; \
      __code = SETJMP(gException->buf); {

#define TRY \
   { \
      ExceptionContext_t __curr, *__old = gException; \
      int __code; \
      gException = &__curr; \
      if ((__code = SETJMP(gException->buf)) == 0) {

#define CATCH(n) \
         gException = __old; \
      } else { \
         int n = __code; \
         gException = __old;

#define ENDTRY \
      } \
      gException = __old; \
   }

R__EXTERN ExceptionContext_t *gException;

extern void Throw(int code);

#endif
