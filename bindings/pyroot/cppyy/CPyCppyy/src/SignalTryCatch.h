// Partial reproduction of ROOT's TException.h

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef CPYCPPYY_SIGNALTRYCATCH_H
#define CPYCPPYY_SIGNALTRYCATCH_H

#include <setjmp.h>
#include "CPyCppyy/CommonDefs.h"

#ifndef _WIN32
#define NEED_SIGJMP 1
#endif

// By default, the ExceptionContext_t class is expected in the namespace
// CppyyLegacy, If it is expected in no namespace, one can explicitly define
// NO_CPPYY_LEGACY_NAMESPACE at build time (e.g. if one wants to use ROOT).

#ifndef NO_CPPYY_LEGACY_NAMESPACE
namespace CppyyLegacy {
#endif
struct ExceptionContext_t {
#ifdef NEED_SIGJMP
    sigjmp_buf fBuf;
#else
    jmp_buf fBuf;
#endif
};
#ifndef NO_CPPYY_LEGACY_NAMESPACE
}

using CppyyExceptionContext_t = CppyyLegacy::ExceptionContext_t;
#else
using CppyyExceptionContext_t = ExceptionContext_t;
#endif

#ifdef NEED_SIGJMP
# define CLING_EXCEPTION_SETJMP(buf) sigsetjmp(buf,1)
#else
# define CLING_EXCEPTION_SETJMP(buf) setjmp(buf)
#endif

#define CLING_EXCEPTION_RETRY \
    { \
        static CppyyExceptionContext_t R__curr, *R__old = gException; \
        int R__code; \
        gException = &R__curr; \
        R__code = CLING_EXCEPTION_SETJMP(gException->fBuf); if (R__code) { }; {

#define CLING_EXCEPTION_TRY \
    { \
        static CppyyExceptionContext_t R__curr, *R__old = gException; \
        int R__code; \
        gException = &R__curr; \
        if ((R__code = CLING_EXCEPTION_SETJMP(gException->fBuf)) == 0) {

#define CLING_EXCEPTION_CATCH(n) \
            gException = R__old; \
        } else { \
            int n = R__code; \
            gException = R__old;

#define CLING_EXCEPTION_ENDTRY \
        } \
        gException = R__old; \
    }

// extern, defined in ROOT Core
#ifdef _MSC_VER
extern __declspec(dllimport) CppyyExceptionContext_t *gException;
#else
extern CppyyExceptionContext_t *gException;
#endif

#endif
