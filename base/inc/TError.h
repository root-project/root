// @(#)root/base:$Name:  $:$Id: TError.h,v 1.3 2002/11/18 23:02:19 rdm Exp $
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

typedef void (*ErrorHandlerFunc_t)(int level, Bool_t abort, const Char_t *location,
              const Char_t *msg);

extern "C" void ErrorHandler(int level, const Char_t *location, const Char_t *fmt,
                             va_list va);

extern void DefaultErrorHandler(int level, Bool_t abort, const Char_t *location,
                                const Char_t *msg);

extern ErrorHandlerFunc_t SetErrorHandler(ErrorHandlerFunc_t newhandler);
extern ErrorHandlerFunc_t GetErrorHandler();

extern void Info(const Char_t *location, const Char_t *msgfmt, ...);
extern void Warning(const Char_t *location, const Char_t *msgfmt, ...);
extern void Error(const Char_t *location, const Char_t *msgfmt, ...);
extern void Break(const Char_t *location, const Char_t *msgfmt, ...);
extern void SysError(const Char_t *location, const Char_t *msgfmt, ...);
extern void Fatal(const Char_t *location, const Char_t *msgfmt, ...);

extern void AbstractMethod(const Char_t *method);
extern void MayNotUse(const Char_t *method);

R__EXTERN const Char_t *kAssertMsg;
R__EXTERN const Char_t *kCheckMsg;
#define Assert(e) \
        if (!(e)) Fatal("", kAssertMsg, _QUOTE_(e), __LINE__, __FILE__)
#define Check(e) \
        if (!(e)) Warning("", kCheckMsg, _QUOTE_(e), __LINE__, __FILE__)

R__EXTERN Int_t gErrorIgnoreLevel;
R__EXTERN Int_t gErrorAbortLevel;

#endif
