// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Cintex_CINTFunctional
#define ROOT_Cintex_CINTFunctional

#include "StubContext.h"

#include "G__ci.h"

class TMemberInspector;

namespace ROOT {
namespace Cintex {

//______________________________________________________________________________
int Method_stub_with_context(StubContext_t*, G__value*, G__CONST char*, G__param*, int);
int Constructor_stub_with_context(StubContext_t*, G__value*, G__CONST char*, G__param*, int);
int Destructor_stub_with_context(StubContext_t*, G__value*, G__CONST char*, G__param*, int);

//______________________________________________________________________________
typedef int (*StubFuncPtr_t)(StubContext_t*, G__value*, G__CONST char*, G__param*, int);
G__InterfaceMethod Allocate_stub_function(StubContext_t* obj, StubFuncPtr_t fun);

//______________________________________________________________________________
typedef void (*FuncVoidPtr_t)(void);
FuncVoidPtr_t Allocate_void_function(void* obj, void (*fun)(void*));

//______________________________________________________________________________
typedef void* (*FuncArg1Ptr_t)(void*);
FuncArg1Ptr_t Allocate_1arg_function(void* obj, void* (*fun)(void*, void*));

//______________________________________________________________________________
typedef void (*FuncArg3Ptr_t)(void*, TMemberInspector&, char*);
FuncArg3Ptr_t Allocate_3arg_function(void* obj, void (*fun)(void*, void*, TMemberInspector&, char*));

//______________________________________________________________________________
void Free_code(void*);

} // namespace Cintex
} // namespace ROOT

#endif // ROOT_Cintex_CINTFunctional
