// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Cintex_StubContext
#define ROOT_Cintex_StubContext

#include "CINTdefs.h"
#include "Reflex/Kernel.h"
#include "Reflex/Member.h"
#include "Reflex/Type.h"
#include "G__ci.h"
#include "Api.h"
#include <vector>

namespace ROOT {
namespace Cintex {

//______________________________________________________________________________
typedef void* (*NewFunc_t)(void*);
typedef void* (*NewArrFunc_t)(long size, void *arena);
typedef void (*DelFunc_t)(void*);
typedef void (*DelArrFunc_t)(void*);
typedef void (*DesFunc_t)(void*);

//______________________________________________________________________________
class NewDelFunctions_t {
public:
   NewFunc_t    fNew;         //pointer to a function newing one object.
   NewArrFunc_t fNewArray;    //pointer to a function newing an array of objects.
   DelFunc_t    fDelete;      //pointer to a function deleting one object.
   DelArrFunc_t fDeleteArray; //pointer to a function deleting an array of objects.
   DesFunc_t    fDestructor;  //pointer to a function call an object's destructor.
};

//______________________________________________________________________________
class StubContext_t {
   // Packet of information about a compiled function.
public:
   ROOT::Reflex::Member fMbr; // this function as a reflex member 
   G__InterfaceMethod fMethodCode; // cint stub code
   std::vector<void*> fParam; // function arguments in reflex stub format
   std::vector<char> fTreat; // coded treatment of function arguments
   NewDelFunctions_t* fNewdelfuncs; // the NewDelFunctions structure
   bool fInitialized; // flag, Initialize() has been called
public:
   StubContext_t(ROOT::Reflex::Member);
   virtual ~StubContext_t();
   void Initialize();
   void ProcessParam(G__param*);
   void ProcessResult(G__value* result, void* obj);
   void* GetReturnAddress(G__value* result) const;
};

} // namespace Cintex
} // namespace ROOT

#endif // ROOT_Cintex_StubContext
