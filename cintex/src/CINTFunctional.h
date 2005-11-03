// @(#)root/reflex:$Name:$:$Id:$
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

#include "Reflex/Type.h"
#include "Reflex/Member.h"
#include "CINTdefs.h"
#include <vector>

namespace ROOT {
  namespace Cintex {

    typedef void(*FuncVoidPtr)(void);
    typedef void*(*FuncArg1Ptr)(void*);
     
    struct StubContext {
      /// Constructor. It prepares the necessary information such that the run-time processing is optimal
      StubContext(const ROOT::Reflex::Member& mem, const ROOT::Reflex::Type& cl );
      /// Destructor
      virtual ~StubContext();
      
      /// Initialization
      void Initialize();
      /// Process the function parameters to adapt from CINT to Reflex interfaces
      void ProcessParam(G__param* libp);
      /// Process the return value to adapt from Reflex to CINT
      void ProcessResult(G__value* result, void * obj);
      
      G__InterfaceMethod fMethodCode;   ///< method allocated code
      std::vector<void*> fParam;        ///< Reflex ParameterNth vector
      std::vector<G__value> fParcnv;    ///< CINT ParameterNth conversions vector
      std::vector<char> fTreat;         ///< Coded treatment of parameters
      CintTypeDesc   fRet_desc;         ///< Coded treatment of parameters
      int            fRet_tag;          ///< Return TypeNth tag number
      bool           fRet_byvalue;      ///< Return by value flag
      int            fClass_tag;        ///< Class TypeNth tag number
      ROOT::Reflex::Member fMember;     ///< Reflex FunctionMember 
      ROOT::Reflex::Type   fClass;      ///< Declaring Reflex class
      ROOT::Reflex::Type   fFunction;   ///< Reflex Function TypeNth
      int    fNpar;                     ///< number of function parameters
      ROOT::Reflex::StubFunction fStub; ///< pointer to the stub function 
      void* fStubctx;                   ///< stub function context 
      bool fInitialized;                ///< Initialized flag
    };

    int Constructor_stub(G__value*, G__CONST char*, G__param*, int );
    int Destructor_stub(G__value*, G__CONST char*, G__param*, int );
    int Method_stub(G__value*, G__CONST char*, G__param*, int );
    int Constructor_stub_with_context(StubContext*, G__value*, G__CONST char*, G__param*, int );
    int Destructor_stub_with_context(StubContext*, G__value*, G__CONST char*, G__param*, int );
    int Method_stub_with_context(StubContext*, G__value*, G__CONST char*, G__param*, int );
    char* Allocate_code(const void* src, size_t len);
    G__InterfaceMethod Allocate_stub_function( StubContext* obj, 
       int (*fun)(StubContext*, G__value*, G__CONST char*, G__param*, int ) );
    FuncVoidPtr Allocate_void_function( void* obj, void (*fun)(void*) );
    FuncArg1Ptr Allocate_1arg_function( void* obj, void* (*fun)(void*, void*) );
    void Free_function( void* );
  }
}

#endif // ROOT_Cintex_CINTFunctional
