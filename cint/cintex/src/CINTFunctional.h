// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
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

      typedef void(*FuncVoidPtr_t)(void);
      typedef void*(*FuncArg1Ptr_t)(void*);
    
      typedef void* (*NewFunc_t)( void* );
      typedef void* (*NewArrFunc_t)( long size, void *arena );
      typedef void  (*DelFunc_t)( void* );
      typedef void  (*DelArrFunc_t)( void* );
      typedef void  (*DesFunc_t)( void* ); 
    
      struct NewDelFunctions_t {
         NewFunc_t    fNew;             //pointer to a function newing one object.
         NewArrFunc_t fNewArray;        //pointer to a function newing an array of objects.
         DelFunc_t    fDelete;          //pointer to a function deleting one object.
         DelArrFunc_t fDeleteArray;     //pointer to a function deleting an array of objects.
         DesFunc_t    fDestructor;      //pointer to a function call an object's destructor.
      };

     
      struct StubContext_t {
         struct ParCnvInfo_t {
            ParCnvInfo_t(): fTreat(0) {
               fValCINT.obj.i = 0;
               fValCINT.ref = 0;
               fValCINT.type = 0;
               fValCINT.tagnum = -1;
               fValCINT.typenum = -1;
               fValCINT.isconst = 0;
            }
            G__value fValCINT; ///< CINT parameter value
            char     fTreat;   ///< Coded treatment of parameters
         };

         /// Constructor. It prepares the necessary information such that the run-time processing is optimal
         StubContext_t(const ROOT::Reflex::Member& mem, const ROOT::Reflex::Type& cl );
         /// Destructor
         virtual ~StubContext_t();
      
         /// Initialization
         void Initialize();
         /// Process the function parameters to adapt from CINT to Reflex interfaces
         void ProcessParam(G__param* libp);
         /// Process the return value to adapt from Reflex to CINT
         void ProcessResult(G__value* result, void * obj);
         /// Return the address of the return value within result
         void* GetReturnAddress(G__value* result) const;
      
         G__InterfaceMethod fMethodCode;   ///< method allocated code
         std::vector<void*> fParam;        ///< Reflex parameter vector
         static const int fgNumParCnvFirst = 5; ///< Entries in fParInfFirst
         ParCnvInfo_t fParCnvFirst[fgNumParCnvFirst]; ///< Conversion info for first five parameters
         std::vector<ParCnvInfo_t>* fParCnvLast; ///< Conversion info for parameters beyond fParInfFirst
         CintTypeDesc   fRet_desc;         ///< Coded treatment of parameters
         int            fRet_tag;          ///< Return TypeNth tag number
         bool           fRet_byvalue;      ///< Return by value flag
         bool           fRet_byref;        ///< Return by reference flag
         int            fRet_plevel;       ///< Pointer/Reference level
         int            fClass_tag;        ///< Class TypeNth tag number
         size_t         fRet_Sizeof;       ///< Sizeof returned by value type
         ROOT::Reflex::Type   fClass;      ///< Declaring Reflex class
         ROOT::Reflex::Type   fFunction;   ///< Reflex Function TypeNth
         int    fNpar;                     ///< number of function parameters
         ROOT::Reflex::StubFunction fStub; ///< pointer to the stub function 
         void* fStubctx;                   ///< stub function context 
         NewDelFunctions_t* fNewdelfuncs;    ///< Pointer to the NewDelFunctions structure
         bool fInitialized;                ///< Initialized flag
      };

      int Constructor_stub(G__value*, G__CONST char*, G__param*, int );
      int Destructor_stub(G__value*, G__CONST char*, G__param*, int );
      int Method_stub(G__value*, G__CONST char*, G__param*, int );
      int Constructor_stub_with_context(StubContext_t*, G__value*, G__CONST char*, G__param*, int );
      int Destructor_stub_with_context(StubContext_t*, G__value*, G__CONST char*, G__param*, int );
      int Method_stub_with_context(StubContext_t*, G__value*, G__CONST char*, G__param*, int );
      char* Allocate_code(const void* src, size_t len);
      G__InterfaceMethod Allocate_stub_function( StubContext_t* obj, 
                                                 int (*fun)(StubContext_t*, G__value*, G__CONST char*, G__param*, int ) );
      FuncVoidPtr_t Allocate_void_function( void* obj, void (*fun)(void*) );
      FuncArg1Ptr_t Allocate_1arg_function( void* obj, void* (*fun)(void*, void*) );
      void Free_function( void* );
   }
}

#endif // ROOT_Cintex_CINTFunctional
