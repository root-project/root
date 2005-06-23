// @(#)root/reflex:$Name:$:$Id:$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Type.h"
#include "CINTdefs.h"
#include "Api.h"
#include "common.h"
#include "CINTFunctional.h"

#ifdef __linux
  #include <sys/mman.h>
#endif

using namespace ROOT::Reflex;
using namespace std;

namespace ROOT { namespace Cintex {

class StubContexts : public vector<StubContext*>  {
  public: 
    static StubContexts& Instance() {
      static StubContexts s_cont;
      return s_cont;
    }
  private:
    StubContexts() {}
    ~StubContexts()  {
      for( vector<StubContext*>::iterator j = begin(); j != end(); ++j)
        delete (*j);
      clear();
    }
  };

StubContext::StubContext(const Member& mem, const Type& cl )
  :  fMethodCode(0), fMember(mem), fClass(cl), fInitialized(false)
{
  StubContexts::Instance().push_back(this);
}

StubContext::~StubContext() {
  if ( fMethodCode ) Free_function( (void*)fMethodCode );
}

void StubContext::Initialize() {
  fFunction = fMember.TypeGet();
  fNpar    = fFunction.ParameterCount();
  fStub    = fMember.Stubfunction();
  fStubctx = fMember.Stubcontext();
  fParam.resize(fNpar);
  fParcnv.resize(fNpar);
  fTreat.resize(fNpar);
  // pre-process paramters and remember the treatment that is needed to be done
  for (int i = 0; i < fNpar; i++ ) {
    Type pt = fFunction.ParameterNth(i);
    while ( pt.IsTypedef() ) pt = pt.ToType();
    if ( pt.IsFundamental() || pt.IsEnum() )
      if      ( pt.TypeInfo() == typeid(float) )  fTreat[i] = 'f';
      else if ( pt.TypeInfo() == typeid(double) ) fTreat[i] = 'd';
      else                                        fTreat[i] = 'i';
    else if ( pt.IsReference() )
      if( pt.IsPointer() ) fTreat[i] = '*';
      else                 fTreat[i] = '&';
    else fTreat[i] = 'u';
  }

  // pre-process result block
  Type rt = fFunction.ReturnType();
  while ( rt.IsTypedef() ) rt = rt.ToType();
  fRet_desc = CintType( rt );
  fRet_tag  = CintTag( fRet_desc.second );
  if ( rt.IsPointer() ) fRet_desc.first = (fRet_desc.first - ('a'-'A'));

  // for constructor the result block is the class itself
  if( fClass) fClass_tag = CintTag( CintType(fClass).second );
  else         fClass_tag = 0;
  // Set initialized flag
  fInitialized = true;
}

void StubContext::ProcessParam(G__param* libp) {
  fParam.resize(libp->paran);
  for (int i = 0; i < libp->paran; i++ ) {
    switch(fTreat[i]) {
      case 'd': fParcnv[i].obj.d  = G__double(libp->para[i]);fParam[i] = &fParcnv[i].obj.d; break;
      case 'f': fParcnv[i].obj.fl = (float)G__double(libp->para[i]);fParam[i] = &fParcnv[i].obj.fl; break;
      case 'i': fParcnv[i].obj.i  = G__int(libp->para[i]);   fParam[i] = &fParcnv[i].obj.i; break;
      case '*': fParam[i] = &libp->para[i].obj.i; break;
      case '&': fParam[i] = (void*)libp->para[i].ref; break;
      case 'u': fParam[i] = (void*)libp->para[i].obj.i; break;
    }
  }
}

void StubContext::ProcessResult(G__value* result, void* obj) { 
  char t = fRet_desc.first;
  result->type = t;
  switch( t ) {
    case 'y': G__setnull(result); break;
    case 'g': Converter<bool>::toCint          (result, obj); break;
    case 'G': Converter<int>::toCint           (result, obj); break;
    case 'c': Converter<char>::toCint          (result, obj); break;
    case 'C': Converter<int>::toCint           (result, obj); break;
    case 'b': Converter<unsigned char>::toCint (result, obj); break;
    case 'B': Converter<int>::toCint           (result, obj); break;
    case 's': Converter<short>::toCint         (result, obj); break;
    case 'S': Converter<int>::toCint           (result, obj); break;
    case 'r': Converter<unsigned short>::toCint(result, obj); break;
    case 'R': Converter<int>::toCint           (result, obj); break;
    case 'i': Converter<int>::toCint           (result, obj); break;
    case 'I': Converter<int>::toCint           (result, obj); break;
    case 'h': Converter<unsigned int>::toCint  (result, obj); break;
    case 'H': Converter<int>::toCint           (result, obj); break;
    case 'l': Converter<long>::toCint          (result, obj); break;
    case 'L': Converter<int>::toCint           (result, obj); break;
    case 'k': Converter<unsigned long>::toCint (result, obj); break;
    case 'K': Converter<int>::toCint           (result, obj); break;
    case 'f': Converter<float>::toCint         (result, obj); break;
    case 'F': Converter<int>::toCint           (result, obj); break;
    case 'd': Converter<double>::toCint        (result, obj); break;
    case 'D': Converter<int>::toCint           (result, obj); break;
    case 'u': Converter<long>::toCint          (result, obj);
              result->ref = (long)obj;
              result->tagnum = fRet_tag;
              break;
    case 'U': Converter<long>::toCint          (result, obj);
              result->ref = 0;
              result->tagnum = fRet_tag;
              break;
  }
}

//------------------Stub adpater functions--------------------------------------------------------
int Method_stub(G__value* result,
                G__CONST char* /*funcname*/,
                G__param *libp,
                int hash ) 
{
  StubContext*    context;
  G__ifunc_table* ifunc = 0;
  int             indx  = hash;

  G__CurrentCall(G__RECMEMFUNCENV, &ifunc, indx);

  if ( ifunc ) context = (StubContext*)ifunc->userparam[indx];
  else throw RuntimeError("Unable to obtain the function context");

  if ( !context->fInitialized ) context->Initialize();

  context->ProcessParam(libp);
  void* r = (*context->fStub)((void*)G__getstructoffset(), context->fParam, context->fStubctx);
  context->ProcessResult(result, r);
  return(1);
}
//------------------Stub adpater functions--------------------------------------------------------
int Method_stub_with_context(StubContext* context,
                             G__value* result,
                             G__CONST char* /*funcname*/,
                             G__param* libp,
                             int /*hash*/ ) 
{
  if ( !context->fInitialized ) context->Initialize();
  context->ProcessParam(libp);
  void* r = (*context->fStub)((void*)G__getstructoffset(), context->fParam, context->fStubctx);
  context->ProcessResult(result, r);
  return(1);
}

//------------------------------------------------------------------------------------------------
int Constructor_stub(G__value* result,
                     G__CONST char *funcname,
                     G__param *libp,
                     int indx ) 
{
  StubContext* context;
  int tagnum;
  if ( funcname == NULL ) {
    G__ifunc_table* table = 0;
    int idx;
    G__CurrentCall(G__RECMEMFUNCENV, &table, idx);
    context = (StubContext*)G__get_linked_user_param(idx);
    tagnum = idx;
  } else {
    G__ifunc_table* ifunc = (G__ifunc_table*)funcname;
    context = (StubContext*)ifunc->userparam[indx];
    tagnum = ifunc->tagnum;
  }
  if ( !context->fInitialized ) context->Initialize();
  
  context->ProcessParam(libp);

  void* p = context->fClass.Allocate();
  (*context->fStub)(p, context->fParam, 0);
  
  result->obj.i = (long)p;
  result->ref = (long)p;
  result->type = 'u';
  result->tagnum = tagnum;

  return(1);
}
//-------------------------------------------------------------------------------------
int Constructor_stub_with_context(StubContext* context, 
                                  G__value* result,
                                  G__CONST char* /*funcname*/,
                                  G__param *libp,
                                  int /*indx*/ ) 
{
 if ( !context->fInitialized ) context->Initialize();
  
  context->ProcessParam(libp);

  void* p = context->fClass.Allocate();
  (*context->fStub)(p, context->fParam, 0);
  
  result->obj.i = (long)p;
  result->ref = (long)p;
  result->type = 'u';
  result->tagnum = context->fClass_tag;

  return(1);
}

//-------------------------------------------------------------------------------------------------
int Destructor_stub(G__value* result,
                    G__CONST char *funcname,
                    G__param* /*libp*/,
                    int indx ) 
{
  void* obj = (void*)G__getstructoffset();
  if( 0 == obj ) return 1;

  StubContext* context;
  G__ifunc_table* ifunc;
  if ( funcname == NULL ) {
    int idx;
    G__CurrentCall(G__RECMEMFUNCENV, &ifunc, idx);
  } else {
    ifunc = (G__ifunc_table*)funcname;
  }
  context = (StubContext*)ifunc->userparam[indx];
  if ( !context->fInitialized ) context->Initialize();

  if( G__getaryconstruct() ) {
    if( G__PVOID == G__getgvp() )
      operator delete[](obj); //delete[] (A::B::C::Calling *)(G__getstructoffset());
    else {
      size_t size = context->fClass.SizeOf();
      for(int i = G__getaryconstruct()-1; i>=0 ; i--)
        (*context->fStub)((char*)obj + size*i, context->fParam, 0);
      operator delete (obj);
    }
  }
  else {
    long G__Xtmp = G__getgvp();
    G__setgvp(G__PVOID);
    (*context->fStub)(obj, context->fParam, 0);
    G__setgvp(G__Xtmp);
    operator delete (obj); //G__operator_delete(obj);
  }
  G__setnull(result);
  return 1;
}//-------------------------------------------------------------------------------------------------
int Destructor_stub_with_context( StubContext* context,
                                  G__value* result,
                                  G__CONST char* /*funcname*/,
                                  G__param* /*libp*/,
                                  int /*indx*/ ) 
{
  void* obj = (void*)G__getstructoffset();
  if( 0 == obj ) return 1;
  if ( !context->fInitialized ) context->Initialize();

  if( G__getaryconstruct() ) {
    if( G__PVOID == G__getgvp() )
      operator delete[](obj); //delete[] (A::B::C::Calling *)(G__getstructoffset());
    else {
      size_t size = context->fClass.SizeOf();
      for(int i = G__getaryconstruct()-1; i>=0 ; i--)
        (*context->fStub)((char*)obj + size*i, context->fParam, 0);
      operator delete (obj);
    }
  }
  else {
    long G__Xtmp = G__getgvp();
    G__setgvp(G__PVOID);
    (*context->fStub)(obj, context->fParam, 0);
    G__setgvp(G__Xtmp);
    operator delete (obj); //G__operator_delete(obj);
  }
  G__setnull(result);
  return 1;
}

// Static function call with stored function....
unsigned char s_code_int_staticfunc_4arg[] = {
  0x55,                               // 0  push        ebp  
  0x8B, 0xEC,                         // 1  mov         ebp,esp
  0x8B, 0x45, 0x14,                   // 3  mov         eax,dword ptr [p2]   ; arg[3]
  0x50,                               // 6  push        eax                  ; puch arg
  0x8B, 0x45, 0x10,                   // 7  mov         eax,dword ptr [p2]   ; arg[2]
  0x50,                               // 10 push        eax                  ; puch arg
  0x8B, 0x45, 0x0C,                   // 11 mov         eax,dword ptr [p1]   ; arg[1]
  0x50,                               // 14 push        eax                  ; puch arg
  0x8B, 0x45, 0x08,                   // 15 mov         eax,dword ptr [p0]   ; arg[0]
  0x50,                               // 18 push        eax                  ; push arg
  0x68, 0xDE, 0xAD, 0xDE, 0xAD,       // 19 push        <object-ptr> 
  0xBA, 0xDE, 0xAD, 0xDE, 0xAD,       // 24 mov         edx, <function-pointer>
  0xFF, 0xD2,                         // 29 call        edx                  ; Off we go!
  0x83, 0xC4, 0x14,                   // 31 add         esp,20 
  0x5D,                               // 34 pop         ebp
  0xC3,                               // 35 ret
  0x90, 0x90, 0x90                    // 36 nop
};

// static void __setup(void) {  ((fun)(0xFEEDBABE))((void*)0xDEADCAFE);  }
static unsigned char s_code_void_staticfunc_0arg[] = {
  0x55,                                // 0  push        ebp  
  0x8B, 0xEC,                          // 1  mov         ebp,esp 
  0x68, 0xFE, 0xCA, 0xAD, 0xDE,        // 3  push        <argument> 
  0xB8, 0xBE, 0xBA, 0xED, 0xFE,        // 8  mov         eax, <function-pointer> 
  0xFF, 0xD0,                          // 13 call        eax  
  0x83, 0xC4, 0x04,                    // 15 add         esp,4 
  0x5D,                                // 18 pop         ebp  
  0xC3                                 // 19 ret              
};

// static void __setup(void) {  ((fun)(0xFEEDBABE))((void*)0xDEADCAFE);  }
unsigned char s_code_int_staticfunc_1arg[] = {
  0x55,                               // 0  push        ebp  
  0x8B, 0xEC,                         // 1  mov         ebp,esp
  0x8B, 0x45, 0x08,                   // 3  mov         eax,dword ptr [p0]   ; arg[0]
  0x50,                               // 6  push        eax                  ; push arg
  0x68, 0xDE, 0xAD, 0xDE, 0xAD,       // 7  push        <object-ptr> 
  0xBA, 0xDE, 0xAD, 0xDE, 0xAD,       // 12 mov         edx, <function-pointer>
  0xFF, 0xD2,                         //    call        edx                  ; Off we go!
  0x83, 0xC4, 0x08,                   //    add         esp,12 
  0x5D,                               //    pop         ebp
  0xC3                                //    ret
};

char* Allocate_code(const void* src, size_t len)  {
  char* code = new char[len+1];
  if ( !code ) return 0;
  ::memcpy(code, src, len);
  //---The following lines take care of unprotecting for execution the allocated data
  //   The usage of mprotect taken from ffcall package (http://directory.fsf.org/libs/c/ffcall.html)
#if defined(__linux) && ! defined(DATA_EXECUTABLE)
  {
    static long pagesize = 0;
    if ( !pagesize ) pagesize = getpagesize();
    unsigned long start_addr = (unsigned long) code;
    unsigned long end_addr   = (unsigned long) (code + len);
    start_addr = start_addr & -pagesize;
    end_addr   = (end_addr + pagesize-1) & -pagesize;
    unsigned long len = end_addr - start_addr;
    if ( mprotect( (void*)start_addr, len, PROT_READ|PROT_WRITE|PROT_EXEC) < 0 ) {
      return 0;
    }
  }
#endif
  return code;
}

G__InterfaceMethod Allocate_stub_function( StubContext* obj, 
       int (*fun)(StubContext*, G__value*, G__CONST char*, G__param*, int ) )
{
  char* code = Allocate_code(s_code_int_staticfunc_4arg,sizeof(s_code_int_staticfunc_4arg));
  *(void**)&code[20] = (void*)obj;
  *(void**)&code[25] = (void*)fun;
  obj->fMethodCode = (G__InterfaceMethod)code;
  return obj->fMethodCode;
}


FuncVoidPtr Allocate_void_function( void* obj, void (*fun)(void*) )
{
  char* code = Allocate_code(s_code_void_staticfunc_0arg,sizeof(s_code_void_staticfunc_0arg));
  *(void**)&code[4] = (void*)obj;
  *(void**)&code[9] = (void*)fun;
  return (FuncVoidPtr)code;
}

FuncArg1Ptr Allocate_1arg_function( void* obj, void* (*fun)(void*, void*) )
{
  char* code = Allocate_code(s_code_int_staticfunc_1arg,sizeof(s_code_int_staticfunc_1arg));
  *(void**)&code[8]  = (void*)obj;
  *(void**)&code[13] = (void*)fun;
  return (FuncArg1Ptr)code;
}

void Free_function( void* code )
{
  char* scode = (char*)code;
  delete [] scode;
}




} }   // seal and cintex namepaces
