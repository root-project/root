// @(#)root/cintex:$Name:  $:$Id: CINTFunctional.cxx,v 1.19 2006/11/24 14:24:54 rdm Exp $
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
#include "CINTFunctional.h"

#ifdef __linux
#include <sys/mman.h>
#endif

using namespace ROOT::Reflex;
using namespace std;

namespace ROOT { namespace Cintex {

   class StubContexts : public vector<StubContext_t*>  {
   public: 
      static StubContexts& Instance() {
         static StubContexts s_cont;
         return s_cont;
      }
   private:
      StubContexts() {}
      ~StubContexts()  {
         for( vector<StubContext_t*>::iterator j = begin(); j != end(); ++j)
            delete (*j);
         clear();
      }
   };

   StubContext_t::StubContext_t(const Member& mem, const Type& cl )
      :  fMethodCode(0), fMember(mem), fClass(cl), fNewdelfuncs(0), fInitialized(false)
   {
      // Push back a context.
      StubContexts::Instance().push_back(this);

      fFunction = mem.TypeOf();
      fNpar    = fFunction.FunctionParameterSize();
      fStub    = mem.Stubfunction();
      fStubctx = mem.Stubcontext();
      
      // for constructor or destructor locate newdelfunctions pointers
      if ( mem.IsConstructor() || mem.IsDestructor() ) {
         Member getnewdelfuncs = fClass.MemberByName("__getNewDelFunctions");
         if( getnewdelfuncs ) {
            fNewdelfuncs = (NewDelFunctions_t*)( getnewdelfuncs.Invoke().Address() );
         }
      }

   }

   StubContext_t::~StubContext_t() {
      // Destructor.
      if ( fMethodCode ) Free_function( (void*)fMethodCode );
   }

   void StubContext_t::Initialize() {
      // Initialise a context.
      fParam.resize(fNpar);
      fParcnv.resize(fNpar);
      fTreat.resize(fNpar);
      // pre-process paramters and remember the treatment that is needed to be done
      for (int i = 0; i < fNpar; i++ ) {
         Type pt = fFunction.FunctionParameterAt(i);
         while ( pt.IsTypedef() ) pt = pt.ToType();
         if ( pt.IsReference() )
            if( pt.IsPointer() ) fTreat[i] = '*';
            else                 fTreat[i] = '&';
         else if ( pt.IsFundamental() || pt.IsEnum() )
            if      ( pt.TypeInfo() == typeid(float) )       fTreat[i] = 'f';
            else if ( pt.TypeInfo() == typeid(double) )      fTreat[i] = 'd';
            else if ( pt.TypeInfo() == typeid(long double) ) fTreat[i] = 'q';
            else if ( pt.TypeInfo() == typeid(long long) )   fTreat[i] = 'n';
            else if ( pt.TypeInfo() == typeid(unsigned long long) ) fTreat[i] = 'm';
            else                                             fTreat[i] = 'i';
         else fTreat[i] = 'u';
      }

      // pre-process result block
      Type rt = fFunction.ReturnType();
      fRet_byref   = rt.IsReference();
      while ( rt.IsTypedef() ) rt = rt.ToType();
      fRet_desc = CintType( rt );
      fRet_tag  = CintTag( fRet_desc.second );
      fRet_byvalue = !fRet_byref && !rt.IsFundamental() && !rt.IsPointer() &&
         !rt.IsArray() && !rt.IsEnum(); 
      if ( rt.IsPointer() ) fRet_desc.first = (fRet_desc.first - ('a'-'A'));

      // for constructor the result block is the class itself
      if( fClass) fClass_tag = CintTag( CintType(fClass).second );
      else         fClass_tag = 0;
      // Set initialized flag
      fInitialized = true;
   }

   void StubContext_t::ProcessParam(G__param* libp) {
      // Process param type.
      fParam.resize(libp->paran);
      for (int i = 0; i < libp->paran; i++ ) {
         switch(fTreat[i]) {
         case 'd': fParcnv[i].obj.d  = G__double(libp->para[i]);fParam[i] = &fParcnv[i].obj.d; break;
         case 'f': fParcnv[i].obj.fl = (float)G__double(libp->para[i]);fParam[i] = &fParcnv[i].obj.fl; break;
         case 'n': fParcnv[i].obj.ll = G__Longlong(libp->para[i]);fParam[i] = &fParcnv[i].obj.ll; break;
         case 'm': fParcnv[i].obj.ull= G__ULonglong(libp->para[i]);fParam[i] = &fParcnv[i].obj.ull; break;
         case 'q': fParcnv[i].obj.ld = G__Longdouble(libp->para[i]);fParam[i] = &fParcnv[i].obj.ld; break;
         case 'i': fParcnv[i].obj.i  = G__int(libp->para[i]);   fParam[i] = &fParcnv[i].obj.i; break;
         case '*': fParam[i] = libp->para[i].ref ? (void*)libp->para[i].ref : &libp->para[i].obj.i; break;
         case '&': fParam[i] = (void*)libp->para[i].ref; break;
         case 'u': fParam[i] = (void*)libp->para[i].obj.i; break;
         }
      }
   }

   void StubContext_t::ProcessResult(G__value* result, void* obj) { 
      // Process ctx result.
      char t = fRet_desc.first;
      result->type = t;
      switch( t ) {
      case 'y': G__setnull(result); break;
      case 'Y': Converter<long>::toCint          (result, obj); break;
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
      case 'n': Converter<long long>::toCint     (result, obj); break;
      case 'N': Converter<int>::toCint           (result, obj); break;
      case 'm': Converter<unsigned long long>::toCint (result, obj); break;
      case 'M': Converter<int>::toCint           (result, obj); break;
      case 'f': Converter<float>::toCint         (result, obj); break;
      case 'F': Converter<int>::toCint           (result, obj); break;
      case 'd': Converter<double>::toCint        (result, obj); break;
      case 'D': Converter<int>::toCint           (result, obj); break;
      case 'q': Converter<long double>::toCint   (result, obj); break;
      case 'Q': Converter<int>::toCint           (result, obj); break;
      case 'u': Converter<long>::toCint          (result, obj);
         result->ref = (long)obj;
         result->tagnum = fRet_tag;
         break;
      case 'U': 
         if ( fRet_byref) {
            Converter<long>::toCint(result, *(void**)obj);
            result->ref = (long)obj;
         }
         else {
            Converter<long>::toCint(result, obj);
            result->ref = 0;
         }
         result->tagnum = fRet_tag;
         break;
      }
   }

   //------------------Stub adpater functions--------------------------------------------------------
   int Method_stub_with_context(StubContext_t* context,
                                G__value* result,
                                G__CONST char* /*funcname*/,
                                G__param* libp,
                                int /*hash*/ ) 
   {
      // Process method, catch exceptions.
      if ( !context->fInitialized ) context->Initialize();
      context->ProcessParam(libp);
  
      // Catch here everything since going through the adaptor in the data section
      // does not transmit the exception 
      try {
         void* r = (*context->fStub)((void*)G__getstructoffset(), context->fParam, context->fStubctx);
         context->ProcessResult(result, r);
         if ( context->fRet_byvalue )  G__store_tempobject(*result);
      } 
      catch ( std::exception& e ) {
         string errtxt("Exception: ");
         errtxt += e.what();
         errtxt += " (C++ exception)";
         G__genericerror(errtxt.c_str());
         G__setnull(result);
      } 
      catch (...) {
         G__genericerror("Exception: Unknown C++ exception");
         G__setnull(result);
      }
      return(1);
   }

   //-------------------------------------------------------------------------------------
   int Constructor_stub_with_context(StubContext_t* context, 
                                     G__value* result,
                                     G__CONST char* /*funcname*/,
                                     G__param *libp,
                                     int /*indx*/ ) 
   {
      // Process constructor, catch exceptions.
      if ( !context->fInitialized ) context->Initialize();
      context->ProcessParam(libp);
  
      void* obj=0;

      // Catch here everything since going through the adaptor in the data section
      // does not transmit the exception 
      try {
         long nary = G__getaryconstruct();
         size_t size = context->fClass.SizeOf();
         if ( nary ) {
            if( context->fNewdelfuncs ) {
               obj = context->fNewdelfuncs->fNewArray(nary, 0);
            }
            else {
               obj = ::operator new( nary * size);
               long p = (long)obj; 
               for( long i = 0; i < nary; ++i, p += size )
                  (*context->fStub)((void*)p, context->fParam, 0);
            }
         }
         else {
            obj = ::operator new( size );
            (*context->fStub)(obj, context->fParam, 0);
         }
      }
      catch ( std::exception& e ) {
         string errtxt("Exception: ");
         errtxt += e.what();
         errtxt += " (C++ exception)";
         G__genericerror(errtxt.c_str());
         ::operator delete (obj);
         obj = 0; 
      } 
      catch (...) {
         G__genericerror("Exception: Unknown C++ exception");
         ::operator delete (obj);
         obj = 0; 
      }
     
      result->obj.i = (long)obj;
      result->ref = (long)obj;
      result->type = 'u';
      result->tagnum = context->fClass_tag;
      return(1);
   }

   //-------------------------------------------------------------------------------------------------
   int Destructor_stub_with_context( StubContext_t* context,
                                     G__value* result,
                                     G__CONST char* /*funcname*/,
                                     G__param* /*libp*/,
                                     int /*indx*/ ) 
   {
      // Process destructor.
      void* obj = (void*)G__getstructoffset();
      if( 0 == obj ) return 1;
      if ( !context->fInitialized ) context->Initialize();

      if( G__getaryconstruct() ) {
         if( G__PVOID == G__getgvp() ) { //  delete[] (TYPE*)(G__getstructoffset());
            if( context->fNewdelfuncs ) context->fNewdelfuncs->fDeleteArray(obj);
         }
         else {
            size_t size = context->fClass.SizeOf();
            for(int i = G__getaryconstruct()-1; i>=0 ; i--)
               (*context->fStub)((char*)obj + size*i, context->fParam, 0);
            ::operator delete (obj);
         }
      }
      else {
         long g__Xtmp = G__getgvp();
         G__setgvp(G__PVOID);
         (*context->fStub)(obj, context->fParam, 0);
         G__setgvp(g__Xtmp);
         if( !(long(obj) == G__getgvp() && G__PVOID != G__getgvp()) )  {
            ::operator delete (obj); //G__operator_delete(obj);
         }
      }
      G__setnull(result);
      return 1;
   }


   //------ Support for functions a state -------------------------------------------------------

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

   //------ Function models-------------------------------------------------------------------
#if INT_MAX < LONG_MAX
#define FUNCPATTERN 0xFAFAFAFAFAFAFAFAL
#define DATAPATTERN 0xDADADADADADADADAL
#else
#define FUNCPATTERN 0xFAFAFAFAL
#define DATAPATTERN 0xDADADADAL
#endif

   static void f0a() {
      typedef void (*f_t)(void*);
      ((f_t)FUNCPATTERN)((void*)DATAPATTERN);
   }
   static void f1a(void* a0) {
      typedef void (*f_t)(void*,void*);
      ((f_t)FUNCPATTERN)((void*)DATAPATTERN, a0);
   }
   static void f4a(void* a0, void* a1, void* a2, void* a3) {
      typedef void (*f_t)(void*,void*,void*,void*,void*);
      ((f_t)FUNCPATTERN)((void*)DATAPATTERN, a0, a1, a2, a3);
   }

   struct FunctionCode_t {
      FunctionCode_t(int narg) : f_offset(0), fa_offset(0), fSize(0) {
         if (narg == 0)      fCode = (char*)f0a;
         else if (narg == 1) fCode = (char*)f1a;
         else if (narg == 4) fCode = (char*)f4a;
         char* b = fCode;
         for ( size_t o = 0; o < 1000; o++, b++) {
            if ( *(size_t*)b == DATAPATTERN ) fa_offset = o;
            if ( *(size_t*)b == FUNCPATTERN ) f_offset = o;
            if ( f_offset && fa_offset ) {
               fSize = (o + 32) & ~0xF;
               break;
            }
         }
      }
      size_t f_offset;
      size_t fa_offset;
      size_t fSize;
      char*  fCode;
   };

#undef DATAPATTERN
#undef FUNCPATTERN

   G__InterfaceMethod Allocate_stub_function( StubContext_t* obj, 
                                              int (*fun)(StubContext_t*, G__value*, G__CONST char*, G__param*, int ) )
   {
      // Allocate a stub function.
      static FunctionCode_t s_func4arg(4);
      char* code = Allocate_code(s_func4arg.fCode, s_func4arg.fSize );
      *(void**)&code[s_func4arg.fa_offset] = (void*)obj;
      *(void**)&code[s_func4arg.f_offset] = (void*)fun;
      obj->fMethodCode = (G__InterfaceMethod)code;
      return obj->fMethodCode;
   }


   FuncVoidPtr_t Allocate_void_function( void* obj, void (*fun)(void*) )
   {
      // Allocate a stub function.
      static FunctionCode_t s_func0arg(0);
      char* code = Allocate_code(s_func0arg.fCode, s_func0arg.fSize);
      *(void**)&code[s_func0arg.fa_offset] = (void*)obj;
      *(void**)&code[s_func0arg.f_offset] = (void*)fun;
      return (FuncVoidPtr_t)code;
   }

   FuncArg1Ptr_t Allocate_1arg_function( void* obj, void* (*fun)(void*, void*) )
   {
      // Allocate a stub function.
      static FunctionCode_t s_func1arg(1);
      char* code = Allocate_code(s_func1arg.fCode, s_func1arg.fSize);
      *(void**)&code[s_func1arg.fa_offset] = (void*)obj;
      *(void**)&code[s_func1arg.fa_offset] = (void*)fun;
      return (FuncArg1Ptr_t)code;
   }

   void Free_function( void* code )
   {
      // Free function code.
      char* scode = (char*)code;
      delete [] scode;
   }

} }   // seal and cintex namepaces
