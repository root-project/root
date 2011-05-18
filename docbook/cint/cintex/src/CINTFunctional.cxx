// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Type.h"
#include "Reflex/Tools.h"
#include "CINTdefs.h"
#include "Api.h"
#include "CINTFunctional.h"
#include "RConfig.h"

#ifndef CINTEX_USE_MMAP
# if (defined(__linux) || defined(R__MACOSX)) && ! defined(DATA_EXECUTABLE)
#  define CINTEX_USE_MMAP 1
# else
#  define CINTEX_USE_MMAP 0
# endif
#endif


#if CINTEX_USE_MMAP
#include <sys/mman.h>
#elif defined(_WIN32)
typedef unsigned long WIN_DWORD;
typedef void *WIN_LPVOID;
extern "C" {
__declspec(dllimport) WIN_LPVOID __stdcall
VirtualAlloc(WIN_LPVOID, size_t, WIN_DWORD, WIN_DWORD);

__declspec(dllimport) int __stdcall
VirtualFree(WIN_LPVOID, size_t, WIN_DWORD);
}
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
      :  fMethodCode(0), fParam(mem.FunctionParameterSize(), 0),
         fParCnvLast(0), fRet_tag(-1), fRet_byvalue(false),
         fRet_byref(false), fRet_plevel(0), fClass_tag(-1), fRet_Sizeof(0),
         fClass(cl), fNpar(0), fStub(0), fStubctx(0), fNewdelfuncs(0),
         fInitialized(false)
   {
      // Push back a context.
      StubContexts::Instance().push_back(this);

      fFunction = mem.TypeOf();
      fNpar    = fFunction.FunctionParameterSize();
      fStub    = mem.Stubfunction();
      fStubctx = mem.Stubcontext();
      
      // for constructor or destructor locate newdelfunctions pointers
      if ( mem.IsConstructor() || mem.IsDestructor() ) {
         Member getnewdelfuncs = fClass.FunctionMemberByName("__getNewDelFunctions", Reflex::Type(),
                                                             0, INHERITEDMEMBERS_NO, DELAYEDLOAD_OFF);
         if( getnewdelfuncs ) {
            static Type tNewdelfuncs = Type::ByTypeInfo(typeid(fNewdelfuncs));
            Object ret( tNewdelfuncs, (void*)&fNewdelfuncs);
            getnewdelfuncs.Invoke(&ret);
         }
      }

   }

   StubContext_t::~StubContext_t() {
      // Destructor.
      if ( fMethodCode ) Free_function( (void*)fMethodCode );
      if (fParCnvLast) delete fParCnvLast;
   }

   void StubContext_t::Initialize() {
      // Initialize a context.
      if (fNpar > fgNumParCnvFirst) {
         if (!fParCnvLast) {
            fParCnvLast = new std::vector<ParCnvInfo_t>(fNpar - fgNumParCnvFirst);
         } else {
            fParCnvLast->resize(fNpar - fgNumParCnvFirst);
         }
      } else {
         delete fParCnvLast;
         fParCnvLast = 0;
      }

      // pre-process paramters and remember the treatment that is needed to be done
      for (int i = 0; i < fNpar; i++ ) {
         ParCnvInfo_t* parInfo =  i < fgNumParCnvFirst ? &fParCnvFirst[i] : &((*fParCnvLast)[i - fgNumParCnvFirst]);
         Type pt = fFunction.FunctionParameterAt(i);
         while ( pt.IsTypedef() ) pt = pt.ToType();
         if ( pt.IsReference() && ! pt.IsConst() )
            if( pt.IsPointer() ) parInfo->fTreat = '*';
            else                 parInfo->fTreat = '&';
         else if ( pt.IsFundamental() || pt.IsEnum() )
            if      ( pt.TypeInfo() == typeid(float) )       parInfo->fTreat = 'f';
            else if ( pt.TypeInfo() == typeid(double) )      parInfo->fTreat = 'd';
            else if ( pt.TypeInfo() == typeid(long double) ) parInfo->fTreat = 'q';
            else if ( pt.TypeInfo() == typeid(long long) )   parInfo->fTreat = 'n';
            else if ( pt.TypeInfo() == typeid(unsigned long long) ) parInfo->fTreat = 'm';
            else                                             parInfo->fTreat = 'i';
         else parInfo->fTreat = 'u';
      }

      // pre-process result block
      Type rt = fFunction.ReturnType();
      fRet_Sizeof = rt.SizeOf();
      if (fRet_Sizeof==0) {
         // Humm a type with sizeof 0 ... more likely to be a type unknown to reflex ...
         fRet_Sizeof = G__Lsizeof( rt.Name( Reflex::SCOPED ).c_str() );
      }
      fRet_byref   = rt.IsReference();
      while ( rt.IsTypedef() ) rt = rt.ToType();
      fRet_desc = CintType( rt );
      fRet_tag  = CintTag( fRet_desc.second );
      fRet_byvalue = !fRet_byref && !rt.IsFundamental() && !rt.IsPointer() &&
         !rt.IsArray() && !rt.IsEnum(); 
      int plevel = 0;
      Type frt = rt.FinalType();
      while (frt.IsPointer()) {
         // Note: almost right (does not handle array or reference in between)
         ++plevel;
         frt = frt.ToType();
      }
      if ( rt.IsPointer() ) {
         fRet_desc.first = (fRet_desc.first - ('a'-'A'));
         --plevel;
      }
      fRet_plevel = plevel;

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
         ParCnvInfo_t* parInfo =  i < fgNumParCnvFirst ? &fParCnvFirst[i] : &((*fParCnvLast)[i - fgNumParCnvFirst]);
         switch(parInfo->fTreat) {
         case 'd':
            parInfo->fValCINT.obj.d  = G__double(libp->para[i]);
            fParam[i] = &parInfo->fValCINT.obj.d;
            break;
         case 'f':
            parInfo->fValCINT.obj.fl = (float)G__double(libp->para[i]);
            fParam[i] = &parInfo->fValCINT.obj.fl;
            break;
         case 'n':
            parInfo->fValCINT.obj.ll = G__Longlong(libp->para[i]);
            fParam[i] = &parInfo->fValCINT.obj.ll;
            break;
         case 'm':
            parInfo->fValCINT.obj.ull= G__ULonglong(libp->para[i]);
            fParam[i] = &parInfo->fValCINT.obj.ull;
            break;
         case 'q':
            parInfo->fValCINT.obj.ld = G__Longdouble(libp->para[i]);
            fParam[i] = &parInfo->fValCINT.obj.ld;
            break;
         case 'i':
            parInfo->fValCINT.obj.i  = G__int(libp->para[i]);
            fParam[i] = &parInfo->fValCINT.obj.i;
            break;
         case '*':
            fParam[i] = libp->para[i].ref ? (void*)libp->para[i].ref : &libp->para[i].obj.i;
            break;
         case '&':
            fParam[i] = (void*)libp->para[i].ref;
            break;
         case 'u':
            fParam[i] = (void*)libp->para[i].obj.i;
            break;
         }
      }
   }

   void* StubContext_t::GetReturnAddress(G__value* result) const {
      // Extract the memory location of the return value given the return type of fMethod
      Type ret = fFunction.ReturnType().FinalType();
      if (ret.IsPointer())
         return &result->obj.i;
      if (ret.IsReference())
         return &result->ref;
      switch (Tools::FundamentalType(ret)) {
      case kFLOAT:
      case kDOUBLE: return &result->obj.d;
      case kLONG_DOUBLE: return &result->obj.ld;
      case kVOID: return 0;
      case kLONGLONG:
      case kULONGLONG: return &result->obj.ll;
      default: ;
      }
      return &result->obj.i;
   }

   void StubContext_t::ProcessResult(G__value* result, void* objaddr) { 
      // Process ctx result.
      
      char t = fRet_desc.first;
      result->type = t;
      void *obj = objaddr;
      if ( fRet_byref ) { 
         result->ref = (long) *(void**)objaddr;
         obj = *(void**)objaddr;
         result->tagnum = fRet_tag;
      } else {
         result->ref = 0;
      }
      switch( t ) {
      case 'y': G__setnull(result); break;
      case 'g': Converter<bool>::toCint          (result, obj); break;
      case 'c': Converter<char>::toCint          (result, obj); break;

      case 'Y': Converter<long>::toCint           (result, *(void**)obj); break;
      case 'G': Converter<long>::toCint           (result, *(void**)obj); break;
      case 'C': Converter<long>::toCint           (result, *(void**)obj); break;
      case 'B': Converter<long>::toCint           (result, *(void**)obj); break;
      case 'R': Converter<long>::toCint           (result, *(void**)obj); break;
      case 'I': Converter<long>::toCint           (result, *(void**)obj); break;
      case 'H': Converter<long>::toCint           (result, *(void**)obj); break;
      case 'K': Converter<long>::toCint           (result, *(void**)obj); break;
      case 'M': Converter<long>::toCint           (result, *(void**)obj); break;
      case 'N': Converter<long>::toCint           (result, *(void**)obj); break;
      case 'S': Converter<long>::toCint           (result, *(void**)obj); break;
      case 'F': Converter<long>::toCint           (result, *(void**)obj); break;
      case 'D': Converter<long>::toCint           (result, *(void**)obj); break;
      case 'L': Converter<long>::toCint           (result, *(void**)obj); break;

      case 'b': Converter<unsigned char>::toCint (result, obj); break;
      case 's': Converter<short>::toCint         (result, obj); break;
      case 'r': Converter<unsigned short>::toCint(result, obj); break;
      case 'i': Converter<int>::toCint           (result, obj); break;
      case 'h': Converter<unsigned int>::toCint  (result, obj); break;
      case 'l': Converter<long>::toCint          (result, obj); break;
      case 'k': Converter<unsigned long>::toCint (result, obj); break;
      case 'n': Converter<long long>::toCint     (result, obj); break;
      case 'm': Converter<unsigned long long>::toCint (result, obj); break;
      case 'f': Converter<float>::toCint         (result, obj); break;
      case 'd': Converter<double>::toCint        (result, obj); break;
      case 'q': Converter<long double>::toCint   (result, obj); break;
      case 'Q': Converter<int>::toCint           (result, obj); break;
      case 'u': 
         Converter<long>::toCint(result, obj);
         if ( !fRet_byref ) {
            // Cint stores the address of the object in .ref even if
            // the value is not a reference.
            result->ref = (long)obj;
         }
         result->tagnum = fRet_tag;
         break;
      case 'U': 
         Converter<long>::toCint(result, *(void**)obj);
         result->tagnum = fRet_tag;
         break;
      }
      if (isupper(t) && fRet_plevel) {
         result->obj.reftype.reftype = fRet_plevel;
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
  
      if(!G__GetCatchException()) {

         // Stub Calling
         void* retaddr = 0;
         if ( context->fRet_byvalue ) {
            // Intentionally use operator new here, we do NOT need to run
            // the constructor since the function itself will run 
            // a new with placement.
            retaddr = ::operator new( context->fRet_Sizeof );
         } else
            retaddr = context->GetReturnAddress(result);
         (*context->fStub)(retaddr, (void*)G__getstructoffset(), context->fParam, context->fStubctx);
         context->ProcessResult(result, retaddr);
         if ( context->fRet_byvalue )  G__store_tempobject(*result);

         return 1;
   
      }  

      // Catch here everything since going through the adaptor in the data section
      // does not transmit the exception 
      try {
         void* retaddr = 0;
         if ( context->fRet_byvalue ) {
            // Intentionally use operator new here, we do NOT need to run
            // the constructor since the function itself will run 
            // a new with placement.
            retaddr = ::operator new( context->fRet_Sizeof );
         } else
            retaddr = context->GetReturnAddress(result);
         (*context->fStub)(retaddr, (void*)G__getstructoffset(), context->fParam, context->fStubctx);
         context->ProcessResult(result, retaddr);
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
                  (*context->fStub)(0, (void*)p, context->fParam, 0);
            }
         }
         else {
            obj = ::operator new( size );
            (*context->fStub)(0, obj, context->fParam, 0);
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
               (*context->fStub)(0, (char*)obj + size*i, context->fParam, 0);
            ::operator delete (obj);
         }
      }
      else {
         long g__Xtmp = G__getgvp();
         G__setgvp(G__PVOID);
         (*context->fStub)(0, obj, context->fParam, 0);
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
#if CINTEX_USE_MMAP
      char* code = (char*) ::mmap(NULL, len + sizeof(size_t), PROT_READ | PROT_WRITE | PROT_EXEC,
                                  MAP_PRIVATE | MAP_ANON, -1, 0);
      if (!code || code == ((void *) -1)) return 0;
      // write the size of the allocation into the first few bytes; we need
      // it for munmap.
      *((size_t*)code) = len + sizeof(size_t);
      code += sizeof(size_t);
#elif defined(_WIN32)
      char* code = (char*)VirtualAlloc(NULL, len + sizeof(size_t),
      /*MEM_COMMIT*/ 0x1000 | /*MEM_RESERVE*/ 0x2000,
      /*PAGE_EXECUTE_READWRITE*/0x40);
      if (!code || code == ((void *) -1)) return 0;
#else
      char* code = new char[len+1];
      if ( !code ) return 0;
#endif
      ::memcpy(code, src, len);
      return code;
   }

   //------ Function models-------------------------------------------------------------------
#ifdef R__B64
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
               fSize = (o + 256) & ~0xF;
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
#if CINTEX_USE_MMAP
      if (!code) return;
      scode -= sizeof(size_t);
      munmap(scode, *((size_t*)scode));
#elif defined(_WIN32)
      if (!code) return;
      VirtualFree(scode, 0, /*MEM_RELEASE*/ 0x8000);
#else
      delete [] scode;
#endif
   }

} }   // seal and cintex namepaces
