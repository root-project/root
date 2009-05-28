// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "CINTFunctional.h"

#include "StubContext.h"

#include "G__ci.h"
#include "Api.h"

#include <exception>

#ifdef __linux
#include <sys/mman.h>
#endif // __linux

using namespace ROOT::Reflex;
using namespace std;

namespace ROOT {
namespace Cintex {

static char* Allocate_code(const void* src, size_t len);

//______________________________________________________________________________
int Method_stub_with_context(StubContext_t* context, G__value* result, G__CONST char* /*funcname*/, G__param* libp, int /*hash*/)
{
   if (!context->fInitialized) {
      context->Initialize();
   }
   context->ProcessParam(libp);
   Type ret_final_type = context->fMbr.TypeOf().ReturnType().FinalType();
   bool by_value = !ret_final_type.IsReference() && !ret_final_type.IsPointer() && !ret_final_type.IsArray() && !ret_final_type.IsFundamental() && !ret_final_type.IsEnum();
   size_t ret_size = ret_final_type.SizeOf();
   if (!ret_size) {
      // A type with a zero size is most likely unknown
      // to reflex, so try again using cint.
      ret_size = G__Lsizeof(ret_final_type.Name(Reflex::SCOPED).c_str());
   }
   //if (!G__GetCatchException()) {
      void* retaddr = 0;
      if (by_value) {
         // Intentionally use global operator new here, we do NOT need to run
         // the constructor since the function itself will run
         // a new with placement.
         retaddr = ::operator new(ret_size);
      }
      else {
         retaddr = context->GetReturnAddress(result);
      }
      (*context->fMbr.Stubfunction())(retaddr, (void*) G__getstructoffset(), context->fParam, context->fMbr.Stubcontext());
      context->ProcessResult(result, retaddr);
      if (by_value) {
         G__store_tempobject(*result);
      }
      return 1;
   //}
   return 1;
#if 0
   try {
      void* retaddr = 0;
      if (by_value) {
         // Intentionally use global operator new here, we do NOT need to run
         // the constructor since the function itself will run
         // a new with placement.
         retaddr = ::operator new(ret_size);
      }
      else {
         retaddr = context->GetReturnAddress(result);
      }
      (*context->fMbr.Stubfunction())(retaddr, (void*) G__getstructoffset(), context->fParam, context->fMbr.Stubcontext());
      context->ProcessResult(result, retaddr);
      if (by_value) {
         G__store_tempobject(*result);
      }
   }
   catch (std::exception& e) {
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
   return 1;
#endif // 0
   // --
}

//______________________________________________________________________________
int Constructor_stub_with_context(StubContext_t* context, G__value* result, G__CONST char* /*funcname*/, G__param* libp, int /*indx*/)
{
   if (!context->fInitialized) {
      context->Initialize();
   }
   context->ProcessParam(libp);
   void* obj = 0;
   try {
      long nary = G__getaryconstruct();
      size_t size = context->fMbr.DeclaringType().SizeOf();
      if (nary) {
         if (context->fNewdelfuncs) {
            obj = context->fNewdelfuncs->fNewArray(nary, 0);
         }
         else {
            obj = ::operator new(nary * size);
            long p = (long) obj;
            for (long i = 0; i < nary; ++i, p += size) {
               (*context->fMbr.Stubfunction())(0, (void*) p, context->fParam, 0);
            }
         }
      }
      else {
         obj = ::operator new(size);
         (*context->fMbr.Stubfunction())(0, obj, context->fParam, 0);
      }
   }
   catch (std::exception& e) {
      string errtxt("Exception: ");
      errtxt += e.what();
      errtxt += " (C++ exception)";
      G__genericerror(errtxt.c_str());
      ::operator delete(obj);
      obj = 0;
   }
   catch (...) {
      G__genericerror("Exception: Unknown C++ exception");
      ::operator delete(obj);
      obj = 0;
   }
   result->obj.i = (long) obj;
   result->ref = (long) obj;
   return 1;
}

//______________________________________________________________________________
int Destructor_stub_with_context(StubContext_t* context, G__value* result, G__CONST char* /*funcname*/, G__param* /*libp*/, int /*indx*/)
{
   void* obj = (void*) G__getstructoffset();
   if (!obj) {
      return 1;
   }
   if (!context->fInitialized) {
      context->Initialize();
   }
   if (G__getaryconstruct()) {
      if (G__getgvp() == (long) G__PVOID) { // delete[] (TYPE*)(G__getstructoffset());
         if (context->fNewdelfuncs) {
            context->fNewdelfuncs->fDeleteArray(obj);
         }
      }
      else {
         size_t size = context->fMbr.DeclaringType().SizeOf();
         for (int i = G__getaryconstruct() - 1; i > -1; --i) {
            (*context->fMbr.Stubfunction())(0, (char*) obj + (size * i), context->fParam, 0);
         }
         ::operator delete(obj);
      }
   }
   else {
      long g__Xtmp = G__getgvp();
      G__setgvp((long) G__PVOID);
      (*context->fMbr.Stubfunction())(0, obj, context->fParam, 0);
      G__setgvp(g__Xtmp);
      if (!((G__getgvp() == (long) obj) && (G__getgvp() != (long) G__PVOID))) {
         ::operator delete(obj); // G__operator_delete(obj);
      }
   }
   G__setnull(result);
   return 1;
}

//______________________________________________________________________________
//
// Function models
//

#if INT_MAX < LONG_MAX // 64-bit machine
#define FUNCPATTERN 0xFAFAFAFAFAFAFAFAL
#define DATAPATTERN 0xDADADADADADADADAL
#else // INT_MAX < LONG_MAX // 32-bit machine
#define FUNCPATTERN 0xFAFAFAFAL
#define DATAPATTERN 0xDADADADAL
#endif // INT_MAX < LONG_MAX

//______________________________________________________________________________
static void f0a()
{
   typedef void (*f_t)(void*);
   ((f_t)FUNCPATTERN)((void*)DATAPATTERN);
}

//______________________________________________________________________________
static void f1a(void* a0)
{
   typedef void (*f_t)(void*, void*);
   ((f_t)FUNCPATTERN)((void*)DATAPATTERN, a0);
}

//______________________________________________________________________________
static void f3a(void* a0, TMemberInspector& a1, char* a2)
{
   typedef void (*f_t)(void*, void*, TMemberInspector&, char*);
   ((f_t)FUNCPATTERN)((void*)DATAPATTERN, a0, a1, a2);
}

//______________________________________________________________________________
static void f4a(void* a0, void* a1, void* a2, void* a3)
{
   typedef void (*f_t)(void*, void*, void*, void*, void*);
   ((f_t)FUNCPATTERN)((void*)DATAPATTERN, a0, a1, a2, a3);
}

//______________________________________________________________________________
class FunctionCode_t {
   // --
public:
   size_t f_offset; // offset to function ptr
   size_t fa_offset; // offset to this ptr
   size_t fSize;
   char*  fCode;
public:
   FunctionCode_t(int narg)
   : f_offset(0)
   , fa_offset(0)
   , fSize(0)
   , fCode(0)
   {
      if (narg == 0) {
         fCode = (char*) f0a;
      }
      else if (narg == 1) {
         fCode = (char*) f1a;
      }
      else if (narg == 3) {
         fCode = (char*) f3a;
      }
      else if (narg == 4) {
         fCode = (char*) f4a;
      }
      char* b = fCode;
      for (size_t o = 0; o < 1000; ++o, ++b) {
         if (*(size_t*)b == DATAPATTERN) {
            fa_offset = o;
         }
         if (*(size_t*)b == FUNCPATTERN) {
            f_offset = o;
         }
         if (f_offset && fa_offset) {
            fSize = (o + 32) & ~0xF; // FIXME: Weird, is this supposed to round to a multiple of 32 or 16???
            break;
         }
      }
   }
};

//______________________________________________________________________________
#undef DATAPATTERN
#undef FUNCPATTERN

//______________________________________________________________________________
static char* Allocate_code(const void* src, size_t len)
{
   // --
#if defined(__linux) && ! defined(DATA_EXECUTABLE)
   char* code = (char*) ::mmap(NULL, len + sizeof(size_t), PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   if (!code || code == ((void *) - 1)) {
      return 0;
   }
   // write the size of the allocation into the
   // first few bytes; we need it for munmap.
   *((size_t*)code) = len + sizeof(size_t);
   code += sizeof(size_t);
#else // __linux && !DATA_EXECUTABLE
   char* code = new char[len+1];
   if (!code) {
      return 0;
   }
#endif // __linux && !DATA_EXECUTABLE
   ::memcpy(code, src, len);
   return code;
}

//______________________________________________________________________________
void Free_code(void* code)
{
   if (!code) {
      return;
   }
   char* p = (char*) code;
#if defined(__linux) && ! defined(DATA_EXECUTABLE)
   p -= sizeof(size_t);
   munmap(p, *(size_t*)p);
#else // __linux && !DATA_EXECUTABLE
   delete[] p;
#endif // __linux && !DATA_EXECUTABLE
   // --
}

//______________________________________________________________________________
G__InterfaceMethod Allocate_stub_function(StubContext_t* obj, StubFuncPtr_t fun)
{
   static FunctionCode_t s_func4arg(4);
   char* code = Allocate_code(s_func4arg.fCode, s_func4arg.fSize);
   *(void**)&code[s_func4arg.fa_offset] = (void*) obj;
   *(void**)&code[s_func4arg.f_offset] = (void*) fun;
   obj->fMethodCode = (G__InterfaceMethod) code;
   return obj->fMethodCode;
}

//______________________________________________________________________________
FuncVoidPtr_t Allocate_void_function(void* obj, void (*fun)(void*))
{
   static FunctionCode_t s_func0arg(0);
   char* code = Allocate_code(s_func0arg.fCode, s_func0arg.fSize);
   *(void**)&code[s_func0arg.fa_offset] = (void*) obj;
   *(void**)&code[s_func0arg.f_offset] = (void*) fun;
   return (FuncVoidPtr_t) code;
}

//______________________________________________________________________________
FuncArg1Ptr_t Allocate_1arg_function(void* obj, void* (*fun)(void*, void*))
{
   static FunctionCode_t s_func1arg(1);
   char* code = Allocate_code(s_func1arg.fCode, s_func1arg.fSize);
   *(void**)&code[s_func1arg.fa_offset] = (void*) obj;
   *(void**)&code[s_func1arg.f_offset] = (void*) fun;
   return (FuncArg1Ptr_t) code;
}

//______________________________________________________________________________
FuncArg3Ptr_t Allocate_3arg_function(void* obj, void (*fun)(void*, void*, TMemberInspector&, char*))
{
   static FunctionCode_t s_func3arg(3);
   char* code = Allocate_code(s_func3arg.fCode, s_func3arg.fSize);
   *(void**)&code[s_func3arg.fa_offset] = (void*) obj;
   *(void**)&code[s_func3arg.f_offset] = (void*) fun;
   return (FuncArg3Ptr_t) code;
}

} // namespace Cintex
} // namespace ROOT
