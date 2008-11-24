/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file CallFunc.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~2003  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "common.h"
#include "Dict.h"

#include <cstring>

using namespace Cint::Internal;
using namespace std;

//
//  class G__CallFunc
//
//  Example 1:
//
//    TCanvas canvas;
//    // set pointer to interface method and set argument list
//    G__CallFunc func;
//    G__ClassInfo ci("TCanvas");
//    long offset;
//    func.SetFunc(&ci, "Draw", "\"ABC\",1234,3.14", &offset);
//    // call function
//    void* address = reinterpret_cast<char*>(&canvas) + offset;
//    func.Exec(address);
//    // reset everything
//    func.Init();
//
//
//  Example 2:
//
//    TCanvas canvas;
//    // set pointer to interface method
//    G__CallFunc func;
//    G__ClassInfo ci("TCanvas");
//    long offset;
//    func.SetFunc(ci.GetMethod("Draw", "char*,int,double", &offset).InterfaceMethod());
//    // set arguments
//    const char* title = "ABC";
//    func.SetArg(reinterpret_cast<long>(title));
//    func.SetArg(1234L);
//    func.SetArg(3.14);
//    // call function
//    void* address = reinterpret_cast<char*>(&canvas) + offset;
//    func.Exec(address);
//    // reset everything
//    func.Init();
//

//______________________________________________________________________________
Cint::G__CallFunc::~G__CallFunc()
{
}

//______________________________________________________________________________
Cint::G__CallFunc::G__CallFunc()
{
   G__LockCriticalSection();
   Init();
   G__UnlockCriticalSection();
}

//______________________________________________________________________________
Cint::G__CallFunc::G__CallFunc(const G__CallFunc& rhs)
: method(rhs.method)
, para(rhs.para)
, result(rhs.result)
, pfunc(rhs.pfunc)
#ifdef G__ASM_WHOLEFUNC
, bytecode(rhs.bytecode)
#endif // G__ASM_WHOLEFUNC
{
}

//______________________________________________________________________________
G__CallFunc& Cint::G__CallFunc::operator=(const G__CallFunc& rhs)
{
   if (this != &rhs) {
      method = rhs.method;
      para = rhs.para;
      result = rhs.result;
      pfunc = rhs.pfunc;
#ifdef G__ASM_WHOLEFUNC
      bytecode = rhs.bytecode;
#endif // G__ASM_WHOLEFUNC
   }
   return *this;
}

//______________________________________________________________________________
void Cint::G__CallFunc::Init()
{
   para.paran = 0;
   para.para[0] = G__null;
   para.para[1] = G__null;
   result = G__null;
   pfunc = 0;
#ifdef G__ASM_WHOLEFUNC
   bytecode = 0;
#endif
   // --
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetFunc(G__ClassInfo* cls, const char* fname, const char* args, long* poffset, MatchMode mode)
{
   //
   //  Separate and evaluate argument list.
   //
   G__StrBuf argtype_sb(G__ONELINE);
   char* argtype = argtype_sb;
   argtype[0] = '\0';
   int pos = 0;
   int isrc = 0;
   para.paran = 0;
   for (int c = ','; c == ',';) {
      c = G__getstream(const_cast<char*>(args), &isrc, para.parameter[para.paran], const_cast<char*>(","));
      if (para.parameter[para.paran][0]) {
         para.para[para.paran] = G__calc(para.parameter[para.paran]); // evaluate arg
         G__value* buf = &para.para[para.paran];
         if (pos) {
            argtype[pos++] = ',';
         }
         // set type string
         ::Reflex::Type ty = G__value_typenum(*buf);
         char type = 0;
         int tagnum = -1;
         int typenum = -1;
         int reftype = 0;
         int isconst = 0;
         G__get_cint5_type_tuple(ty, &type, &tagnum, &typenum, &reftype, &isconst);
         if (islower(type)) {
            strcpy(argtype + pos, G__type2string(type, tagnum, typenum, 0, 0));
         }
         else {
            strcpy(argtype + pos, G__type2string(type, tagnum, typenum, reftype, 0));
         }
         pos = strlen(argtype);
         ++para.paran; // increment argument count
      }
   }
   method = cls->GetMethod(fname, argtype, poffset, (G__ClassInfo::MatchMode) mode);
   pfunc = method.InterfaceMethod(); // get compiled interface method
#ifdef G__ASM_WHOLEFUNC
   if (!pfunc) {
      int store_paran = para.paran;
      SetBytecode(method.GetBytecode()); // try to compile bytecode
      para.paran = store_paran;
   }
#endif // G__ASM_WHOLEFUNC
   // --
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetFuncProto(G__ClassInfo* cls, const char* fname, const char* argtype, long* poffset)
{
   G__LockCriticalSection();
   method = cls->GetMethod(fname, argtype, poffset);
   pfunc = method.InterfaceMethod(); // get compiled interface method
   // FIXME: We need the same kind of call to SetBytecode() that is done in SetFunc().
   para.paran = 0; // reset parameters, not needed actually, done in SetBytecode
   G__UnlockCriticalSection();
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetFunc(G__InterfaceMethod f)
{
   pfunc = f; // Set pointer to interface method
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetFunc(G__MethodInfo m)
{
   method = m;
   pfunc = m.InterfaceMethod();
}

#ifdef G__ASM_WHOLEFUNC
//______________________________________________________________________________
void Cint::G__CallFunc::SetBytecode(G__bytecodefunc* bc)
{
   bytecode = bc;
   if (bytecode) {
      pfunc = (G__InterfaceMethod) G__exec_bytecode;
   }
   else {
      pfunc = (G__InterfaceMethod) 0;
#ifndef G__ROOT
      if (G__asm_dbg) {
         if (G__dispmsg >= G__DISPWARN) {
            G__fprinterr(G__serr, "Warning: Bytecode compilation of %s failed. Cint::G__CallFunc::Exec may be slow\n", method.Name());
         }
      }
#endif // G__ROOT
      // --
   }
   para.paran = 0;
}
#endif // G__ASM_WHOLEFUNC

//______________________________________________________________________________
int Cint::G__CallFunc::IsValid()
{
   return method.IsValid();
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetArgArray(long* p, int narg /*= -1*/)
{
   if (!method.IsValid()) {
      G__fprinterr(G__serr, "Error: Cint::G__CallFunc::SetArgArray() must be initialized with 'G__CallFunc::SetFunc(G__ClassInfo* cls,char* fname,char* args,long* poffset)' first\n");
      return;
   }
   int n = 0;
   if (narg < 0) {
      n = method.NArg();
   }
   else {
      n = narg;
      if (narg > method.NArg()) {
         G__fprinterr(G__serr, "Warning: Cint::G__CallFunc::SetArgArray() too many arguments specified (%d expected %d)\n", narg, method.NArg());
         G__printlinenum();
         n = method.NArg();
      }
      else if (n < (method.NArg() - method.NDefaultArg())) {
         G__fprinterr(G__serr, "Error: Cint::G__CallFunc::SetArgArray() too few arguments");
         G__printlinenum();
         n = method.NArg();
      }
   }
   G__MethodArgInfo arginfo;
   arginfo.Init(method);
   para.paran = 0;
   for (int i = 0; i < n; ++i) {
      para.para[i].obj.i = p[i];
      para.para[i].ref = p[i];
      // Following data shouldn't matter, but set just in case
      arginfo.Next();
      {
         G__TypeInfo* ti = arginfo.Type();
         int type = (int) ti->Type();
         int tagnum = (int) ti->Tagnum();
         int typenum = (int) ti->Typenum();
         // FIXME: We need to use the reftype and isconst here!
         G__value_typenum(para.para[i]) = G__cint5_tuple_to_type(type, tagnum, typenum, 0, 0);
      }
      ++para.paran;
   }
}

//______________________________________________________________________________
void Cint::G__CallFunc::ResetArg()
{
   para.paran = 0;
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetArg(long l)
{
   para.para[para.paran].obj.i = l;
   para.para[para.paran].ref = l;
   // Following data shouldn't matter, but set just in case
   G__value_typenum(para.para[para.paran]) = G__get_from_type('l', 0);
   ++para.paran;
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetArg(unsigned long ul)
{
   para.para[para.paran].obj.ulo = ul;
   para.para[para.paran].ref = (long)ul;
   // Following data shouldn't matter, but set just in case
   G__value_typenum(para.para[para.paran]) = G__get_from_type('k', 0);
   ++para.paran;
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetArg(double d)
{
   para.para[para.paran].obj.d = d;
   // Following data shouldn't matter, but set just in case
   para.para[para.paran].ref = 0 ;
   G__value_typenum(para.para[para.paran]) = G__get_from_type('d', 0);
   ++para.paran;
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetArgRef(long& l)
{
   para.para[para.paran].ref = (long) &l;
   para.para[para.paran].obj.i = l;
   // Following data shouldn't matter, but set just in case
   // FIXME: Need to add a REFERENCE modifier here???  Cint5 does not.
   G__value_typenum(para.para[para.paran]) =  G__get_from_type('l', 0);
   ++para.paran;
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetArgRef(double& d)
{
   para.para[para.paran].ref = (long) &d;
   // Following data shouldn't matter, but set just in case
   para.para[para.paran].obj.d = d;
   // FIXME: Need to add a REFERENCE modifier here???  Cint5 does not.
   G__value_typenum(para.para[para.paran]) =  G__get_from_type('d', 0);
   ++para.paran;
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetArg(G__value v)
{
   para.para[para.paran] = v;
   ++para.paran;
}

#ifdef G__NATIVELONGLONG
//______________________________________________________________________________
void Cint::G__CallFunc::SetArg(G__int64 ll)
{
   para.para[para.paran].obj.ll = ll;
   para.para[para.paran].ref = 0;
   // Following data shouldn't matter, but set just in case
   G__value_typenum(para.para[para.paran]) =  G__get_from_type('n', 0);
   ++para.paran;
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetArg(G__uint64 ull)
{
   para.para[para.paran].obj.ull = ull;
   para.para[para.paran].ref = 0;
   // Following data shouldn't matter, but set just in case
   G__value_typenum(para.para[para.paran]) =  G__get_from_type('m', 0);
   ++para.paran;
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetArg(long double ld)
{
   para.para[para.paran].obj.ld = ld;
   // Following data shouldn't matter, but set just in case
   para.para[para.paran].ref = 0 ;
   G__value_typenum(para.para[para.paran]) = G__get_from_type('q', 0);
   ++para.paran;
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetArgRef(G__int64& ll)
{
   para.para[para.paran].ref = (long) &ll;
   // Following data shouldn't matter, but set just in case
   para.para[para.paran].obj.ll = ll;
   // FIXME: Need to add a REFERENCE modifier here???  Cint5 does not.
   G__value_typenum(para.para[para.paran]) = G__get_from_type('n', 0);
   ++para.paran;
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetArgRef(G__uint64& ull)
{
   para.para[para.paran].ref = (long) &ull;
   // Following data shouldn't matter, but set just in case
   para.para[para.paran].obj.ull = ull;
   // FIXME: Need to add a REFERENCE modifier here???  Cint5 does not.
   G__value_typenum(para.para[para.paran]) = G__get_from_type('m', 0);
   ++para.paran;
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetArgRef(long double& ld)
{
   para.para[para.paran].ref = (long) &ld;
   // Following data shouldn't matter, but set just in case
   para.para[para.paran].obj.ld = ld;
   // FIXME: Need to add a REFERENCE modifier here???  Cint5 does not.
   G__value_typenum(para.para[para.paran]) = G__get_from_type('q', 0);
   ++para.paran;
}
#endif // G__NATIVELONGLONG

//______________________________________________________________________________
G__value Cint::G__CallFunc::Execute(void* pobject)
{
   G__LockCriticalSection();
   char* store_struct_offset = G__store_struct_offset;
   G__store_struct_offset = reinterpret_cast<char*>(pobject) + method.GetThisPointerOffset();
   SetFuncType();
   {
      long index = method.Index();
      G__CurrentCall(G__SETMEMFUNCENV, method.ifunc(), &index);
   }
   int ret = 0;
   if (!pfunc) {
      ret = ExecInterpretedFunc(&result);
   }
   else {
      // --
#ifdef G__ASM_WHOLEFUNC
      if (pfunc == G__DLL_direct_globalfunc) {
         ret = (*pfunc)(&result, (char*) method.ifunc(), &para, method.Index());
      }
      else {
         ret = (*pfunc)(&result, (char*) bytecode, &para, 0);
      }
#else // G__ASM_WHOLEFUNC
      ret = (*pfunc)(&result, 0, &para, 0);
#endif // G__ASM_WHOLEFUNC
      // --
   }
   G__store_struct_offset -= method.GetThisPointerOffset();
   G__CurrentCall(G__NOP, 0, 0);
   G__store_struct_offset = store_struct_offset; // Restore  object address
   G__UnlockCriticalSection();
   return result;
}

//______________________________________________________________________________
void Cint::G__CallFunc::Exec(void* pobject)
{
   Execute(pobject);
}

//______________________________________________________________________________
long Cint::G__CallFunc::ExecInt(void* pobject)
{
   return G__int(Execute(pobject));
}

//______________________________________________________________________________
double Cint::G__CallFunc::ExecDouble(void* pobject)
{
   return G__double(Execute(pobject));
}

#ifdef G__NATIVELONGLONG
//______________________________________________________________________________
G__int64 Cint::G__CallFunc::ExecInt64(void* pobject)
{
   return G__Longlong(Execute(pobject));
}
#endif // G__NATIVELONGLONG

//______________________________________________________________________________
G__InterfaceMethod Cint::G__CallFunc::InterfaceMethod()
{
   return pfunc;
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetArgs(const char* args)
{
   para.paran = 0;
   if (!args || !args[0]) {
      return;
   }
   // separate and evaluate argument list
   int isrc = 0;
   char* tmp = new char[strlen(args)+2];
   int c = ',';
   for (; c == ','; ) {
      c = G__getstream((char*) args, &isrc, tmp, (char*) ",");
      if (tmp[0]) {
         para.para[para.paran] = G__calc(tmp); // evaluate arg
         if (strlen(tmp) < (G__ONELINE - 1)) { // FIXME: This code implicitly accepts that a buffer overwrite is ok.
            strcpy(para.parameter[para.paran], tmp);
         }
         else {
            para.parameter[para.paran][0] = 0;
         }
         ++para.paran; // increment argument count
      }
   }
   delete[] tmp;
}

//______________________________________________________________________________
void G__CallFunc::SetArgs(const G__param &p)
{
   // Set the arguments for the function calls.

   para = p;
}

//______________________________________________________________________________
G__MethodInfo Cint::G__CallFunc::GetMethodInfo()
{
   return method;
}

//______________________________________________________________________________
void Cint::G__CallFunc::SetFuncType()
{
   if (!method.IsValid()) {
      return;
   }
   ::Reflex::Member mbr = G__Dict::GetDict().GetFunction(method.ifunc(), method.Index());
   G__value_typenum(result) = mbr.TypeOf().ReturnType();
}

//______________________________________________________________________________
int Cint::G__CallFunc::ExecInterpretedFunc(G__value* presult)
{
   if (!method.IsValid()) {
      return 0;
   }
   int ret = 0;
   G__ClassInfo* pcls = method.MemberOf();
   if (pcls && pcls->Name() && method.Name() && !strcmp(pcls->Name(), method.Name())) {
      G__store_struct_offset = new char[pcls->Size()];
   }
   int store_asm_exec = G__asm_exec;
   ::Reflex::Member store_asm_index = G__asm_index;
   int store_asm_noverflow = G__asm_noverflow;
   G__asm_exec = 1;
   G__asm_index = G__Dict::GetDict().GetFunction(method.ifunc(), method.Index());
   G__asm_noverflow = 0;
   ret = G__interpret_func(presult, (char*) method.Name(), &para, method.Hash(), G__asm_index.DeclaringScope(), G__EXACT, G__TRYNORMAL);
   G__asm_exec = store_asm_exec;
   G__asm_index = store_asm_index;
   G__asm_noverflow = store_asm_noverflow;
   return ret;
}

