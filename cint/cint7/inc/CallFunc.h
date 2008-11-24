/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file CallFunc.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~2003  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__CALLFUNC_H
#define G__CALLFUNC_H

#include "Api.h"

namespace Cint {

class
#ifndef __CINT__
G__EXPORT
#endif // __CINT__
G__CallFunc
{
public: // -- Types
   enum MatchMode
   {
      ExactMatch = 0,
      ConversionMatch = 1
   };
public: // -- Member Functions.
   ~G__CallFunc();
   G__CallFunc();
   G__CallFunc(const G__CallFunc&);
   G__CallFunc& operator=(const G__CallFunc& cf);
   void Init();
   void SetFunc(G__ClassInfo* cls, const char* fname, const char* args, long* poffset, MatchMode mode = ConversionMatch);
   void SetFuncProto(G__ClassInfo* cls, const char* fname, const char* argtype, long* poffset);
   void SetFunc(G__InterfaceMethod f);
   void SetFunc(G__MethodInfo m);
#ifdef G__ASM_WHOLEFUNC
   void SetBytecode(struct G__bytecodefunc* bc);
#endif // G__ASM_WHOLEFUNC
   int IsValid();
   void SetArgArray(long* p, int narg = -1);
   void ResetArg();
   void SetArg(long);
   void SetArg(unsigned long);
   void SetArg(double);
   void SetArgRef(long&);
   void SetArgRef(double&);
   void SetArg(G__value);
#ifdef G__NATIVELONGLONG
   void SetArg(G__int64);
   void SetArg(G__uint64);
   void SetArg(long double);
   void SetArgRef(G__int64&);
   void SetArgRef(G__uint64&);
   void SetArgRef(long double&);
#endif // G__NATIVELONGLONG

   G__value Execute(void * pobject);
   void Exec(void* pobject);
   long ExecInt(void* pobject);
   double ExecDouble(void* pobject);
#ifdef G__NATIVELONGLONG
   G__int64 ExecInt64(void* pobject);
#endif // G__NATIVELONGLONG
   G__InterfaceMethod InterfaceMethod();
   void SetArgs(const char* args);
   void SetArgs(const G__param &p);
   G__MethodInfo GetMethodInfo();
private: // -- Private utility functions.
   void SetFuncType();
   int ExecInterpretedFunc(G__value* presult);
private: // -- Data members.
   G__MethodInfo method; // primary key, our function
   G__param para; // arguments for the function call
   G__value result; // result of the function call, return value
   G__InterfaceMethod pfunc; // compiled function pointer, can be overridden by SetFunc(), or replaced with G__exec_bytecode by SetByteCode()
#ifdef G__ASM_WHOLEFUNC
   struct G__bytecodefunc* bytecode; // if whole function compiled to bytecode, pointer to bytecode
#endif // G__ASM_WHOLEFUNC
};

} // namespace Cint

using namespace Cint;
#endif // G__CALLFUNC_H
