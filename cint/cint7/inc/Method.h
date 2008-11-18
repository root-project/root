/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file Method.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__METHODINFO_H
#define G__METHODINFO_H

#include "Api.h"

#ifndef __CINT__
#include "Reflex/Member.h"
#else // __CINT__
namespace Reflex {
class Member;
} // namespace Reflex
#endif // __CINT__

#include <string>

namespace Cint {
class
#ifndef __CINT__
G__EXPORT
#endif // __CINT__
G__MethodInfo
{
   friend class G__MethodArgInfo;
public:
   ~G__MethodInfo();
   G__MethodInfo();
   G__MethodInfo(G__ClassInfo&);
   G__MethodInfo(const G__MethodInfo&);
   //G__MethodInfo(const ::Reflex::Member&);
   G__MethodInfo& operator=(const G__MethodInfo&);
   void Init();
   void Init(G__ClassInfo&);
   void Init(long handlein, long indexin, G__ClassInfo* belongingclassin);
   void Init(G__ClassInfo* belongingclassin, long funcpage, long indexin);
   const char* Name();
#ifndef __MAKECINT__
   int Hash();
   G__ifunc_table* ifunc();
#endif // __MAKECINT__
   long Handle();
   //::Reflex::Member ReflexFunction() const;
   int Index();
   const char* Title();
   G__TypeInfo* Type();
   long Property();
   int NArg();
   int NDefaultArg();
   int HasVarArgs();
   G__InterfaceMethod InterfaceMethod();
#ifdef G__ASM_WHOLEFUNC
   G__bytecodefunc* GetBytecode();
#endif // G__ASM_WHOLEFUNC
   G__DataMemberInfo GetLocalVariable();
#ifdef G__TRUEP2F
   void* PointerToFunc();
#endif // G__TRUEP2F
   G__ClassInfo* MemberOf();
   int GetDefiningScopeTagnum();
   G__friendtag* GetFriendInfo();
   void SetGlobalcomp(int globalcomp);
   void SetForceStub();
   int IsValid();
   int SetFilePos(const char* fname);
   int Next();
   const char* FileName();
   int LineNumber();
   int Size();
   int IsBusy();
   FILE* FilePointer();
   long FilePosition();
   char* GetPrototype();
   char* GetMangledName();
   int LoadDLLDirect(const char* filename, const char* funcname);
   void SetVtblIndex(int vtblindex);
   void SetIsVirtual(int isvirtual);
   void SetVtblBasetagnum(int basetagnum);
   void SetUserParam(void*);
   void* GetUserParam();
   long GetThisPointerOffset();
protected:
   // -- Attributes.
   ::Reflex::Scope fScope; // attribute, the scope whose member functions we are iterating over
   // -- Internal Iterator.
   long fIndex; // internal iterator, index into  member function table for our class or namespace
   long fUsingIndex; // internal iterator, index into "using namespace" inheritance table.
   // -- Cached Information.
   G__ClassInfo* fClassInfo; // cached info, we own, fScope as a classinfo
   ::Reflex::Member fFunc; // cached info, the member function the internal iterator is currently pointing at.
   G__TypeInfo* fTypeInfo; // cached info, we own, return type of the current function
};

} // namespace Cint

using namespace Cint;
#endif // G__METHODINFO_H
