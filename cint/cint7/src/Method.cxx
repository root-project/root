/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file Method.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Author                  Masaharu Goto
 * Copyright(c) 1995~2005  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "Dict.h"
#include "common.h"
#include "fproto.h"

using namespace Cint::Internal;
using namespace std;

//______________________________________________________________________________
static char G__buf[G__LONGLINE];

//______________________________________________________________________________
Cint::G__MethodInfo::~G__MethodInfo()
{
   delete fClassInfo;
   fClassInfo = 0;
   delete fTypeInfo;
   fTypeInfo = 0;
}

//______________________________________________________________________________
Cint::G__MethodInfo::G__MethodInfo() : fIndex(0), fUsingIndex(0), fClassInfo(0), fTypeInfo(0)
{
   Init();
}

//______________________________________________________________________________
Cint::G__MethodInfo::G__MethodInfo(G__ClassInfo& a) : fIndex(0), fUsingIndex(0), fClassInfo(0), fTypeInfo(0)
{
   Init(a);
}

//______________________________________________________________________________
Cint::G__MethodInfo::G__MethodInfo(const G__MethodInfo& mi) : fScope(mi.fScope), fIndex(mi.fIndex), fUsingIndex(mi.fUsingIndex), fClassInfo(0), fFunc(mi.fFunc), fTypeInfo(0)
{
}

#if 0
//______________________________________________________________________________
Cint::G__MethodInfo::G__MethodInfo(const ::Reflex::Member& m) : fScope(m.DeclaringScope()), fFunc(m), fIndex(-1), fUsingIndex(0), fClassInfo(0), fTypeInfo(0)
{
}
#endif // 0

//______________________________________________________________________________
G__MethodInfo& Cint::G__MethodInfo::operator=(const G__MethodInfo& mi)
{
   if (this != &mi) {
      fScope = mi.fScope;
      fIndex = mi.fIndex;
      fUsingIndex = mi.fUsingIndex;
      fClassInfo = 0;
      fFunc = mi.fFunc;
      fTypeInfo = 0;
   }
   return *this;
}

//______________________________________________________________________________
void Cint::G__MethodInfo::Init()
{
   fScope = ::Reflex::Scope::GlobalScope();
   fIndex = -1;
   fUsingIndex = -1;
   delete fClassInfo;
   fClassInfo = 0;
   fFunc = ::Reflex::Member();
   delete fTypeInfo;
   fTypeInfo = 0;
}

//______________________________________________________________________________
void Cint::G__MethodInfo::Init(G__ClassInfo& a)
{
   fScope = Reflex::Scope();
   fIndex = -1;
   fUsingIndex = -1;
   delete fClassInfo;
   fClassInfo = 0;
   fFunc = ::Reflex::Member();
   delete fTypeInfo;
   fTypeInfo = 0;
   if (a.IsValid()) {
      fScope = G__Dict::GetDict().GetScope((int) a.Tagnum());
      G__incsetup_memfunc((int) a.Tagnum());
   }
}

//______________________________________________________________________________
void Cint::G__MethodInfo::Init(long handlein, long indexin, G__ClassInfo* /*belongingclassin*/)
{
   fScope = Reflex::Scope();
   fIndex = -1;
   fUsingIndex = -1;
   delete fClassInfo;
   fClassInfo = 0;
   fFunc = Reflex::Member();
   delete fTypeInfo;
   fTypeInfo = 0;
   if (handlein) {
      fFunc = ::Reflex::Member(reinterpret_cast<const Reflex::MemberBase*>(handlein));
      fScope = fFunc.DeclaringScope();
      fIndex = indexin;
      //fClassInfo = new G__ClassInfo(belongingclassin);
      fClassInfo = new G__ClassInfo(G__get_tagnum(fScope)); // FIXME: What if fScope is the global scope?
      // TODO: Remove use of friendship.
      ::Reflex::Type ty = fFunc.TypeOf().ReturnType();
      fTypeInfo = new G__TypeInfo();
      G__get_cint5_type_tuple_long(ty, &fTypeInfo->fType, &fTypeInfo->fTagnum, &fTypeInfo->fTypenum, &fTypeInfo->fReftype, &fTypeInfo->fIsconst);
      fTypeInfo->fClassProperty = 0L;
   }
}

//______________________________________________________________________________
void Cint::G__MethodInfo::Init(G__ClassInfo* belongingclassin, long funcpage, long /*indexin*/)
{
   delete fClassInfo;
   fClassInfo = 0;
   fScope = Reflex::Scope::GlobalScope();
   if (belongingclassin && belongingclassin->IsValid()) { // member function
      fScope = G__Dict::GetDict().GetScope(belongingclassin->Tagnum());
   }
   delete fTypeInfo;
   fTypeInfo = 0;
   fFunc = fScope.FunctionMemberAt(funcpage);
   if (fFunc) {
      fIndex = funcpage;
   }
   else {
      fScope = Reflex::Scope();
      fFunc = Reflex::Member();
      fIndex = -1;
   }
}

//______________________________________________________________________________
const char* Cint::G__MethodInfo::Name()
{
   if (IsValid()) {
      strcpy(G__buf, fFunc.Name().c_str()); // FIXME: Possible buffer overflow here!
      return G__buf;
   }
   return 0;
}

//______________________________________________________________________________
int Cint::G__MethodInfo::Hash()
{
   if (IsValid()) {
      return 0; // FIXME: Should we caculate the hash code here?
   }
   return 0;
}

//______________________________________________________________________________
G__ifunc_table* Cint::G__MethodInfo::ifunc()
{
   if (IsValid()) {
      return (G__ifunc_table*) fScope.Id();
   }
   return 0;
}

//______________________________________________________________________________
long Cint::G__MethodInfo::Handle()
{
   return (long) fScope.Id();
}

#if 0
//______________________________________________________________________________
::Reflex::Member Cint::G__MethodInfo::ReflexFunction() const
{
   return fFunc;
}
#endif // 0

//______________________________________________________________________________
int Cint::G__MethodInfo::Index()
{
   return (int) fIndex;
}

//______________________________________________________________________________
const char* Cint::G__MethodInfo::Title()
{
   static char buf[G__INFO_TITLELEN];
   buf[0] = '\0';
   if (IsValid()) {
      G__getcomment(buf, &G__get_funcproperties(fFunc)->comment, G__get_tagnum(fScope));
      return buf;
   }
   return 0;
}

//______________________________________________________________________________
G__TypeInfo* Cint::G__MethodInfo::Type()
{
   if (!fTypeInfo) {
      // TODO: Remove use of friendship.
      ::Reflex::Type ty = fFunc.TypeOf().ReturnType();
      fTypeInfo = new G__TypeInfo();
      G__get_cint5_type_tuple_long(ty, &fTypeInfo->fType, &fTypeInfo->fTagnum, &fTypeInfo->fTypenum, &fTypeInfo->fReftype, &fTypeInfo->fIsconst);
      fTypeInfo->fClassProperty = 0L;
   }
   return fTypeInfo;
}

//______________________________________________________________________________
long Cint::G__MethodInfo::Property()
{
   if (!IsValid()) {
      return 0;
   }
   long property = 0L;
   switch (G__get_access(fFunc)) {
      case G__PUBLIC:
         property |= G__BIT_ISPUBLIC;
         break;
      case G__PROTECTED:
         property |= G__BIT_ISPROTECTED;
         break;
      case G__PRIVATE:
         property |= G__BIT_ISPRIVATE;
         break;
   }
   if (G__get_isconst(fFunc.TypeOf()) & G__CONSTFUNC) {
      property |= (G__BIT_ISCONSTANT | G__BIT_ISMETHCONSTANT);
   }
   if (G__get_isconst(fFunc.TypeOf()) & G__CONSTVAR) {
      property |= G__BIT_ISCONSTANT;
   }
   if (G__get_isconst(fFunc.TypeOf()) & G__PCONSTVAR) {
      property |= G__BIT_ISPCONSTANT;
   }
   if (isupper(G__get_type(fFunc.TypeOf().ReturnType()))) {
      property |= G__BIT_ISPOINTER;
   }
   if (fFunc.IsStatic()) {
      property |= G__BIT_ISSTATIC;
   }
   if (fFunc.IsVirtual()) {
      property |= G__BIT_ISVIRTUAL;
   }
   if (fFunc.IsAbstract()) {
      property |= G__BIT_ISPUREVIRTUAL;
   }
   if (G__get_funcproperties(fFunc)->entry.size < 0) {
      property |= G__BIT_ISCOMPILED;
   }
   if (G__get_funcproperties(fFunc)->entry.bytecode) {
      property |= G__BIT_ISBYTECODE;
   }
   if (fFunc.IsExplicit()) {
      property |= G__BIT_ISEXPLICIT;
   }
   return property;
}

//______________________________________________________________________________
int Cint::G__MethodInfo::NArg()
{
   if (IsValid()) {
      return fFunc.FunctionParameterSize();
   }
   return -1;
}

//______________________________________________________________________________
int Cint::G__MethodInfo::NDefaultArg()
{
   if (IsValid()) {
      return fFunc.FunctionParameterSize() - fFunc.FunctionParameterSize(true);
   }
   return -1;
}

//______________________________________________________________________________
int Cint::G__MethodInfo::HasVarArgs()
{
   if (IsValid()) {
      return (2 == G__get_funcproperties(fFunc)->entry.ansi ? 1 : 0);
   }
   return -1;
}

//______________________________________________________________________________
G__InterfaceMethod Cint::G__MethodInfo::InterfaceMethod()
{
   G__LockCriticalSection();
   if (IsValid()) {
      int ret = G__get_funcproperties(fFunc)->entry.size;
      if (ret == -1) { // this means compiled class
         G__UnlockCriticalSection();
         return (G__InterfaceMethod) G__get_funcproperties(fFunc)->entry.p; // FIXME: Wrong, cannot unlock before fetching value!!!
      }
      else {
         G__UnlockCriticalSection();
         return 0;
      }
   }
   G__UnlockCriticalSection();
   return 0;
}

#ifdef G__ASM_WHOLEFUNC
//______________________________________________________________________________
G__bytecodefunc* G__MethodInfo::GetBytecode()
{
   if (IsValid()) {
      int store_asm_loopcompile = G__asm_loopcompile;
      G__asm_loopcompile = 4;
      if (
         !G__get_funcproperties(fFunc)->entry.bytecode &&
         (G__get_funcproperties(fFunc)->entry.size != -1) &&
         (G__get_funcproperties(fFunc)->entry.bytecodestatus == G__BYTECODE_NOTYET) &&
         (G__asm_loopcompile > 3)
      ) {
         G__compile_function_bytecode(fFunc);
      }
      G__asm_loopcompile = store_asm_loopcompile;
      return G__get_funcproperties(fFunc)->entry.bytecode;
   }
   return 0;
}
#endif // G__ASM_WHOLEFUNC

//______________________________________________________________________________
Cint::G__DataMemberInfo Cint::G__MethodInfo::GetLocalVariable()
{
   G__DataMemberInfo localvar;
   localvar.Init(0L, -1L, 0);
   if (!IsValid()) {
      return localvar;
   }
   int store_fixedscope = G__fixedscope;
   G__xrefflag = 1;
   G__fixedscope = 1;
   G__bytecodefunc* pbc = GetBytecode();
   G__xrefflag = 0;
   G__fixedscope = store_fixedscope;
   if (!pbc) {
      if (Property() & G__BIT_ISCOMPILED) {
         G__fprinterr(G__serr, "Limitation: can not get local variable information for compiled function %s\n", Name());
      }
      else {
         G__fprinterr(G__serr, "Limitation: function %s , failed to get local variable information\n", Name());
      }
      return localvar;
   }
   localvar.Init((long) pbc->frame.Id(), -1L, 0);
   return localvar;
}

#ifdef G__TRUEP2F
//______________________________________________________________________________
void* Cint::G__MethodInfo::PointerToFunc()
{
   if (IsValid()) {
      if (
         (G__get_funcproperties(fFunc)->entry.size != -1) &&
         (G__get_funcproperties(fFunc)->entry.bytecodestatus == G__BYTECODE_NOTYET) &&
         (G__asm_loopcompile > 3)
      ) {
         G__compile_function_bytecode(fFunc);
      }
      if (G__get_funcproperties(fFunc)->entry.bytecodestatus == G__BYTECODE_SUCCESS) {
         return (void*) G__get_funcproperties(fFunc)->entry.bytecode;
      }
      return G__get_funcproperties(fFunc)->entry.tp2f;
   }
   return 0;
}
#endif // G__TRUEP2F

//______________________________________________________________________________
G__ClassInfo* Cint::G__MethodInfo::MemberOf()
{
   // Return a G__ClassInfo representing the declaring fScope.
   if (!fClassInfo) {
      fClassInfo = new G__ClassInfo(G__get_tagnum(fScope)); // FIXME: What if fScope is the global scope?
   }
   return fClassInfo;
}

//______________________________________________________________________________
int Cint::G__MethodInfo::GetDefiningScopeTagnum()
{
   if (IsValid()) {
      return G__get_tagnum(fFunc.DeclaringScope());
   }
   return -1;
}

//______________________________________________________________________________
G__friendtag* Cint::G__MethodInfo::GetFriendInfo()
{
   if (IsValid()) {
      return G__get_funcproperties(fFunc)->entry.friendtag;
   }
   return 0;
}

//______________________________________________________________________________
void Cint::G__MethodInfo::SetGlobalcomp(int globalcomp)
{
   if (IsValid()) {
      G__get_funcproperties(fFunc)->globalcomp = globalcomp;
   }
}

//______________________________________________________________________________
void Cint::G__MethodInfo::SetForceStub()
{
   // FIXME: This needs to be implemented!
}

//______________________________________________________________________________
int Cint::G__MethodInfo::IsValid()
{
   return (bool) fFunc;
}

//______________________________________________________________________________
int Cint::G__MethodInfo::SetFilePos(const char* fname)
{
   G__dictposition* dict = G__get_dictpos((char*)fname);
   if (!dict) {
      return 0;
   }
   fIndex = dict->ifn - 1;
   delete fClassInfo;
   fClassInfo = 0;
   fScope = dict->ifunc;
   delete fTypeInfo;
   fTypeInfo = 0;
   fFunc = fScope.DataMemberAt(fIndex);
   return 1;
}

//______________________________________________________________________________
int Cint::G__MethodInfo::Next()
{
   if (!fScope) {
      return 0;
   }
   if ((fIndex == -1) && fFunc) {
      for (fIndex = 0; fIndex < (int) fScope.FunctionMemberSize(); ++fIndex) {
         if (fFunc == fScope.FunctionMemberAt(fIndex)) {
            break;
         }
      }
   }
   ++fIndex;
   delete fTypeInfo;
   fTypeInfo = 0;
   if (fIndex < (int) fScope.FunctionMemberSize()) {
      fFunc = fScope.FunctionMemberAt(fIndex);
   }
   else {
      fIndex = -1;
      fFunc = ::Reflex::Member();
   }
   if (!fFunc && !fScope && (fUsingIndex < (long) G__globalusingnamespace.vec.size())) {
      ++fUsingIndex;
      G__incsetup_memfunc(G__globalusingnamespace.vec[fUsingIndex].basetagnum);
      delete fClassInfo;
      fClassInfo = 0;
      fScope = G__Dict::GetDict().GetScope(G__globalusingnamespace.vec[fUsingIndex].basetagnum);
      return Next();
   }
   return (bool) fFunc;
}

//______________________________________________________________________________
const char* Cint::G__MethodInfo::FileName()
{
   if (IsValid()) {
      if (G__get_funcproperties(fFunc)->filenum >= 0) {
         return G__srcfile[G__get_funcproperties(fFunc)->filenum].filename;
      }
      return "(compiled)";
   }
   return 0;
}

//______________________________________________________________________________
int Cint::G__MethodInfo::LineNumber()
{
   if (IsValid()) {
      if (
         (G__get_funcproperties(fFunc)->filenum >= 0) &&
         (G__get_funcproperties(fFunc)->entry.size >= 0)
      ) {
         return G__get_funcproperties(fFunc)->linenum;
      }
      return 0;
   }
   return -1;
}

//______________________________________________________________________________
int Cint::G__MethodInfo::Size()
{
   if (IsValid()) {
      if (G__get_funcproperties(fFunc)->entry.size >= 0) {
         return G__get_funcproperties(fFunc)->entry.size;
      }
      return 0;
   }
   return -1;
}

//______________________________________________________________________________
int Cint::G__MethodInfo::IsBusy()
{
   if (IsValid()) {
      return G__get_funcproperties(fFunc)->entry.busy;
   }
   return -1;
}

//______________________________________________________________________________
FILE* Cint::G__MethodInfo::FilePointer()
{
   if (IsValid()) {
      if (
         (G__get_funcproperties(fFunc)->filenum >= 0) &&
         (G__get_funcproperties(fFunc)->entry.size >= 0)
      ) {
         return(G__srcfile[G__get_funcproperties(fFunc)->filenum].fp);
      }
   }
   return 0;
}

//______________________________________________________________________________
long Cint::G__MethodInfo::FilePosition()
{
   // returns  'type fname(type p1,type p2)'
   //                      ^
   long invalid = 0L;
   if (IsValid()) {
      if (
         (G__get_funcproperties(fFunc)->filenum >= 0) &&
         (G__get_funcproperties(fFunc)->entry.size >= 0)
      ) {
         // --
#if defined(G__NONSCALARFPOS2)
         return (long) G__get_funcproperties(fFunc)->entry.pos.__pos;
#elif defined(G__NONSCALARFPOS_QNX)
         return (long) G__get_funcproperties(fFunc)->entry.pos._Off;
#else // G__NONSCALARFPOSxxx
         return (long) G__get_funcproperties(fFunc)->entry.pos;
#endif // G__NONSCALARFPOSxxx
         // --
      }
   }
   return invalid;
}

//______________________________________________________________________________
char* Cint::G__MethodInfo::GetPrototype()
{
   if (!IsValid()) {
      return 0;
   }
   strcpy(G__buf, Type()->Name());
   strcat(G__buf, " ");
   if (fScope && !fScope.IsTopScope()) {
      strcat(G__buf, fScope.Name(::Reflex::SCOPED).c_str());
      strcat(G__buf, "::");
   }
   strcat(G__buf, fFunc.Name().c_str());
   strcat(G__buf, "(");
   G__MethodArgInfo arg(*this);
   int flag = 0;
   while (arg.Next()) {
      if (flag) strcat(G__buf, ",");
      flag = 1;
      strcat(G__buf, arg.Type()->Name());
      strcat(G__buf, " ");
      if (arg.Name()) strcat(G__buf, arg.Name());
      if (arg.DefaultValue()) {
         strcat(G__buf, "=");
         strcat(G__buf, arg.DefaultValue());
      }
   }
   strcat(G__buf, ")");
   return G__buf;
}

//______________________________________________________________________________
char* Cint::G__MethodInfo::GetMangledName()
{
   if (!IsValid()) {
      return 0;
   }
   return G__map_cpp_name(GetPrototype());
}

//______________________________________________________________________________
int Cint::G__MethodInfo::LoadDLLDirect(const char* filename, const char* funcname)
{
   void* p2f = G__FindSym(filename, funcname);
   if (p2f) {
      G__RflxFuncProperties* prop = G__get_funcproperties(fFunc);
      prop->entry.tp2f = p2f;
      prop->entry.p = (void*) G__DLL_direct_globalfunc;
      prop->entry.size = -1;
      prop->linenum = -1;
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
void Cint::G__MethodInfo::SetVtblIndex(int vtblindex)
{
   if (!IsValid()) {
      return;
   }
   G__get_funcproperties(fFunc)->entry.vtblindex = (short) vtblindex;
}

//______________________________________________________________________________
void Cint::G__MethodInfo::SetIsVirtual(int /*isvirtual*/)
{
   if (!IsValid()) {
      return;
   }
   fprintf(stderr, "Cint::G__MethodInfo::SetIsVirtual: Reflex is read-only\n");
   assert(0); // FIXME: Production code should never just abort!  Error message, error codes, something.
   //ifunc->isvirtual[index] = isvirtual;
}

//______________________________________________________________________________
void Cint::G__MethodInfo::SetVtblBasetagnum(int basetagnum)
{
   if (!IsValid()) {
      return;
   }
   G__get_funcproperties(fFunc)->entry.vtblbasetagnum = (short) basetagnum;
}

//______________________________________________________________________________
void Cint::G__MethodInfo::SetUserParam(void* user)
{
   if (IsValid()) {
      G__get_funcproperties(fFunc)->entry.userparam = user;
   }
}

//______________________________________________________________________________
void* Cint::G__MethodInfo::GetUserParam()
{
   if (IsValid()) {
      return G__get_funcproperties(fFunc)->entry.userparam;
   }
   return 0;
}

//______________________________________________________________________________
long Cint::G__MethodInfo::GetThisPointerOffset()
{
   // Return: Return the this-pointer offset, to adjust it in case of non left-most multiple inheritance
   if (IsValid()) {
      return G__get_funcproperties(fFunc)->entry.ptradjust;
   }
   return 0;
}

//______________________________________________________________________________
//______________________________________________________________________________
//
//  FIXME: What are these functions doing in this file?
//

//______________________________________________________________________________
int Cint::G__SetGlobalcomp(char* funcname, char* param, int globalcomp)
{
   G__ClassInfo globalscope;
   G__MethodInfo method;
   long dummy = 0;
   G__StrBuf classname_sb(G__LONGLINE);
   char* classname = classname_sb;
   // Actually find the last :: to get the full classname, including
   // namespace and/or containing classes.
   strcpy(classname, funcname);
   char* fname = 0;
   char* tmp = classname;
   while ((tmp = strstr(tmp, "::"))) {
      fname = tmp;
      tmp += 2;
   }
   if (fname) {
      *fname = 0;
      fname += 2;
      globalscope.Init(classname);
   }
   else {
      fname = funcname;
   }
   if (!strcmp(fname, "*")) {
      method.Init(globalscope);
      while (method.Next()) {
         method.SetGlobalcomp(globalcomp);
      }
      return 0;
   }
   method = globalscope.GetMethod(fname, param, &dummy);
   if (method.IsValid()) {
      method.SetGlobalcomp(globalcomp);
      return 0;
   }
   G__fprinterr(G__serr, "Warning: #pragma link, function %s(%s) not found", fname, param);
   G__printlinenum();
   return 1;
}

//______________________________________________________________________________
int Cint::G__ForceBytecodecompilation(char* funcname, char* param)
{
   G__ClassInfo globalscope;
   G__MethodInfo method;
   long dummy = 0;
   G__StrBuf classname_sb(G__LONGLINE);
   char* classname = classname_sb;
   // Actually find the last :: to get the full classname, including
   // namespace and/or containing classes.
   strcpy(classname, funcname);
   char* fname = 0;
   char* tmp = classname;
   while ((tmp = strstr(tmp, "::"))) {
      fname = tmp;
      tmp += 2;
   }
   if (fname) {
      *fname = 0;
      fname += 2;
      globalscope.Init(classname);
   }
   else {
      fname = funcname;
   }
   method = globalscope.GetMethod(fname, param, &dummy);
   if (method.IsValid()) {
      G__ifunc_table* ifunc = method.ifunc();
      int ifn = method.Index();
      int stat;
      int store_asm_loopcompile = G__asm_loopcompile;
      int store_asm_loopcompile_mode = G__asm_loopcompile_mode;
      G__asm_loopcompile_mode = G__asm_loopcompile = 4;
      stat = G__compile_bytecode(ifunc, ifn);
      G__asm_loopcompile = store_asm_loopcompile;
      G__asm_loopcompile_mode = store_asm_loopcompile_mode;
      if (stat) {
         return 0;
      }
      return 1;
   }
   G__fprinterr(G__serr, "Warning: function %s(%s) not found", fname, param);
   G__printlinenum();
   return 1;
}

