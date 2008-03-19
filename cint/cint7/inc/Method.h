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
#else
namespace Reflex {
   class Member;
} // namespace Reflex
#endif
namespace Cint {

/*********************************************************************
* class G__MethodInfo
*
* 
*********************************************************************/
class 
#ifndef __CINT__
G__EXPORT
#endif
G__MethodInfo {
  friend class G__MethodArgInfo;
 public:
  ~G__MethodInfo() { delete fClassInfo; }
  G__MethodInfo(): iter(-1),usingIndex(0),fClassInfo(0),fTypeInfo(0) { Init(); }
  G__MethodInfo(G__ClassInfo &a): usingIndex(0),fClassInfo(0),fTypeInfo(0) { Init(a); } 
  G__MethodInfo(const G__MethodInfo& mi)
    : fScope(mi.fScope), fFunc(mi.fFunc), iter(mi.iter), usingIndex(mi.usingIndex), fClassInfo(0),fTypeInfo(0) {}
  G__MethodInfo( const ::Reflex::Member & m ) : fScope(m.DeclaringScope()), fFunc(m), iter(-1),usingIndex(0),fClassInfo(0),fTypeInfo(0) {}
  G__MethodInfo& operator=(const G__MethodInfo& mi) {
    fScope=mi.fScope; fFunc=mi.fFunc; iter=mi.iter; usingIndex=mi.usingIndex; fClassInfo = 0; fTypeInfo = 0; fName = ""; return *this;}

  void Init();
  void Init(G__ClassInfo &a);
  void Init(long handlein /* go via G__Dict, ignore indexin, belongingclassin */,long indexin,G__ClassInfo *belongingclassin);
  void Init(G__ClassInfo *belongingclassin,long funcpage /* use to access funcpage'th fFunc member */,long indexin /* ignore */);

  const char *Name() ;
#ifndef __MAKECINT__
  int Hash() ;
  struct G__ifunc_table* ifunc();
#endif
  size_t Handle() { return (size_t)fFunc.Id(); }
  Reflex::Member ReflexFunction() const { return fFunc; }
  int Index() { return -2; } // signal that Hash has the fFunc's Id()
  const char *Title() ;
  G__TypeInfo* Type();
  long Property();
  int NArg();
  int NDefaultArg();
  int HasVarArgs();
  G__InterfaceMethod InterfaceMethod();
#ifdef G__ASM_WHOLEFUNC
  struct G__bytecodefunc *GetBytecode();
#endif
  G__DataMemberInfo GetLocalVariable(); /* ON1163 */
#ifdef G__TRUEP2F
  void* PointerToFunc();
#endif
  G__ClassInfo* MemberOf();
  int GetDefiningScopeTagnum();
  struct G__friendtag* GetFriendInfo();
  void SetGlobalcomp(int globalcomp);
  int IsValid();
  int SetFilePos(const char* fname);
  int Next();

  const char *FileName();
  int LineNumber();
  int Size();
  int IsBusy();
  FILE* FilePointer();
  long FilePosition();
  char* GetPrototype();
  char* GetMangledName();

  int LoadDLLDirect(const char* filename,const char* funcname);

  void SetVtblIndex(int vtblindex);
  void SetIsVirtual(int isvirtual);
  void SetVtblBasetagnum(int basetagnum);

  void  SetUserParam(void*);
  void *GetUserParam();
  long  GetThisPointerOffset();

 protected:
    Reflex::Scope fScope;
    Reflex::Member fFunc;
    int iter;
    //long handle;
    //long index;
#ifndef G__OLDIMPLEMENTATION2194
    long usingIndex;
#endif
    //Members from backward compatibility
    std::string fName;
    G__ClassInfo *fClassInfo;
    G__TypeInfo *fTypeInfo;
};

} // namespace Cint

using namespace Cint;
#endif

