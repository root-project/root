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

#ifndef G__API_H
#include "Api.h"
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
  ~G__MethodInfo() { if (memberOf != belongingclass) delete memberOf; }
  G__MethodInfo() 
    : handle(0), index(0), usingIndex(0), belongingclass(NULL), memberOf(0), type() 
    { Init(); }
  G__MethodInfo(G__ClassInfo &a)
    : handle(0), index(0), usingIndex(0), belongingclass(NULL), memberOf(0), type() 
    { Init(a); } 
  G__MethodInfo(const G__MethodInfo& mi)
    : handle(mi.handle), index(mi.index), usingIndex(mi.usingIndex), 
     belongingclass(mi.belongingclass), memberOf(0), type(mi.type) {}
  G__MethodInfo& operator=(const G__MethodInfo& mi) {
    handle=mi.handle; index=mi.index; usingIndex=mi.usingIndex; 
    belongingclass=mi.belongingclass; memberOf=NULL; type=mi.type; return *this;}

  void Init();
  void Init(G__ClassInfo &a);
  void Init(long handlein,long indexin,G__ClassInfo *belongingclassin);
  void Init(G__ClassInfo *belongingclassin,long funcpage,long indexin);

  const char *Name() ;
#ifndef __MAKECINT__
  int Hash() ;
  struct G__ifunc_table* ifunc();
#endif
  long Handle() { return(handle); }
  int Index() { return ((int)index); }
  const char *Title() ;
  G__TypeInfo* Type() { return(&type); }
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
  G__ClassInfo* CreatedBy() { return(belongingclass); }
  G__ClassInfo* MemberOf();
  int GetDefiningScopeTagnum();
  struct G__friendtag* GetFriendInfo();
  void SetGlobalcomp(int globalcomp);
  void SetForceStub();
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
  long handle;
  long index;
#ifndef G__OLDIMPLEMENTATION2194
  long usingIndex;
#endif
  G__ClassInfo* belongingclass;
  G__ClassInfo* memberOf;
  G__TypeInfo type;
  
};

} // namespace Cint

using namespace Cint;
#endif

