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
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/


#ifndef G__METHODINFO_H
#define G__METHODINFO_H

#include "Api.h"


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
  ~G__MethodInfo() {}
  G__MethodInfo() : type() { Init(); }
  G__MethodInfo(G__ClassInfo &a) : type() { Init(a); } 
  void Init();
  void Init(G__ClassInfo &a);
  void Init(long handlein,long indexin,G__ClassInfo *belongingclassin);
#ifndef G__OLDIMPLEMENTATION644
  void Init(G__ClassInfo *belongingclassin,long funcpage,long indexin);
#endif

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
  G__ClassInfo* MemberOf() { return(belongingclass); }
#ifndef G__OLDIMPLEMENTATION1020
  struct G__friendtag* GetFriendInfo() { 
    if(IsValid()) {
      struct G__ifunc_table *ifunc=(struct G__ifunc_table*)handle;
      return(ifunc->friendtag[index]);
    }
    else return 0;
  }
#endif
  void SetGlobalcomp(int globalcomp);
  int IsValid();
  int SetFilePos(const char* fname);
  int Next();

  const char *FileName();
  int LineNumber();
  int Size();
  int IsBusy();
#ifndef G__OLDIMPLEMENTATION644
  FILE* FilePointer();
  long FilePosition();
#endif
  char* GetPrototype();
  char* GetMangledName();

#ifndef G__OLDIMPLEMENTATION1908
  int LoadDLLDirect(const char* filename,const char* funcname);
#endif

#ifndef G__OLDIMPLEMENTATION2073
  void SetVtblIndex(int vtblindex);
  void SetIsVirtual(int isvirtual);
#endif
#ifndef G__OLDIMPLEMENTATION2084
  void SetVtblBasetagnum(int basetagnum);
#endif

 protected:
  long handle;
  long index;
#ifndef G__OLDIMPLEMENTATION2194
  long usingIndex;
#endif
  G__ClassInfo* belongingclass;
  G__TypeInfo type;
};

extern "C" int G__SetGlobalcomp(char *funcname,char *param,int globalcomp);

extern "C" int G__ForceBytecodecompilation(char *funcname,char *param);

#endif

