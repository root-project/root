/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file Method.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
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
class G__MethodInfo {
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
  int Index() { return ((int)index); }
#endif
  const char *Title() ;
  G__TypeInfo* Type() { return(&type); }
  long Property();
  int NArg();
  int NDefaultArg();
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

 protected:
  long handle;
  long index;
  G__ClassInfo* belongingclass;
  G__TypeInfo type;
};

extern "C" int G__SetGlobalcomp(char *funcname,char *param,int globalcomp);

#endif

