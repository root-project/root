/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file DataMbr.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~1998  Masaharu Goto 
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/


#ifndef G__DATAMEMBER_H
#define G__DATAMEMBER_H

#include "Api.h"

/*********************************************************************
* class G__DataMemberInfo
*
*
*********************************************************************/
class 
#ifndef __CINT__
G__EXPORT
#endif
G__DataMemberInfo {
 public:
  ~G__DataMemberInfo() {}
  G__DataMemberInfo() : type() { Init(); }
  G__DataMemberInfo(class G__ClassInfo &a) : type() { Init(a); }
  void Init();
  void Init(class G__ClassInfo &a);
  void Init(long handlinin,long indexin,G__ClassInfo *belongingclassin);

  long Handle() { return(handle); }
  int Index() { return ((int)index); }
  const char *Name() ;
  const char *Title() ;
  G__TypeInfo* Type() { return(&type); }
  long Property();
  long Offset() ;
  int Bitfield();
  int ArrayDim() ;
  int MaxIndex(int dim) ;
  G__ClassInfo* MemberOf() { return(belongingclass); }
  void SetGlobalcomp(int globalcomp);
  int IsValid();
  int SetFilePos(const char* fname);
  int Next();
  int Prev();

#ifdef G__ROOTSPECIAL
  enum error_code { VALID, NOT_INT, NOT_DEF, IS_PRIVATE, UNKNOWN };
  const char *ValidArrayIndex(int *errnum = 0, char **errstr = 0);
#endif

  const char *FileName();
  int LineNumber();

 private:
  long handle;
  long index;
  G__ClassInfo *belongingclass;
  G__TypeInfo type;
};

#endif
