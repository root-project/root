/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file MethodAr.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/


#ifndef G__METHODARGINFO_H
#define G__METHODARGINFO_H

#include "Api.h"

/*********************************************************************
* class G__MethodArgInfo
*
* 
*********************************************************************/
class 
#ifndef __CINT__
G__EXPORT
#endif
G__MethodArgInfo {
 public:
  ~G__MethodArgInfo() {}
  void Init(class G__MethodInfo &a);
  G__MethodArgInfo(class G__MethodInfo &a) : type() { Init(a); } 

  const char *Name();
  G__TypeInfo* Type() { return(&type); }
  long Property();
  char *DefaultValue();
  G__MethodInfo* ArgOf() { return(belongingmethod); }
  int IsValid();
  int Next();

 private:
  long argn;
  G__MethodInfo *belongingmethod;
  G__TypeInfo type;

 public:
  G__MethodArgInfo() {} 

};

#endif
