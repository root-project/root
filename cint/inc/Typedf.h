/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file Typedf.h
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


#ifndef G__TYPEDEFINFO_H
#define G__TYPEDEFINFO_H 


#include "Api.h"

/*********************************************************************
* class G__TypedefInfo
*
* 
*********************************************************************/
class 
#ifndef __CINT__
G__EXPORT
#endif
G__TypedefInfo : public G__TypeInfo {
 public:
  ~G__TypedefInfo() {}
  G__TypedefInfo() { Init(); }
  void Init();
  G__TypedefInfo(const char *typenamein) { Init(typenamein); } 
  void Init(const char *typenamein);
  G__TypedefInfo(int typenumin) { Init(typenumin); } 
  void Init(int typenumin);
  G__ClassInfo EnclosingClassOfTypedef();

  const char *Title() ;
  void SetGlobalcomp(int globalcomp);
  int IsValid();
  int SetFilePos(const char* fname);
  int Next();

  /* added with G__TYPEDEFFPOS */
  const char *FileName();
  int LineNumber();
};

#endif
