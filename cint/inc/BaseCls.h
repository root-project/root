/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file BaseCls.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Author                  Masaharu Goto 
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


#ifndef G__BaseClassInfo_H
#define G__BaseClassInfo_H


#include "Api.h"

/*********************************************************************
* class G__BaseClassInfo
*
* Rene says OK
* 
*********************************************************************/
class 
#ifndef __CINT__
G__EXPORT
#endif
G__BaseClassInfo : public G__ClassInfo {
 public:
  ~G__BaseClassInfo() {}
  G__BaseClassInfo(G__ClassInfo &a);
  void Init(G__ClassInfo &a);

  long Offset() ;
  long Property();
  int IsValid();
  int Next();
  int Next(int onlydirect);
  int Prev();
  int Prev(int onlydirect);

 private:
  long basep;
  long derivedtagnum;
};


#endif
