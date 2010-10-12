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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#ifndef G__BaseClassInfo_H
#define G__BaseClassInfo_H

#ifndef G__API_H
#include "Api.h"
#endif

namespace Cint {

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
} // namespace Cint

using namespace Cint;
#endif
