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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#ifndef G__METHODARGINFO_H
#define G__METHODARGINFO_H

#ifndef G__API_H
#include "Api.h"
#endif

namespace Cint {

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
  G__MethodArgInfo(class G__MethodInfo &a)
    : argn(0), belongingmethod(NULL), type() { Init(a); } 
  G__MethodArgInfo(const G__MethodArgInfo& mai)
    : argn(mai.argn), belongingmethod(mai.belongingmethod), type(mai.type) 
      { } 
  G__MethodArgInfo& operator=(const G__MethodArgInfo& mai) {
    argn=mai.argn; belongingmethod=mai.belongingmethod; 
    type=mai.type; return *this;}

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
  G__MethodArgInfo(): argn(0), belongingmethod(NULL), type() {} 

};

} // namespace Cint

using namespace Cint;
#endif
