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

#include "Api.h"

namespace Cint {

class
#ifndef __CINT__
G__EXPORT
#endif // __CINT__
G__MethodArgInfo
{
public:
   ~G__MethodArgInfo();
   G__MethodArgInfo();
   G__MethodArgInfo(G__MethodInfo&);
   void Init(G__MethodInfo&);
   G__MethodArgInfo(const G__MethodArgInfo&);
   G__MethodArgInfo& operator=(const G__MethodArgInfo&);
   const char* Name();
   G__TypeInfo* Type();
   long Property();
   const char* DefaultValue();
   G__MethodInfo* ArgOf();
   int IsValid();
   int Next();
private:
   long argn;
   G__MethodInfo* belongingmethod;
   G__TypeInfo type;
};

} // namespace Cint

using namespace Cint;
#endif // G__METHODARGINFO_H
