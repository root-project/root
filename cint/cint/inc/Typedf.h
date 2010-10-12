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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#ifndef G__TYPEDEFINFO_H
#define G__TYPEDEFINFO_H 


#ifndef G__API_H
#include "Api.h"
#endif

namespace Cint {

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

  static int GetNumTypedefs();
};

} // namespace Cint

using namespace Cint;
#endif
