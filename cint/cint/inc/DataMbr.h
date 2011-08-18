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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#ifndef G__DATAMEMBER_H
#define G__DATAMEMBER_H

#ifndef G__API_H
#include "Api.h"
#endif

namespace Cint {

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
   G__DataMemberInfo();
   G__DataMemberInfo(const G__DataMemberInfo& dmi);
   G__DataMemberInfo(class G__ClassInfo &a);
   G__DataMemberInfo& operator=(const G__DataMemberInfo& dmi);

  void Init();
  void Init(class G__ClassInfo &a);
  void Init(long handlinin,long indexin,G__ClassInfo *belongingclassin);

  long Handle() { return(handle); }
  int Index() { return ((int)index); }
  const char *Name();
  const char *Title();
  G__TypeInfo* Type();
  long Property();
  long Offset() ;
  int Bitfield();
  int ArrayDim() ;
  long MaxIndex(int dim) ;
  G__ClassInfo* MemberOf();
  void SetGlobalcomp(G__SIGNEDCHAR_T globalcomp);
  int IsValid();
  int SetFilePos(const char* fname);
  int Next();
  int Prev();

  enum error_code { VALID, NOT_INT, NOT_DEF, IS_PRIVATE, UNKNOWN };
  const char *ValidArrayIndex(int *errnum = 0, char **errstr = 0);

  const char *FileName();
  int LineNumber();

  static int SerialNumber();

 private:
  long handle;
  long index;
  G__ClassInfo *belongingclass;
  G__TypeInfo type;
};

}

using namespace Cint;
#endif
