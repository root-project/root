/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file Type.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~2003  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#ifndef G__TYPEINFOX_H
#define G__TYPEINFOX_H 


#include "Api.h"

namespace Cint {

/*********************************************************************
* class G__TypeInfo
* 
*********************************************************************/
class 
#ifndef __CINT__
G__EXPORT
#endif
G__TypeInfo : public G__ClassInfo  {
  friend class G__DataMemberInfo;
  friend class G__MethodInfo;
  friend class G__MethodArgInfo;
 public:
  ~G__TypeInfo();
  G__TypeInfo(const char *typenamein);
  G__TypeInfo();
  void Init(const char *typenamein);
#ifndef __MAKECINT__
  G__TypeInfo(G__value buf);
  void Init(G__value& buf);
  void Init(struct G__var_array *var,int ig15);
  G__TypeInfo(const G__TypeInfo&);
  G__TypeInfo& operator=(const G__TypeInfo&);
#endif
  int operator==(const G__TypeInfo& a);
  int operator!=(const G__TypeInfo& a);
  const char *Name() ;
  const char *TrueName() ;
  int Size() const; 
  long Property();
  int IsValid();
  void *New();

  int Typenum() const;
  int Type() const;
  int Reftype() const;
  int Isconst() const;

  G__value Value() const;
 protected:
  long type;
  long typenum;
  long reftype;
  long isconst;

 private:
  int Next() { return(0); } // prohibit use of next
};

} // namespace Cint

using namespace Cint;
#endif
