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
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/


#ifndef G__TYPEINFOX_H
#define G__TYPEINFOX_H 


#include "Api.h"

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
  ~G__TypeInfo() {}
  G__TypeInfo(const char *typenamein) : G__ClassInfo() { Init(typenamein); }
  G__TypeInfo() : G__ClassInfo() { type=0; typenum= -1; reftype=0; isconst=0; }
  void Init(const char *typenamein);
#ifndef __MAKECINT__
  G__TypeInfo(G__value buf) : G__ClassInfo() { Init(buf); }
  void Init(G__value& buf) { 
    type    = buf.type; 
    typenum = buf.typenum; 
    tagnum  = buf.tagnum;
#ifndef G__OLDIMPLEMENTATION2063
    if(type!='d' && type!='f') reftype=buf.obj.reftype.reftype;
    else              reftype=0;
    isconst = buf.isconst;
#else
    if(isupper((int)type)) reftype=buf.obj.reftype.reftype;
    else              reftype=0;
    isconst = 0;
#endif
  }
  void Init(struct G__var_array *var,int ig15) {
    type    = var->type[ig15]; 
    typenum = var->p_typetable[ig15]; 
    tagnum  = var->p_tagtable[ig15];
    reftype = var->reftype[ig15];
    isconst = var->constvar[ig15];
  }
#endif
  int operator==(const G__TypeInfo& a);
  int operator!=(const G__TypeInfo& a);
  const char *Name() ;
  const char *TrueName() ;
  int Size() const; 
  long Property();
  int IsValid();
  void *New();

  int Typenum() const { return(typenum); }
  int Type() const { return(type); }
  int Reftype() const { return(reftype); }
  int Isconst() const { return(isconst); }

  G__value Value() const {
    G__value buf;
    buf.type=type;
    buf.tagnum=tagnum;
    buf.typenum=typenum;
    buf.isconst=isconst;
    buf.obj.reftype.reftype = reftype;
    buf.obj.i = 1;
    buf.ref = 0;
    return(buf);
  }
 protected:
  long type;
  long typenum;
  long reftype;
#ifndef G__OLDIMPLEMENTATION401
  long isconst;
#endif

 private:
  int Next() { return(0); } // prohibit use of next
};


#endif
