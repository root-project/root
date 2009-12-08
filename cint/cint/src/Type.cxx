/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file Type.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Author                  Masaharu Goto 
 * Copyright(c) 1995~2005  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "common.h"
#include "FastAllocString.h"

/*********************************************************************
* class G__TypeInfo
* 
*********************************************************************/
///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::G__TypeInfo(const char *typenamein):
   G__ClassInfo(), type(0), typenum(-1), reftype(0), isconst(0)
{
   Init(typenamein);
}
///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::G__TypeInfo():
   G__ClassInfo(), type(0), typenum(-1), reftype(0), isconst(0)
{}
///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::G__TypeInfo(G__value buf):
   G__ClassInfo(), type(0), typenum(-1), reftype(0), isconst(0)
{
   Init(buf);
}
///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::~G__TypeInfo() {}
#ifndef __MAKECINT__
///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::G__TypeInfo(const Cint::G__TypeInfo& rhs)
: G__ClassInfo(rhs)
{
   type = rhs.type;
   typenum = rhs.typenum;
   reftype = rhs.reftype;
   isconst = rhs.isconst;
}

///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo& Cint::G__TypeInfo::operator=(const Cint::G__TypeInfo& rhs)
{
   if (this != &rhs) {
      type = rhs.type;
      typenum = rhs.typenum;
      reftype = rhs.reftype;
      isconst = rhs.isconst;
   }
   return *this;
}
#endif // __MAKECINT__
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypeInfo::Init(G__value& buf) { 
   type    = buf.type; 
   typenum = buf.typenum; 
   tagnum  = buf.tagnum;
   if(type!='d' && type!='f') reftype=buf.obj.reftype.reftype;
   else              reftype=0;
   isconst = buf.isconst;
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypeInfo::Init(struct G__var_array *var,int ig15) {
   type    = var->type[ig15]; 
   typenum = var->p_typetable[ig15]; 
   tagnum  = var->p_tagtable[ig15];
   reftype = var->reftype[ig15];
   isconst = var->constvar[ig15];
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypeInfo::Init(const char *typenamein)
{
  G__value buf;
  buf = G__string2type_body(typenamein,2);
  type = buf.type;
  tagnum = buf.tagnum;
  typenum = buf.typenum;
  reftype = buf.obj.reftype.reftype;
  isconst = buf.obj.i;
  class_property = 0;
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::operator==(const G__TypeInfo& a)
{
  if(type==a.type && tagnum==a.tagnum && typenum==a.typenum &&
     reftype==a.reftype) {
    return(1);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::operator!=(const G__TypeInfo& a)
{
  if(type==a.type && tagnum==a.tagnum && typenum==a.typenum &&
     reftype==a.reftype) {
    return(0);
  }
  else {
    return(1);
  }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__TypeInfo::TrueName() 
{
  static G__FastAllocString buf(G__LONGLINE);
  buf = G__type2string((int)type,(int)tagnum,-1,(int)reftype,(int)isconst);
  return(buf);
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__TypeInfo::Name() 
{
  static G__FastAllocString buf(G__LONGLINE);
  buf = G__type2string((int)type,(int)tagnum,(int)typenum,(int)reftype
                       ,(int)isconst);
  return(buf);
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Size() const
{
  G__value buf;
  buf.type=(int)type;
  buf.tagnum=(int)tagnum;
  buf.typenum=(int)typenum;
  buf.ref=reftype;
  if(isupper(type)) {
    buf.obj.reftype.reftype=reftype;
    return(sizeof(void*));
  }
  return(G__sizeof(&buf));
}
///////////////////////////////////////////////////////////////////////////
long Cint::G__TypeInfo::Property()
{
  long property = 0;
  if(-1!=typenum) property|=G__BIT_ISTYPEDEF;
  if(-1==tagnum) property|=G__BIT_ISFUNDAMENTAL;
  else {
    if(strcmp(G__struct.name[tagnum],"G__longlong")==0 ||
       strcmp(G__struct.name[tagnum],"G__ulonglong")==0 ||
       strcmp(G__struct.name[tagnum],"G__longdouble")==0) {
      property|=G__BIT_ISFUNDAMENTAL;
      if(-1!=typenum && 
	 (strcmp(G__newtype.name[typenum],"long long")==0 ||
	  strcmp(G__newtype.name[typenum],"unsigned long long")==0 ||
	  strcmp(G__newtype.name[typenum],"long double")==0)) {
	property &= (~G__BIT_ISTYPEDEF);
      }
    }
    else {
      if(G__ClassInfo::IsValid()) property|=G__ClassInfo::Property();
    }
  }
  if(isupper((int)type)) property|=G__BIT_ISPOINTER;
  if(reftype==G__PARAREFERENCE||reftype>G__PARAREF) 
    property|=G__BIT_ISREFERENCE;
  if(isconst&G__CONSTVAR)  property|=G__BIT_ISCONSTANT;
  if(isconst&G__PCONSTVAR) property|=G__BIT_ISPCONSTANT;
  return(property);
}
///////////////////////////////////////////////////////////////////////////
void* Cint::G__TypeInfo::New() {
  if(G__ClassInfo::IsValid()) {
    return(G__ClassInfo::New());
  }
  else {
    size_t size;
    void *p;
    size = Size();
    p = new char[size];
    return(p);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::IsValid() {
  if(G__ClassInfo::IsValid()) {
    return(1);
  }
  else if(type) {
    return(1);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Typenum() const { return(typenum); }
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Type() const { return(type); }
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Reftype() const { return(reftype); }
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Isconst() const { return(isconst); }
///////////////////////////////////////////////////////////////////////////
G__value Cint::G__TypeInfo::Value() const {
   G__value buf;
   buf.type=type;
   buf.tagnum=tagnum;
   buf.typenum=typenum;
   buf.isconst=(G__SIGNEDCHAR_T)isconst;
   buf.obj.reftype.reftype = reftype;
   buf.obj.i = 1;
   buf.ref = 0;
   return(buf);
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Next()
{
   return 0;
}

