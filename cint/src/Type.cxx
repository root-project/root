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

#ifndef G__OLDIMPLEMENTATION1586
// This length should match or exceed the length in G__type2string
static char G__buf[G__LONGLINE];
#endif

/*********************************************************************
* class G__TypeInfo
* 
*********************************************************************/
///////////////////////////////////////////////////////////////////////////
void G__TypeInfo::Init(const char *typenamein)
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
int G__TypeInfo::operator==(const G__TypeInfo& a)
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
int G__TypeInfo::operator!=(const G__TypeInfo& a)
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
const char* G__TypeInfo::TrueName() 
{
#if !defined(G__OLDIMPLEMENTATION1586)
  strcpy(G__buf,
	 G__type2string((int)type,(int)tagnum,-1,(int)reftype,(int)isconst));
  return(G__buf);
#elif  !defind(G__OLDIMPLEMENTATION401)
  return(G__type2string((int)type,(int)tagnum,-1,(int)reftype,(int)isconst));
#else
  return(G__type2string((int)type,(int)tagnum,-1,(int)reftype));
#endif
}
///////////////////////////////////////////////////////////////////////////
const char* G__TypeInfo::Name() 
{
#if !defined(G__OLDIMPLEMENTATION1586)
  strcpy(G__buf,G__type2string((int)type,(int)tagnum,(int)typenum,(int)reftype
			       ,(int)isconst));
  return(G__buf);
#elif  !defind(G__OLDIMPLEMENTATION401)
  return(G__type2string((int)type,(int)tagnum,(int)typenum,(int)reftype
	,(int)isconst));
#else
  return(G__type2string((int)type,(int)tagnum,(int)typenum,(int)reftype));
#endif
}
///////////////////////////////////////////////////////////////////////////
int G__TypeInfo::Size() const
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
long G__TypeInfo::Property()
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
void* G__TypeInfo::New() {
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
int G__TypeInfo::IsValid() {
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
