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
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#include "Api.h"
#include "common.h"

#ifndef G__OLDIMPLEMENTATION1586
static char G__buf[G__ONELINE];
#endif

/*********************************************************************
* class G__TypeInfo
* 
*********************************************************************/
///////////////////////////////////////////////////////////////////////////
void G__TypeInfo::Init(const char *typenamein)
{
#ifndef G__OLDIMPLEMENTATION707
  G__value buf;
  buf = G__string2type_body(typenamein,2);
  type = buf.type;
  tagnum = buf.tagnum;
  typenum = buf.typenum;
  reftype = buf.obj.reftype.reftype;
  isconst = buf.obj.i;
#else /* ON707 */
  typenum = G__defined_typename(typenamein);
  if(-1!=typenum) {
    tagnum = G__newtype.tagnum[typenum];
    type = G__newtype.type[typenum];
    reftype = G__newtype.reftype[typenum];
    isconst = 0;
  }
  else {
    tagnum = G__defined_tagname(typenamein,2);
    if(-1!=tagnum) {
      switch(G__struct.type[tagnum]) {
      case 'e': type = 'i'; break;
      default:  type = 'u'; break;
      }
#ifndef G__OLDIMPLEMENTATION602
      reftype = 0;
      isconst = 0;
#endif
    }
    else {
      G__value buf;
      buf = G__castvalue(typenamein,G__null);
      type = buf.type;
      tagnum = buf.tagnum;
      typenum = buf.typenum;
      reftype = buf.ref;
      isconst = 0;
    }
  }
#endif /* ON707 */
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
#ifndef G__OLDIMPLEMENTATION401
#ifndef G__OLDIMPLEMENTATION1586
  strcpy(G__buf,
	 G__type2string((int)type,(int)tagnum,-1,(int)reftype,(int)isconst));
  return(G__buf);
#else
  return(G__type2string((int)type,(int)tagnum,-1,(int)reftype,(int)isconst));
#endif
#else
  return(G__type2string((int)type,(int)tagnum,-1,(int)reftype));
#endif
}
///////////////////////////////////////////////////////////////////////////
const char* G__TypeInfo::Name()
{
#ifndef G__OLDIMPLEMENTATION401
#ifndef G__OLDIMPLEMENTATION1586
  strcpy(G__buf,G__type2string((int)type,(int)tagnum,(int)typenum,(int)reftype
			       ,(int)isconst));
  return(G__buf);
#else
  return(G__type2string((int)type,(int)tagnum,(int)typenum,(int)reftype
	,(int)isconst));
#endif
#else
  return(G__type2string((int)type,(int)tagnum,(int)typenum,(int)reftype));
#endif
}
///////////////////////////////////////////////////////////////////////////
int G__TypeInfo::Size()
{
  G__value buf;
  buf.type=(int)type;
  buf.tagnum=(int)tagnum;
  buf.typenum=(int)typenum;
  buf.ref=reftype;
  return(G__sizeof(&buf));
}
///////////////////////////////////////////////////////////////////////////
long G__TypeInfo::Property()
{
  long property = 0;
  if(-1!=typenum) property|=G__BIT_ISTYPEDEF;
  if(-1==tagnum) property|=G__BIT_ISFUNDAMENTAL;
  if(G__ClassInfo::IsValid()) property|=G__ClassInfo::Property();
  if(isupper((int)type)) property|=G__BIT_ISPOINTER;
#ifndef G__OLDIMPLEMENTATION1453
  if (reftype) property |= G__BIT_ISREFERENCE;
#endif
#ifndef G__OLDIMPLEMENTATION401
  if(isconst&G__CONSTVAR)  property|=G__BIT_ISCONSTANT;
  if(isconst&G__PCONSTVAR) property|=G__BIT_ISPCONSTANT;
#endif
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
    p = malloc(size);
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
