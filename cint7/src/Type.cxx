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

// Class header.
//#include "Type.h"

// Non-system headers.
#include "Api.h"
#include "common.h"

// System provided headers.

// Using declarations.
using namespace Cint::Internal;

/*********************************************************************
* class G__TypeInfo
*
*********************************************************************/

///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::~G__TypeInfo()
{
   delete reflexInfo;
   reflexInfo = 0;
}

///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::G__TypeInfo(const char* typenamein)
: G__ClassInfo()
, reftype(0)
, isconst(0)
, reflexInfo(0)
{
   Init(typenamein);
}

///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::G__TypeInfo()
: G__ClassInfo()
, reftype(0)
, isconst(0)
, reflexInfo(0)
{
   reflexInfo = new ReflexInfo;
   reflexInfo->type = 0;
   reflexInfo->typeiter = -1;
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__TypeInfo::Init(const char* typenamein)
{
   G__value buf = G__string2type_body(typenamein, 2);
   // G__ClassInfo part.
   tagnum = buf.tagnum;
   class_property = 0;
   // G__TypeInfo part.
   reftype = buf.obj.reftype.reftype;
   isconst = buf.obj.i;
   reflexInfo = new ReflexInfo;
   reflexInfo->type = buf.type;
   reflexInfo->typenum = G__value_typenum(buf);
   reflexInfo->typeiter = -1;
}

///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::G__TypeInfo(G__value buf)
: G__ClassInfo()
, reftype(0)
, isconst(0)
, reflexInfo(0)
{
   Init(buf);
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__TypeInfo::Init(G__value& buf)
{
   // G__ClassInfo part.
   tagnum = buf.tagnum;
   // G__TypeInfo part.
   if ((buf.type == 'd') || (buf.type == 'f')) {
      reftype = 0;
   } else {
      reftype = buf.obj.reftype.reftype;
   }
   isconst = buf.isconst;
   if (!reflexInfo) {
      reflexInfo = new ReflexInfo;
   }
   reflexInfo->type = buf.type;
   reflexInfo->typenum = Internal::G__value_typenum(buf);
   reflexInfo->typeiter = -1;
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__TypeInfo::Init(struct G__var_array* var, int i)
{
   // G__ClassInfo part.
   tagnum = var->p_tagtable[i];
   // G__TypeInfo part.
   reftype = var->reftype[i];
   isconst = var->constvar[i];
   if (!reflexInfo) {
      reflexInfo = new ReflexInfo;
   }
   reflexInfo->type = var->type[i];
   reflexInfo->typenum = var->p_typetable[i];
   reflexInfo->typeiter = -1;
}

#ifndef __MAKECINT__
///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::G__TypeInfo(const Cint::G__TypeInfo& rhs)
: G__ClassInfo(rhs)
{
   reftype = rhs.reftype;
   isconst = rhs.isconst;
   reflexInfo = new ReflexInfo(*rhs.reflexInfo);
}

///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo& Cint::G__TypeInfo::operator=(const Cint::G__TypeInfo& rhs)
{
   if (this != &rhs) {
      reftype = rhs.reftype;
      isconst = rhs.isconst;
      delete reflexInfo;
      reflexInfo = new ReflexInfo(*rhs.reflexInfo);
   }
   return *this;
}
#endif // __MAKECINT__

///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::operator==(const G__TypeInfo& a)
{
   if (
      reflexInfo->type == a.reflexInfo->type &&
      tagnum == a.tagnum &&
      reflexInfo->typenum == a.reflexInfo->typenum &&
      reftype == a.reftype
   ) {
      return 1;
   }
   return 0;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::operator!=(const G__TypeInfo& a)
{
   if (
      reflexInfo->type == a.reflexInfo->type &&
      tagnum == a.tagnum &&
      reflexInfo->typenum == a.reflexInfo->typenum &&
      reftype == a.reftype
   ) {
      return 0;
   }
   return 1;
}

///////////////////////////////////////////////////////////////////////////
const char* Cint::G__TypeInfo::Name()
{
   // This length should match or exceed the length in G__type2string
   static char G__buf[G__LONGLINE];

#ifdef __GNUC__
#else
#pragma message(FIXME("Get rid of static G__buf by proper lookup"))
#endif

   strcpy(G__buf,G__type2string((int)reflexInfo->type,(int)tagnum,G__get_typenum(reflexInfo->typenum),(int)reftype,(int)isconst));
   return G__buf;
}

///////////////////////////////////////////////////////////////////////////
const char* Cint::G__TypeInfo::TrueName()
{
   // This length should match or exceed the length in G__type2string
   static char G__buf[G__LONGLINE];

#ifdef __GNUC__
#else
#pragma message(FIXME("Get rid of static G__buf by proper lookup"))
#endif

   strcpy(G__buf, G__type2string((int)reflexInfo->type,(int)tagnum,-1,(int)reftype,(int)isconst));
   return G__buf;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Size() const
{
   G__value buf;
   buf.type = (int) reflexInfo->type;
   buf.tagnum = (int) tagnum;
   G__value_typenum(buf) = reflexInfo->typenum;
   buf.ref = reftype;
   if (isupper(reflexInfo->type)) {
     buf.obj.reftype.reftype = reftype;
     return sizeof(void*);
   }
   return G__sizeof(&buf);
}

///////////////////////////////////////////////////////////////////////////
long Cint::G__TypeInfo::Property()
{
   long property = 0L;
   if (reflexInfo->typenum) {
      property |= G__BIT_ISTYPEDEF;
   }
   if (tagnum == -1) {
      property |= G__BIT_ISFUNDAMENTAL;
   } else {
      if (
         strcmp(G__struct.name[tagnum],"G__longlong") == 0 ||
         strcmp(G__struct.name[tagnum],"G__ulonglong") == 0 ||
         strcmp(G__struct.name[tagnum],"G__longdouble") == 0
      ) {
         property |= G__BIT_ISFUNDAMENTAL;
         if (
            reflexInfo->typenum &&
            (
               reflexInfo->typenum.Name() == "long long" ||
               reflexInfo->typenum.Name() == "unsigned long long" ||
               reflexInfo->typenum.Name() == "long double"
            )
         ) {
            property &= (~G__BIT_ISTYPEDEF);
         }
      } else {
         if (G__ClassInfo::IsValid()) {
            property |= G__ClassInfo::Property();
         }
      }
   }
   if (isupper((int) reflexInfo->type)) {
      property |= G__BIT_ISPOINTER;
   }
   if ((reftype == G__PARAREFERENCE) || (reftype > G__PARAREF)) {
      property |= G__BIT_ISREFERENCE;
   }
   if (isconst & G__CONSTVAR) {
      property |= G__BIT_ISCONSTANT;
   }
   if (isconst & G__PCONSTVAR) {
      property |= G__BIT_ISPCONSTANT;
   }
   return property;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::IsValid()
{
   if (G__ClassInfo::IsValid() || reflexInfo->type) {
      return 1;
   }
   return 0;
}

///////////////////////////////////////////////////////////////////////////
void* Cint::G__TypeInfo::New()
{
   if (G__ClassInfo::IsValid()) {
     return G__ClassInfo::New();
   } else {
     size_t size = Size();
     void* p = (void*) new char[size];
     return p;
   }
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Typenum() const
{
   return G__get_typenum(reflexInfo->typenum);
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Type() const
{
   return reflexInfo->type;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Reftype() const
{
   return reftype;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Isconst() const
{
   return isconst;
}

///////////////////////////////////////////////////////////////////////////
ROOT::Reflex::Type Cint::G__TypeInfo::ReflexType() const
{
   return reflexInfo->typenum;
}

///////////////////////////////////////////////////////////////////////////
G__value Cint::G__TypeInfo::Value() const
{
   G__value buf;
   buf.type = reflexInfo->type;
   buf.tagnum = tagnum;
   Internal::G__value_typenum(buf) = reflexInfo->typenum;
   buf.isconst = (G__SIGNEDCHAR_T) isconst;
   buf.obj.reftype.reftype = reftype;
   buf.obj.i = 1;
   buf.ref = 0;
   return buf;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Next()
{
   return 0;
}

