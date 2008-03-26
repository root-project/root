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
#include "Dict.h"
#include "common.h"
#include "fproto.h"

using namespace Cint::Internal;

/*********************************************************************
* class G__TypeInfo
* 
*********************************************************************/

///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::~G__TypeInfo()
{}

///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::G__TypeInfo(const char* typenamein): 
  G__ClassInfo(), typeiter(-1)
{
   Init(typenamein);
}

///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::G__TypeInfo():
G__ClassInfo(), typeiter(-1)
{}

///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::G__TypeInfo(G__value buf):
   G__ClassInfo(), typeiter(-1)
{
   Init(buf);
}

///////////////////////////////////////////////////////////////////////////
Cint::G__TypeInfo::G__TypeInfo(const ::Reflex::Type &in):
   G__ClassInfo(), typeiter(-1)
{
   Init(in);
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__TypeInfo::Init(G__value& buf) { 
   typenum = Internal::G__value_typenum(buf);
   typeiter = -1;
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypeInfo::Init(const char *typenamein)
{
   G__value buf = G__string2type_body(typenamein,2);
   // G__TypeInfo part. 
   typenum = G__value_typenum(buf);
   typeiter= -1;
   // G__ClassInfo part.
   tagnum = G__get_tagnum(typenum);
   class_property = 0;
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypeInfo::Init(const ::Reflex::Type &in)
{
   // G__TypeInfo part. 
   typenum = in;
   typeiter = -1;
   // G__ClassInfo part.
   tagnum = G__get_tagnum(typenum);
   class_property = 0;
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__TypeInfo::Init(struct G__var_array *var,int ig15) {
   ::Reflex::Member m = G__Dict::GetDict().GetDataMember(var,ig15);
   typenum = m.TypeOf();

   // G__ClassInfo part.
   tagnum  = G__get_tagnum(typenum);
   // G__TypeInfo part.
   typeiter= -1;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::operator==(const G__TypeInfo& a)
{
  if(typenum == a.typenum) {
    return(1);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::operator!=(const G__TypeInfo& a)
{
  if(typenum != a.typenum) {
    return(0);
  }
  else {
    return(1);
  }
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Type() const
{
   return G__get_type(typenum);
}

int Cint::G__TypeInfo::Typenum() const
{
   return G__get_typenum(typenum);
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Isconst() const
{
   return G__get_isconst(typenum);
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Reftype() const
{
   return G__get_reftype(typenum);
}

///////////////////////////////////////////////////////////////////////////
Reflex::Type Cint::G__TypeInfo::ReflexType() const
{
   return typenum;
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

  strcpy(G__buf,
	 G__type2string(G__get_type(typenum),(int)tagnum,-1,G__get_reftype(typenum),G__get_isconst(typenum)));
  return G__buf;
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
  char type = '\0';
  int cint5_tagnum = -1;
  int cint5_typenum = -1;
  int reftype = 0;
  int constvar = 0;
  G__get_cint5_type_tuple(typenum, &type, &cint5_tagnum, &cint5_typenum, &reftype, &constvar);
  strcpy(G__buf, G__type2string(type, cint5_tagnum, cint5_typenum, reftype, constvar));
  return G__buf;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Size() const
{
  G__value buf;
  G__value_typenum(buf) = typenum;
  return(G__sizeof(&buf));
}
///////////////////////////////////////////////////////////////////////////
long Cint::G__TypeInfo::Property()
{
   long property = 0L;
   if (typenum.IsTypedef())
      property |= G__BIT_ISTYPEDEF;
   if (-1==tagnum)
      property |= G__BIT_ISFUNDAMENTAL;
   else {
       std::string cname( typenum.RawType().Name() );
       if(cname == "G__longlong" ||
          cname == "G__ulonglong"||
          cname == "G__longdouble") {
             property |= G__BIT_ISFUNDAMENTAL;
             cname = typenum.FinalType().Name(Reflex::FINAL);
             if((property&G__BIT_ISTYPEDEF) && 
                (cname == "long long" ||
                cname == "unsigned long long" ||
                cname == "long double")) {
                   property &= (~G__BIT_ISTYPEDEF);
             }
       } else {
          if (G__ClassInfo::IsValid())
             property |= G__ClassInfo::Property();
       }
   }
   if (isupper(G__get_type(typenum)))
      property |= G__BIT_ISPOINTER;
   if (G__get_reftype(typenum) == G__PARAREFERENCE || G__get_reftype(typenum) > G__PARAREF)
      property |= G__BIT_ISREFERENCE;
   if (G__test_const(typenum,G__CONSTVAR))
      property |= G__BIT_ISCONSTANT;
   if (G__test_const(typenum,G__PCONSTVAR))
      property |= G__BIT_ISPCONSTANT;

   return property;
}
///////////////////////////////////////////////////////////////////////////
void* Cint::G__TypeInfo::New() {
  if(G__ClassInfo::IsValid()) {
    return(G__ClassInfo::New());
  }

  size_t size = Size();
  void *p = (void*) new char[size];
  return p;
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::IsValid() {
  if (G__ClassInfo::IsValid() || typenum) {
    return(1);
  }
   return(0);
}
///////////////////////////////////////////////////////////////////////////
G__value Cint::G__TypeInfo::Value() const {
   G__value buf;
   Internal::G__value_typenum(buf)=typenum;
   buf.obj.i = 1;
   buf.ref = 0;
   return(buf);
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypeInfo::Next()
{
   return 0;
}
