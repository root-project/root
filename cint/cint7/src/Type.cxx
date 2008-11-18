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
using namespace std;

//______________________________________________________________________________
static char G__buf[G__LONGLINE]; // This length should match or exceed the length in G__type2string

//______________________________________________________________________________
Cint::G__TypeInfo::~G__TypeInfo()
{
}

#if 0
//______________________________________________________________________________
Cint::G__TypeInfo::G__TypeInfo(const ::Reflex::Type in) : G__ClassInfo()
{
   Init(in);
}
#endif // 0

//______________________________________________________________________________
Cint::G__TypeInfo::G__TypeInfo(const char* typenamein) : G__ClassInfo(), fType(0), fTypenum(-1), fReftype(0), fIsconst(0)
{
   Init(typenamein);
}

//______________________________________________________________________________
Cint::G__TypeInfo::G__TypeInfo() : G__ClassInfo(), fType(0), fTypenum(-1), fReftype(0), fIsconst(0)
{
}

//______________________________________________________________________________
void Cint::G__TypeInfo::Init(const char* typenamein)
{
   G__value buf = G__string2type_body(typenamein, 2);
   ::Reflex::Type ty = G__value_typenum(buf);
   // G__ClassInfo part.
   fClassProperty = 0;
   // G__TypeInfo part.
   G__get_cint5_type_tuple_long(ty, &fType, &fTagnum, &fTypenum, &fReftype, &fIsconst);
   fIsconst = buf.obj.i;
}

#if 0
//______________________________________________________________________________
void Cint::G__TypeInfo::Init(const ::Reflex::Type in)
{
   // G__ClassInfo part.
   fTagnum = G__get_tagnum(fTypenum);
   class_property = 0;
   // G__TypeInfo part.
   fTypenum = in;
}
#endif // 0

//______________________________________________________________________________
Cint::G__TypeInfo::G__TypeInfo(G__value buf) : G__ClassInfo(), fType(0), fTypenum(-1), fReftype(0), fIsconst(0)
{
   Init(buf);
}

//______________________________________________________________________________
void Cint::G__TypeInfo::Init(G__value& buf)
{
   ::Reflex::Type ty = G__value_typenum(buf);
   G__get_cint5_type_tuple_long(ty, &fType, &fTagnum, &fTypenum, &fReftype, &fIsconst);
   if ((fType == 'd') || (fType == 'f')) {
      fReftype = 0;
   }
}

//______________________________________________________________________________
void Cint::G__TypeInfo::Init(G__var_array* var, int idx)
{
   ::Reflex::Member m = G__Dict::GetDict().GetDataMember(var, idx);
   ::Reflex::Type ty = m.TypeOf();
   G__get_cint5_type_tuple_long(ty, &fType, &fTagnum, &fTypenum, &fReftype, &fIsconst);
}

//______________________________________________________________________________
Cint::G__TypeInfo::G__TypeInfo(const G__TypeInfo& rhs) : G__ClassInfo(rhs), fType(rhs.fType), fTypenum(rhs.fTypenum), fReftype(rhs.fReftype), fIsconst(rhs.fIsconst)
{
}

//______________________________________________________________________________
G__TypeInfo& Cint::G__TypeInfo::operator=(const G__TypeInfo& rhs)
{
   if (this != &rhs) {
      fType = rhs.fType;
      fTypenum = rhs.fTypenum;
      fReftype = rhs.fReftype;
      fIsconst = rhs.fIsconst;
   }
   return *this;
}

//______________________________________________________________________________
int Cint::G__TypeInfo::operator==(const G__TypeInfo& a)
{
   if ((fType == a.fType) && (fTagnum == a.fTagnum) && (fTypenum == a.fTypenum) && (fReftype == a.fReftype)) {
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
int Cint::G__TypeInfo::operator!=(const G__TypeInfo& a)
{
   return !this->operator==(a);
}

//______________________________________________________________________________
const char* Cint::G__TypeInfo::Name()
{
   strcpy(G__buf, G__type2string((int) fType, (int) fTagnum, (int) fTypenum, (int) fReftype, (int) fIsconst));
   return G__buf;
}

//______________________________________________________________________________
const char* Cint::G__TypeInfo::TrueName()
{
   strcpy(G__buf, G__type2string((int) fType, (int) fTagnum, -1, (int) fReftype, (int) fIsconst));
   return G__buf;
}

//______________________________________________________________________________
int Cint::G__TypeInfo::Size() const
{
   G__value buf;
   ::Reflex::Type ty = G__cint5_tuple_to_type((int) fType, (int) fTagnum, (int) fTypenum, (int) fReftype, 0);
   G__value_typenum(buf) = ty;
   buf.ref = fReftype;
   if (isupper(fType)) {
      return sizeof(void*);
   }
   return G__sizeof(&buf);
}

//______________________________________________________________________________
long Cint::G__TypeInfo::Property()
{
   long property = 0L;
   if (fTypenum != -1L) {
      property |= G__BIT_ISTYPEDEF;
   }
   if (fTagnum == -1L) {
      property |= G__BIT_ISFUNDAMENTAL;
   }
   else {
      std::string cname = G__Dict::GetDict().GetScope(fTagnum).Name();
      if (
         (cname == "G__longlong") ||
         (cname == "G__ulonglong") ||
         (cname == "G__longdouble")
      ) {
         property |= G__BIT_ISFUNDAMENTAL;
         if (fTypenum != -1L) {
            std::string tname = G__Dict::GetDict().GetTypedef(fTypenum).Name();
            if (
               (tname == "long long") ||
               (tname == "unsigned long long") ||
               (tname == "long double")
            ) {
               property &= (~G__BIT_ISTYPEDEF);
            }
         }
      }
      else {
         if (G__ClassInfo::IsValid()) {
            property |= G__ClassInfo::Property();
         }
      }
   }
   if (isupper((int) fType)) {
      property |= G__BIT_ISPOINTER;
   }
   if ((fReftype == G__PARAREFERENCE) || (fReftype > G__PARAREF)) {
      property |= G__BIT_ISREFERENCE;
   }
   if (fIsconst & G__CONSTVAR) {
      property |= G__BIT_ISCONSTANT;
   }
   if (fIsconst & G__PCONSTVAR) {
      property |= G__BIT_ISPCONSTANT;
   }
   return property;
}

//______________________________________________________________________________
int Cint::G__TypeInfo::IsValid()
{
   if (G__ClassInfo::IsValid()) {
      return 1;
   }
   else if (fType) {
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
void* Cint::G__TypeInfo::New()
{
   if (G__ClassInfo::IsValid()) {
      return G__ClassInfo::New();
   }
   size_t size = Size();
   void* p = new char[size];
   return p;
}

//______________________________________________________________________________
int Cint::G__TypeInfo::Typenum() const
{
   return fTypenum;
}

//______________________________________________________________________________
int Cint::G__TypeInfo::Type() const
{
   return fType;
}

//______________________________________________________________________________
int Cint::G__TypeInfo::Reftype() const
{
   return fReftype;
}

//______________________________________________________________________________
int Cint::G__TypeInfo::Isconst() const
{
   return fIsconst;
}

#if 0
//______________________________________________________________________________
Reflex::Type Cint::G__TypeInfo::ReflexType() const
{
   return fTypenum;
}
#endif // 0

//______________________________________________________________________________
G__value Cint::G__TypeInfo::Value() const
{
   G__value buf;
   ::Reflex::Type ty = G__cint5_tuple_to_type((int) fType, (int) fTagnum, (int) fTypenum, (int) fReftype, (int) fIsconst);
   G__value_typenum(buf) = ty;
   buf.obj.i = 1;
   buf.ref = 0;
   return buf;
}

//______________________________________________________________________________
int Cint::G__TypeInfo::Next()
{
   return 0;
}

