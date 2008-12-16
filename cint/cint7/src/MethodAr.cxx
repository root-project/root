/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file MethodAr.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Author                  Masaharu Goto
 * Copyright(c) 1995~1999  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "common.h"
#include "fproto.h"

using namespace Cint::Internal;
using namespace std;

//______________________________________________________________________________
Cint::G__MethodArgInfo::~G__MethodArgInfo()
{
}

//______________________________________________________________________________
Cint::G__MethodArgInfo::G__MethodArgInfo() : argn(0), belongingmethod(0), type()
{
}

//______________________________________________________________________________
Cint::G__MethodArgInfo::G__MethodArgInfo(G__MethodInfo& a) : argn(0), belongingmethod(0), type()
{
   Init(a);
}

//______________________________________________________________________________
void Cint::G__MethodArgInfo::Init(G__MethodInfo &a)
{
   belongingmethod = 0;
   if (a.IsValid()) {
      belongingmethod = &a;
      argn = -1;
   }
}

//______________________________________________________________________________
Cint::G__MethodArgInfo::G__MethodArgInfo(const G__MethodArgInfo& mai) : argn(mai.argn), belongingmethod(mai.belongingmethod), type(mai.type)
{
}

//______________________________________________________________________________
G__MethodArgInfo& Cint::G__MethodArgInfo::operator=(const G__MethodArgInfo& mai)
{
   if (this != &mai) {
      argn = mai.argn;
      belongingmethod = mai.belongingmethod;
      type = mai.type;
   }
   return *this;
}

//______________________________________________________________________________
const char* Cint::G__MethodArgInfo::Name()
{
   if (!IsValid()) {
      return 0;
   }
   // TODO: Remove use of friendship.
   
   // Note: We need a static buffer because the string must continue
   //       to exist after we exit and FunctionParameterNameAt returns
   //       a std::string by value.     
   static std::string static_buf;
   static_buf = belongingmethod->fFunc.FunctionParameterNameAt(argn);
   return static_buf.c_str();
}

//______________________________________________________________________________
G__TypeInfo* Cint::G__MethodArgInfo::Type()
{
   return &type;
}

//______________________________________________________________________________
long Cint::G__MethodArgInfo::Property()
{
   if (!IsValid()) {
      return 0L;
   }
   long result = 0L;
   // TODO: Remove use of friendship.
   ::Reflex::Member func = belongingmethod->fFunc;
   ::Reflex::Type param_type = func.TypeOf().FunctionParameterAt(argn);
   if (isupper(G__get_type(param_type))) {
      result |= G__BIT_ISPOINTER;
   }
   if (func.FunctionParameterDefaultAt(argn).size()) {
      result |= G__BIT_ISDEFAULT;
   }
   if (param_type.FinalType().IsReference()) {
      result |= G__BIT_ISREFERENCE;
   }
   if (G__get_isconst(param_type) & G__CONSTVAR) {
      result |= G__BIT_ISCONSTANT;
   }
   if (G__get_isconst(param_type) & G__PCONSTVAR) {
      result |= G__BIT_ISPCONSTANT;
   }
   return result;
}

//______________________________________________________________________________
const char* Cint::G__MethodArgInfo::DefaultValue()
{
   if (!IsValid()) {
      return 0;
   }
   // TODO: Remove use of friendship.
   // Note: We need a static buffer because the string must continue
   //       to exist after we exit and FunctionParameterDefaultAt returns
   //       a std::string by value.     
   static std::string static_buf;
   static_buf = belongingmethod->fFunc.FunctionParameterDefaultAt(argn);
   return static_buf.c_str();
}

//______________________________________________________________________________
G__MethodInfo* Cint::G__MethodArgInfo::ArgOf()
{
   return belongingmethod;
}

//______________________________________________________________________________
int Cint::G__MethodArgInfo::IsValid()
{
   if (belongingmethod && belongingmethod->IsValid()) {
      if ((argn > -1) && (argn < belongingmethod->NArg())) {
         return 1;
      }
   }
   return 0;
}

//______________________________________________________________________________
int Cint::G__MethodArgInfo::Next()
{
   ++argn;
   if (!IsValid()) {
      return 0;
   }
   // TODO: Remove use of friendship.
   ::Reflex::Type ty = belongingmethod->fFunc.TypeOf().FunctionParameterAt(argn);
   G__get_cint5_type_tuple_long(ty, &type.fType, &type.fTagnum, &type.fTypenum, &type.fReftype, &type.fIsconst);
   type.fClassProperty = 0;
   return 1;
}

