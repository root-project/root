/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file Typedf.cxx
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
#include "Dict.h"
#include <algorithm>

using namespace Internal;
using namespace std;

//______________________________________________________________________________
Cint::G__TypedefInfo::~G__TypedefInfo()
{
}

//______________________________________________________________________________
Cint::G__TypedefInfo::G__TypedefInfo()
{
   Init();
}

//______________________________________________________________________________
void Cint::G__TypedefInfo::Init()
{
   fType = 0;
   fTypenum = -1;
   fTagnum = -1;
   fIsconst = 0;
}

//______________________________________________________________________________
Cint::G__TypedefInfo::G__TypedefInfo(const char* typenamein)
{
   Init(typenamein);
}

//______________________________________________________________________________
void Cint::G__TypedefInfo::Init(const char* typenamein)
{
   char store_var_type = G__var_type;
   ::Reflex::Type ty = G__find_typedef(typenamein);
   if (ty) {
      G__get_cint5_type_tuple_long(ty, &fType, &fTagnum, &fTypenum, &fReftype, &fIsconst);
      fIsconst = 0;
   }
   else {
      fType = 0;
      fTagnum = -1;
      fTypenum = -1;
      fIsconst = 0;
   }
   G__var_type = store_var_type;
}

//______________________________________________________________________________
Cint::G__TypedefInfo::G__TypedefInfo(int typenumin)
{
   Init(typenumin);
}

//______________________________________________________________________________
void Cint::G__TypedefInfo::Init(int typenumin)
{
   fTypenum = typenumin;
   ::Reflex::Type ty = G__Dict::GetDict().GetTypedef(fTypenum);
   if (ty) {
      long junk;
      G__get_cint5_type_tuple_long(ty, &fType, &fTagnum, &junk, &fReftype, &fIsconst);
      fIsconst = 0;
   }
   else {
      fType = 0;
      fTagnum = -1;
      fTypenum = -1;
      fIsconst = 0;
   }
}

#if 0
//______________________________________________________________________________
Cint::G__TypedefInfo::G__TypedefInfo(const ::Reflex::Type typein)
{
   Init(typein);
}
#endif // 0

#if 0
//______________________________________________________________________________
void Cint::G__TypedefInfo::Init(const ::Reflex::Type typein)
{
   fTypenum = typein;
   typeiter = -1;
   if (fTypenum) {
      fTagnum =  G__get_tagnum(fTypenum);
   }
   else {
      fTagnum = -1;
   }
}
#endif // 0

//______________________________________________________________________________
Cint::G__ClassInfo Cint::G__TypedefInfo::EnclosingClassOfTypedef()
{
   if (IsValid()) {
      G__ClassInfo enclosingclass(G__get_tagnum(G__Dict::GetDict().GetTypedef(fTypenum).DeclaringScope()));
      return enclosingclass;
   }
   G__ClassInfo enclosingclass;
   return enclosingclass;
}

//______________________________________________________________________________
const char* Cint::G__TypedefInfo::Title()
{
   static char buf[G__INFO_TITLELEN];
   buf[0] = '\0';
   if (!IsValid()) {
      return 0;
   }
   ::Reflex::Type ty = G__Dict::GetDict().GetTypedef(fTypenum);
   G__getcommenttypedef(buf, &G__get_properties(ty)->comment, ty);
   return buf;
}

//______________________________________________________________________________
void Cint::G__TypedefInfo::SetGlobalcomp(int globalcomp)
{
   if (IsValid()) {
      ::Reflex::Type ty = G__Dict::GetDict().GetTypedef(fTypenum);
      G__get_properties(ty)->globalcomp = globalcomp;
   }
}

//______________________________________________________________________________
int Cint::G__TypedefInfo::IsValid()
{
   return (bool) G__Dict::GetDict().GetTypedef(fTypenum);
}

//______________________________________________________________________________
int Cint::G__TypedefInfo::SetFilePos(const char* fname)
{
   struct G__dictposition* dict = G__get_dictpos(const_cast<char*>(fname));
   if (!dict) {
      return 0;
   }
   Init((int) dict->typenum - 1);
   return 1;
}

//______________________________________________________________________________
int Cint::G__TypedefInfo::Next()
{
   Init((int) fTypenum + 1);
   return IsValid();
}

//______________________________________________________________________________
const char* Cint::G__TypedefInfo::FileName()
{
   // --
#ifdef G__TYPEDEFFPOS
   if (IsValid()) {
      ::Reflex::Type ty = G__Dict::GetDict().GetTypedef(fTypenum);
      return G__srcfile[G__get_properties(ty)->filenum].filename;
   }
   return 0;
#else // G__TYPEDEFFPOS
   G__fprinterr("Warning: Cint::G__TypedefInfo::FileName() not supported in this configuration. define G__TYPEDEFFPOS macro in platform dependency file and recompile cint");
   G__printlinenum();
   return 0;
#endif // G__TYPEDEFFPOS
}

//______________________________________________________________________________
int Cint::G__TypedefInfo::LineNumber()
{
   // --
#ifdef G__TYPEDEFFPOS
   if (IsValid()) {
      ::Reflex::Type ty = G__Dict::GetDict().GetTypedef(fTypenum);
      return G__get_properties(ty)->linenum;
   }
   return -1;
#else // G__TYPEDEFFPOS
   G__fprinterr("Warning: Cint::G__TypedefInfo::LineNumber() not supported in this configuration. define G__TYPEDEFFPOS macro in platform dependency file and recompile cint");
   G__printlinenum();
   return -1;
#endif // G__TYPEDEFFPOS
}

