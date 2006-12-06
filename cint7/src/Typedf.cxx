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

/*********************************************************************
* class G__TypedefInfo
* 
*********************************************************************/
///////////////////////////////////////////////////////////////////////////
Cint::G__TypedefInfo::~G__TypedefInfo()
{
}
///////////////////////////////////////////////////////////////////////////
Cint::G__TypedefInfo::G__TypedefInfo()
{
   Init();
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypedefInfo::Init() 
{
   // G__ClassInfo part.
   tagnum = -1;
   // G__TypedefInfo part.
   reflexInfo->type = 0;
   // FIXME: What about reftype?
   reftype = 0;
   isconst = 0;
   reflexInfo->typenum = ::ROOT::Reflex::Type();
   reflexInfo->typeiter = -1;
}
///////////////////////////////////////////////////////////////////////////
Cint::G__TypedefInfo::G__TypedefInfo(const char* typenamein)
{
   Init(typenamein);
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypedefInfo::Init(const char* typenamein)
{
   char store_var_type = G__var_type;
   reflexInfo->typenum = G__find_typedef(typenamein);
   reflexInfo->typeiter = -1;
   if (reflexInfo->typenum) {
      tagnum = G__get_tagnum(reflexInfo->typenum);
      reflexInfo->type = G__get_type(reflexInfo->typenum);
      reftype = G__get_reftype(reflexInfo->typenum);
      isconst = 0;
   }
   else {
      reflexInfo->type = 0;
      tagnum = -1;
      reflexInfo->typenum = ::ROOT::Reflex::Type();
      // FIXME: What about reftype?
      reftype = 0;
      isconst = 0;
   }
   G__var_type = store_var_type;
}
///////////////////////////////////////////////////////////////////////////
Cint::G__TypedefInfo::G__TypedefInfo(int typenumin)
{
   Init(typenumin);
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypedefInfo::Init(int typenumin)
{
   Init(G__Dict::GetDict().GetTypedef(typenumin));
}
///////////////////////////////////////////////////////////////////////////
Cint::G__TypedefInfo::G__TypedefInfo(const ::ROOT::Reflex::Type& typein)
{
   Init(typein);
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypedefInfo::Init(const ::ROOT::Reflex::Type& typein)
{
   reflexInfo->typenum = typein;
   reflexInfo->typeiter = -1;
   if (reflexInfo->typenum) {
      tagnum = G__get_tagnum(reflexInfo->typenum);
      reflexInfo->type = G__get_type(reflexInfo->typenum);
      reftype = G__get_reftype(reflexInfo->typenum);
      isconst = 0;
   }
   else {
      reflexInfo->type = 0;
      tagnum = -1;
      // FIXME: What about reftype?
      reftype = 0;
      isconst = 0;
   }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypedefInfo::SetGlobalcomp(int globalcomp)
{
   if (IsValid() && G__get_properties(reflexInfo->typenum)) {
      G__get_properties(reflexInfo->typenum)->globalcomp = globalcomp;
   }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypedefInfo::IsValid()
{
   return (bool) reflexInfo->typenum;
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypedefInfo::SetFilePos(const char* fname)
{
   struct G__dictposition* dict = G__get_dictpos((char*) fname);
   if (!dict) {
      return 0;
   }
   Init(G__Dict::GetDict().GetTypedef(dict->typenum));
   return 1;
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypedefInfo::Next()
{
   // When the iterator reaches the max it must stay there.
   int max_index = (int) ROOT::Reflex::Type::TypeSize();
   if (reflexInfo->typeiter != max_index) {
      ++reflexInfo->typeiter;
   }
   if (reflexInfo->typeiter != max_index) {
      Reflex::Type ty;
      do {
         ty = Reflex::Type::TypeAt(reflexInfo->typeiter);
      } while (!ty && (++reflexInfo->typeiter < max_index));
      if (reflexInfo->typeiter == max_index) {
         Init(ROOT::Reflex::Type());
         reflexInfo->typeiter = max_index;
      } else { 
         int saved_typeiter = reflexInfo->typeiter;
         Init(ty);
         reflexInfo->typeiter = saved_typeiter;
      }
   } else {
      Init(ROOT::Reflex::Type());
      reflexInfo->typeiter = max_index;
   }
   return IsValid();
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__TypedefInfo::Title()
{
   static char buf[G__INFO_TITLELEN];
   buf[0] = '\0';
   if (IsValid()) {
#ifdef __GNUC__
#else
#pragma message(FIXME("G__getcommenttypedef(buf,&G__newtype.comment[typenum],(int)typenum);"))
#endif
      strcat(buf, "Gotta fix this in cint/Typedf.cxx");
      return buf;
   }
   return 0;
}
///////////////////////////////////////////////////////////////////////////
Cint::G__ClassInfo Cint::G__TypedefInfo::EnclosingClassOfTypedef()
{
   if (IsValid()) {
#ifdef __GNUC__
#else
#pragma message(FIXME("// don't use the G__ClassInfo(const char*) here!"))
#endif
      G__ClassInfo enclosingclass(reflexInfo->typenum.DeclaringScope().Name(::ROOT::Reflex::SCOPED).c_str());
      return enclosingclass;
   }
   G__ClassInfo enclosingclass;
   return enclosingclass;
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__TypedefInfo::FileName()
{
#ifdef G__TYPEDEFFPOS
   if (IsValid() && G__get_properties(reflexInfo->typenum)) {
      return G__srcfile[G__get_properties(reflexInfo->typenum)->filenum].filename;
   }
   return 0;
#else
   G__fprinterr("Warning: Cint::G__TypedefInfo::FileName() not supported in this configuration. define G__TYPEDEFFPOS macro in platform dependency file and recompile cint");
   G__printlinenum();
   return 0;
#endif
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypedefInfo::LineNumber()
{
#ifdef G__TYPEDEFFPOS
   if (IsValid() && G__get_properties(reflexInfo->typenum)) {
      return G__get_properties(reflexInfo->typenum)->linenum;
   }
   return -1;
#else
   G__fprinterr("Warning: Cint::G__TypedefInfo::LineNumber() not supported in this configuration. define G__TYPEDEFFPOS macro in platform dependency file and recompile cint");
   G__printlinenum();
   return -1;
#endif
}
///////////////////////////////////////////////////////////////////////////
