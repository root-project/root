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
   typenum = ::Reflex::Type();
   typeiter = -1;
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
  typenum = G__find_typedef(typenamein);
  typeiter = -1;
  if(typenum) {
    tagnum = G__get_tagnum(typenum);
  }
  else {
    tagnum= -1;
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
Cint::G__TypedefInfo::G__TypedefInfo(const ::Reflex::Type& typein)
{
   Init(typein);
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypedefInfo::Init(const ::Reflex::Type& typein)
{
   typenum = typein;
   typeiter = -1;
   if (typenum) 
      tagnum =  G__get_tagnum(typenum);
   else tagnum= -1;
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__TypedefInfo::SetGlobalcomp(int globalcomp)
{
   if (IsValid() && G__get_properties(typenum)) {
      G__get_properties(typenum)->globalcomp = globalcomp;
   }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TypedefInfo::IsValid()
{
   return (bool) typenum;
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
   int max_index = (int) Reflex::Type::TypeSize();
   if (typeiter != max_index) {
      ++typeiter;
   }
   if (typeiter == max_index) {
      Init(Reflex::Type());
      // Init changed it, make it max again.
      typeiter = max_index;
   } else {
      Reflex::Type ty;
      for (; typeiter < max_index; ++typeiter) {
         ty = Reflex::Type::TypeAt(typeiter);
         if (ty && ty.IsTypedef()) {
            break;
         }
      }
      if (typeiter == max_index) {
         Init(Reflex::Type());
         // Init changed it, make it max again.
         typeiter = max_index;
      } else { 
         int saved_typeiter = typeiter;
         Init(ty);
         typeiter = saved_typeiter;
      }
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
      G__ClassInfo enclosingclass(typenum.DeclaringScope().Name(::Reflex::SCOPED).c_str());
      return enclosingclass;
   }
   G__ClassInfo enclosingclass;
   return enclosingclass;
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__TypedefInfo::FileName()
{
#ifdef G__TYPEDEFFPOS
   if (IsValid() && G__get_properties(typenum)) {
      return G__srcfile[G__get_properties(typenum)->filenum].filename;
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
   if (IsValid() && G__get_properties(typenum)) {
      return G__get_properties(typenum)->linenum;
   }
   return -1;
#else
   G__fprinterr("Warning: Cint::G__TypedefInfo::LineNumber() not supported in this configuration. define G__TYPEDEFFPOS macro in platform dependency file and recompile cint");
   G__printlinenum();
   return -1;
#endif
}
///////////////////////////////////////////////////////////////////////////
