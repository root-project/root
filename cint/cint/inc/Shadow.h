/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file Shadow.h
 ************************************************************************
 * Description:
 *  Shadow class generator for dictionaries
 ************************************************************************
 * Copyright(c) 1995~2007  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*********************************************************************
* Shadow class functions
*********************************************************************/

#ifndef G__SHADOW_H
#define G__SHADOW_H 


#include <string>
#include <ostream>
#ifndef G__API
#include "Api.h"
#endif

using std::string;
using std::ostream;

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable: 4251 ) // fNSPrefix needs to have dll-interface
#endif

namespace Cint {

class
#ifndef __CINT__
G__EXPORT
#endif
G__ShadowMaker {
public:
   static bool NeedShadowClass(G__ClassInfo& cl);
   G__ShadowMaker(std::ostream& out, const char* nsprefix,
      bool(*needShadowClass)(G__ClassInfo &cl)=G__ShadowMaker::NeedShadowClass,
      bool(*needTypedefShadow)(G__ClassInfo &cl)=0);

   void WriteAllShadowClasses();

   void WriteShadowClass(G__ClassInfo &cl, int level = 0);
   int WriteNamespaceHeader(G__ClassInfo &cl);

   char NeedShadowCached(int tagnum);
   
   static bool IsSTLCont(const char *type);
   static bool IsStdPair(G__ClassInfo &cl);

   static void GetFullyQualifiedName(const char *originalName, std::string &fullyQualifiedName);
   static void GetFullyQualifiedName(G__ClassInfo &cl, std::string &fullyQualifiedName);
   static void GetFullyQualifiedName(G__TypeInfo &type, std::string &fullyQualifiedName);
   static std::string GetNonConstTypeName(G__DataMemberInfo &m, bool fullyQualified = false);
   void GetFullShadowName(G__ClassInfo &cl, std::string &fullname);

   static void VetoShadow(bool veto=true);

private:
   G__ShadowMaker(const G__ShadowMaker&); // intentionally not implemented
   G__ShadowMaker& operator =(const G__ShadowMaker&); // intentionally not implemented
   void GetFullShadowNameRecurse(G__ClassInfo &cl, std::string &fullname);
   
   void UpdateCachedNeedShadow();  
#ifndef __CINT__
   std::ostream& fOut; // where to write to
#endif
   std::string fNSPrefix; // shadow classes are in this namespace's namespace "Shadow"
   char fCacheNeedShadow[G__MAXSTRUCT]; // whether we need a shadow for a tagnum
   int fMaxCachedNeedShadow; // Up to where in G__struct did we calculate the value.
   bool (*fNeedShadowClass)(G__ClassInfo &cl); // func deciding whether the shadow is a tyepdef
   bool (*fNeedTypedefShadow)(G__ClassInfo &cl); // func deciding whether the shadow is a tyepdef

   static bool fgVetoShadow; // whether WritaAllShadowClasses should write the shadow
};

} // end namespace Cint

using namespace Cint;
#ifdef _MSC_VER
#pragma warning( pop )
#endif

#endif // G__SHADOW_H
