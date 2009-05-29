// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "CINTScopeBuilder.h"

#include "CINTdefs.h"
#include "CINTClassBuilder.h"
#include "CINTTypedefBuilder.h"
#include "CINTEnumBuilder.h"
#include "CINTFunctional.h"

#include "TClass.h"
#include "TGenericClassInfo.h"
#include "Reflex/Reflex.h"
#include "Reflex/Tools.h"
#include "Api.h"


using namespace ROOT::Reflex;
using namespace std;


namespace ROOT {

class TForNamespace {}; // Dummy class to give a typeid to namespace.

namespace Cintex {

//______________________________________________________________________________
void CINTScopeBuilder::Setup(const Scope scope)
{
   if (scope) {
      if (scope.IsTopScope()) {
         return;
      }
      Setup(scope.DeclaringScope());
   }
   else {
      if (scope.Name() == "") {
         return;
      }
      Scope declaring_scope = Scope::ByName(Tools::GetScopeName(scope.Name(SCOPED)));
      if (declaring_scope.Id()) {
         Setup(declaring_scope);
      }
   }
   string sname = CintName(scope.Name(SCOPED));
   G__linked_taginfo taginfo;
   taginfo.tagname = sname.c_str();
   taginfo.tagtype = 'a'; // init to autoload for invalid case
   taginfo.tagnum = -1; // init to invalid
   if (scope.IsNamespace()) {
      taginfo.tagtype = 'n'; // namespace
   }
   else if (scope.IsClass()) {
      taginfo.tagtype = 'c'; // class
   }
   else if (sname.find('<') != string::npos) { // check for class template-id
      taginfo.tagtype = 'c'; // class template-id
   }
   int tagnum = G__defined_tagname(taginfo.tagname, 2); // check if cint knows this scope
   if (tagnum != -1) {
      return;
   }
   G__get_linked_tagnum(&taginfo); // have cint create a tagnum for the new scope
   if (scope.IsClass()) { // scope is a class, use the class builder
      CINTClassBuilder::Get(Type::ByName(sname));
      return;
   }
   if (!scope.IsNamespace()) {
      return;
   }
   //
   //  Since we do not have a namespace
   //  builder, we do the work here.
   //
   //--
   // Do the cint part of the dictionary.
   G__tagtable_setup(
        taginfo.tagnum // tagnum, tag number
      , 0 // size
      , -1 // cpplink
      , 0 // isabstract
      , 0 // comment
      , 0 // setup_memvar, member variable setup func
      , 0 // setup_memfunc, member function setup func
   );
   // Do the root part of the dictionary.
   ROOT::AddClass(
        sname.c_str() // cname, class name
      , 0 // id, version number
      , typeid(ROOT::TForNamespace) // info, type info
      , 0 // dict, dictionary getter
      , 0 // pragmabits
   ); // Add the class to the global class table (TODO: We should have a dictionary function!)
   ROOT::CreateClass(
        sname.c_str() // Name
      , 0 // version
      , typeid(ROOT::TForNamespace) // typeid
      , 0 // TVirtualIsAProxy *isa,
      , 0 // ShowMembersFunc_t show,
      , "" // definition file
      , "" // implementation file
      , 1 // definition line number
      , 1 // implementation line number
   ); // Do what the dictionary getter would have done, create the root class.
   return;
}

//______________________________________________________________________________
void CINTScopeBuilder::Setup(const Type type)
{
   if (type.IsFunction()) {
      Setup(type.ReturnType());
      for (size_t i = 0; i < type.FunctionParameterSize(); ++i) {
         Setup(type.FunctionParameterAt(i));
      }
   }
   else if (type.IsTypedef()) {
      CINTTypedefBuilder::Setup(type);
      Setup(type.ToType());
   }
   else if (type.IsEnum()) {
      CINTEnumBuilder::Setup(type);
      Setup(type.DeclaringScope());
   }
   else {
      Scope scope = type.DeclaringScope();
      if (scope) {
         Setup(scope);
      }
      else {
         // Type not yet defined. Get the Scope anyway.
         scope = Scope::ByName(Tools::GetScopeName(type.Name(SCOPED)));
         if (scope.Id()) {
            Setup(scope);
         }
      }
   }
}

} // namespace Cintex
} // namespace ROOT
