// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Reflex.h"
#include "Reflex/Tools.h"
#include "CINTdefs.h"
#include "CINTScopeBuilder.h"
#include "CINTClassBuilder.h"
#include "CINTTypedefBuilder.h"
#include "CINTEnumBuilder.h"
#include "CINTFunctional.h"
#include "ROOTClassEnhancer.h"
#include "Api.h"


using namespace ROOT::Reflex;
using namespace std;

namespace ROOT { namespace Cintex {

   void CINTScopeBuilder::Setup(const Scope& scope) {
      if ( scope ) {
         if (scope.IsTopScope() ) return;
         Setup( scope.DeclaringScope() );
      }
      else {
         if ( scope.Name() == "" ) return;
         Scope dcl_scope = Scope::ByName(Tools::GetScopeName(scope.Name(SCOPED)));
         if( dcl_scope.Id() ) Setup(dcl_scope);
      }
      string sname = CintName(scope.Name(SCOPED));
      G__linked_taginfo taginfo;
      taginfo.tagnum  = -1;   // >> need to be pre-initialized to be understood by CINT
      if (scope.IsNamespace() )  taginfo.tagtype = 'n';
      else if (scope.IsClass() ) taginfo.tagtype = 'c';
      else  {
         if ( sname.find('<') != string::npos )
            taginfo.tagtype = 'c'; // Is a templated class
         else
            taginfo.tagtype = 'a'; // Undefined. Do not assume namespace
      }
      taginfo.tagname = sname.c_str();
      int tagnum = G__defined_tagname(taginfo.tagname, 2);
      G__ClassInfo info(tagnum);
      if ( !info.IsLoaded() )  {
         G__get_linked_tagnum(&taginfo);
         //--Setup the namespace---
         if ( scope.IsClass() )  {                  //--Setup the class scope
            CINTClassBuilder::Get(Type::ByName(sname));
         }
         else if (taginfo.tagtype == 'n' ) {
            G__tagtable_setup( taginfo.tagnum,       // tag number
                               0,                    // size
                               G__CPPLINK,           // cpplink
                               9600,                 // isabstract
                               0,                    // comment
                               0,                    // Variable Setup func
                               0);                   // Function Setup func
            //-- Create a TClass Instance to please PyROOT adnd ROOT that also wats to have
            //   TClass for namespaces
            if (scope) ROOTClassEnhancer::CreateClassForNamespace(sname);
         }
         else {
            //--Tag_table not possible to be setup at this moment....
         }
      }
      return;
   }

   void CINTScopeBuilder::Setup(const Type& type) {
      if ( type.IsFunction() ) {
         Setup(type.ReturnType());
         for ( size_t i = 0; i < type.FunctionParameterSize(); i++ ) Setup(type.FunctionParameterAt(i));
      }
      else if ( type.IsTypedef() ) {
         CINTTypedefBuilder::Setup(type);
         Setup(type.ToType());
      }
      else if ( type.IsEnum() ) {
         CINTEnumBuilder::Setup(type);
         Setup(type.DeclaringScope());
      }
      else {
         Scope scope = type.DeclaringScope();
         if ( scope ) Setup(scope);
         else {
            // Type not yet defined. Get the ScopeNth anyway...
            scope = Scope::ByName(Tools::GetScopeName(type.Name(SCOPED)));
            if( scope.Id() ) Setup(scope);
         }
      }
   }

}}
