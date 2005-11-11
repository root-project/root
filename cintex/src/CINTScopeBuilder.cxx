// @(#)root/reflex:$Name:  $:$Id: CINTScopeBuilder.cxx,v 1.2 2005/11/03 15:29:47 roiser Exp $
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

  void CINTScopeBuilder::Setup(const Scope& ScopeNth) {
    if ( ScopeNth ) {
      if ( ScopeNth.IsTopScope() ) return;
      Setup( ScopeNth.DeclaringScope() );
    }
    else {
      if ( ScopeNth.Name() == "" ) return;
      Scope dcl_scope = Scope::ByName(Tools::GetScopeName(ScopeNth.Name(SCOPED)));
      if( dcl_scope.Id() ) Setup(dcl_scope);
    }
    string sname = CintName(ScopeNth.Name(SCOPED));
    G__linked_taginfo taginfo;
    taginfo.tagnum  = -1;   // >> need to be pre-initialized to be understood by CINT
    if (ScopeNth.IsNamespace() )  taginfo.tagtype = 'n';
    else if (ScopeNth.IsClass() ) taginfo.tagtype = 'c';
    else                       taginfo.tagtype = 'u'; // Undefined!!!
    taginfo.tagname = sname.c_str();
    taginfo.tagnum = G__defined_tagname(taginfo.tagname, 2);
    G__ClassInfo info(taginfo.tagnum);
    if ( !info.IsLoaded() )  {
      G__get_linked_tagnum(&taginfo);
      //--Setup the namespace---
      if ( ScopeNth.IsClass() )  {                  //--Setup the class ScopeNth
        CINTClassBuilder::Get(Type::ByName(sname));
      }
      else {
        G__tagtable_setup( taginfo.tagnum,       // tag number
                           0,                    // size
                           G__CPPLINK,           // cpplink
                           9600,                 // isabstract
                           0,                    // comment
                           0,                    // Variable Setup func
                           0);                   // Function Setup func
        //-- Create a TClass Instance to please PyROOT adnd ROOT that also wats to have
        //   TClass for namespaces
        ROOTClassEnhancer::CreateClassForNamespace(sname);
      }
    }
    return;
  }

  void CINTScopeBuilder::Setup(const Type& TypeNth) {
    if ( TypeNth.IsFunction() ) {
      Setup(TypeNth.ReturnType());
      for ( size_t i = 0; i < TypeNth.FunctionParameterSize(); i++ ) Setup(TypeNth.FunctionParameterAt(i));
    }
    else if ( TypeNth.IsTypedef() ) {
      CINTTypedefBuilder::Setup(TypeNth);
      Setup(TypeNth.ToType());
    }
    else if ( TypeNth.IsEnum() ) {
      CINTEnumBuilder::Setup(TypeNth);
      Setup(TypeNth.DeclaringScope());
    }
    else {
      Scope ScopeNth = TypeNth.DeclaringScope();
      if ( ScopeNth ) Setup(ScopeNth);
      else {
        // Type not yet defined. Get the ScopeNth anyway...
        ScopeNth = Scope::ByName(Tools::GetScopeName(TypeNth.Name(SCOPED)));
        if( ScopeNth.Id() ) Setup(ScopeNth);
      }
    }
  }

}}
