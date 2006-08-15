// @(#)root/reflex:$Name:  $:$Id: Namespace.cxx,v 1.10 2006/08/11 06:31:59 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
#define REFLEX_BUILD
#endif

#include "Namespace.h"
#include "Reflex/internal/OwnedMember.h"
#include "Reflex/DictionaryGenerator.h"


//-------------------------------------------------------------------------------
ROOT::Reflex::Namespace::Namespace( const char * scop ) 
//-------------------------------------------------------------------------------
   : ScopeBase( scop, NAMESPACE ) {
   // Create dictionary info for a namespace scope.
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Namespace::Namespace() 
//-------------------------------------------------------------------------------
   : ScopeBase() {
   // Destructor.
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Scope & ROOT::Reflex::Namespace::GlobalScope() {
//-------------------------------------------------------------------------------
// Initialise the global namespace at startup.
   static Scope s = (new Namespace())->ThisScope();
   return s;
}




//-------------------------------------------------------------------------------
void ROOT::Reflex::Namespace::GenerateDict( DictionaryGenerator & generator ) const {
//-------------------------------------------------------------------------------
// Generate Dictionary information about itself.

   
 
   if( (*this).Name()!="" && generator.IsNewType((*this)) )
      {
         std::stringstream tempcounter;
         tempcounter << generator.fMethodCounter;
         
         generator.fStr_namespaces<<"NamespaceBuilder nsb" + tempcounter.str() + 
            " (\"" << (*this).Name(SCOPED) << "\");\n" ;
         
         ++generator.fMethodCounter;
      }
      
   
   for (Member_Iterator mi = (*this).Member_Begin(); mi != (*this).Member_End(); ++mi) 
      {
         (*mi).GenerateDict(generator); // call Members' own gendict
      }
      
   this->ScopeBase::GenerateDict(generator);
   
   
}

