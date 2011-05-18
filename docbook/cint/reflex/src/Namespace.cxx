// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
# define REFLEX_BUILD
#endif

#include "Namespace.h"
#include "Reflex/internal/OwnedMember.h"
#include "Reflex/DictionaryGenerator.h"


//-------------------------------------------------------------------------------
Reflex::Namespace::Namespace(const char* scop)
//-------------------------------------------------------------------------------
   : ScopeBase(scop, NAMESPACE),
   fPropertyList(OwnedPropertyList(new PropertyListImpl())) {
   // Create dictionary info for a namespace scope.
}


//-------------------------------------------------------------------------------
Reflex::Namespace::Namespace()
//-------------------------------------------------------------------------------
   : ScopeBase(),
   fPropertyList(OwnedPropertyList(new PropertyListImpl())) {
   // Default Constructore (for the global namespace)
}


//-------------------------------------------------------------------------------
Reflex::Namespace::~Namespace() {
//-------------------------------------------------------------------------------
// Default destructor
   fPropertyList.Delete();
}


//-------------------------------------------------------------------------------
const Reflex::Scope&
Reflex::Namespace::GlobalScope() {
//-------------------------------------------------------------------------------
// Initialise the global namespace at startup.
   static Scope s = (new Namespace())->ThisScope();
   return s;
}


//-------------------------------------------------------------------------------
void
Reflex::Namespace::GenerateDict(DictionaryGenerator& generator) const {
//-------------------------------------------------------------------------------
// Generate Dictionary information about itself.

   if ((*this).Name() != "" && generator.IsNewType((*this))) {
      std::stringstream tempcounter;
      tempcounter << generator.fMethodCounter;

      generator.fStr_namespaces << "NamespaceBuilder nsb" + tempcounter.str() +
      " (\"" << (*this).Name(SCOPED) << "\");\n";

      ++generator.fMethodCounter;
   }

   for (Member_Iterator mi = (*this).Member_Begin(); mi != (*this).Member_End(); ++mi) {
      (*mi).GenerateDict(generator);    // call Members' own gendict
   }

   this->ScopeBase::GenerateDict(generator);


} // GenerateDict
