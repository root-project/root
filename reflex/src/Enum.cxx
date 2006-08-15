// @(#)root/reflex:$Name:  $:$Id: Enum.cxx,v 1.9 2006/08/01 09:14:33 roiser Exp $
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

#include "Enum.h"

#include "Reflex/Tools.h"
#include "Reflex/DictionaryGenerator.h"

#include <sstream>


//-------------------------------------------------------------------------------
ROOT::Reflex::Enum::Enum( const char * enumType,
                          const std::type_info & ti,
                          unsigned int modifiers )
//-------------------------------------------------------------------------------
// Construct the dictionary information for an enum
   : TypeBase( enumType, sizeof(int), ENUM, ti ),
     ScopeBase( enumType, ENUM ),
     fModifiers( modifiers ) {}


//-------------------------------------------------------------------------------
ROOT::Reflex::Enum::~Enum() {
//-------------------------------------------------------------------------------
// Destructor for enum dictionary information.
}




//-------------------------------------------------------------------------------
void ROOT::Reflex::Enum::GenerateDict( DictionaryGenerator & generator ) const {
//-------------------------------------------------------------------------------
// Generate Dictionary information about itself.
         
   size_t lastMember = DataMemberSize()-1;

   if( !(DeclaringScope().IsNamespace()) ) {  

      generator.AddIntoFree("\n.AddEnum(\"" + Name() + "\", \"");

      for ( size_t i = 0; i < DataMemberSize(); ++i ) {
         DataMemberAt(i).GenerateDict(generator);
         if ( i < lastMember ) generator.AddIntoFree(";");
      }

      generator.AddIntoFree("\",");
      if      ( IsPublic())    generator.AddIntoFree("typeid(" + Name(SCOPED) + "), PUBLIC)");
      else if ( IsProtected()) generator.AddIntoFree("typeid(ROOT::Reflex::ProtectedEnum), PROTECTED)");
      else if ( IsPrivate())   generator.AddIntoFree("typeid(ROOT::Reflex::PrivateEnum), PRIVATE)");
   }
   else {

      generator.AddIntoInstances("      EnumBuilder(\"" + Name(SCOPED) + "\", typeid(" + Name(SCOPED) + "), PUBLIC)");
      for ( size_t i = 0; i < DataMemberSize(); ++i ) DataMemberAt(i).GenerateDict(generator);
      generator.AddIntoInstances(";\n");

   }   
}

