// @(#)root/reflex:$Name:  $:$Id: TypeTemplateImpl.cxx,v 1.9 2006/07/04 15:02:55 roiser Exp $
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

#include "Reflex/internal/TypeTemplateImpl.h"

#include "Reflex/Type.h"
#include "Reflex/Member.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplateImpl::TypeTemplateImpl( const std::string & templateName,
                                                  const Scope & scop,
                                                  std::vector < std::string > parameterNames,
                                                  std::vector < std::string > parameterDefaults )
//------------------------------------------------------------------------------- 
   : fTemplateName( templateName ),
     fScope( scop ),
     fTemplateInstances( std::vector < Type >() ),
     fParameterNames( parameterNames ),
     fParameterDefaults( parameterDefaults ),
     fReqParameters( parameterNames.size() - parameterDefaults.size() ) {
   // Construct the type template family info.
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplateImpl::~TypeTemplateImpl() {
//-------------------------------------------------------------------------------
// Destructor.
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Type & ROOT::Reflex::TypeTemplateImpl::TemplateInstanceAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth template instance of this family.
   if ( nth < fTemplateInstances.size() ) return fTemplateInstances[ nth ];
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::TypeTemplateImpl::TemplateInstanceSize() const {
//-------------------------------------------------------------------------------
// Return the number of template instances of this family.
   return fTemplateInstances.size();
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::TypeTemplateImpl::AddTemplateInstance( const Type & templateInstance ) const {
//-------------------------------------------------------------------------------
// Add template instance to this family.
   fTemplateInstances.push_back( templateInstance );
}
