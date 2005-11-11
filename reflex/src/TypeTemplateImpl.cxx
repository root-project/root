// @(#)root/reflex:$Name:  $:$Id: TypeTemplateImpl.cxx,v 1.2 2005/11/03 15:24:40 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/TypeTemplateImpl.h"

#include "Reflex/Type.h"

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
    fReqParameters( parameterNames.size() - parameterDefaults.size() ) {}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplateImpl::~TypeTemplateImpl() {}
//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeTemplateImpl::TemplateInstanceAt( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( nth < fTemplateInstances.size() ) return fTemplateInstances[ nth ];
  return Type();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::TypeTemplateImpl::TemplateInstanceSize() const {
//-------------------------------------------------------------------------------
  return fTemplateInstances.size();
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::TypeTemplateImpl::AddTemplateInstance( const Type & templateInstance ) const {
//-------------------------------------------------------------------------------
  fTemplateInstances.push_back( templateInstance );
}
