// @(#)root/reflex:$Name:  $:$Id: MemberTemplateImpl.cxx,v 1.8 2006/03/06 12:51:46 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#define REFLEX_BUILD

#include "Reflex/MemberTemplateImpl.h"
#include "Reflex/Member.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplateImpl::MemberTemplateImpl( const std::string & templateName,
                                                      const Scope & scope,
                                                      std::vector < std::string > parameterNames,
                                                      std::vector < std::string > parameterDefaults )
//------------------------------------------------------------------------------- 
   : fTemplateName( templateName ),
     fScope( scope ),
     fTemplateInstances( std::vector < Member >() ),
     fParameterNames( parameterNames ),
     fParameterDefaults( parameterDefaults ),
     fReqParameters( parameterNames.size() - parameterDefaults.size() ) {}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplateImpl::~MemberTemplateImpl() {}
//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
bool ROOT::Reflex::MemberTemplateImpl::operator == ( const MemberTemplateImpl & mt ) const {
//-------------------------------------------------------------------------------
   return ( ( fTemplateName == mt.fTemplateName ) && 
            ( fParameterNames.size() == mt.fParameterNames.size() ) );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::MemberTemplateImpl::TemplateInstanceAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( nth < fTemplateInstances.size() ) return Member(fTemplateInstances[ nth ]);
   return Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::MemberTemplateImpl::TemplateInstanceSize() const {
//-------------------------------------------------------------------------------
   return fTemplateInstances.size();
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::MemberTemplateImpl::AddTemplateInstance( const Member & templateInstance ) const {
//-------------------------------------------------------------------------------
   fTemplateInstances.push_back( templateInstance );
}

