// @(#)root/reflex:$Name: HEAD $:$Id: MemberTemplateImpl.cxx,v 1.11 2006/07/04 15:02:55 roiser Exp $
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

#include "Reflex/MemberTemplateImpl.h"
#include "Reflex/Member.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplateImpl::MemberTemplateImpl( const std::string & templateName,
                                                      const Scope & scope,
                                                      std::vector < std::string > parameterNames,
                                                      std::vector < std::string > parameterDefaults )
//------------------------------------------------------------------------------- 
// Construct dictionary info for this template member function.
   : fTemplateName( templateName ),
     fScope( scope ),
     fTemplateInstances( std::vector < Member >() ),
     fParameterNames( parameterNames ),
     fParameterDefaults( parameterDefaults ),
     fReqParameters( parameterNames.size() - parameterDefaults.size() ) {}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplateImpl::~MemberTemplateImpl() {
//-------------------------------------------------------------------------------
// Destructor.
}

//-------------------------------------------------------------------------------
bool ROOT::Reflex::MemberTemplateImpl::operator == ( const MemberTemplateImpl & mt ) const {
//-------------------------------------------------------------------------------
// Equal operator.
   return ( ( fTemplateName == mt.fTemplateName ) && 
            ( fParameterNames.size() == mt.fParameterNames.size() ) );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::MemberTemplateImpl::TemplateInstanceAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth template instance of this template family.
   if ( nth < fTemplateInstances.size() ) return Member(fTemplateInstances[ nth ]);
   return Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::MemberTemplateImpl::TemplateInstanceSize() const {
//-------------------------------------------------------------------------------
// Return number of template instances of this family.
   return fTemplateInstances.size();
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::MemberTemplateImpl::AddTemplateInstance( const Member & templateInstance ) const {
//-------------------------------------------------------------------------------
// Add template instance to this family.
   fTemplateInstances.push_back( templateInstance );
}

