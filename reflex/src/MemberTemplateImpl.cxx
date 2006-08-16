// @(#)root/reflex:$Name:  $:$Id: MemberTemplateImpl.cxx,v 1.14 2006/08/11 06:31:59 roiser Exp $
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

#include "Reflex/internal/MemberTemplateImpl.h"

#include "Reflex/MemberTemplate.h"
#include "Reflex/internal/OwnedMember.h"


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplateImpl::MemberTemplateImpl( const char * templateName,
                                                      const Scope & scope,
                                                      const std::vector < std::string > & parameterNames,
                                                      const std::vector < std::string > & parameterDefaults )
//------------------------------------------------------------------------------- 
   : fScope( scope ),
     fTemplateInstances( std::vector < Member >() ),
     fParameterNames( parameterNames ),
     fParameterDefaults( parameterDefaults ),
     fReqParameters( parameterNames.size() - parameterDefaults.size() ) {
// Construct dictionary info for this template member function.
   MemberTemplate mt = MemberTemplate::ByName( templateName, parameterNames.size() );
   if ( mt.Id() == 0 ) {
      fMemberTemplateName = new MemberTemplateName( templateName, this );
   }
   else {
      fMemberTemplateName = (MemberTemplateName*)mt.Id();
      if ( fMemberTemplateName->fMemberTemplateImpl ) delete fMemberTemplateName->fMemberTemplateImpl;
      fMemberTemplateName->fMemberTemplateImpl = this;
   }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplateImpl::~MemberTemplateImpl() {
//-------------------------------------------------------------------------------
  if ( fMemberTemplateName->fMemberTemplateImpl == this ) fMemberTemplateName->fMemberTemplateImpl = 0;
// Destructor.
}


//-------------------------------------------------------------------------------
bool ROOT::Reflex::MemberTemplateImpl::operator == ( const MemberTemplateImpl & mt ) const {
//-------------------------------------------------------------------------------
// Equal operator.
   return ( ( fMemberTemplateName->fName == mt.fMemberTemplateName->fName ) && 
            ( fParameterNames.size() == mt.fParameterNames.size() ) );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member_Iterator ROOT::Reflex::MemberTemplateImpl::TemplateInstance_Begin() const {
//-------------------------------------------------------------------------------
   // Return the begin iterator of the instance container of this member template.
   return fTemplateInstances.begin();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Member_Iterator ROOT::Reflex::MemberTemplateImpl::TemplateInstance_End() const {
//-------------------------------------------------------------------------------
   // Return the end iterator of the instance container of this member template.
   return fTemplateInstances.end();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::MemberTemplateImpl::TemplateInstance_RBegin() const {
//-------------------------------------------------------------------------------
   // Return the rbegin iterator of the instance container of this member template.
   return fTemplateInstances.rbegin();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::MemberTemplateImpl::TemplateInstance_REnd() const {
//-------------------------------------------------------------------------------
   // Return the rend iterator of the instance container of this member template.
   return fTemplateInstances.rend();
}

                                             
//-------------------------------------------------------------------------------
const ROOT::Reflex::Member & ROOT::Reflex::MemberTemplateImpl::TemplateInstanceAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth template instance of this template family.
   if ( nth < fTemplateInstances.size() ) return fTemplateInstances[ nth ];
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::MemberTemplateImpl::TemplateInstanceSize() const {
//-------------------------------------------------------------------------------
// Return number of template instances of this family.
   return fTemplateInstances.size();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::MemberTemplate & ROOT::Reflex::MemberTemplateImpl::ThisMemberTemplate() const {
//-------------------------------------------------------------------------------
   // Return a ref to this member template.
   return fMemberTemplateName->ThisMemberTemplate();
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::MemberTemplateImpl::AddTemplateInstance( const Member & templateInstance ) const {
//-------------------------------------------------------------------------------
// Add template instance to this family.
   fTemplateInstances.push_back( templateInstance );
}

