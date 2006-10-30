// @(#)root/reflex:$Name:  $:$Id: MemberTemplate.cxx,v 1.12 2006/08/16 06:42:35 roiser Exp $
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

#include "Reflex/MemberTemplate.h"
#include "Reflex/internal/OwnedMember.h"
#include "Reflex/internal/MemberTemplateName.h"


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate ROOT::Reflex::MemberTemplate::ByName( const std::string & name,
                                                                   size_t nTemplateParams ) {
//-------------------------------------------------------------------------------
   // Return a member template by name.
   return MemberTemplateName::ByName( name, nTemplateParams );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate ROOT::Reflex::MemberTemplate::MemberTemplateAt( size_t nth ) {
//-------------------------------------------------------------------------------
   // Return the nth member template defined.
   return MemberTemplateName::MemberTemplateAt( nth );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate_Iterator ROOT::Reflex::MemberTemplate::MemberTemplate_Begin() {
//-------------------------------------------------------------------------------
   // Return the begin iterator of the member template container.
   return MemberTemplateName::MemberTemplate_Begin();
}
 
                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate_Iterator ROOT::Reflex::MemberTemplate::MemberTemplate_End() {
//-------------------------------------------------------------------------------
   // Return the end iterator of the member template container.
   return MemberTemplateName::MemberTemplate_End();
}
 
                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::MemberTemplate::MemberTemplate_RBegin() {
//-------------------------------------------------------------------------------
   // Return the rbegin iterator of the member template container.
   return MemberTemplateName::MemberTemplate_RBegin();
}
 
                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::MemberTemplate::MemberTemplate_REnd() {
//-------------------------------------------------------------------------------
   // Return the rend iterator of the member template container.
   return MemberTemplateName::MemberTemplate_REnd();
}
 
                                             
//-------------------------------------------------------------------------------
std::string ROOT::Reflex::MemberTemplate::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
   // Return the name of the member template.
   if ( fMemberTemplateName ) return fMemberTemplateName->Name( mod );
   else                       return "";
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member_Iterator ROOT::Reflex::MemberTemplate::TemplateInstance_Begin() const {
//-------------------------------------------------------------------------------
   // Return the begin iterator of the instance container of this member template.
   if ( * this ) return fMemberTemplateName->fMemberTemplateImpl->TemplateInstance_Begin();
   return Dummy::MemberCont().begin();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Member_Iterator ROOT::Reflex::MemberTemplate::TemplateInstance_End() const {
//-------------------------------------------------------------------------------
   // Return the end iterator of the instance container of this member template.
   if ( * this ) return fMemberTemplateName->fMemberTemplateImpl->TemplateInstance_End();
   return Dummy::MemberCont().end();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::MemberTemplate::TemplateInstance_RBegin() const {
//-------------------------------------------------------------------------------
   // Return the rbegin iterator of the instance container of this member template.
   if ( * this ) return fMemberTemplateName->fMemberTemplateImpl->TemplateInstance_RBegin();
   return Dummy::MemberCont().rbegin();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::MemberTemplate::TemplateInstance_REnd() const {
//-------------------------------------------------------------------------------
   // Return the rend iterator of the instance container of this member template.
   if ( * this ) return fMemberTemplateName->fMemberTemplateImpl->TemplateInstance_REnd();
   return Dummy::MemberCont().rend();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::MemberTemplate::TemplateInstanceAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth template instance of this family.
   if ( * this ) return fMemberTemplateName->fMemberTemplateImpl->TemplateInstanceAt( nth );
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::MemberTemplate::AddTemplateInstance( const Member & templateInstance ) const {
//-------------------------------------------------------------------------------
// Add member templateInstance to this template family.
   if ( * this ) fMemberTemplateName->fMemberTemplateImpl->AddTemplateInstance( templateInstance );
}

