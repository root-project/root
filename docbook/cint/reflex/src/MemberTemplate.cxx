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

#include "Reflex/MemberTemplate.h"
#include "Reflex/internal/OwnedMember.h"
#include "Reflex/internal/MemberTemplateName.h"


//-------------------------------------------------------------------------------
Reflex::MemberTemplate
Reflex::MemberTemplate::ByName(const std::string& name,
                               size_t nTemplateParams) {
//-------------------------------------------------------------------------------
// Return a member template by name.
   return MemberTemplateName::ByName(name, nTemplateParams);
}


//-------------------------------------------------------------------------------
Reflex::MemberTemplate
Reflex::MemberTemplate::MemberTemplateAt(size_t nth) {
//-------------------------------------------------------------------------------
// Return the nth member template defined.
   return MemberTemplateName::MemberTemplateAt(nth);
}


//-------------------------------------------------------------------------------
Reflex::MemberTemplate_Iterator
Reflex::MemberTemplate::MemberTemplate_Begin() {
//-------------------------------------------------------------------------------
// Return the begin iterator of the member template container.
   return MemberTemplateName::MemberTemplate_Begin();
}


//-------------------------------------------------------------------------------
Reflex::MemberTemplate_Iterator
Reflex::MemberTemplate::MemberTemplate_End() {
//-------------------------------------------------------------------------------
// Return the end iterator of the member template container.
   return MemberTemplateName::MemberTemplate_End();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_MemberTemplate_Iterator
Reflex::MemberTemplate::MemberTemplate_RBegin() {
//-------------------------------------------------------------------------------
// Return the rbegin iterator of the member template container.
   return MemberTemplateName::MemberTemplate_RBegin();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_MemberTemplate_Iterator
Reflex::MemberTemplate::MemberTemplate_REnd() {
//-------------------------------------------------------------------------------
// Return the rend iterator of the member template container.
   return MemberTemplateName::MemberTemplate_REnd();
}


//-------------------------------------------------------------------------------
std::string
Reflex::MemberTemplate::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
// Return the name of the member template.
   if (fMemberTemplateName) {
      return fMemberTemplateName->Name(mod);
   } else { return ""; }
}


//-------------------------------------------------------------------------------
Reflex::Member_Iterator
Reflex::MemberTemplate::TemplateInstance_Begin() const {
//-------------------------------------------------------------------------------
// Return the begin iterator of the instance container of this member template.
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateInstance_Begin();
   }
   return Dummy::MemberCont().begin();
}


//-------------------------------------------------------------------------------
Reflex::Member_Iterator
Reflex::MemberTemplate::TemplateInstance_End() const {
//-------------------------------------------------------------------------------
// Return the end iterator of the instance container of this member template.
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateInstance_End();
   }
   return Dummy::MemberCont().end();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_Member_Iterator
Reflex::MemberTemplate::TemplateInstance_RBegin() const {
//-------------------------------------------------------------------------------
// Return the rbegin iterator of the instance container of this member template.
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateInstance_RBegin();
   }
   return Dummy::MemberCont().rbegin();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_Member_Iterator
Reflex::MemberTemplate::TemplateInstance_REnd() const {
//-------------------------------------------------------------------------------
// Return the rend iterator of the instance container of this member template.
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateInstance_REnd();
   }
   return Dummy::MemberCont().rend();
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::MemberTemplate::TemplateInstanceAt(size_t nth) const {
//-------------------------------------------------------------------------------
// Return the nth template instance of this family.
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateInstanceAt(nth);
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
void
Reflex::MemberTemplate::AddTemplateInstance(const Member& templateInstance) const {
//-------------------------------------------------------------------------------
// Add member templateInstance to this template family.
   if (*this) {
      fMemberTemplateName->fMemberTemplateImpl->AddTemplateInstance(templateInstance);
   }
}
