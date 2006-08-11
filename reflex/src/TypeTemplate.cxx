// @(#)root/reflex:$Name:  $:$Id: TypeTemplate.cxx,v 1.12 2006/08/03 16:49:21 roiser Exp $
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

#include "Reflex/TypeTemplate.h"
#include "Reflex/Type.h"
#include "Reflex/internal/OwnedMember.h"
#include "Reflex/internal/TypeTemplateName.h"                                                             

///-------------------------------------------------------------------------------
std::string ROOT::Reflex::TypeTemplate::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
   if ( fTypeTemplateName ) return fTypeTemplateName->Name( mod );
   else                     return "";
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::TypeTemplate & ROOT::Reflex::TypeTemplate::ByName( const std::string & name,
                                                                       size_t nTemplateParams ) {
//-------------------------------------------------------------------------------
   return TypeTemplateName::ByName( name, nTemplateParams );
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::TypeTemplate & ROOT::Reflex::TypeTemplate::TypeTemplateAt( size_t nth ) {
//-------------------------------------------------------------------------------
   return TypeTemplateName::TypeTemplateAt( nth );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::TypeTemplate::TypeTemplate_Begin() {
//-------------------------------------------------------------------------------
   return TypeTemplateName::TypeTemplate_Begin();
}
 
                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::TypeTemplate::TypeTemplate_End() {
//-------------------------------------------------------------------------------
   return TypeTemplateName::TypeTemplate_End();
}
 
                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::TypeTemplate::TypeTemplate_RBegin() {
//-------------------------------------------------------------------------------
   return TypeTemplateName::TypeTemplate_RBegin();
}
 
                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::TypeTemplate::TypeTemplate_REnd() {
//-------------------------------------------------------------------------------
   return TypeTemplateName::TypeTemplate_REnd();
}
 

//-------------------------------------------------------------------------------
ROOT::Reflex::Type_Iterator ROOT::Reflex::TypeTemplate::TemplateInstance_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateName->fTypeTemplateImpl->TemplateInstance_Begin();
   return Dummy::TypeCont().begin();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Type_Iterator ROOT::Reflex::TypeTemplate::TemplateInstance_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateName->fTypeTemplateImpl->TemplateInstance_End();
   return Dummy::TypeCont().end();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TypeTemplate::TemplateInstance_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateName->fTypeTemplateImpl->TemplateInstance_RBegin();
   return Dummy::TypeCont().rbegin();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TypeTemplate::TemplateInstance_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateName->fTypeTemplateImpl->TemplateInstance_REnd();
   return Dummy::TypeCont().rend();
}

                                             
//-------------------------------------------------------------------------------
const ROOT::Reflex::Type & ROOT::Reflex::TypeTemplate::TemplateInstanceAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   // Return the nth template instance of this family.
   if ( * this ) return fTypeTemplateName->fTypeTemplateImpl->TemplateInstanceAt( nth );
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::TypeTemplate::AddTemplateInstance( const Type & templateInstance ) const {
//-------------------------------------------------------------------------------
   // Add template instance to this template family.
   if ( * this ) fTypeTemplateName->fTypeTemplateImpl->AddTemplateInstance( templateInstance );
}
