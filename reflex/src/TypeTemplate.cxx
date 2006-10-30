// @(#)root/reflex:$Name:  $:$Id: TypeTemplate.cxx,v 1.14 2006/08/16 06:42:36 roiser Exp $
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
   // Return the name of this type template.
   if ( fTypeTemplateName ) return fTypeTemplateName->Name( mod );
   else                     return "";
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate ROOT::Reflex::TypeTemplate::ByName( const std::string & name,
                                                               size_t nTemplateParams ) {
//-------------------------------------------------------------------------------
   // Lookup a type template by name.
   return TypeTemplateName::ByName( name, nTemplateParams );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate ROOT::Reflex::TypeTemplate::TypeTemplateAt( size_t nth ) {
//-------------------------------------------------------------------------------
   // Return the nth type template defined.
   return TypeTemplateName::TypeTemplateAt( nth );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::TypeTemplate::TypeTemplate_Begin() {
//-------------------------------------------------------------------------------
   // Return the begin iterator of the type template container.
   return TypeTemplateName::TypeTemplate_Begin();
}
 
                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::TypeTemplate::TypeTemplate_End() {
//-------------------------------------------------------------------------------
   // Return the end iterator of the type template container.
   return TypeTemplateName::TypeTemplate_End();
}
 
                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::TypeTemplate::TypeTemplate_RBegin() {
//-------------------------------------------------------------------------------
   // Return the rbegin iterator of the type template container.
   return TypeTemplateName::TypeTemplate_RBegin();
}
 
                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::TypeTemplate::TypeTemplate_REnd() {
//-------------------------------------------------------------------------------
   // Return the rend iterator of the type template container.
   return TypeTemplateName::TypeTemplate_REnd();
}
 

//-------------------------------------------------------------------------------
ROOT::Reflex::Type_Iterator ROOT::Reflex::TypeTemplate::TemplateInstance_Begin() const {
//-------------------------------------------------------------------------------
   // Return the begin iterator of the instances container of this type template.
   if ( * this ) return fTypeTemplateName->fTypeTemplateImpl->TemplateInstance_Begin();
   return Dummy::TypeCont().begin();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Type_Iterator ROOT::Reflex::TypeTemplate::TemplateInstance_End() const {
//-------------------------------------------------------------------------------
   // Return the end iterator of the instances container of this type template.
   if ( * this ) return fTypeTemplateName->fTypeTemplateImpl->TemplateInstance_End();
   return Dummy::TypeCont().end();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TypeTemplate::TemplateInstance_RBegin() const {
//-------------------------------------------------------------------------------
   // Return the rbegin iterator of the instances container of this type template.
   if ( * this ) return fTypeTemplateName->fTypeTemplateImpl->TemplateInstance_RBegin();
   return Dummy::TypeCont().rbegin();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TypeTemplate::TemplateInstance_REnd() const {
//-------------------------------------------------------------------------------
   // Return the rend iterator of the instances container of this type template.
   if ( * this ) return fTypeTemplateName->fTypeTemplateImpl->TemplateInstance_REnd();
   return Dummy::TypeCont().rend();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeTemplate::TemplateInstanceAt( size_t nth ) const {
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
