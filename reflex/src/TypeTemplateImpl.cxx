// @(#)root/reflex:$Name:  $:$Id: TypeTemplateImpl.cxx,v 1.12 2006/08/11 06:31:59 roiser Exp $
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
#include "Reflex/TypeTemplate.h"
#include "Reflex/internal/OwnedMember.h"


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplateImpl::TypeTemplateImpl( const char * templateName,
                                                  const Scope & scop,
                                                  std::vector < std::string > parameterNames,
                                                  std::vector < std::string > parameterDefaults )
//------------------------------------------------------------------------------- 
   : fScope( scop ),
     fTemplateInstances( std::vector < Type >() ),
     fParameterNames( parameterNames ),
     fParameterDefaults( parameterDefaults ),
     fReqParameters( parameterNames.size() - parameterDefaults.size()) {
   // Construct the type template family info.

   TypeTemplate tt = TypeTemplate::ByName( templateName, parameterNames.size() );
   if ( tt.Id() == 0 ) {
      fTypeTemplateName = new TypeTemplateName( templateName, this );
   }
   else {
      fTypeTemplateName = (TypeTemplateName*)tt.Id();
      if ( fTypeTemplateName->fTypeTemplateImpl ) delete fTypeTemplateName->fTypeTemplateImpl;
      fTypeTemplateName->fTypeTemplateImpl = this;
   }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplateImpl::~TypeTemplateImpl() {
//-------------------------------------------------------------------------------
// Destructor.
   for ( Type_Iterator ti = TemplateInstance_Begin(); ti != TemplateInstance_End(); ++ti ) {
      ti->Unload();
   }
   if ( fTypeTemplateName->fTypeTemplateImpl == this ) fTypeTemplateName->fTypeTemplateImpl = 0;
}


//-------------------------------------------------------------------------------
bool ROOT::Reflex::TypeTemplateImpl::operator == ( const TypeTemplateImpl & tt ) const {
//-------------------------------------------------------------------------------
   // Equal operator.
   return ( ( fTypeTemplateName->fName == tt.fTypeTemplateName->fName ) && 
            ( fParameterNames.size() == tt.fParameterNames.size() ) );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type_Iterator ROOT::Reflex::TypeTemplateImpl::TemplateInstance_Begin() const {
//-------------------------------------------------------------------------------
   // Return the begin iterator of the instance container of this type template.
   return fTemplateInstances.begin();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Type_Iterator ROOT::Reflex::TypeTemplateImpl::TemplateInstance_End() const {
//-------------------------------------------------------------------------------
   // Return the end iterator of the instance container of this type template.
   return fTemplateInstances.end();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TypeTemplateImpl::TemplateInstance_RBegin() const {
//-------------------------------------------------------------------------------
   // Return the rbegin iterator of the instance container of this type template.
   return fTemplateInstances.rbegin();
}

                                             
//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TypeTemplateImpl::TemplateInstance_REnd() const {
//-------------------------------------------------------------------------------
   // Return the rend iterator of the instance container of this type template.
   return fTemplateInstances.rend();
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
const ROOT::Reflex::TypeTemplate & ROOT::Reflex::TypeTemplateImpl::ThisTypeTemplate() const {
//-------------------------------------------------------------------------------
   // Return a ref to this type template.
   return fTypeTemplateName->ThisTypeTemplate();
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::TypeTemplateImpl::AddTemplateInstance( const Type & templateInstance ) const {
//-------------------------------------------------------------------------------
// Add template instance to this family.
   fTemplateInstances.push_back( templateInstance );
}
