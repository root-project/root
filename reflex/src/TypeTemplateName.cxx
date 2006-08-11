// @(#)root/reflex:$Name:  $:$Id: ScopeName.cxx,v 1.17 2006/08/03 16:49:21 roiser Exp $
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

#include "Reflex/internal/TypeTemplateName.h"

#include "Reflex/TypeTemplate.h"
#include "Reflex/internal/TypeTemplateImpl.h"
#include "Reflex/Type.h"
#include "Reflex/Tools.h"

#include "stl_hash.h"
#include <vector>

//-------------------------------------------------------------------------------
typedef __gnu_cxx::hash_multimap < const char *, ROOT::Reflex::TypeTemplate > Name2TypeTemplate_t;
typedef std::vector< ROOT::Reflex::TypeTemplate > TypeTemplateVec_t;


//-------------------------------------------------------------------------------
static Name2TypeTemplate_t & sTypeTemplates() {
//-------------------------------------------------------------------------------
   // Static wrapper around the type template map.
   static Name2TypeTemplate_t t;
   return t;
}


//-------------------------------------------------------------------------------
static TypeTemplateVec_t & sTypeTemplateVec() {
//-------------------------------------------------------------------------------
   // Static wrapper around the type template vector.
   static TypeTemplateVec_t t;
   return t;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplateName::TypeTemplateName( const char * name,
                                                      TypeTemplateImpl * typeTemplateImpl )
//-------------------------------------------------------------------------------
   : fName( name ),
     fTypeTemplateImpl( typeTemplateImpl ) {
   // Constructor.
   fThisTypeTemplate = new TypeTemplate( this );
   sTypeTemplates().insert(std::make_pair<const char * const, TypeTemplate>(fName.c_str(),*fThisTypeTemplate));
   sTypeTemplateVec().push_back( * fThisTypeTemplate );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplateName::~TypeTemplateName() {
//-------------------------------------------------------------------------------
   // Destructor.
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::TypeTemplate & ROOT::Reflex::TypeTemplateName::ByName( const std::string & name,
                                                                               size_t nTemplateParams ) {
//-------------------------------------------------------------------------------
   // Lookup a type template by its name.
   typedef Name2TypeTemplate_t::iterator IT;
   const char * cname = name.c_str();
   IT lower = sTypeTemplates().find(cname);
   if ( lower != sTypeTemplates().end()) {
      if ( ! nTemplateParams ) return lower->second;
      else {
         std::pair<IT,IT> bounds = sTypeTemplates().equal_range(cname);
         for ( IT it = bounds.first; it != bounds.second; ++it ) {
            if ( it->second.TemplateParameterSize() == nTemplateParams ) {
               return it->second;
            }
         }
      }
   }
   return Dummy::TypeTemplate();
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::TypeTemplateName::CleanUp() {
//-------------------------------------------------------------------------------
   // Do the final cleanup for the type templates.  
   for ( TypeTemplateVec_t::iterator it = sTypeTemplateVec().begin(); it != sTypeTemplateVec().end(); ++it ) {
      TypeTemplateName * tn = (TypeTemplateName*)it->Id();
      TypeTemplate * t = tn->fThisTypeTemplate;
      if ( t ) t->Unload();
      delete t;
      delete tn;
   }
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::TypeTemplateName::DeleteTypeTemplate() const {
//-------------------------------------------------------------------------------
   // Remove a type template dictionary information.
   delete fTypeTemplateImpl;
   fTypeTemplateImpl = 0;
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::TypeTemplateName::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
   if ( 0 != ( mod & ( SCOPED | S ))) return fName;
   else                               return Tools::GetBaseName( fName );
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::TypeTemplate & ROOT::Reflex::TypeTemplateName::ThisTypeTemplate() const {
//-------------------------------------------------------------------------------
   // Return the type template corresponding to this type template name.
   return * fThisTypeTemplate;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::TypeTemplate & ROOT::Reflex::TypeTemplateName::TypeTemplateAt( size_t nth ) {
//-------------------------------------------------------------------------------
   // Return teh nth type template.
   if ( nth < sTypeTemplateVec().size()) return sTypeTemplateVec()[nth];
   return Dummy::TypeTemplate();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::TypeTemplateName::TypeTemplateSize() {
//-------------------------------------------------------------------------------
   // Return the number of type templates declared.
   return sTypeTemplateVec().size();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::TypeTemplateName::TypeTemplate_Begin() {
//-------------------------------------------------------------------------------
   // Return the begin iterator of the type template collection
   return sTypeTemplateVec().begin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::TypeTemplateName::TypeTemplate_End() {
//-------------------------------------------------------------------------------
   // Return the end iterator of the type template collection
   return sTypeTemplateVec().end();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::TypeTemplateName::TypeTemplate_RBegin() {
//-------------------------------------------------------------------------------
   // Return the RBegin iterator of the type template collection
   return sTypeTemplateVec().rbegin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::TypeTemplateName::TypeTemplate_REnd() {
//-------------------------------------------------------------------------------
   // Return the rend iterator of the type template collection
   return sTypeTemplateVec().rend();
}


