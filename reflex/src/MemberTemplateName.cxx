// @(#)root/reflex:$Name:  $:$Id: MemberTemplateName.cxx,v 1.4 2006/09/05 17:13:15 roiser Exp $
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

#include "Reflex/internal/MemberTemplateName.h"

#include "Reflex/MemberTemplate.h"
#include "Reflex/internal/MemberTemplateImpl.h"
#include "Reflex/Member.h"
#include "Reflex/Tools.h"

#include "stl_hash.h"
#include <vector>

//-------------------------------------------------------------------------------
typedef __gnu_cxx::hash_multimap < const std::string *, ROOT::Reflex::MemberTemplate > Name2MemberTemplate_t;
typedef std::vector< ROOT::Reflex::MemberTemplate > MemberTemplateVec_t;


//-------------------------------------------------------------------------------
static Name2MemberTemplate_t & sMemberTemplates() {
//-------------------------------------------------------------------------------
   // Static wrapper around the member template map.
   static Name2MemberTemplate_t t;
   return t;
}


//-------------------------------------------------------------------------------
static MemberTemplateVec_t & sMemberTemplateVec() {
//-------------------------------------------------------------------------------
   // Static wrapper around the member template vector.
   static MemberTemplateVec_t t;
   return t;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplateName::MemberTemplateName( const char * name,
                                                      MemberTemplateImpl * memberTemplateImpl )
//-------------------------------------------------------------------------------
   : fName( name ),
     fMemberTemplateImpl( memberTemplateImpl ) {
   // Constructor.
   fThisMemberTemplate = new MemberTemplate( this );
   sMemberTemplates().insert(std::make_pair<const std::string * const,MemberTemplate>( &fName, *fThisMemberTemplate));
   sMemberTemplateVec().push_back( * fThisMemberTemplate );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplateName::~MemberTemplateName() {
//-------------------------------------------------------------------------------
   // Destructor.
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate ROOT::Reflex::MemberTemplateName::ByName( const std::string & name,
                                                                       size_t nTemplateParams ) {
//-------------------------------------------------------------------------------
   // Lookup a member template by its name.
   typedef Name2MemberTemplate_t::iterator IT;
   IT lower = sMemberTemplates().find(&name);
   if ( lower != sMemberTemplates().end()) {
      if ( ! nTemplateParams ) return lower->second;
      else {
         std::pair<IT,IT> bounds = sMemberTemplates().equal_range(&name);
         for ( IT it = bounds.first; it != bounds.second; ++it ) {
            if ( it->second.TemplateParameterSize() == nTemplateParams ) {
               return it->second;
            }
         }
      }
   }
   return Dummy::MemberTemplate();
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::MemberTemplateName::CleanUp() {
//-------------------------------------------------------------------------------
   // Do the final cleanup for the member templates.  
   for ( MemberTemplateVec_t::iterator it = sMemberTemplateVec().begin(); it != sMemberTemplateVec().end(); ++it ) {
      MemberTemplateName * tn = (MemberTemplateName*)it->Id();
      MemberTemplate * t = tn->fThisMemberTemplate;
      tn->DeleteMemberTemplate();
      delete t;
      delete tn;
   }
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::MemberTemplateName::DeleteMemberTemplate() const {
//-------------------------------------------------------------------------------
   // Remove a member template dictionary information.
   delete fMemberTemplateImpl;
   fMemberTemplateImpl = 0;
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::MemberTemplateName::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
   // Print the name of this member template.
   if ( 0 != ( mod & ( SCOPED | S ))) return fName;
   else                               return Tools::GetBaseName( fName );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate ROOT::Reflex::MemberTemplateName::ThisMemberTemplate() const {
//-------------------------------------------------------------------------------
   // Return the member template corresponding to this member template name.
   return * fThisMemberTemplate;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate ROOT::Reflex::MemberTemplateName::MemberTemplateAt( size_t nth ) {
//-------------------------------------------------------------------------------
   // Return teh nth member template.
   if ( nth < sMemberTemplateVec().size()) return sMemberTemplateVec()[nth];
   return Dummy::MemberTemplate();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::MemberTemplateName::MemberTemplateSize() {
//-------------------------------------------------------------------------------
   // Return the number of member templates declared.
   return sMemberTemplateVec().size();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate_Iterator ROOT::Reflex::MemberTemplateName::MemberTemplate_Begin() {
//-------------------------------------------------------------------------------
   // Return the begin iterator of the member template collection
   return sMemberTemplateVec().begin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate_Iterator ROOT::Reflex::MemberTemplateName::MemberTemplate_End() {
//-------------------------------------------------------------------------------
   // Return the end iterator of the member template collection
   return sMemberTemplateVec().end();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::MemberTemplateName::MemberTemplate_RBegin() {
//-------------------------------------------------------------------------------
   // Return the RBegin iterator of the member template collection
   return ((const std::vector<MemberTemplate>&)sMemberTemplateVec()).rbegin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::MemberTemplateName::MemberTemplate_REnd() {
//-------------------------------------------------------------------------------
   // Return the rend iterator of the member template collection
   return ((const std::vector<MemberTemplate>&)sMemberTemplateVec()).rend();
}


