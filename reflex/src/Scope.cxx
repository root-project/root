// @(#)root/reflex:$Name:  $:$Id: Scope.cxx,v 1.8 2006/06/08 16:05:14 roiser Exp $
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

#include "Reflex/Scope.h"

#include "Reflex/ScopeBase.h"
#include "Reflex/Member.h"
#include "Reflex/Type.h"
#include "Reflex/TypeTemplate.h"
#include "Reflex/MemberTemplate.h"
#include "Reflex/Base.h"

#include "Reflex/Tools.h"
#include "Class.h"


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope::operator ROOT::Reflex::Type () const {
//-------------------------------------------------------------------------------
   if ( * this ) return *(fScopeName->fScopeBase);
   return Type();
}



//-------------------------------------------------------------------------------
ROOT::Reflex::Base ROOT::Reflex::Scope::BaseAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->BaseAt( nth );
   return Base();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::BaseSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->BaseSize();
   return 0;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::Scope::ByName( const std::string & name ) {
//-------------------------------------------------------------------------------
   return ScopeName::ByName( name );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Scope::DataMemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->DataMemberAt( nth ); 
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Scope::DataMemberByName( const std::string & name ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->DataMemberByName( name ); 
   return Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::DataMemberSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->DataMemberSize(); 
   return 0;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Scope::FunctionMemberAt( size_t nth ) const {
//------------------------------------------------------------------------------- 
   if ( * this ) return fScopeName->fScopeBase->FunctionMemberAt(nth); 
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Scope::FunctionMemberByName( const std::string & name ) const {
//------------------------------------------------------------------------------- 
   if ( * this ) return fScopeName->fScopeBase->FunctionMemberByName( name, Type() ); 
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Scope::FunctionMemberByName( const std::string & name,
                                                                const Type & signature ) const {
//------------------------------------------------------------------------------- 
   if ( * this ) return fScopeName->fScopeBase->FunctionMemberByName( name, signature ); 
   return Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::FunctionMemberSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->FunctionMemberSize(); 
   return 0;
}



//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::Scope::GlobalScope() {
//-------------------------------------------------------------------------------
   return fg__GLOBALSCOPE__;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member 
ROOT::Reflex::Scope::LookupMember( const std::string & nam ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->LookupMember( nam, *this );
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type
ROOT::Reflex::Scope::LookupType( const std::string & nam ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->LookupType( nam, *this );
   return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member 
ROOT::Reflex::Scope::MemberByName( const std::string & name ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->MemberByName(name, Type()); 
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member 
ROOT::Reflex::Scope::MemberByName( const std::string & name,
                                   const Type & signature ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->MemberByName(name, signature); 
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Scope::MemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->MemberAt(nth); 
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate ROOT::Reflex::Scope::MemberTemplateAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->MemberTemplateAt( nth );
   return MemberTemplate();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::MemberTemplateSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->MemberTemplateSize();
   return 0;
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Scope::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->Name( mod );
   else if ( fScopeName ) {
      if ( 0 != ( mod & ( SCOPED | S ))) return fScopeName->Name();
      else                               return Tools::GetBaseName( fScopeName->Name());
   }
   else {
      return "";
   }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::Scope::ScopeAt( size_t nth ) {
//-------------------------------------------------------------------------------
   return ScopeName::ScopeAt( nth );
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::ScopeSize() {
//-------------------------------------------------------------------------------
   return ScopeName::ScopeSize();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::Scope::SubTypeAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->SubTypeAt( nth ); 
   return Type();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::SubTypeSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->SubTypeSize(); 
   return 0;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::Scope::TemplateArgumentAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->TemplateArgumentAt( nth );
   return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate ROOT::Reflex::Scope::TemplateFamily() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->TemplateFamily();
   return TypeTemplate();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate ROOT::Reflex::Scope::SubTypeTemplateAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->SubTypeTemplateAt( nth ); 
   return TypeTemplate();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::SubTypeTemplateSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fScopeName->fScopeBase->SubTypeTemplateSize();
   return 0;
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
   if ( * this) fScopeName->fScopeBase->AddDataMember( dm );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddDataMember( const char * name,
                                         const Type & type,
                                         size_t offset,
                                         unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fScopeName->fScopeBase->AddDataMember( name, 
                                                        type, 
                                                        offset, 
                                                        modifiers );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::RemoveDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
   if ( * this) fScopeName->fScopeBase->RemoveDataMember( dm );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddFunctionMember( const Member & fm ) const {
//-------------------------------------------------------------------------------
   if ( * this) fScopeName->fScopeBase->AddFunctionMember( fm );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddFunctionMember( const char * nam,
                                             const Type & typ,
                                             StubFunction stubFP,
                                             void * stubCtx,
                                             const char * params,
                                             unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fScopeName->fScopeBase->AddFunctionMember( nam, 
                                                            typ, 
                                                            stubFP, 
                                                            stubCtx, 
                                                            params, 
                                                            modifiers );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::RemoveFunctionMember( const Member & fm ) const {
//-------------------------------------------------------------------------------
   if ( * this) fScopeName->fScopeBase->RemoveFunctionMember( fm );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddSubType( const Type & ty ) const {
//-------------------------------------------------------------------------------
   if ( * this) fScopeName->fScopeBase->AddSubType( ty );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddSubType( const char * type,
                                      size_t size,
                                      TYPE typeType,
                                      const std::type_info & typeInfo,
                                      unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fScopeName->fScopeBase->AddSubType( type, 
                                                     size, 
                                                     typeType, 
                                                     typeInfo, 
                                                     modifiers );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::RemoveSubType( const Type & ty ) const {
//-------------------------------------------------------------------------------
   if ( * this) fScopeName->fScopeBase->RemoveSubType( ty );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddMemberTemplate( const MemberTemplate & mt ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fScopeName->fScopeBase->AddMemberTemplate( mt );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::RemoveMemberTemplate( const MemberTemplate & mt ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fScopeName->fScopeBase->RemoveMemberTemplate( mt );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddSubTypeTemplate( const TypeTemplate & tt ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fScopeName->fScopeBase->AddSubTypeTemplate( tt );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::RemoveSubTypeTemplate( const TypeTemplate & tt ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fScopeName->fScopeBase->RemoveSubTypeTemplate( tt );
}
