// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Scope.h"

#include "Reflex/ScopeBase.h"
#include "Reflex/Member.h"
#include "Reflex/Type.h"
#include "Reflex/TypeTemplate.h"
#include "Reflex/MemberTemplate.h"

#include "Reflex/Tools.h"
#include "Reflex/Tools.h"
#include "Class.h"


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::Scope::ByName( const std::string & Name ) {
//-------------------------------------------------------------------------------
  return ScopeName::ByName( Name );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Scope::DataMemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fScopeName->fScopeBase->DataMemberNth( nth ); 
  return Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::DataMemberCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fScopeName->fScopeBase->DataMemberCount(); 
  return 0;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Scope::FunctionMemberNth( size_t nth ) const {
//------------------------------------------------------------------------------- 
  if ( * this ) return fScopeName->fScopeBase->FunctionMemberNth(nth); 
  return Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::FunctionMemberCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fScopeName->fScopeBase->FunctionMemberCount(); 
  return 0;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member 
ROOT::Reflex::Scope::MemberNth( const std::string & Name ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fScopeName->fScopeBase->MemberNth(Name); 
  return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Scope::MemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fScopeName->fScopeBase->MemberNth(nth); 
  return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate ROOT::Reflex::Scope::MemberTemplateNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fScopeName->fScopeBase->MemberTemplateNth( nth );
  return MemberTemplate();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::MemberTemplateCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fScopeName->fScopeBase->MemberTemplateCount();
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
ROOT::Reflex::Scope ROOT::Reflex::Scope::ScopeNth( size_t nth ) {
//-------------------------------------------------------------------------------
  return ScopeName::ScopeNth( nth );
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::ScopeCount() {
//-------------------------------------------------------------------------------
  return ScopeName::ScopeCount();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::Scope::SubTypeNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fScopeName->fScopeBase->SubTypeNth( nth ); 
  return Type();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::SubTypeCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fScopeName->fScopeBase->SubTypeCount(); 
  return 0;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::Scope::TemplateArgumentNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fScopeName->fScopeBase->TemplateArgumentNth( nth );
  return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate ROOT::Reflex::Scope::TemplateFamily() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fScopeName->fScopeBase->TemplateFamily();
  return TypeTemplate();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate ROOT::Reflex::Scope::TypeTemplateNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fScopeName->fScopeBase->TypeTemplateNth( nth ); 
  return TypeTemplate();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::TypeTemplateCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fScopeName->fScopeBase->TypeTemplateCount();
  return 0;
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
  if ( * this) fScopeName->fScopeBase->AddDataMember( dm );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddDataMember( const char * Name,
                                         const Type & TypeNth,
                                         size_t Offset,
                                         unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
  if ( * this ) fScopeName->fScopeBase->AddDataMember( Name, 
                                                         TypeNth, 
                                                         Offset, 
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
void ROOT::Reflex::Scope::AddFunctionMember( const char * Name,
                                             const Type & TypeNth,
                                             StubFunction stubFP,
                                             void * stubCtx,
                                             const char * params,
                                             unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
  if ( * this ) fScopeName->fScopeBase->AddFunctionMember( Name, 
                                                             TypeNth, 
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
void ROOT::Reflex::Scope::AddSubType( const char * TypeNth,
                                      size_t size,
                                      TYPE TypeType,
                                      const std::type_info & TypeInfo,
                                      unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
  if ( * this ) fScopeName->fScopeBase->AddSubType( TypeNth, 
                                                      size, 
                                                      TypeType, 
                                                      TypeInfo, 
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
void ROOT::Reflex::Scope::AddTypeTemplate( const TypeTemplate & tt ) const {
//-------------------------------------------------------------------------------
  if ( * this ) fScopeName->fScopeBase->AddTypeTemplate( tt );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::RemoveTypeTemplate( const TypeTemplate & tt ) const {
//-------------------------------------------------------------------------------
  if ( * this ) fScopeName->fScopeBase->RemoveTypeTemplate( tt );
}
