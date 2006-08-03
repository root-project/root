// @(#)root/reflex:$Name:  $:$Id: Scope.cxx,v 1.14 2006/08/03 16:49:21 roiser Exp $
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

#include "Reflex/internal/ScopeBase.h"
#include "Reflex/Member.h"
#include "Reflex/Type.h"
#include "Reflex/TypeTemplate.h"
#include "Reflex/MemberTemplate.h"
#include "Reflex/Base.h"

#include "Reflex/Tools.h"
#include "Class.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::Scope & ROOT::Reflex::Scope::__NIRVANA__() {
//-------------------------------------------------------------------------------
// static wraper around NIRVANA, the base of the top scope.
   static Scope s = Scope( new ScopeName( "@N@I@R@V@A@N@A@", 0 ));
   return s;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope & ROOT::Reflex::Scope::__GLOBALSCOPE__() {
//-------------------------------------------------------------------------------
// static wrapper for the global scope.
   static Scope s = Scope::ByName("");
   return s;
}



//-------------------------------------------------------------------------------
ROOT::Reflex::Scope::operator const ROOT::Reflex::Type & () const {
//-------------------------------------------------------------------------------
// Conversion operator to Type. If this scope is not a Type, returns the empty type.
   if ( * this ) return *(fScopeName->fScopeBase);
   return Dummy::Type();
}



//-------------------------------------------------------------------------------
const ROOT::Reflex::Base & ROOT::Reflex::Scope::BaseAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return nth base class info.
   if ( * this ) return fScopeName->fScopeBase->BaseAt( nth );
   return Dummy::Base();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::BaseSize() const {
//-------------------------------------------------------------------------------
// Return number of base classes.
   if ( * this ) return fScopeName->fScopeBase->BaseSize();
   return 0;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Scope & ROOT::Reflex::Scope::ByName( const std::string & name ) {
//-------------------------------------------------------------------------------
// Lookup a Scope by it's fully qualified name.
   return ScopeName::ByName( name );
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Member & ROOT::Reflex::Scope::DataMemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth data member of this scope.
   if ( * this ) return fScopeName->fScopeBase->DataMemberAt( nth ); 
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Member & ROOT::Reflex::Scope::DataMemberByName( const std::string & name ) const {
//-------------------------------------------------------------------------------
// Return a data member by it's name.
   if ( * this ) return fScopeName->fScopeBase->DataMemberByName( name ); 
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::DataMemberSize() const {
//-------------------------------------------------------------------------------
// Return number of data mebers of this scope.
   if ( * this ) return fScopeName->fScopeBase->DataMemberSize(); 
   return 0;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Member & ROOT::Reflex::Scope::FunctionMemberAt( size_t nth ) const {
//------------------------------------------------------------------------------- 
// Return nth function member of this socpe.
   if ( * this ) return fScopeName->fScopeBase->FunctionMemberAt(nth); 
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Member & ROOT::Reflex::Scope::FunctionMemberByName( const std::string & name ) const {
//------------------------------------------------------------------------------- 
// Return a function member by it's name.
   if ( * this ) return fScopeName->fScopeBase->FunctionMemberByName( name, Type() ); 
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Member & ROOT::Reflex::Scope::FunctionMemberByName( const std::string & name,
                                                                const Type & signature ) const {
//------------------------------------------------------------------------------- 
// Return a function member by it's name, qualified by it's signature type.
   if ( * this ) return fScopeName->fScopeBase->FunctionMemberByName( name, signature ); 
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::FunctionMemberSize() const {
//-------------------------------------------------------------------------------
// Return number of function members of this scope.
   if ( * this ) return fScopeName->fScopeBase->FunctionMemberSize(); 
   return 0;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Scope & ROOT::Reflex::Scope::GlobalScope() {
//-------------------------------------------------------------------------------
// Return global scope representaiton
   return Scope::__GLOBALSCOPE__();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Member & 
ROOT::Reflex::Scope::LookupMember( const std::string & nam ) const {
//-------------------------------------------------------------------------------
// Lookup a member from this scope.
   if ( * this ) return fScopeName->fScopeBase->LookupMember( nam, *this );
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Type &
ROOT::Reflex::Scope::LookupType( const std::string & nam ) const {
//-------------------------------------------------------------------------------
// Lookup a type from this scope.
   if ( * this ) return fScopeName->fScopeBase->LookupType( nam, *this );
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Member & 
ROOT::Reflex::Scope::MemberByName( const std::string & name ) const {
//-------------------------------------------------------------------------------
// Return a member from this scope, by name.
   if ( * this ) return fScopeName->fScopeBase->MemberByName(name, Type()); 
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Member & 
ROOT::Reflex::Scope::MemberByName( const std::string & name,
                                   const Type & signature ) const {
//-------------------------------------------------------------------------------
// Return a member in this scope, looked up by name and signature (for functions)
   if ( * this ) return fScopeName->fScopeBase->MemberByName(name, signature); 
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member_Iterator ROOT::Reflex::Scope::Member_Begin() const {
//-------------------------------------------------------------------------------
// Return the begin iterator of member container.
   if ( * this ) return fScopeName->fScopeBase->Member_Begin();
   return Dummy::MemberCont().begin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member_Iterator ROOT::Reflex::Scope::Member_End() const {
//-------------------------------------------------------------------------------
// Return the end iterator of member container.
   if ( * this ) return fScopeName->fScopeBase->Member_End();
   return Dummy::MemberCont().end();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Scope::Member_RBegin() const {
//-------------------------------------------------------------------------------
// Return the rbegin iterator of member container.
   if ( * this ) return fScopeName->fScopeBase->Member_RBegin();
   return Dummy::MemberCont().rbegin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Scope::Member_REnd() const {
//-------------------------------------------------------------------------------
// Return the rend iterator of member container.
   if ( * this ) return fScopeName->fScopeBase->Member_REnd();
   return Dummy::MemberCont().rend();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Member & ROOT::Reflex::Scope::MemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth member of this scope.
   if ( * this ) return fScopeName->fScopeBase->MemberAt(nth); 
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::MemberTemplate & ROOT::Reflex::Scope::MemberTemplateAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth memer template in this scope.
   if ( * this ) return fScopeName->fScopeBase->MemberTemplateAt( nth );
   return Dummy::MemberTemplate();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::MemberTemplateSize() const {
//-------------------------------------------------------------------------------
// Return the number of member templates in this scope.
   if ( * this ) return fScopeName->fScopeBase->MemberTemplateSize();
   return 0;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::MemberTemplate & ROOT::Reflex::Scope::MemberTemplateByName( const std::string & nam ) const {
//-------------------------------------------------------------------------------
// Look up a member template in this scope by name and return it.
   if ( * this ) return fScopeName->fScopeBase->MemberTemplateByName( nam );
   return Dummy::MemberTemplate();
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Scope::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
// Return the name of this scope, scoped if requested.
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
const ROOT::Reflex::Scope & ROOT::Reflex::Scope::ScopeAt( size_t nth ) {
//-------------------------------------------------------------------------------
// Return the nth scope in the Reflex database.
   return ScopeName::ScopeAt( nth );
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::ScopeSize() {
//-------------------------------------------------------------------------------
// Return the number of scopes defined.
   return ScopeName::ScopeSize();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Type & ROOT::Reflex::Scope::SubTypeAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth sub type of this scope.
   if ( * this ) return fScopeName->fScopeBase->SubTypeAt( nth ); 
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::SubTypeSize() const {
//-------------------------------------------------------------------------------
// Return the number of sub types.
   if ( * this ) return fScopeName->fScopeBase->SubTypeSize(); 
   return 0;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Type & ROOT::Reflex::Scope::SubTypeByName( const std::string & nam ) const {
//-------------------------------------------------------------------------------
// Look up a sub type by name and return it.
   if ( * this ) return fScopeName->fScopeBase->SubTypeByName( nam );
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Type & ROOT::Reflex::Scope::TemplateArgumentAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth template argument of this scope (ie. class).
   if ( * this ) return fScopeName->fScopeBase->TemplateArgumentAt( nth );
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::TypeTemplate & ROOT::Reflex::Scope::TemplateFamily() const {
//-------------------------------------------------------------------------------
// Return the template family related to this scope.
   if ( * this ) return fScopeName->fScopeBase->TemplateFamily();
   return Dummy::TypeTemplate();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::TypeTemplate & ROOT::Reflex::Scope::SubTypeTemplateAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth sub type template.
   if ( * this ) return fScopeName->fScopeBase->SubTypeTemplateAt( nth ); 
   return Dummy::TypeTemplate();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Scope::SubTypeTemplateSize() const {
//-------------------------------------------------------------------------------
// Return the number of type templates in this scope.
   if ( * this ) return fScopeName->fScopeBase->SubTypeTemplateSize();
   return 0;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::TypeTemplate & ROOT::Reflex::Scope::SubTypeTemplateByName( const std::string & nam ) const {
//-------------------------------------------------------------------------------
// Lookup a sub type template by string and return it.
   if ( * this ) return fScopeName->fScopeBase->SubTypeTemplateByName( nam );
   return Dummy::TypeTemplate();
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
// Add data member dm to this scope.
   if ( * this) fScopeName->fScopeBase->AddDataMember( dm );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddDataMember( const char * name,
                                         const Type & type,
                                         size_t offset,
                                         unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
// Add data member to this scope.
   if ( * this ) fScopeName->fScopeBase->AddDataMember( name, 
                                                        type, 
                                                        offset, 
                                                        modifiers );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::RemoveDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
// Remove data member dm from this scope.
   if ( * this) fScopeName->fScopeBase->RemoveDataMember( dm );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddFunctionMember( const Member & fm ) const {
//-------------------------------------------------------------------------------
// Add function member fm to this scope.
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
// Add function member to this scope.
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
// Remove function member fm from this scope.
   if ( * this) fScopeName->fScopeBase->RemoveFunctionMember( fm );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddSubType( const Type & ty ) const {
//-------------------------------------------------------------------------------
// Add sub type ty to this scope.
   if ( * this) fScopeName->fScopeBase->AddSubType( ty );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddSubType( const char * type,
                                      size_t size,
                                      TYPE typeType,
                                      const std::type_info & typeInfo,
                                      unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
// Add sub type to this scope.
   if ( * this ) fScopeName->fScopeBase->AddSubType( type, 
                                                     size, 
                                                     typeType, 
                                                     typeInfo, 
                                                     modifiers );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::RemoveSubType( const Type & ty ) const {
//-------------------------------------------------------------------------------
// Remove sub type ty from this scope.
   if ( * this) fScopeName->fScopeBase->RemoveSubType( ty );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddMemberTemplate( const MemberTemplate & mt ) const {
//-------------------------------------------------------------------------------
// Add member template mt to this scope.
   if ( * this ) fScopeName->fScopeBase->AddMemberTemplate( mt );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::RemoveMemberTemplate( const MemberTemplate & mt ) const {
//-------------------------------------------------------------------------------
// Remove member template mt from this scope.
   if ( * this ) fScopeName->fScopeBase->RemoveMemberTemplate( mt );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::AddSubTypeTemplate( const TypeTemplate & tt ) const {
//-------------------------------------------------------------------------------
// Add type template tt to this scope.
   if ( * this ) fScopeName->fScopeBase->AddSubTypeTemplate( tt );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Scope::RemoveSubTypeTemplate( const TypeTemplate & tt ) const {
//-------------------------------------------------------------------------------
// Remove type template tt from this scope.
   if ( * this ) fScopeName->fScopeBase->RemoveSubTypeTemplate( tt );
}
