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

#include "Reflex/PropertyList.h"
#include "Reflex/Type.h"
#include "Reflex/Member.h"
#include "Reflex/ScopeName.h"
#include "Reflex/TypeTemplate.h"
#include "Reflex/MemberTemplate.h"
#include "Reflex/Tools.h"

#include "Class.h"
#include "Namespace.h"
#include "DataMember.h"
#include "FunctionMember.h"
#include "Union.h"
#include "Enum.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::ScopeBase::ScopeBase( const char * ScopeNth, 
                                    TYPE ScopeType )
//-------------------------------------------------------------------------------
  : fMembers( Members() ),
    fDataMembers( Members() ),
    fFunctionMembers( Members() ),
    fScopeName( 0 ),
    fScopeType( ScopeType ),
    fDeclaringScope( Scope() ),
    fSubScopes( std::vector<Scope>() ),
    fTypes( std::vector<Type>() ),
    fTypeTemplates( std::vector<TypeTemplate>() ),
    fMemberTemplates( std::vector<MemberTemplate>() ),
    fPropertyList( PropertyList( new PropertyListImpl())),
    fBasePosition( Tools::GetBasePosition( ScopeNth )) {

  std::string sname(ScopeNth);

  std::string declScope = "";
  std::string currScope = sname;

  if ( fBasePosition ) {
    declScope = sname.substr( 0, fBasePosition-2);
    currScope = std::string(sname, fBasePosition);
  }

  // Construct Scope
  Scope scopePtr = Scope::ByName(sname);
  if ( scopePtr.Id() == 0 ) { 
    // create a new ScopeNth
    fScopeName = new ScopeName(ScopeNth, this); 
  }
  else {
    fScopeName = (ScopeName*)scopePtr.Id();
    fScopeName->fScopeBase = this;
  }

  Scope declScopePtr = Scope::ByName(declScope);
  if ( declScopePtr.Id() == 0 ) {
    if ( ScopeType == NAMESPACE ) declScopePtr = (new Namespace( declScope.c_str() ))->ScopeGet();
    else                          declScopePtr = (new ScopeName( declScope.c_str(), 0 ))->ScopeGet();
  }

  // Set declaring ScopeNth and sub-scopes
  fDeclaringScope = declScopePtr; 
  if ( fDeclaringScope )  fDeclaringScope.AddSubScope( this->ScopeGet() );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ScopeBase::ScopeBase() 
//-------------------------------------------------------------------------------
  : fMembers( Members()),
    fDataMembers( Members()),
    fFunctionMembers( Members()),
    fScopeName( 0 ),
    fScopeType( NAMESPACE ),
    fDeclaringScope( Scope::__NIRVANA__ ),
    fSubScopes( std::vector<Scope>()),
    fTypes( std::vector<Type>()),
    fPropertyList( PropertyList() ),
    fBasePosition( 0 ) {
  fScopeName = new ScopeName("", this);
  fPropertyList.AddProperty("Description", "global namespace");
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ScopeBase::~ScopeBase( ) {
//-------------------------------------------------------------------------------
  // Informing Scope that I am going away
  if ( fScopeName->fScopeBase == this ) fScopeName->fScopeBase = 0;

  // Informing declaring ScopeNth that I am going to do away
  if ( fDeclaringScope ) {
    fDeclaringScope.RemoveSubScope(this->ScopeGet());
  }

  fPropertyList.ClearProperties();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ScopeBase::operator ROOT::Reflex::Scope () const {
//-------------------------------------------------------------------------------
  return Scope( fScopeName );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member
ROOT::Reflex::ScopeBase::DataMemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( nth < fDataMembers.size() ) return fDataMembers[ nth ];
  return Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::ScopeBase::DataMemberCount() const {
//-------------------------------------------------------------------------------
  return fDataMembers.size();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member
ROOT::Reflex::ScopeBase::FunctionMemberNth( size_t nth ) const { 
//-------------------------------------------------------------------------------
  if ( nth < fFunctionMembers.size() ) return fFunctionMembers[ nth ];
  return Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::ScopeBase::FunctionMemberCount() const {
//-------------------------------------------------------------------------------
  return fFunctionMembers.size();
}

//-------------------------------------------------------------------------------
bool ROOT::Reflex::ScopeBase::IsTopScope() const {
//-------------------------------------------------------------------------------
  if ( fDeclaringScope == Scope::__NIRVANA__ ) return true;
  return false;
}

//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::ScopeBase::MemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( nth < fMembers.size() ) { return fMembers[ nth ]; };
  return Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::ScopeBase::MemberCount() const {
//-------------------------------------------------------------------------------
  return fMembers.size();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member 
ROOT::Reflex::ScopeBase::MemberNth( const std::string & Name ) const {
//-------------------------------------------------------------------------------
  for ( size_t i = 0; i < fMembers.size() ; i++ ) {
    if ( fMembers[i].Name() == Name ) return fMembers[i];
  }
  return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate ROOT::Reflex::ScopeBase::MemberTemplateNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( nth < fMemberTemplates.size() ) { return MemberTemplate(fMemberTemplates[ nth ]); }
  return MemberTemplate();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::ScopeBase::MemberTemplateCount() const {
//-------------------------------------------------------------------------------
  return fMemberTemplates.size();
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::ScopeBase::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
  if ( 0 != ( mod & ( SCOPED | S ))) return fScopeName->Name();
  return std::string(fScopeName->Name(), fBasePosition);
}


//-------------------------------------------------------------------------------
ROOT::Reflex::PropertyList ROOT::Reflex::ScopeBase::PropertyListGet() const {
//-------------------------------------------------------------------------------
  return fPropertyList;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::ScopeBase::ScopeGet() const {
//-------------------------------------------------------------------------------
  return fScopeName->ScopeGet();
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::ScopeBase::ScopeTypeAsString() const {
//-------------------------------------------------------------------------------
  switch ( fScopeType ) {
    case CLASS:
      return "CLASS";
      break;
    case TYPETEMPLATEINSTANCE:
      return " TYPETEMPLATEINSTANCE";
      break;
    case NAMESPACE:
      return "NAMESPACE";
      break;
    case UNRESOLVED:
      return "UNRESOLVED";
      break;
    default:
      return "Scope " + Name() + "is not assigned to a SCOPE";
  }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::ScopeBase::SubTypeNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( nth < fTypes.size() ) { return fTypes[ nth ]; }
  return Type();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::ScopeBase::SubTypeCount() const {
//-------------------------------------------------------------------------------
  return fTypes.size();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::ScopeBase::TemplateArgumentNth( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
  return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate ROOT::Reflex::ScopeBase::TypeTemplateNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( nth < fTypeTemplates.size() ) { return TypeTemplate(fTypeTemplates[ nth ]); }
  return TypeTemplate();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate ROOT::Reflex::ScopeBase::TemplateFamily() const {
//-------------------------------------------------------------------------------
  return TypeTemplate();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::ScopeBase::TypeTemplateCount() const {
//-------------------------------------------------------------------------------
  return fTypeTemplates.size();
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
  dm.SetScope( ScopeGet() );
  fDataMembers.push_back( dm );
  fMembers.push_back( dm );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddDataMember( const char * Name,
                                             const Type & TypeNth,
                                             size_t Offset,
                                             unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
  AddDataMember(Member(new DataMember(Name, TypeNth, Offset, modifiers)));
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::RemoveDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
  std::vector< Member >::iterator it;
  for ( it = fDataMembers.begin(); it != fDataMembers.end(); ++it) {
    if ( *it == dm ) fDataMembers.erase(it); break;
  }
  std::vector< Member >::iterator im;
  for ( im = fMembers.begin(); im != fMembers.end(); ++im) {
    if ( *im == dm ) fMembers.erase(im); break;
  }
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddFunctionMember( const Member & fm ) const {
//-------------------------------------------------------------------------------
  fm.SetScope( ScopeGet());
  fFunctionMembers.push_back( fm );
  fMembers.push_back( fm );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddFunctionMember( const char * Name,
                                                 const Type & TypeNth,
                                                 StubFunction stubFP,
                                                 void * stubCtx,
                                                 const char * params,
                                                 unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
  AddFunctionMember(Member(new FunctionMember(Name, TypeNth, stubFP, stubCtx, params, modifiers)));
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::RemoveFunctionMember( const Member & fm ) const {
//-------------------------------------------------------------------------------
  std::vector< Member >::iterator it;
  for ( it = fFunctionMembers.begin(); it != fFunctionMembers.end(); ++it) {
    if ( *it == fm ) fFunctionMembers.erase(it); break;
  }
  std::vector< Member >::iterator im;
  for ( im = fMembers.begin(); im != fMembers.end(); ++im) {
    if ( *im == fm ) fMembers.erase(im); break;
  }
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddMemberTemplate( const MemberTemplate & mt ) const {
//-------------------------------------------------------------------------------
  fMemberTemplates.push_back( mt );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::RemoveMemberTemplate( const MemberTemplate & mt ) const {
//-------------------------------------------------------------------------------
  std::vector< MemberTemplate >::iterator it;
  for ( it = fMemberTemplates.begin(); it != fMemberTemplates.end(); ++it ) {
    if ( *it == mt ) fMemberTemplates.erase(it); break;
  }
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddSubScope( const Scope & subscope ) const {
//-------------------------------------------------------------------------------
  RemoveSubScope(subscope);
  fSubScopes.push_back(subscope);
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddSubScope( const char * ScopeNth,
                                           TYPE ScopeType ) const {
//-------------------------------------------------------------------------------
  AddSubScope(*(new ScopeBase( ScopeNth, ScopeType )));
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::RemoveSubScope( const Scope & subscope ) const {
//-------------------------------------------------------------------------------
  std::vector< Scope >::iterator it;
  for ( it = fSubScopes.begin(); it != fSubScopes.end(); ++it) {
    if ( *it == subscope ) {
      fSubScopes.erase(it); break;
    }
  }
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddSubType( const Type & TypeNth ) const {
//-------------------------------------------------------------------------------
  RemoveSubType(TypeNth);
  fTypes.push_back(TypeNth);
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddSubType( const char * TypeNth,
                                          size_t size,
                                          TYPE TypeType,
                                          const std::type_info & ti,
                                          unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
  TypeBase * tb = 0;
  switch ( TypeType ) {
  case CLASS:
    tb = new Class(TypeNth,size,ti,modifiers,STRUCT);
    break;
  case STRUCT:
    tb = new Class(TypeNth,size,ti,modifiers);
    break;
  case ENUM:
    tb = new Enum(TypeNth,ti);
    break;
  case FUNCTION:
    break;
  case ARRAY:
    break;
  case FUNDAMENTAL:
    break;
  case  POINTER:
    break;
  case POINTERTOMEMBER:
    break;
  case TYPEDEF:
    break;
  case UNION:
    tb = new Union(TypeNth,size,ti); 
    break;
  default:
    tb = new TypeBase( TypeNth, size, TypeType, ti );
  }
  if ( tb ) AddSubType( * tb );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::RemoveSubType( const Type & TypeNth ) const {
//-------------------------------------------------------------------------------
  std::vector< Type >::iterator it;
  for ( it = fTypes.begin(); it != fTypes.end(); ++it) {
    if ( *it == TypeNth ) {
      fTypes.erase(it); break;
    }
  }
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddTypeTemplate( const TypeTemplate & tt ) const {
//-------------------------------------------------------------------------------
  fTypeTemplates.push_back( tt );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::RemoveTypeTemplate( const TypeTemplate & tt ) const {
//-------------------------------------------------------------------------------
  std::vector< TypeTemplate >::iterator it;
  for ( it = fTypeTemplates.begin(); it != fTypeTemplates.end(); ++it) {
    if (*it == tt) {
      fTypeTemplates.erase(it); break;
    }
  }
}


