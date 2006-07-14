// @(#)root/reflex:$Name:  $:$Id: ScopeBase.cxx,v 1.17 2006/07/13 14:45:59 roiser Exp $
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
#include "NameLookup.h"


//-------------------------------------------------------------------------------
ROOT::Reflex::ScopeBase::ScopeBase( const char * scope, 
                                    TYPE scopeType )
   : fMembers( Members() ),
     fDataMembers( Members() ),
     fFunctionMembers( Members() ),
     fScopeName( 0 ),
     fScopeType( scopeType ),
     fDeclaringScope( Scope() ),
     fSubScopes( std::vector<Scope>() ),
     fSubTypes( std::vector<Type>() ),
     fTypeTemplates( std::vector<TypeTemplate>() ),
     fMemberTemplates( std::vector<MemberTemplate>() ),
     fPropertyList( PropertyList( new PropertyListImpl())),
     fBasePosition( Tools::GetBasePosition( scope )) {
//-------------------------------------------------------------------------------
// Construct the dictionary information for a scope.
   std::string sname(scope);

   std::string declScope = "";
   std::string currScope = sname;

   if ( fBasePosition ) {
      declScope = sname.substr( 0, fBasePosition-2);
      currScope = std::string(sname, fBasePosition);
   }

   // Construct Scope
   Scope scopePtr = Scope::ByName(sname);
   if ( scopePtr.Id() == 0 ) { 
      // create a new Scope
      fScopeName = new ScopeName(scope, this); 
   }
   else {
      fScopeName = (ScopeName*)scopePtr.Id();
      fScopeName->fScopeBase = this;
   }

   Scope declScopePtr = Scope::ByName(declScope);
   if ( declScopePtr.Id() == 0 ) {
      if ( scopeType == NAMESPACE ) declScopePtr = (new Namespace( declScope.c_str() ))->ThisScope();
      else                          declScopePtr = (new ScopeName( declScope.c_str(), 0 ))->ThisScope();
   }

   // Set declaring Scope and sub-scopes
   fDeclaringScope = declScopePtr; 
   if ( fDeclaringScope )  fDeclaringScope.AddSubScope( this->ThisScope() );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ScopeBase::ScopeBase() 
   : fMembers( Members()),
     fDataMembers( Members()),
     fFunctionMembers( Members()),
     fScopeName( 0 ),
     fScopeType( NAMESPACE ),
     fDeclaringScope( Scope::__NIRVANA__() ),
     fSubScopes( std::vector<Scope>()),
     fSubTypes( std::vector<Type>()),
     fPropertyList( PropertyList() ),
     fBasePosition( 0 ) {
//-------------------------------------------------------------------------------
// Default constructor for the ScopeBase (used at init time for the global scope)
   fScopeName = new ScopeName("", this);
   fPropertyList.AddProperty("Description", "global namespace");
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ScopeBase::~ScopeBase( ) {
//-------------------------------------------------------------------------------
// Destructor.
   // Informing Scope that I am going away
   if ( fScopeName->fScopeBase == this ) fScopeName->fScopeBase = 0;

   // Informing declaring Scope that I am going to do away
   if ( fDeclaringScope ) {
      fDeclaringScope.RemoveSubScope(ThisScope());
   }

   fPropertyList.ClearProperties();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ScopeBase::operator ROOT::Reflex::Scope () const {
//-------------------------------------------------------------------------------
// Conversion operator to Scope.
   return Scope( fScopeName );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ScopeBase::operator ROOT::Reflex::Type() const {
//-------------------------------------------------------------------------------
// Conversion operator to Type.
   switch ( fScopeType ) {
   case CLASS:
   case TYPETEMPLATEINSTANCE:
   case UNION:
   case ENUM:
      return *(dynamic_cast<const TypeBase*>(this));
   default:
      return Type();
   }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Base ROOT::Reflex::ScopeBase::BaseAt( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
// Return nth base info.
   return Base();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member
ROOT::Reflex::ScopeBase::DataMemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return nth data member info.
   if ( nth < fDataMembers.size() ) return fDataMembers[ nth ];
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member
ROOT::Reflex::ScopeBase::DataMemberByName( const std::string & nam ) const {
//-------------------------------------------------------------------------------
// Return data member info by name.
   for ( Members::const_iterator it = fDataMembers.begin(); it != fDataMembers.end(); ++it) {
      if (it->Name() == nam) return (*it);
   }
   return Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::ScopeBase::DataMemberSize() const {
//-------------------------------------------------------------------------------
// Return number of data members.
   return fDataMembers.size();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member
ROOT::Reflex::ScopeBase::FunctionMemberAt( size_t nth ) const { 
//-------------------------------------------------------------------------------
// Return nth function member.
   if ( nth < fFunctionMembers.size() ) return fFunctionMembers[ nth ];
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member
ROOT::Reflex::ScopeBase::FunctionMemberByName( const std::string & name,
                                               const Type & signature ) const {
//-------------------------------------------------------------------------------
// Return function member by name and signature.
   for (Members::const_iterator it = fFunctionMembers.begin(); it != fFunctionMembers.end(); ++it ) {
      if (it->Name() == name) {
         if (signature) {
           if (signature.IsEquivalentTo(it->TypeOf())) return (*it);
         }
         else {
            return (*it);
         }
      }
   }
   return Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::ScopeBase::FunctionMemberSize() const {
//-------------------------------------------------------------------------------
// Return number of function members.
   return fFunctionMembers.size();
}

//-------------------------------------------------------------------------------
bool ROOT::Reflex::ScopeBase::IsTopScope() const {
//-------------------------------------------------------------------------------
// Check if this scope is the top scope.
   if ( fDeclaringScope == Scope::__NIRVANA__() ) return true;
   return false;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member 
ROOT::Reflex::ScopeBase::LookupMember( const std::string & nam,
                                       const Scope & current ) const {
//------------------------------------------------------------------------------- 
// Lookup a member name from this scope.
   return NameLookup::LookupMember( nam, current );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type
ROOT::Reflex::ScopeBase::LookupType( const std::string & nam,
                                     const Scope & current ) const {
//-------------------------------------------------------------------------------
// Lookup a type name from this scope.
   return NameLookup::LookupType( nam, current );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::ScopeBase::MemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth member of this scope.
   if ( nth < fMembers.size() ) { return fMembers[ nth ]; };
   return Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::ScopeBase::MemberSize() const {
//-------------------------------------------------------------------------------
// Return the number of members.
   return fMembers.size();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member 
ROOT::Reflex::ScopeBase::MemberByName( const std::string & name,
                                       const Type & signature ) const {
//-------------------------------------------------------------------------------
// Return member by name and signature.
   if (signature) return FunctionMemberByName(name, signature);
   for ( size_t i = 0; i < fMembers.size() ; i++ ) {
      if ( fMembers[i].Name() == name ) return fMembers[i];
   }
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate ROOT::Reflex::ScopeBase::MemberTemplateAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return nth member template of this scope.
   if ( nth < fMemberTemplates.size() ) { return MemberTemplate(fMemberTemplates[ nth ]); }
   return MemberTemplate();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::ScopeBase::MemberTemplateSize() const {
//-------------------------------------------------------------------------------
// Return number of member templates.
   return fMemberTemplates.size();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate ROOT::Reflex::ScopeBase::MemberTemplateByName( const std::string & nam ) const {
//-------------------------------------------------------------------------------
   // Lookup a member template by name and return it.
   for ( size_t i = 0; i < fMemberTemplates.size(); ++i ) {
      if ( fMemberTemplates[i].Name() == nam ) return fMemberTemplates[i];
   }
   return MemberTemplate();
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::ScopeBase::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
// Return name of this scope.
   if ( 0 != ( mod & ( SCOPED | S ))) return fScopeName->Name();
   return std::string(fScopeName->Name(), fBasePosition);
}


//-------------------------------------------------------------------------------
ROOT::Reflex::PropertyList ROOT::Reflex::ScopeBase::Properties() const {
//-------------------------------------------------------------------------------
// Return property list attached to this scope.
   return fPropertyList;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::ScopeBase::ThisScope() const {
//-------------------------------------------------------------------------------
// Return the scope of this scope base.
   return fScopeName->ThisScope();
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::ScopeBase::ScopeTypeAsString() const {
//-------------------------------------------------------------------------------
// Return the type of the scope as a string.
   switch ( fScopeType ) {
   case CLASS:
      return "CLASS";
      break;
   case TYPETEMPLATEINSTANCE:
      return "TYPETEMPLATEINSTANCE";
      break;
   case NAMESPACE:
      return "NAMESPACE";
      break;
   case ENUM:
      return "ENUM";
      break;
   case UNION:
      return "UNION";
      break;
   case UNRESOLVED:
      return "UNRESOLVED";
      break;
   default:
      return "Scope " + Name() + "is not assigned to a SCOPE";
   }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::ScopeBase::SubTypeAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth sub type of this scope.
   if ( nth < fSubTypes.size() ) { return fSubTypes[ nth ]; }
   return Type();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::ScopeBase::SubTypeSize() const {
//-------------------------------------------------------------------------------
// Return the number of sub types.
   return fSubTypes.size();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::ScopeBase::SubTypeByName( const std::string & nam ) const {
//-------------------------------------------------------------------------------
   // Lookup a sub type by name and return it.
   if ( Tools::GetBasePosition(nam)) return Type::ByName(Name(SCOPED)+"::"+nam);
   for ( size_t i = 0; i < fSubTypes.size(); ++i ) {
      if ( fSubTypes[i].Name() == nam ) return fSubTypes[i];
   }
   return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::ScopeBase::TemplateArgumentAt( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
// Return the nth template argument.
   return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate ROOT::Reflex::ScopeBase::SubTypeTemplateAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth sub type template.
   if ( nth < fTypeTemplates.size() ) { return TypeTemplate(fTypeTemplates[ nth ]); }
   return TypeTemplate();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate ROOT::Reflex::ScopeBase::TemplateFamily() const {
//-------------------------------------------------------------------------------
// Return the template family corresponding to this scope.
   return TypeTemplate();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::ScopeBase::SubTypeTemplateSize() const {
//-------------------------------------------------------------------------------
// Return the number of sub type templates.
   return fTypeTemplates.size();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate ROOT::Reflex::ScopeBase::SubTypeTemplateByName( const std::string & nam ) const {
//-------------------------------------------------------------------------------
   // Lookup a type template in this scope by name and return it.
   for ( size_t i = 0; i < fTypeTemplates.size(); ++i ) {
      if ( fTypeTemplates[i].Name() == nam ) return fTypeTemplates[i];
   }
   return TypeTemplate();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::ScopeBase::SubScopeByName( const std::string & nam ) const {
//-------------------------------------------------------------------------------
// Lookup a sub scope of this scope by name and return it.
   if (Tools::GetBasePosition(nam)) return Scope::ByName(Name(SCOPED)+"::"+nam);
   for ( size_t i = 0; i < fSubScopes.size(); ++i ) {
      if ( fSubScopes[i].Name() == nam ) return fSubScopes[i];
   }
   return Scope();
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
// Add data member dm to this scope.
   dm.SetScope( ThisScope() );
   fDataMembers.push_back( dm );
   fMembers.push_back( dm );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddDataMember( const char * name,
                                             const Type & type,
                                             size_t offset,
                                             unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
// Add data member to this scope.
   AddDataMember(Member(new DataMember(name, type, offset, modifiers)));
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::RemoveDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
// Remove data member dm from this scope.
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
// Add function member fm to this scope.
   fm.SetScope( ThisScope());
   fFunctionMembers.push_back( fm );
   fMembers.push_back( fm );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddFunctionMember( const char * name,
                                                 const Type & type,
                                                 StubFunction stubFP,
                                                 void * stubCtx,
                                                 const char * params,
                                                 unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
// Add function member to this scope.
   AddFunctionMember(Member(new FunctionMember(name, type, stubFP, stubCtx, params, modifiers)));
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::RemoveFunctionMember( const Member & fm ) const {
//-------------------------------------------------------------------------------
// Remove function member fm from this scope.
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
// Add member template mt to this scope.
   fMemberTemplates.push_back( mt );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::RemoveMemberTemplate( const MemberTemplate & mt ) const {
//-------------------------------------------------------------------------------
// Remove member template mt from this scope.
   std::vector< MemberTemplate >::iterator it;
   for ( it = fMemberTemplates.begin(); it != fMemberTemplates.end(); ++it ) {
      if ( *it == mt ) fMemberTemplates.erase(it); break;
   }
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddSubScope( const Scope & subscope ) const {
//-------------------------------------------------------------------------------
// Add sub scope to this scope.
   RemoveSubScope(subscope);
   fSubScopes.push_back(subscope);
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddSubScope( const char * scope,
                                           TYPE scopeType ) const {
//-------------------------------------------------------------------------------
// Add sub scope to this scope.
   AddSubScope(*(new ScopeBase( scope, scopeType )));
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::RemoveSubScope( const Scope & subscope ) const {
//-------------------------------------------------------------------------------
// Remove sub scope from this scope.
   std::vector< Scope >::iterator it;
   for ( it = fSubScopes.begin(); it != fSubScopes.end(); ++it) {
      if ( *it == subscope ) {
         fSubScopes.erase(it); break;
      }
   }
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddSubType( const Type & ty ) const {
//-------------------------------------------------------------------------------
// Add sub type ty to this scope.
   RemoveSubType(ty);
   fSubTypes.push_back(ty);
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddSubType( const char * type,
                                          size_t size,
                                          TYPE typeType,
                                          const std::type_info & ti,
                                          unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
// Add sub type to this scope.
   TypeBase * tb = 0;
   switch ( typeType ) {
   case CLASS:
      tb = new Class(type,size,ti,modifiers,STRUCT);
      break;
   case STRUCT:
      tb = new Class(type,size,ti,modifiers);
      break;
   case ENUM:
      tb = new Enum(type,ti);
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
      tb = new Union(type,size,ti); 
      break;
   default:
      tb = new TypeBase( type, size, typeType, ti );
   }
   if ( tb ) AddSubType( * tb );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::RemoveSubType( const Type & ty ) const {
//-------------------------------------------------------------------------------
// Remove sub type ty from this scope.
   std::vector< Type >::iterator it;
   for ( it = fSubTypes.begin(); it != fSubTypes.end(); ++it) {
      if ( *it == ty ) {
         fSubTypes.erase(it); break;
      }
   }
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddSubTypeTemplate( const TypeTemplate & tt ) const {
//-------------------------------------------------------------------------------
// Add sub type template to this scope.
   fTypeTemplates.push_back( tt );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::RemoveSubTypeTemplate( const TypeTemplate & tt ) const {
//-------------------------------------------------------------------------------
// Remove sub type template tt from this scope.
   std::vector< TypeTemplate >::iterator it;
   for ( it = fTypeTemplates.begin(); it != fTypeTemplates.end(); ++it) {
      if (*it == tt) {
         fTypeTemplates.erase(it); break;
      }
   }
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::AddUsingDirective( const Scope & ud ) const {
//-------------------------------------------------------------------------------
// Add using directive ud to this scope.
   fUsingDirectives.push_back( ud );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeBase::RemoveUsingDirective( const Scope & ud ) const {
//-------------------------------------------------------------------------------
// Remove using directive ud from this scope.
   for ( Scope_Iterator it = UsingDirective_Begin(); it != UsingDirective_End(); ++ it ) {
      if ( *it == ud ) {
         fUsingDirectives.erase( it ); 
         break;
      }
   }
}

