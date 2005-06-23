// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/MemberBase.h"

#include "Reflex/Member.h"
#include "Reflex/Scope.h"
#include "Reflex/Type.h"
#include "Reflex/Base.h"
#include "Reflex/Object.h"
#include "Reflex/PropertyList.h"

#include "Reflex/Tools.h"
#include "Class.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::MemberBase::MemberBase( const char *  Name,
                                      const Type &  TypeNth,
                                      TYPE          MemberType,
                                      unsigned int  modifiers )
//-------------------------------------------------------------------------------
  : fType( TypeNth, modifiers & ( CONST | VOLATILE | REFERENCE )),
    fModifiers( modifiers ),
    fName( Name ),
    fScope( Scope() ),
    fMemberType( MemberType ),
    fPropertyList( PropertyList( new PropertyListImpl())) {}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberBase::~MemberBase() {
//-------------------------------------------------------------------------------
  fPropertyList.ClearProperties();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberBase::operator ROOT::Reflex::Member () const {
//-------------------------------------------------------------------------------
  return Member( this );
}


//-------------------------------------------------------------------------------
void * ROOT::Reflex::MemberBase::CalculateBaseObject( const Object & obj ) const {
//-------------------------------------------------------------------------------
  char * mem = (char*)obj.AddressGet();
  // check if its a dummy object 
  Type cl = obj.TypeGet();
  // if the object TypeNth is not implemented return the AddressGet of the object
  if ( ! cl ) return mem; 
  if ( cl.IsClass() ) {
    if ( ScopeGet() && ( cl.Id() != (dynamic_cast<const Class*>(ScopeGet().ScopeBaseNth()))->TypeGet().Id())) {
      // now we know that the MemberNth is an inherited one
      std::vector < OffsetFunction > basePath = (dynamic_cast<const Class*>(cl.TypeBaseNth()))->PathToBase( ScopeGet());
      if ( basePath.size() ) {
        // there is a path described from the object to the class containing the MemberNth
        std::vector < OffsetFunction >::iterator pIter;
        for ( pIter = basePath.begin(); pIter != basePath.end(); ++pIter ) {
          mem += (*pIter)(mem);
        }
      }
      else {
        throw RuntimeError(std::string(": ERROR: There is no path available from class ")
			   + cl.Name(SCOPED) + " to " + Name(SCOPED));
      }
    }
  }
  else {
    throw RuntimeError(std::string("Object ") + cl.Name(SCOPED) + " does not represent a class");
  }
  return (void*)mem;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::MemberBase::ScopeGet() const {
//-------------------------------------------------------------------------------
  return fScope;
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::MemberBase::MemberTypeAsString() const {
//-------------------------------------------------------------------------------
  switch ( fMemberType ) {
    case DATAMEMBER:
      return "DataMember";
      break;
    case FUNCTIONMEMBER:
      return "FunctionMember";
      break;
    default:
      return Reflex::Argv0() + ": ERROR: Member " + Name() +
         " has no Species associated";
  }
}

//-------------------------------------------------------------------------------
ROOT::Reflex::PropertyList ROOT::Reflex::MemberBase::PropertyListGet() const {
//-------------------------------------------------------------------------------
  return fPropertyList;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::MemberBase::TemplateArgumentNth( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
  return Type();
}

