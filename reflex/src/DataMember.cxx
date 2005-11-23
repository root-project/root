// @(#)root/reflex:$Name:  $:$Id: DataMember.cxx,v 1.3 2005/11/11 07:18:06 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "DataMember.h"

#include "Reflex/Scope.h"
#include "Reflex/Object.h"
#include "Reflex/Member.h"

#include "Reflex/Tools.h"
#include "Class.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::DataMember::DataMember( const char *  nam,
                                      const Type &  typ,
                                      size_t        offs,
                                      unsigned int  modifiers )
//-------------------------------------------------------------------------------
   : MemberBase ( nam, typ, DATAMEMBER, modifiers ),
     fOffset( offs ) { }


//-------------------------------------------------------------------------------
ROOT::Reflex::DataMember::~DataMember() {}
//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::DataMember::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
   std::string s = "";

   if ( 0 != ( mod & ( QUALIFIED | Q ))) {
      if ( IsPublic())          { s += "public ";    }
      if ( IsProtected())       { s += "protected "; }
      if ( IsPrivate())         { s += "private ";   }
      if ( IsExtern())          { s += "extern ";    }
      if ( IsStatic())          { s += "static ";    }
      if ( IsAuto())            { s += "auto ";      }
      if ( IsRegister())        { s += "register ";  }
      if ( IsMutable())         { s += "mutable ";   }
   }

   if ( DeclaringScope().IsEnum()) {
      if ( DeclaringScope().DeclaringScope()) {
         std::string sc = DeclaringScope().DeclaringScope().Name(SCOPED);
         if ( sc != "::" ) s += sc + "::";
      }
      s += MemberBase::Name( mod & FINAL & QUALIFIED );
   }
   else {
      s += MemberBase::Name( mod );
   }

   return s;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Object ROOT::Reflex::DataMember::Get( const Object & obj ) const {
//-------------------------------------------------------------------------------
   if (DeclaringScope().ScopeType() == ENUM ) {
      return Object(Type::ByName("int"), (void*)&fOffset);
   }
   else {
      void * mem = CalculateBaseObject( obj );
      mem = (char*)mem + Offset();
      return Object(TypeOf(),mem);
   }
}


/*/-------------------------------------------------------------------------------
  void ROOT::Reflex::DataMember::Set( const Object & instance,
  const Object & value ) const {
//-------------------------------------------------------------------------------
  void * mem = CalculateBaseObject( instance );
  mem = (char*)mem + Offset();
  if (TypeOf().IsClass() ) {
  // Should use the asigment operator if exists (FIX-ME)
  memcpy( mem, value.Address(), TypeOf().SizeOf());
  }
  else {
  memcpy( mem, value.Address(), TypeOf().SizeOf() );
  }
  }
*/


//-------------------------------------------------------------------------------
void ROOT::Reflex::DataMember::Set( const Object & instance,
                                    const void * value ) const {
//-------------------------------------------------------------------------------
   void * mem = CalculateBaseObject( instance );
   mem = (char*)mem + Offset();
   if (TypeOf().IsClass() ) {
      // Should use the asigment operator if exists (FIX-ME)
      memcpy( mem, value, TypeOf().SizeOf());
   }
   else {
      memcpy( mem, value, TypeOf().SizeOf() );
   }
}
