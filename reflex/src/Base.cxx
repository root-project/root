// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Base.h"
#include "Class.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::Base::Base( const Type &    baseType,
                          OffsetFunction  OffsetFP,
                          unsigned int    modifiers )
//-------------------------------------------------------------------------------
  : fOffsetFP( OffsetFP ),
    fModifiers( modifiers ),
    fBaseType( Type() ),
    fBaseClass( 0 ) {

  fBaseType = baseType;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Class * ROOT::Reflex::Base::BaseClass() const {
//-------------------------------------------------------------------------------
  if ( fBaseClass ) return fBaseClass;
  if ( fBaseType ) {
    fBaseClass = dynamic_cast< const Class * >(fBaseType.TypeBaseNth());
    return fBaseClass;
  }
  return 0;
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Base::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
  std::string s = "";
  if ( 0 != ( mod & ( QUALIFIED | Q ))) {
    if ( IsPublic())    { s += "public "; }
    if ( IsProtected()) { s += "protected "; }
    if ( IsPrivate())   { s += "private "; }
    if ( IsVirtual())   { s += "virtual "; }
  }
  s += fBaseType.Name( mod );
  return s;
}
