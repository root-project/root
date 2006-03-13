// @(#)root/reflex:$Name:  $:$Id: PropertyList.cxx,v 1.4 2006/03/06 12:51:46 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#define REFLEX_BUILD

#include "Reflex/PropertyList.h"

#include "Reflex/PropertyListImpl.h"

//-------------------------------------------------------------------------------
inline std::ostream & ROOT::Reflex::operator<<( std::ostream & s,
                                                const PropertyList & p ) {
//-------------------------------------------------------------------------------
   if ( p.fPropertyListImpl ) s << *(p.fPropertyListImpl); 
   return s;
}





