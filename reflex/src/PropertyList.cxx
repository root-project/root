// @(#)root/reflex:$Name:  $:$Id: PropertyList.cxx,v 1.5 2006/03/13 15:49:51 roiser Exp $
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

#include "Reflex/PropertyList.h"

#include "Reflex/PropertyListImpl.h"

//-------------------------------------------------------------------------------
inline std::ostream & ROOT::Reflex::operator<<( std::ostream & s,
                                                const PropertyList & p ) {
//-------------------------------------------------------------------------------
   if ( p.fPropertyListImpl ) s << *(p.fPropertyListImpl); 
   return s;
}





