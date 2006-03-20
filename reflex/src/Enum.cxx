// @(#)root/reflex:$Name:  $:$Id: Enum.cxx,v 1.6 2006/03/13 15:49:50 roiser Exp $
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

#include "Enum.h"

#include "Reflex/Tools.h"

#include <sstream>


//-------------------------------------------------------------------------------
ROOT::Reflex::Enum::Enum( const char * enumType,
                          const std::type_info & ti )
//-------------------------------------------------------------------------------
   : TypeBase( enumType, sizeof(int), ENUM, ti ),
     ScopeBase( enumType, ENUM ) {}


//-------------------------------------------------------------------------------
ROOT::Reflex::Enum::~Enum() {
//-------------------------------------------------------------------------------
}


