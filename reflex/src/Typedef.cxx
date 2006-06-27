// @(#)root/reflex:$Name:  $:$Id: Typedef.cxx,v 1.6 2006/03/20 09:46:18 roiser Exp $
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

#include "Typedef.h"

#include "Reflex/Tools.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::Typedef::Typedef( const char * typ,
                                const Type & typedefType,
                                TYPE typeTyp )
//-------------------------------------------------------------------------------
   : TypeBase(typ, typedefType.SizeOf() , typeTyp, typeid(UnknownType)),
     fTypedefType(typedefType),
     fFinalType() { }


//-------------------------------------------------------------------------------
ROOT::Reflex::Typedef::~Typedef() {}
//-------------------------------------------------------------------------------
