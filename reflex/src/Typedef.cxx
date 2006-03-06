// @(#)root/reflex:$Name:  $:$Id: Typedef.cxx,v 1.3 2005/11/23 16:08:08 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Typedef.h"

#include "Reflex/Tools.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::Typedef::Typedef( const char * typ,
                                const Type & typedefType,
                                TYPE typeTyp )
//-------------------------------------------------------------------------------
   : TypeBase(typ, typedefType.SizeOf() , typeTyp, typeid(UnknownType)),
     fTypedefType(typedefType)  { }


//-------------------------------------------------------------------------------
ROOT::Reflex::Typedef::~Typedef() {}
//-------------------------------------------------------------------------------
