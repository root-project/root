// @(#)root/reflex:$Id$
// Author: Axel Naumann, 2009

// Copyright CERN, CH-1211 Geneva 23, 2004-2009, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
# define REFLEX_BUILD
#endif

#include "Reflex/Builder/GenreflexMemberBuilder.h"
#include "Class.h"

//-------------------------------------------------------------------------------
void
Reflex::GenreflexMemberBuilder::BuildAll() {
//-------------------------------------------------------------------------------
   // Build the members, return if we added something

   Class* sb = dynamic_cast<Class*>(Context());
   if (sb) {
      fFunc(sb);
   }
}
