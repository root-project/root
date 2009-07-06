// @(#)root/reflex:$Id$
// Author: Axel Naumann, 2009

// Copyright CERN, CH-1211 Geneva 23, 2004-2009, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_GenreflexMemberBuilder
#define Reflex_GenreflexMemberBuilder

#include "Reflex/Builder/OnDemandBuilderForScope.h"

namespace Reflex {
class Class;

class RFLX_API GenreflexMemberBuilder: public OnDemandBuilderForScope {
public:
   typedef void (*SetupFunc_t)(Class* sb);
   GenreflexMemberBuilder(SetupFunc_t func): fFunc(func) {}
   virtual ~GenreflexMemberBuilder() {}

   void BuildAll();

private:
   SetupFunc_t  fFunc;
};
} // namespace Reflex

#endif // Reflex_OnDemandBuilder
