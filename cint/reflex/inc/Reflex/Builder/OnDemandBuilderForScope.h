// @(#)root/reflex:$Id$
// Author: Axel Naumann, 2009

// Copyright CERN, CH-1211 Geneva 23, 2004-2009, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_OnDemandBuilderForScope
#define Reflex_OnDemandBuilderForScope

#include "Reflex/Kernel.h"
#include "Reflex/Builder/OnDemandBuilder.h"

namespace Reflex {
// not part of the interface
class BuilderContainer;
class ScopeBase;

class RFLX_API OnDemandBuilderForScope: public OnDemandBuilder {
public:

   OnDemandBuilderForScope() {}
   OnDemandBuilderForScope(ScopeBase* scope): fContext(scope) {}
   virtual ~OnDemandBuilderForScope() {}

   // return whether the builder has changed reflection data
   virtual void BuildAll() = 0;

   void SetContext(ScopeBase* scope) { fContext = scope; }
   ScopeBase* Context() const { return fContext; }

private:
   ScopeBase* fContext; // which scope to build for
};
} // namespace Reflex

#endif // Reflex_OnDemandBuilderForScope
