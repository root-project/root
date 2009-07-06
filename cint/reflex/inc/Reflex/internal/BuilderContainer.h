// @(#)root/reflex:$Id$
// Author: Axel Naumann, 2009

// Copyright CERN, CH-1211 Geneva 23, 2004-2009, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_BuilderContainer
#define Reflex_BuilderContainer

#include "Reflex/Kernel.h"

namespace Reflex {
class OnDemandBuilder;

class RFLX_API BuilderContainer {
public:

   BuilderContainer(): fFirst(0) {}
   ~BuilderContainer() { Clear(); }

   void Insert(OnDemandBuilder* odb);
   void Remove(OnDemandBuilder* odb);
   void Clear();

   OnDemandBuilder* First() const { return fFirst; }
   bool Empty() const { return !fFirst; }

   void BuildAll();

private:
   OnDemandBuilder* fFirst;
};
} // namespace Reflex

#endif // Reflex_BuilderContainer
