// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Cintex_CINTFunctionBuilder
#define ROOT_Cintex_CINTFunctionBuilder

#include "Reflex/Member.h"
#include "Reflex/Type.h"
#include "CINTdefs.h"

namespace ROOT {
namespace Cintex {

class CINTFunctionBuilder {
   // --
public: // Static Interface
   static void Setup(ROOT::Reflex::Member);
private: // Data Member
   ROOT::Reflex::Member fFunction;
public: // Public Interface
   CINTFunctionBuilder(ROOT::Reflex::Member m);
   ~CINTFunctionBuilder();
   void Setup(void);
};

} // namespace Cintex
} // namespace ROOT

#endif // ROOT_Cintex_CINTFunctionBuilder
