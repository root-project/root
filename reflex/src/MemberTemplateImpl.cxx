// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/MemberTemplateImpl.h"
#include "Reflex/Member.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::MemberTemplateImpl::InstantiationNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( nth < fTemplateInstances.size() ) return Member(fTemplateInstances[ nth ]);
  return Member();
}


