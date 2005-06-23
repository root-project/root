// @(#)root/reflex:$Name:$:$Id:$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Reflex.h"
#include "Reflex/Tools.h"
#include "Cintex/Cintex.h"
#include "CINTdefs.h"
#include "CINTScopeBuilder.h"
#include "CINTClassBuilder.h"
#include "CINTEnumBuilder.h"
#include "Api.h"


using namespace ROOT::Reflex;
using namespace std;

namespace ROOT { namespace Cintex {

  void CINTEnumBuilder::Setup(const Type& t) {
    if ( t.IsEnum() )  {
      string Name = t.Name(SCOPED);
      int tagnum = ::G__defined_tagname(Name.c_str(), 2);
      if( -1 != tagnum ) return;

      if ( Cintex::Debug() )  {
        std::cout << "Building enum " << Name << std::endl;
      }

      Scope ScopeNth = t.ScopeGet();
      CINTScopeBuilder::Setup( ScopeNth );

      G__linked_taginfo taginfo;
      taginfo.tagnum  = -1;
      taginfo.tagtype = 'e';
      taginfo.tagname = Name.c_str();
      G__get_linked_tagnum(&taginfo);
      tagnum = taginfo.tagnum;

      ::G__tagtable_setup( tagnum, sizeof(int), -1, 0,(char*)NULL, NULL, NULL);
    }
  }
}}
