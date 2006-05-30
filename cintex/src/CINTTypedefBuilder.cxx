// @(#)root/cintex:$Name:  $:$Id: CINTTypedefBuilder.cxx,v 1.7 2006/05/30 08:14:13 roiser Exp $
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
#include "CINTTypedefBuilder.h"
#include "Api.h"


using namespace ROOT::Reflex;
using namespace std;

namespace ROOT { 

  namespace Cintex {

    int CINTTypedefBuilder::Setup(const Type& t) {
      if ( t.IsTypedef() )  {

        std::string nam = CintName(t.Name(SCOPED));

        Type rt(t);
        Scope ScopeNth = rt.DeclaringScope();
        CINTScopeBuilder::Setup( ScopeNth );
        while ( rt.IsTypedef() ) rt = rt.ToType();

        Indirection indir = IndirectionGet(rt);
        Scope rscope = indir.second.DeclaringScope();

        if ( ScopeNth != rscope ) {
          if ( rscope ) CINTScopeBuilder::Setup(rscope);
          else {
            rscope = Scope::ByName(Tools::GetScopeName(indir.second.Name(SCOPED)));
            CINTScopeBuilder::Setup(rscope);
          }
        }

        if( -1 != G__defined_typename(nam.c_str()) ) return -1;

        if ( Cintex::Debug() )  {
          std::cout << "Building typedef " << nam << std::endl;
        }

        int typenum;
        int tagnum;
        CintType(rt, typenum, tagnum );

        // If the final type was was not found create a place holder in the G__struct for it
        if (tagnum == -1) tagnum = G__search_tagname(CintName(rt).c_str(), 'a');

        int r = ::G__search_typename2( nam.c_str(), typenum, tagnum, 0, -1);
        ::G__setnewtype(-1,NULL,0);
        return r;
      }
      return -1;
    }
  }
}
