// @(#)root/cintex:$Name:  $:$Id: CINTTypedefBuilder.cxx,v 1.9 2006/06/13 08:19:01 brun Exp $
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
        Scope scope = rt.DeclaringScope();
        CINTScopeBuilder::Setup( scope );
        while ( rt.IsTypedef() ) rt = rt.ToType();

        Indirection indir = IndirectionGet(rt);
        Scope rscope = indir.second.DeclaringScope();

        if ( scope != rscope ) {
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

        int rtypenum;
        int rtagnum;
        CintType(rt, rtypenum, rtagnum );
		
        int stagnum = -1;
	    if ( !scope.IsTopScope() ) stagnum = ::G__defined_tagname(scope.Name(SCOPED).c_str(), 1);
		
        int r = ::G__search_typename2( t.Name().c_str(), rtypenum, rtagnum, 0, stagnum);
        ::G__setnewtype(-1,NULL,0);
		
        return r;
      }
      return -1;
    }
  }
}
