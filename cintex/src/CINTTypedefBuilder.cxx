// @(#)root/cintex:$Name:  $:$Id: CINTTypedefBuilder.cxx,v 1.15 2007/01/13 19:53:41 brun Exp $
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
         // Setup typedef info.
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
               std::cout << "Cintex: Building typedef " << nam << std::endl;
            }

            int rtypenum;
            int rtagnum;
            CintType(rt, rtypenum, rtagnum );
		
            int stagnum = -1;
            if ( !scope.IsTopScope() ) stagnum = ::G__defined_tagname(CintName(scope.Name(SCOPED)).c_str(), 1);
		
            int r = ::G__search_typename2( t.Name().c_str(), rtypenum, rtagnum, 0, stagnum);
            ::G__setnewtype(-1,NULL,0);
		
            return r;
         }
         return -1;
      }

      void CINTTypedefBuilder::Set(const char* name, const char* value) {
         // As the function name indicates
         G__linked_taginfo taginfo;
         taginfo.tagnum  = -1;   // >> need to be pre-initialized to be understood by CINT
         taginfo.tagtype = 'c';
         taginfo.tagname = value;
         G__search_typename2(name, 117, G__get_linked_tagnum(&taginfo),0,-1);
         G__setnewtype(-1,NULL,0);
      }
   }
}
