// @(#)root/cintex:$Id$
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

int CINTTypedefBuilder::Setup(const Type& t)
{
   // Additional setup for this typedef in CINT.

   if (!t.IsTypedef())  {
      return -1;
   }
   std::string nam = CintName(t.Name(SCOPED));
   Scope scope = t.DeclaringScope();
   CINTScopeBuilder::Setup(scope);
   Type rt = t;
   for (; rt.IsTypedef(); rt = rt.ToType()) {}
   Type indir_type = rt;
   for (; indir_type.IsPointer(); indir_type = indir_type.ToType()) {}
   Scope rscope = indir_type.DeclaringScope();
   if (scope != rscope) {
      if (rscope) {
         CINTScopeBuilder::Setup(rscope);
      }
      else {
         rscope = Scope::ByName(Tools::GetScopeName(indir_type.Name(SCOPED)));
         CINTScopeBuilder::Setup(rscope);
      }
   }
   if (G__defined_typename(nam.c_str()) != -1) {
      return -1;
   }
   if (Cintex::Debug())  {
      std::cout << "Cintex: Building typedef " << nam << std::endl;
   }
   int rtypenum;
   int rtagnum;
   CintType(rt, rtypenum, rtagnum);
   int stagnum = -1;
   if (!scope.IsTopScope()) {
      stagnum = G__defined_tagname(CintName(scope.Name(SCOPED)).c_str(), 1);
   }
   int r = G__search_typename2(t.Name().c_str(), rtypenum, rtagnum, 0, stagnum);
   G__setnewtype(-1, 0, 0);
   return r;
}

void CINTTypedefBuilder::Set(const char* name, const char* value)
{
   // Set the tagnum for this typedef.

   G__linked_taginfo taginfo;
   taginfo.tagnum = -1;
   taginfo.tagtype = 'c';
   taginfo.tagname = value;
   G__search_typename2(name, 117, G__get_linked_tagnum(&taginfo), 0, -1);
   G__setnewtype(-1, NULL, 0);
}

} // namespace Cintex
} // namespace ROOT
