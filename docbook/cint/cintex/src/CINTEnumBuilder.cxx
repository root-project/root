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
#include "CINTEnumBuilder.h"
#include "Api.h"

#include <sstream>

using namespace ROOT::Reflex;
using namespace std;

namespace ROOT { namespace Cintex {

   void CINTEnumBuilder::Setup(const Type& e) {
      // Setup enum info.
      if ( e.IsEnum() )  {
         string name = CintName(e.Name(SCOPED));
         G__ClassInfo cintEnum(name.c_str());
         if (cintEnum.IsValid()) {
            // This enum is already known to CINT.
            // But did we just declare it via an EnumTypeBuilder?
            // If so it won't have any members yet, so only return if
            // the e's first member is already known to CINT:
            if (!e.DataMemberSize(INHERITEDMEMBERS_NO) || cintEnum.NDataMembers())
               return;
         }

         if ( Cintex::Debug() )  {
            cout << "Cintex: Building enum " << name << endl;
         }

         Scope scope = e.DeclaringScope();
         CINTScopeBuilder::Setup( scope );
         bool isCPPMacroEnum = name == "$CPP_MACROS";
         int tagnum = cintEnum.Tagnum();

         if (!isCPPMacroEnum) {
            G__linked_taginfo taginfo;
            taginfo.tagnum  = -1;
            taginfo.tagtype = 'e';
            taginfo.tagname = name.c_str();
            G__get_linked_tagnum(&taginfo);
            tagnum = taginfo.tagnum;
            ::G__tagtable_setup( tagnum, sizeof(int), -1, 0,(char*)NULL, NULL, NULL);
         }

         //--- setup enum values -------
         int isstatic;
         if ( scope.IsTopScope() ) {
            isstatic = -1;
            /* Setting up global variables */
            ::G__resetplocal();
         }
         else {
            string sname = CintName(scope.Name(SCOPED));
            int stagnum = ::G__defined_tagname(sname.c_str(), 2);
            isstatic = -2;
            if( -1 == stagnum ) return;
            ::G__tag_memvar_setup(stagnum);
         }
         for ( size_t i = 0; i < e.DataMemberSize(INHERITEDMEMBERS_NO); i++ ) {
            stringstream s;
            s << e.DataMemberAt(i, INHERITEDMEMBERS_NO).Name() << "=";
            if ( isCPPMacroEnum ) s << (const char*)e.DataMemberAt(i, INHERITEDMEMBERS_NO).Offset();
            else                  s << (int)e.DataMemberAt(i, INHERITEDMEMBERS_NO).Offset();
        
            string item = s.str();
            if ( Cintex::Debug() ) cout << "Cintex:          item " << i << " " << item  <<endl;
            if ( isCPPMacroEnum )
               ::G__memvar_setup((void*)G__PVOID, 'p', 0, 0, -1, -1, -1, 1, item.c_str(), 1, (char*)NULL);
            else
               ::G__memvar_setup((void*)G__PVOID, 'i', 0, 1, tagnum, -1, isstatic, 1, item.c_str(), 0, (char*)NULL);
         }
         if ( scope.IsTopScope() ) {
            ::G__resetglobalenv();
         }
         else {
            ::G__tag_memvar_reset();
         }
      
      }
   }
}}
