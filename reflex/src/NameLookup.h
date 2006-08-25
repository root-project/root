// @(#)root/reflex:$Name:  $:$Id: NameLookup.h,v 1.12 2006/08/07 15:02:09 axel Exp $
// Author: Stefan Roiser 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_NameLookup
#define ROOT_Reflex_NameLookup

// Include files
#include <string>
#include <vector>
#include <set>
#include "Reflex/Scope.h"

namespace ROOT {
   namespace Reflex {
    
      // forward declarations
      class Type;
      class Member;
      
      /*
       * point of declaration (3.3.1 [basic.scope.pdecl]) is not taken into account 
       */

      class NameLookup {
      public:

         // 1. Lookup
         static const Type & LookupType( const std::string & nam,
                                         const Scope & current );

         static const Scope & LookupScope( const std::string & nam,
                                           const Scope & current );

         static const Member & LookupMember( const std::string & nam,
                                             const Scope & current );

         static const Member & LookupMemberUnqualified( const std::string & nam,
                                                        const Scope & current );

         static const Member & LookupMemberQualified( const std::string & nam );

         // 2. OverloadResolution
         static const Member & OverloadResultion( const std::string & nam,
                                                  const std::vector< Member > & funcs );
         

         // 3. AccessControl
         static const Type & AccessControl( const Type & typ,
                                            const Scope & current );

      private:
         NameLookup(const std::string& name, const Scope& current);

         template <class T>
         const T & Lookup();
         template <class T>
         const T & LookupInScope();
         template <class T>
         const T & LookupInUnknownScope();

         void FindNextScopePos();

         Scope fCurrentScope; // scope where lookup is carried out
         const std::string fLookupName; // we're looking for a type / member of this name
         bool fPartialSuccess; // found part of the qualified name
         std::set<Scope> fLookedAtUsingDir; // already checked these using directives
         size_t fPosNamePart; // start position in fLookupName of name part to look up
         size_t fPosNamePartLen; // length of name part to look up
      }; // struct  NameLookup

   } //namespace Reflex
} //namespace ROOT


#endif // ROOT_Reflex_NameLookup
