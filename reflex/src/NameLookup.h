// @(#)root/reflex:$Name: HEAD $:$Id: NameLookup.h,v 1.3 2006/06/28 08:33:12 roiser Exp $
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

namespace ROOT {
   namespace Reflex {
    
      // forward declarations
      class Type;
      class Scope;
      class Member;
      
      /*
       * point of declaration (3.3.1 [basic.scope.pdecl]) is not taken into account 
       */

      namespace NameLookup {

         // 1. Lookup
         Type LookupType( const std::string & nam,
                          const Scope & current );

         
         Type LookupTypeQualified( const std::string & nam );

         
         Type LookupTypeUnqualified( const std::string & nam,
                                     const Scope & current );


         Scope LookupScope( const std::string & nam,
                            const Scope & current );


         Scope LookupScopeQualified( const std::string & nam );


         Scope LookupScopeUnqualified( const std::string & nam,
                                       const Scope & current );


         Member LookupMember( const std::string & nam,
                              const Scope & current );


         Member LookupMemberQualified( const std::string & nam );

         
         Member LookupMemberUnqualified( const std::string & nam,
                                         const Scope & current );

         


         // 2. OverloadResolution
         Member OverloadResultion( const std::string & nam,
                                   const std::vector< Member > & funcs );
                                   

         // 3. AccessControl
         bool AccessControl( const Type & typ,
                             const Scope & current );


      } // namespace NameLookup
   } //namespace Reflex
} //namespace ROOT


#endif // ROOT_Reflex_NameLookup
