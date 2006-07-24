// @(#)root/reflex:$Name: HEAD $:$Id: NameLookup.h,v 1.5 2006/07/13 14:45:59 roiser Exp $
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


namespace ROOT {
   namespace Reflex {
    
      // forward declarations
      class Type;
      class Scope;
      class Member;
      
      /*
       * point of declaration (3.3.1 [basic.scope.pdecl]) is not taken into account 
       */

      struct NameLookup {

         // 1. Lookup
         static Type LookupType( const std::string & nam,
                                 const Scope & current );

         
         static Type LookupTypeQualified( const std::string & nam );

         
         static Type LookupTypeUnqualified( const std::string & nam,
                                            const Scope & current );

         static Type LookupTypeInScope( const std::string & nam, 
                                        const Scope & current,
                                        bool &partial_success,
                                        std::set<Scope> & lookedAtUsingDir,
                                        size_t pos_subscope = 0,
                                        size_t pos_scope_end = std::string::npos );
  
           
         static Type LookupTypeInUnknownScope( const std::string & nam,
                                               const Scope & current );

         static Scope LookupScope( const std::string & nam,
                                   const Scope & current );


         static Scope LookupScopeQualified( const std::string & nam );


         static Scope LookupScopeUnqualified( const std::string & nam,
                                              const Scope & current );


         static Member LookupMember( const std::string & nam,
                                     const Scope & current );


         static Member LookupMemberQualified( const std::string & nam );

         
         static Member LookupMemberUnqualified( const std::string & nam,
                                                const Scope & current );

         


         // 2. OverloadResolution
         static Member OverloadResultion( const std::string & nam,
                                          const std::vector< Member > & funcs );
                                   

         // 3. AccessControl
         static bool AccessControl( const Type & typ,
                                    const Scope & current );

         private:

      }; // struct  NameLookup

   } //namespace Reflex
} //namespace ROOT


#endif // ROOT_Reflex_NameLookup
