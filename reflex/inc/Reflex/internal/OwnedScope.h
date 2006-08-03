// @(#)root/reflex:$Name:  $:$Id: OwnedScope.h,v 1.1 2006/08/01 09:14:32 roiser Exp $
// Author: Stefan Roiser 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_OwnedScope
#define ROOT_Reflex_OwnedScope

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Scope.h"
#include "Reflex/internal/ScopeName.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations

      /**
       * @class OwnedScope OwnedScope.h OwnedScope.h
       * @author Stefan Roiser
       * @date 21/07/2006
       * @ingroup Ref
       */
      class RFLX_API OwnedScope : public Scope {

      public:

         /** constructor */
         OwnedScope( const ScopeName * scopeName = 0 )
            : Scope( scopeName ) {}

         
         /** delete info */
         void Delete() {
            fScopeName->DeleteScope();
         }

      }; // class OwnedScope
   
   } // namespace Reflex
} // namespace ROOT


#endif // ROOT_Reflex_OwnedScope
