// @(#)root/reflex:$Name:  $:$Id: $
// Author: Stefan Roiser 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_OwnedBase
#define ROOT_Reflex_OwnedBase

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Base.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations
      class Type;

      /**
       * @class OwnedBase OwnedBase.h OwnedBase.h
       * @author Stefan Roiser
       * @date 21/07/2006
       * @ingroup Ref
       */
      class RFLX_API OwnedBase : public Base {

      public:

         /** constructor */
         OwnedBase( const Type & baseType,
                    OffsetFunction offsetFP,
                    unsigned int modifiers = 0 )
            : Base ( baseType, offsetFP, modifiers ) {}

         
         /** destructor */
         ~OwnedBase() {}

      }; // class OwnedBase
   
   } // namespace Reflex
} // namespace ROOT


#endif // ROOT_Reflex_OwnedBase
