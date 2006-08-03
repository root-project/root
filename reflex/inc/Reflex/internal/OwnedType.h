// @(#)root/reflex:$Name:  $:$Id: OwnedType.h,v 1.1 2006/08/01 09:14:32 roiser Exp $
// Author: Stefan Roiser 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_OwnedType
#define ROOT_Reflex_OwnedType

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Type.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations
      class TypeName;

      /**
       * @class OwnedType OwnedType.h OwnedType.h
       * @author Stefan Roiser
       * @date 21/07/2006
       * @ingroup Ref
       */
      class RFLX_API OwnedType : public Type {

      public:

         /** constructor */
         OwnedType( const TypeName * typeName = 0,
                    unsigned int modifiers = 0 )
            : Type( typeName, modifiers ) {}

         
         /** delete info */
         void Delete() {
            fTypeName->DeleteType();
         }

      }; // class OwnedType
   
   } // namespace Reflex
} // namespace ROOT


#endif // ROOT_Reflex_OwnedType
