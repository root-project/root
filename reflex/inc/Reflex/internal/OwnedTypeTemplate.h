// @(#)root/reflex:$Name:  $:$Id: OwnedTypeTemplate.h,v 1.1 2006/08/01 09:14:32 roiser Exp $
// Author: Stefan Roiser 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_OwnedTypeTemplate
#define ROOT_Reflex_OwnedTypeTemplate

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/TypeTemplate.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations
      class TypeTemplateImpl;

      /**
       * @class OwnedTypeTemplate OwnedTypeTemplate.h OwnedTypeTemplate.h
       * @author Stefan Roiser
       * @date 21/07/2006
       * @ingroup Ref
       */
      class RFLX_API OwnedTypeTemplate : public TypeTemplate {

      public:

         /** constructor */
         OwnedTypeTemplate( const TypeTemplateImpl * typeTemplateImpl = 0 )
            : TypeTemplate( typeTemplateImpl ) {}

         
         /** take ownership */
         OwnedTypeTemplate( const TypeTemplate & rh )
            : TypeTemplate( rh ) {}

         
         /** delete info */
         void Delete() {
            delete fTypeTemplateImpl;
            fTypeTemplateImpl = 0;
         }

      }; // class OwnedTypeTemplate
   
   } // namespace Reflex
} // namespace ROOT


#endif // ROOT_Reflex_OwnedTypeTemplate
