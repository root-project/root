// @(#)root/reflex:$Name:  $:$Id: $
// Author: Stefan Roiser 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_OwnedPropertyList
#define ROOT_Reflex_OwnedPropertyList

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/PropertyList.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations
      class PropertyListImpl;

      /**
       * @class OwnedPropertyList OwnedPropertyList.h OwnedPropertyList.h
       * @author Stefan Roiser
       * @date 21/07/2006
       * @ingroup Ref
       */
      class RFLX_API OwnedPropertyList : public PropertyList {

      public:

         /** constructor */
         OwnedPropertyList( PropertyListImpl * propertyListImpl = 0 )
            : PropertyList( propertyListImpl ) {}

         
         /** destructor */
         ~OwnedPropertyList() {
            delete fPropertyListImpl;
            fPropertyListImpl = 0;
         }

      }; // class OwnedPropertyList
   
   } // namespace Reflex
} // namespace ROOT


#endif // ROOT_Reflex_OwnedPropertyList
