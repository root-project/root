// @(#)root/reflex:$Name:  $:$Id: ValueObject.h,v 1.1 2006/09/08 20:41:29 roiser Exp $
// Author: Pere Mato 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_ValueObject
#define ROOT_Reflex_ValueObject

// Include files
#include "Reflex/Any.h"
#include "Reflex/Object.h"
#include "Reflex/Builder/TypeBuilder.h"


namespace ROOT {
   namespace Reflex {
      
      /** 
       * @class ValueObject ValueObject.h Reflex/ValueObject.h
       * @author Pere Mato
       * @date 01/09/2006
       * @ingroup Ref
       */
      class RFLX_API ValueObject : public Object {
      
      public:

         /** constructor */
         template <typename T> ValueObject( const T& v);
         
         /** constructor */
         ValueObject( const ValueObject& o);
         
         /** destructor */
         ~ValueObject();

         /** get the actual value */
         template<typename T> const T& Value();

      private:

         /** the value of the generic object by value */
         Any fValue;

      }; // class ValueObject
   } // namespace Reflex
} // namespace ROOT


//-------------------------------------------------------------------------------
template <typename T> 
inline ROOT::Reflex::ValueObject::ValueObject( const T& v) 
   : Object( GetType<T>(), 0 ), 
     fValue(v)  {
//-------------------------------------------------------------------------------
   if ( TypeOf().IsPointer() ) fAddress = *(void**)fValue.Address();
   else                        fAddress = fValue.Address();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::ValueObject::ValueObject( const ValueObject& o) 
   : Object( o.TypeOf(), 0 ), 
     fValue(o.fValue)  {
//-------------------------------------------------------------------------------
   if ( TypeOf().IsPointer() ) fAddress = *(void**)fValue.Address();
   else                        fAddress = fValue.Address();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::ValueObject::~ValueObject() {
//-------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
template<typename T> 
inline const T& ROOT::Reflex::ValueObject::Value() { 
//-------------------------------------------------------------------------------
   return *(T*)fAddress; 
}


#endif // ROOT_Reflex_ValueObject
