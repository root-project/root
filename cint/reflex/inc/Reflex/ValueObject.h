// @(#)root/reflex:$Id$
// Author: Pere Mato 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_ValueObject
#define Reflex_ValueObject

// Include files
#include "Reflex/Any.h"
#include "Reflex/Object.h"
#include "Reflex/Builder/TypeBuilder.h"


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
      ValueObject();

      /** constructor */
      template <typename T> explicit ValueObject( T& v);

      /** destructor */
      ~ValueObject();

      /** get the actual value */
      template<typename T> const T& Value();

      template<typename T> ValueObject& operator =(const T&);

   private:

      /** the value of the generic object by value */
      Any fValue;

   }; // class ValueObject


//-------------------------------------------------------------------------------
inline ValueObject::ValueObject() {
//------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
template <typename T> 
inline ValueObject::ValueObject( T& v) 
   : Object( GetType<T>(), 0 ), 
     fValue(v)  {
//-------------------------------------------------------------------------------
   if ( TypeOf().IsPointer() ) fAddress = *(void**)fValue.Address();
   else                        fAddress = fValue.Address();
}

template <> 
inline ValueObject::ValueObject( ValueObject& o) 
   : Object( o.TypeOf(), 0 ), 
     fValue(o.fValue)  {
   if ( TypeOf().IsPointer() ) fAddress = *(void**)fValue.Address();
   else                        fAddress = fValue.Address();
}

template <> 
inline ValueObject::ValueObject( const ValueObject& o) 
   : Object( o.TypeOf(), 0 ), 
     fValue(o.fValue)  {
   if ( TypeOf().IsPointer() ) fAddress = *(void**)fValue.Address();
   else                        fAddress = fValue.Address();
}

//-------------------------------------------------------------------------------
template < typename T >
inline ValueObject& ValueObject::operator=( const T& v)  {
//-------------------------------------------------------------------------------
  fValue = Any(v);
  fType = GetType<T>();
  if ( TypeOf().IsPointer() ) fAddress = *(void**)fValue.Address();
  else                        fAddress = fValue.Address();
  return *this;
}


//-------------------------------------------------------------------------------
inline ValueObject::~ValueObject() {
//-------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
template<typename T> 
inline const T& ValueObject::Value() { 
//-------------------------------------------------------------------------------
   return *(T*)fAddress; 
}

} // namespace Reflex

#endif // Reflex_ValueObject
