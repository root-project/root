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
class RFLX_API ValueObject: public Object {
public:
   /** constructor */
   ValueObject();

   /** constructor */
   template <typename T>
   static ValueObject Create(const T& v);

   /** constructor */
   ValueObject(const ValueObject &o);

   /** destructor */
   ~ValueObject();

   /** assignment op */
   ValueObject& operator=(const ValueObject &o);

   /** get the actual value */
   template <typename T> const T& Value();

   template <typename T> ValueObject& Assign(const T&);

private:
   /** the value of the generic object by value */
   Any fValue;

};    // class ValueObject
} // namespace Reflex


//-------------------------------------------------------------------------------
inline Reflex::ValueObject::ValueObject() {
//-------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
template <typename T>
inline Reflex::ValueObject
Reflex::ValueObject::Create(const T& v) {
//-------------------------------------------------------------------------------
   ValueObject ret;
   ret.Assign(v);
   return ret;
}


//-------------------------------------------------------------------------------
inline Reflex::ValueObject::ValueObject(const ValueObject& o):
   Object(o.TypeOf(), 0),
   fValue(o.fValue) {
//-------------------------------------------------------------------------------
   if (TypeOf().IsPointer()) {
      fAddress = *(void**) fValue.Address();
   } else { fAddress = fValue.Address(); }
}


//-------------------------------------------------------------------------------
template <typename T>
inline Reflex::ValueObject&
Reflex::ValueObject::Assign(const T& v) {
//-------------------------------------------------------------------------------
   fValue = Any(v);
   fType = GetType<T>();

   if (TypeOf().IsPointer()) {
      fAddress = *(void**) fValue.Address();
   } else { fAddress = fValue.Address(); }
   return *this;
}


//-------------------------------------------------------------------------------
inline Reflex::ValueObject::~ValueObject() {
//-------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
inline
Reflex::ValueObject&
Reflex::ValueObject::operator=(const ValueObject& o) {
//-------------------------------------------------------------------------------
   if (&o != this) {
      Object::operator=(Object(o.TypeOf(), 0));
      fValue = o.fValue;
      if (TypeOf().IsPointer()) {
         fAddress = *(void**) fValue.Address();
      } else { fAddress = fValue.Address(); }
   }
   return *this;
}


//-------------------------------------------------------------------------------
template <typename T>
inline const T&
Reflex::ValueObject::Value() {
//-------------------------------------------------------------------------------
   return *(T*) fAddress;
}


#endif // Reflex_ValueObject
