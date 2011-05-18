// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_TypedefBuilder
#define Reflex_TypedefBuilder

// Include files
#include "Reflex/Builder/TypeBuilder.h"
#include "Reflex/Type.h"

namespace Reflex {
// forward declarations

/**
 * @class TypedefBuilderImpl TypedefBuilder.h Reflex/Builder/TypedefBuilderImpl.h
 * @author Stefan Roiser
 * @date 14/3/2005
 * @ingroup RefBld
 */
class RFLX_API TypedefBuilderImpl {
public:
   /** constructor */
   TypedefBuilderImpl(const char* typ,
                      const Type &typedefType);


   /** destructor */
   virtual ~TypedefBuilderImpl() {}


   /**
    * AddProperty will add a property to the typedef currently being built
    * @param  key the PropertyNth key
    * @param  value the value of the PropertyNth
    */
   void AddProperty(const char* key,
                    Any value);


   /**
    * AddProperty will add a property to the typedef currently being built
    * @param  key the PropertyNth key
    * @param  value the value of the PropertyNth
    */
   void AddProperty(const char* key,
                    const char* value);


   /*
    * ToType will return the currently produced Type (class)
    * @return the type currently being built
    */
   Type ToType();

private:
   /** the typedef currently being built */
   Type fTypedef;

};    // class TypdefBuilderImpl


/**
 * @class TypedefBuilder TypedefBuilder.h Reflex/Builder/TypedefBuilder.h
 * @author Stefan Roiser
 * @date 30/3/2004
 * @ingroup RefBld
 */
template <typename T>
class TypedefBuilder  {
public:
   /** constructor */
   TypedefBuilder(const char* nam);


   /** destructor */
   virtual ~TypedefBuilder() {}


   /**
    * AddProperty will add a property to the typedef currently being built
    * @param  key the property key
    * @param  value the value of the property
    * @return a reference to the building class
    */
   template <typename P>
   TypedefBuilder& AddProperty(const char* key,
                               P value);


   /*
    * ToType will return the currently produced Type (class)
    * @return the type currently being built
    */
   Type ToType();

private:
   /** the type of the typedef */
   TypedefBuilderImpl fTypedefBuilderImpl;

};    // class TypedefBuilder

} // namespace Reflex

//-------------------------------------------------------------------------------
template <typename T>
inline Reflex::TypedefBuilder<T>::TypedefBuilder(const char* nam)
//-------------------------------------------------------------------------------
   : fTypedefBuilderImpl(nam, TypeDistiller<T>::Get()) {
}


//-------------------------------------------------------------------------------
template <typename T> template <typename P>
inline Reflex::TypedefBuilder<T>&
Reflex::TypedefBuilder<T
>::AddProperty(const char* key,
               P value) {
//-------------------------------------------------------------------------------
   fTypedefBuilderImpl.AddProperty(key, value);
   return *this;
}


//-------------------------------------------------------------------------------
template <typename T> inline Reflex::Type
Reflex::TypedefBuilder<T
>::ToType() {
//-------------------------------------------------------------------------------
   return fTypedefBuilderImpl.ToType();
}


#endif // Reflex_TypedefBuilder
