// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_EnumBuilder
#define Reflex_EnumBuilder

// Include files
#include "Reflex/Builder/TypeBuilder.h"
#include "Reflex/Member.h"

namespace Reflex {
// forward declarations
class Enum;

/**
 * @class EnumBuilder EnumBuilder.h Reflex/Builder/EnumBuilder.h
 * @author Stefan Roiser
 * @date 14/3/2005
 * @ingroup RefBld
 */
class RFLX_API EnumBuilder {
public:
   /** constructor */
   EnumBuilder(const char* name,
               const std::type_info & ti,
               unsigned int modifiers = 0);


   /** destructor */
   virtual ~EnumBuilder();


   /**
    * AddProperty will add a PropertyNth to the PropertyNth stack
    * which will be emptied with the next enum / item build
    * @param  key the PropertyNth key
    * @param  value the value of the PropertyNth
    * @return a reference to the building class
    */
   EnumBuilder& AddItem(const char* nam,
                        long value);


   /**
    * AddProperty will add a PropertyNth
    * @param  key the PropertyNth key
    * @param  value the value of the PropertyNth
    */
   EnumBuilder& AddProperty(const char* key,
                            Any value);


   /**
    * AddProperty will add a PropertyNth
    * @param  key the PropertyNth key
    * @param  value the value of the PropertyNth
    */
   EnumBuilder& AddProperty(const char* key,
                            const char* value);


   /*
    * ToType will return the currently produced Type (class)
    * @return the type currently being built
    */
   Type ToType();

private:
   /** current enum being built */
   Enum* fEnum;

   /** last added enum item */
   Member fLastMember;

};    // class EnumBuilder


/**
 * @class EnumBuilder EnumBuilder.h Reflex/Builder/EnumBuilder.h
 * @author Stefan Roiser
 * @ingroup RefBld
 * @date 30/3/2004
 */
template <typename T>
class EnumBuilderT  {
public:
   /** constructor */
   EnumBuilderT(unsigned int modifiers = 0);


   /** constructor */
   EnumBuilderT(const char* nam,
                unsigned int modifiers = 0);


   /** destructor */
   virtual ~EnumBuilderT() {}


   /**
    * AddItem add a new item in the enum
    * @param  Name item Name
    * @param  value the value of the item
    * @return a reference to the building class
    */
   EnumBuilderT& AddItem(const char* nam,
                         long value);


   /**
    * AddProperty will add a PropertyNth to the PropertyNth stack
    * which will be emptied with the next enum / item build
    * @param  key the PropertyNth key
    * @param  value the value of the PropertyNth
    * @return a reference to the building class
    */
   template <typename P>
   EnumBuilderT& AddProperty(const char* key,
                             P value);


   /*
    * ToType will return the currently produced Type (class)
    * @return the type currently being built
    */
   Type ToType();

private:
   /** the enums and values */
   EnumBuilder fEnumBuilderImpl;

};    // class EnumBuilder

} // namespace Reflex

//-------------------------------------------------------------------------------
template <typename T>
inline Reflex::EnumBuilderT<T>::EnumBuilderT(unsigned int modifiers)
//-------------------------------------------------------------------------------
   : fEnumBuilderImpl(Tools::Demangle(typeid(T)).c_str(),
                      typeid(T),
                      modifiers) {
}


//-------------------------------------------------------------------------------
template <typename T>
inline Reflex::EnumBuilderT<T>::EnumBuilderT(const char* nam,
                                             unsigned int modifiers)
//-------------------------------------------------------------------------------
   : fEnumBuilderImpl(nam,
                      typeid(UnknownType),
                      modifiers) {
}


//-------------------------------------------------------------------------------
template <typename T>
inline Reflex::EnumBuilderT<T>&
Reflex::EnumBuilderT<T
>::AddItem(const char* nam,
           long value) {
//-------------------------------------------------------------------------------
   fEnumBuilderImpl.AddItem(nam, value);
   return *this;
}


//-------------------------------------------------------------------------------
template <typename T> template <typename P>
inline Reflex::EnumBuilderT<T>&
Reflex::EnumBuilderT<T
>::AddProperty(const char* key,
               P value) {
//-------------------------------------------------------------------------------
   fEnumBuilderImpl.AddProperty(key, value);
   return *this;
}


//-------------------------------------------------------------------------------
template <typename T> inline Reflex::Type
Reflex::EnumBuilderT<T
>::ToType() {
//-------------------------------------------------------------------------------
   return fEnumBuilderImpl.ToType();
}


#endif // Reflex_EnumBuilder
