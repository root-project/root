// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_VariableBuilder
#define Reflex_VariableBuilder

// Include files
#include "Reflex/Reflex.h"
#include "Reflex/Builder/TypeBuilder.h"


namespace Reflex {
// forward declarations

/** @class VariableBuilder VariableBuilder.h Reflex/Builder/VariableBuilder.h
 *  @author Stefan Roiser
 *  @date 6/4/2005
 *  @ingroup RefBld
 */
class RFLX_API VariableBuilder {
public:
   /** constructor */
   VariableBuilder(const char* nam,
                   const Type &typ,
                   size_t offs,
                   unsigned int modifiers = 0);


   /** destructor */
   virtual ~VariableBuilder();


   /**
    * AddProperty will add a property
    * @param  key the property key
    * @param  value the value of the property
    * @return a reference to the building class
    */
   VariableBuilder& AddProperty(const char* key,
                                Any value);
   VariableBuilder& AddProperty(const char* key,
                                const char* value);


   /**
    * ToMember will return the member currently being built
    * @return member currently being built
    */
   Member ToMember();

private:
   /** function member */
   Member fDataMember;

};    // class VariableBuilder


/**
 * @class VariableBuilderImpl VariableBuilder.h Reflex/Builder/VariableBuilder.h
 * @author Stefan Roiser
 * @date 6/4/2005
 * @ingroup RefBld
 */
class RFLX_API VariableBuilderImpl {
public:
   /** constructor */
   VariableBuilderImpl(const char* nam,
                       const Type &typ,
                       size_t offs,
                       unsigned int modifiers = 0);


   /** destructor */
   ~VariableBuilderImpl();


   /** AddProperty will add a property
    * @param  key the property key
    * @param  value the value of the property
    * @return a reference to the building class
    */
   void AddProperty(const char* key,
                    Any value);
   void AddProperty(const char* key,
                    const char* value);


   /**
    * ToMember will return the member currently being built
    * @return member currently being built
    */
   Member ToMember();

private:
   /** member being built */
   Member fDataMember;

};    // class VariableBuilderImpl


/**
 * @class VariableBuilderT VariableBuilder.h Reflex/Builder/VariableBuilder.h
 * @author Stefan Roiser
 * @date 6/4/2005
 * @ingroup RefBld
 */
template <typename D> class VariableBuilderT {
public:
   /** constructor */
   VariableBuilderT(const char* nam,
                    size_t offs,
                    unsigned int modifiers = 0);


   /** destructor */
   virtual ~VariableBuilderT() {}


   /**
    * AddProperty will add a property
    * @param  key the property key
    * @param  value the value of the property
    * @return a reference to the building class
    */
   template <typename P>
   VariableBuilderT& AddProperty(const char* key,
                                 P value);


   /**
    * ToMember will return the member currently being built
    * @return member currently being built
    */
   Member ToMember();

private:
   /** data member builder implementation */
   VariableBuilderImpl fDataMemberBuilderImpl;

};    // class VariableBuilderT


} // namespace Reflex


//-------------------------------------------------------------------------------
template <typename D>
inline Reflex::VariableBuilderT<D>::VariableBuilderT(const char* nam,
                                                     size_t offs,
                                                     unsigned int modifiers)
//-------------------------------------------------------------------------------
   : fDataMemberBuilderImpl(nam,
                            TypeDistiller<D>::Get(),
                            offs,
                            modifiers) {
}


//-------------------------------------------------------------------------------
template <typename D> template <typename P>
inline Reflex::VariableBuilderT<D>&
Reflex::VariableBuilderT<D
>::AddProperty(const char* key,
               P value) {
//-------------------------------------------------------------------------------
   fDataMemberBuilderImpl.AddProperty(key, value);
   return *this;
}


//-------------------------------------------------------------------------------
template <typename D> inline Reflex::Member
Reflex::VariableBuilderT<D
>::ToMember() {
//-------------------------------------------------------------------------------
   return fDataMemberBuilderImpl.ToMember();
}


#endif // Reflex_VariableBuilder
