// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_FunctionBuilder
#define Reflex_FunctionBuilder

// Include files
#include "Reflex/Reflex.h"

namespace Reflex {
// forward declarations
class FunctionMember;
class Type;

/**
 * @class FunctionBuilder FunctionBuilder.h Reflex/Builder/FunctionBuilder.h
 * @author Pere Mato
 * @date 1/8/2004
 * @ingroup RefBld
 */
class RFLX_API FunctionBuilder {
public:
   /** constructor */
   FunctionBuilder(const Type &typ,
                   const char* nam,
                   StubFunction stubFP,
                   void* stubCtx,
                   const char* params,
                   unsigned char modifiers);


   /** destructor */
   virtual ~FunctionBuilder();


   /** AddProperty will add a property
    * @param  key the property key
    * @param  value the value of the property
    * @return a reference to the building class
    */
   FunctionBuilder& AddProperty(const char* key,
                                Any value);
   FunctionBuilder& AddProperty(const char* key,
                                const char* value);


   /**
    * ToMember will return the member currently being built
    * @return member currently being built
    */
   Member ToMember();

private:
   /** function member */
   Member fFunction;

};    // class FunctionBuilder


/**
 * @class FunctionBuilderImpl FunctionBuilder.h Reflex/Builder/FunctionBuilder.h
 * @author Pere Mato
 * @date 3/8/2004
 * @ingroup RefBld
 */
class RFLX_API FunctionBuilderImpl {
public:
   /** constructor */
   FunctionBuilderImpl(const char* nam,
                       const Type &typ,
                       StubFunction stubFP,
                       void* stubCtx,
                       const char* params,
                       unsigned char modifiers = 0);


   /** destructor */
   ~FunctionBuilderImpl();


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
   /** function member being built */
   Member fFunction;

};    // class FunctionBuilderImpl


/**
 * @class FunctionBuilderT FunctionBuilder.h Reflex/Builder/FunctionBuilder.h
 * @author Pere Mato
 * @date 1/8/2004
 * @ingroup RefBld
 */
template <typename F> class FunctionBuilderT {
public:
   /** constructor */
   FunctionBuilderT(const char* nam,
                    StubFunction stubFP,
                    void* stubCtx,
                    const char* params,
                    unsigned char modifiers);

   /** destructor */
   virtual ~FunctionBuilderT() {}


   /** AddProperty will add a property
    * @param  key the property key
    * @param  value the value of the property
    * @return a reference to the building class
    */
   template <typename P>
   FunctionBuilderT& AddProperty(const char* key,
                                 P value);


   /**
    * ToMember will return the member currently being built
    * @return member currently being built
    */
   Member ToMember();

private:
   /** function builder implemenation */
   FunctionBuilderImpl fFunctionBuilderImpl;

};    //class FunctionBuilderT

} // namespace Reflex

#include "Reflex/Builder/TypeBuilder.h"

//-------------------------------------------------------------------------------
template <typename  F>
inline Reflex::FunctionBuilderT<F>::FunctionBuilderT(const char* nam,
                                                     StubFunction stubFP,
                                                     void* stubCtx,
                                                     const char* params,
                                                     unsigned char modifiers)
//-------------------------------------------------------------------------------
   : fFunctionBuilderImpl(nam,
                          FunctionDistiller<F>::Get(),
                          stubFP,
                          stubCtx,
                          params,
                          modifiers) {
}


//-------------------------------------------------------------------------------
template <typename F> template <typename P>
inline Reflex::FunctionBuilderT<F>&
Reflex::FunctionBuilderT<F
>::AddProperty(const char* key,
               P value) {
//-------------------------------------------------------------------------------
   fFunctionBuilderImpl.AddProperty(key, value);
   return *this;
}


//-------------------------------------------------------------------------------
template <typename F> inline Reflex::Member
Reflex::FunctionBuilderT<F
>::ToMember() {
//-------------------------------------------------------------------------------
   return fFunctionBuilderImpl.ToMember();
}


#endif // Reflex_FunctionBuilder
