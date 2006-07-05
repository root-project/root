// @(#)root/reflex:$Name: HEAD $:$Id: FunctionBuilder.h,v 1.7 2006/03/13 15:49:50 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_FunctionBuilder
#define ROOT_Reflex_FunctionBuilder

// Include files
#include "Reflex/Reflex.h"

namespace ROOT {
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
         FunctionBuilder( const Type & typ,
                          const char * nam,
                          StubFunction stubFP,
                          void * stubCtx,
                          const char * params, 
                          unsigned char modifiers );
      

         /** destructor */
         virtual ~FunctionBuilder();


         /** AddProperty will add a PropertyNth 
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          * @return a reference to the building class
          */
         FunctionBuilder & AddProperty( const char * key, 
                                        Any value );
         FunctionBuilder & AddProperty( const char * key,
                                        const char * value );

      private:

         /** function MemberAt */
         Member fFunction;

      }; // class FunctionBuilder
    

      /** 
       * @class FunctionBuilderImpl FunctionBuilder.h Reflex/Builder/FunctionBuilder.h
       * @author Pere Mato
       * @date 3/8/2004
       * @ingroup RefBld
       */
      class RFLX_API FunctionBuilderImpl {
    
      public:
      
         /** constructor */
         FunctionBuilderImpl( const char * nam, 
                              const Type & typ,
                              StubFunction stubFP,
                              void * stubCtx,
                              const char * params, 
                              unsigned char modifiers = 0 );
                          

         /** destructor */
         ~FunctionBuilderImpl();


         /** AddProperty will add a PropertyNth 
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          * @return a reference to the building class
          */
         void AddProperty( const char * key, 
                           Any value );
         void AddProperty( const char * key, 
                           const char * value );

         /** string containing the union information */
         Member fFunction;

      }; // class FunctionBuilderImpl


      /** 
       * @class FunctionBuilderT FunctionBuilder.h Reflex/Builder/FunctionBuilder.h
       * @author Pere Mato
       * @date 1/8/2004
       * @ingroup RefBld
       */
      template < typename F > class FunctionBuilderT {

      public:    

         /** constructor */
         FunctionBuilderT( const char * nam,
                           StubFunction stubFP,
                           void * stubCtx,
                           const char * params, 
                           unsigned char modifiers );
      
         /** destructor */
         virtual ~FunctionBuilderT() {}

      
         /** AddProperty will add a PropertyNth 
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          * @return a reference to the building class
          */
         template < typename P >
            FunctionBuilderT & AddProperty( const char * key, P value );


      private:

         /** function builder implemenation */
         FunctionBuilderImpl fFunctionBuilderImpl;

      }; //class FunctionBuilderT

   } // namespace Reflex
} // namespace ROOT

#include "Reflex/Builder/TypeBuilder.h"

//-------------------------------------------------------------------------------
template < typename  F > 
inline ROOT::Reflex::FunctionBuilderT<F>::FunctionBuilderT( const char * nam, 
                                                            StubFunction stubFP,
                                                            void * stubCtx,
                                                            const char * params, 
                                                            unsigned char modifiers )
//-------------------------------------------------------------------------------
   : fFunctionBuilderImpl( nam,
                           FunctionDistiller<F>::Get(),
                           stubFP,
                           stubCtx,
                           params,
                           modifiers ) { }
      

//-------------------------------------------------------------------------------
template <  typename F > template < typename P >
inline ROOT::Reflex::FunctionBuilderT<F> & 
ROOT::Reflex::FunctionBuilderT<F>::AddProperty( const char * key, 
                                                P value )
//-------------------------------------------------------------------------------
{ 
   fFunctionBuilderImpl.AddProperty(key , value);
   return * this;
}

#endif // ROOT_Reflex_FunctionBuilder
