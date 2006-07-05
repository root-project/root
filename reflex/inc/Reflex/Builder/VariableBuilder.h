// @(#)root/reflex:$Name: HEAD $:$Id: VariableBuilder.h,v 1.7 2006/03/13 15:49:50 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_VariableBuilder
#define ROOT_Reflex_VariableBuilder

// Include files
#include "Reflex/Reflex.h"
#include "Reflex/Builder/TypeBuilder.h"

namespace ROOT {
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
         VariableBuilder( const char * nam,
                          const Type & typ,
                          size_t offs,
                          unsigned int modifiers = 0 );


         /** destructor */
         virtual ~VariableBuilder();


         /** 
          * AddProperty will add a PropertyNth 
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          * @return a reference to the building class
          */
         VariableBuilder & AddProperty( const char * key, 
                                        Any value );
         VariableBuilder & AddProperty( const char * key,
                                        const char * value );

      private:

         /** function MemberAt */
         Member fDataMember;

      }; // class VariableBuilder


      /** 
       * @class VariableBuilderImpl VariableBuilder.h Reflex/Builder/VariableBuilder.h
       * @author Stefan Roiser
       * @date 6/4/2005
       * @ingroup RefBld
       */
      class RFLX_API VariableBuilderImpl {

      public:

         /** constructor */
         VariableBuilderImpl( const char * nam,
                              const Type & typ,
                              size_t offs,
                              unsigned int modifiers = 0 );


         /** destructor */
         ~VariableBuilderImpl();


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
         Member fDataMember;

      }; // class VariableBuilder


      /** 
       * @class VariableBuilderT VariableBuilder.h Reflex/Builder/VariableBuilder.h
       * @author Stefan Roiser
       * @date 6/4/2005
       * @ingroup RefBld
       */
      template < typename D > class VariableBuilderT {

      public:

         /** constructor */
         VariableBuilderT( const char * nam,
                           size_t offs,
                           unsigned int modifiers = 0 );


         /** destructor */
         virtual ~VariableBuilderT() {}
      

         /** 
          * AddProperty will add a PropertyNth 
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          * @return a reference to the building class
          */
         template < typename P >
            VariableBuilderT & AddProperty( const char * key, P value );

      private:

         /** data MemberAt builder implementation */
         VariableBuilderImpl fDataMemberBuilderImpl;
    
      }; // class VariableBuilderT


   } // namespace Reflex
} // namespace ROOT


//-------------------------------------------------------------------------------
template < typename D > 
inline ROOT::Reflex::VariableBuilderT<D>::VariableBuilderT( const char * nam,
                                                            size_t offs,
                                                            unsigned int modifiers ) 
//-------------------------------------------------------------------------------
   : fDataMemberBuilderImpl( nam,
                             TypeDistiller<D>::Get(),
                             offs,
                             modifiers ) {}


//-------------------------------------------------------------------------------
template < typename D > template < typename P >
inline ROOT::Reflex::VariableBuilderT<D> &
ROOT::Reflex::VariableBuilderT<D>::AddProperty( const char * key, 
                                                P value ) {
//-------------------------------------------------------------------------------
   fDataMemberBuilderImpl.AddProperty(key, value);
   return * this;
}

#endif // ROOT_Reflex_VariableBuilder
