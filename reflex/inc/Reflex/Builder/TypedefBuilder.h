// @(#)root/reflex:$Name: HEAD $:$Id: TypedefBuilder.h,v 1.6 2006/03/13 15:49:50 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_TypedefBuilder
#define ROOT_Reflex_TypedefBuilder

// Include files
#include "Reflex/Builder/TypeBuilder.h"

namespace ROOT{
   namespace Reflex{

      // forward declarations
      class Typedef;

      /**
       * @class TypedefBuilderImpl TypedefBuilder.h Reflex/Builder/TypedefBuilderImpl.h
       * @author Stefan Roiser
       * @date 14/3/2005
       * @ingroup RefBld
       */
      class RFLX_API TypedefBuilderImpl {
      
      public:
      
         /** constructor */
         TypedefBuilderImpl( const char * typ,
                             const Type & typedefType );


         /** destructor */
         virtual ~TypedefBuilderImpl() {}

      
         /** 
          * AddProperty will add a PropertyNth to the typedef currently being built
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          */
         void AddProperty( const char * key,
                           Any value );


         /** 
          * AddProperty will add a PropertyNth to the typedef currently being built
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          */
         void AddProperty( const char * key,
                           const char * value );

      private:

         /** the typedef currently being built */
         Typedef * fTypedef;

      }; // class TypdefBuilderImpl


      /** 
       * @class TypedefBuilder TypedefBuilder.h Reflex/Builder/TypedefBuilder.h
       * @author Stefan Roiser
       * @date 30/3/2004
       * @ingroup RefBld
       */
      template < typename T >
         class TypedefBuilder  {

         public:            

         /** constructor */
         TypedefBuilder(const char * nam);


         /** destructor */
         virtual ~TypedefBuilder() {}


         /** 
          * AddProperty will add a PropertyNth to the typedef currently being built
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          * @return a reference to the building class
          */
         template < typename P >
            TypedefBuilder & AddProperty( const char * key, 
                                          P value );

         private:

         /** the At of the typedef */
         TypedefBuilderImpl fTypedefBuilderImpl;

      }; // class TypedefBuilder

   } // namespace Reflex
} // namespace ROOT

//-------------------------------------------------------------------------------
template < typename T >
inline ROOT::Reflex::TypedefBuilder<T>::TypedefBuilder( const char * nam ) 
//-------------------------------------------------------------------------------
   : fTypedefBuilderImpl( nam, TypeDistiller<T>::Get()) {}


//-------------------------------------------------------------------------------
template < typename T > template < typename P >
inline ROOT::Reflex::TypedefBuilder<T> & 
ROOT::Reflex::TypedefBuilder<T>::AddProperty( const char * key, 
                                              P value ) {
//-------------------------------------------------------------------------------
   fTypedefBuilderImpl.AddProperty( key, value );
   return * this;
}

#endif // ROOT_Reflex_TypedefBuilder

