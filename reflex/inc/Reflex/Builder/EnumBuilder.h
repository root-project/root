// @(#)root/reflex:$Name:  $:$Id: EnumBuilder.h,v 1.2 2005/11/03 15:24:40 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_EnumBuilder
#define ROOT_Reflex_EnumBuilder

// Include files
#include "Reflex/Builder/TypeBuilder.h"
#include "Reflex/Member.h"

namespace ROOT{
   namespace Reflex{

      // forward declarations
      class Enum;

      /**
       * @class EnumBuilderImpl EnumBuilder.h Reflex/Builder/EnumBuilder.h
       * @author Stefan Roiser
       * @date 14/3/2005
       * @ingroup RefBld
       */
      class EnumBuilderImpl {

      public:
      
         /** constructor */
         EnumBuilderImpl( const char * nam,
                          const std::type_info & ti );


         /** destructor */
         virtual ~EnumBuilderImpl() {}


         /** 
          * AddProperty will add a PropertyNth to the PropertyNth stack
          * which will be emptied with the next enum / item build
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          * @return a reference to the building class
          */
         void AddItem ( const char * nam,
                        long value );


         /** 
          * AddProperty will add a PropertyNth 
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          */
         void AddProperty( const char * key,
                           Any value );


         /** 
          * AddProperty will add a PropertyNth 
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          */
         void AddProperty( const char * key,
                           const char * value );

      private:

         /** current enum being built */
         Enum * fEnum;

         /** last added enum item */
         Member fLastMember;

      }; // class EnumBuilderImpl


      /** 
       * @class EnumBuilder EnumBuilder.h Reflex/Builder/EnumBuilder.h
       * @author Stefan Roiser
       * @ingroup RefBld
       * @date 30/3/2004
       */
      template < typename T >
         class EnumBuilder  {

         public:            

         /** constructor */
         EnumBuilder();


         /** constructor */
         EnumBuilder( const char * nam );


         /** destructor */
         virtual ~EnumBuilder() {}


         /** 
          * AddItem add a new item in the enum
          * @param  Name item Name
          * @param  value the value of the item
          * @return a reference to the building class
          */
         EnumBuilder & AddItem( const char * nam, 
                                long value );


         /** 
          * AddProperty will add a PropertyNth to the PropertyNth stack
          * which will be emptied with the next enum / item build
          * @param  key the PropertyNth key
          * @param  value the value of the PropertyNth
          * @return a reference to the building class
          */
         template  < typename P >
            EnumBuilder & AddProperty( const char * key, 
                                       P value );

         private:

         /** the enums and values */
         EnumBuilderImpl fEnumBuilderImpl;

      }; // class EnumBuilder

   } // namespace Reflex
} // namespace ROOT


//-------------------------------------------------------------------------------
template < typename T >
inline ROOT::Reflex::EnumBuilder<T>::EnumBuilder() 
//-------------------------------------------------------------------------------
   : fEnumBuilderImpl( Tools::Demangle( typeid(T) ).c_str(), 
                       typeid(T) ) {}


//-------------------------------------------------------------------------------
template < typename T >
inline ROOT::Reflex::EnumBuilder<T>::EnumBuilder( const char * nam )
//-------------------------------------------------------------------------------
   : fEnumBuilderImpl( nam, 
                       typeid(UnknownType) ) {}


//-------------------------------------------------------------------------------
template < typename T >
inline ROOT::Reflex::EnumBuilder<T> & 
ROOT::Reflex::EnumBuilder<T>::AddItem( const char * nam, 
                                       long value ) {
//-------------------------------------------------------------------------------
   fEnumBuilderImpl.AddItem( nam, value );
   return * this;
}


//-------------------------------------------------------------------------------
template < typename T > template < typename P >
inline ROOT::Reflex::EnumBuilder<T> & 
ROOT::Reflex::EnumBuilder<T>::AddProperty( const char * key, 
                                           P value ) {
//-------------------------------------------------------------------------------
   fEnumBuilderImpl.AddProperty( key, value );
   return * this;
}


#endif // ROOT_Reflex_EnumBuilder
