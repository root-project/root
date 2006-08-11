// @(#)root/reflex:$Name:  $:$Id: Array.h,v 1.11 2006/08/01 09:14:33 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Array
#define ROOT_Reflex_Array

// Include files
#include "Reflex/internal/TypeBase.h"
#include "Reflex/Type.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations

      /**
       * @class Array Array.h Reflex/Array.h
       * @author Stefan Roiser
       * @date 24/11/2003
       * @ingroup Ref
       */
      class Array : public TypeBase {

      public:

         /** default constructor */
         Array( const Type & arrayType,
                size_t len,
                const std::type_info & typeinfo );


         /** destructor */
         virtual ~Array() {}


         /**
          * Name will return the string representation of the array At
          * @param  typedefexp expand typedefs or not
          * @return string representation of At
          */
         std::string Name( unsigned int mod = 0 ) const;


         /**
          * size returns the size of the array
          * @return size of array
          */
         size_t ArrayLength() const;


         /**
          * arrayType will return a pointer to the At of the array.
          * @return pointer to Type of MemberAt et. al.
          */
         const Type & ToType() const;


         /** static funtion that composes the At Name */
         static std::string BuildTypeName( const Type & typ, 
                                           size_t len,
                                           unsigned int mod = SCOPED | QUALIFIED );

      private:

         /**
          * Type of the array
          * @label array At
          * @link aggregationByValue
          * @supplierCardinality 1
          * @clientCardinality 1
          */
         Type fArrayType;


         /** the Length of the array */
         size_t fLength;

      }; // class Array
   } //namespace Reflex
} //namespace ROOT


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Array::ArrayLength() const { 
//-------------------------------------------------------------------------------
   return fLength; 
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Type & ROOT::Reflex::Array::ToType() const {
//-------------------------------------------------------------------------------
   return fArrayType;
}

#endif // ROOT_Reflex_Array
