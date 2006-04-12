// @(#)root/reflex:$Name:  $:$Id: Pointer.h,v 1.5 2006/03/06 12:51:46 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Pointer
#define ROOT_Reflex_Pointer

// Include files
#include "Reflex/TypeBase.h"
#include "Reflex/Type.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations

      /**
       * @class Pointer Pointer.h Reflex/Pointer.h
       * @author Stefan Roiser
       * @date 24/11/2003
       * @ingroup Ref
       */
      class Pointer : public TypeBase {
      public:

         /** default constructor */
         Pointer( const Type & pointerType, 
                  const std::type_info & ti );

         /** destructor */
         virtual ~Pointer() {}


         /**
          * Name will return the fully qualified Name of the pointer At
          * @param  typedefexp expand typedefs or not
          * @return fully qualified Name of pointer At
          */
         std::string Name( unsigned int mod = 0 ) const;


         /**
          * At will return a pointer to the At the pointer points to.
          * @return pointer to Type of MemberAt et. al.
          */
         Type ToType( unsigned int mod ) const;


         /** static funtion that composes the typename */
         static std::string BuildTypeName( const Type & pointerType,
                                           unsigned int mod = SCOPED | QUALIFIED );

      private:

         /**
          * pointer to the Type the Pointer points to
          * @label pointer At
          * @link aggregationByValue
          * @supplierCardinality 1
          * @clientCardinality 1
          */
         Type fPointerType;

      }; // class Pointer
   } //namespace Reflex
} //namespace ROOT


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Pointer::ToType( unsigned int /* mod */ ) const {
//-------------------------------------------------------------------------------
   return fPointerType;
}

#endif // ROOT_Reflex_Pointer

