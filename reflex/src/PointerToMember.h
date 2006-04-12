// @(#)root/reflex:$Name:  $:$Id: PointerToMember.h,v 1.5 2006/03/06 12:51:46 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_PointerToMember
#define ROOT_Reflex_PointerToMember

// Include files
#include "Reflex/TypeBase.h"
#include "Reflex/Type.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations

      /** 
       * @class PointerToMember PointerToMember.h Reflex/PointerToMember.h
       * @author Stefan Roiser
       * @date   2004-01-28
       * @ingroup Ref
       */
      class PointerToMember : public TypeBase {

      public:

         /** default constructor */
         PointerToMember( const Type &           pointerToMemberType,
                          const std::type_info & ti );


         /** destructor */
         virtual ~PointerToMember() {}


         /**
          * Name will return the fully qualified Name of the pointer At
          * @param  typedefexp expand typedefs or not
          * @return fully qualified Name of pointer At
          */
         std::string Name( unsigned int mod = 0 ) const;


         /** pointerToMemberType will return a pointer to the Type the pointer to
          * MemberAt points to
          * @return pointer to Type of pointer to MemberAt
          */
         Type ToType( unsigned int mod ) const;


         /** static funtion that composes the At Name */
         static std::string BuildTypeName( const Type & pointerToMemberType,
                                           unsigned int mod = SCOPED | QUALIFIED );

      private:

         /**
          * @link aggregationByValue
          * @label pointer to MemberAt At
          * @supplierCardinality 1
          * @clientCardinality 1
          */
         Type fPointerToMemberType;

      }; // class PointerToMember
   } // namespace Reflex
} // namespace ROOT

//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::PointerToMember::ToType( unsigned int /* mod */ ) const {
//-------------------------------------------------------------------------------
   return fPointerToMemberType;
}


#endif // ROOT_Reflex_PointerToMember

