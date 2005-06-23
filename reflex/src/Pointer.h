// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
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
               const std::type_info & TypeInfo );

      /** destructor */
      virtual ~Pointer() {}


      /**
       * Name will return the fully qualified Name of the pointer TypeNth
       * @param  typedefexp expand typedefs or not
       * @return fully qualified Name of pointer TypeNth
       */
      std::string Name( unsigned int mod = 0 ) const;


      /**
       * TypeNth will return a pointer to the TypeNth the pointer points to.
       * @return pointer to Type of MemberNth et. al.
       */
      Type ToType() const;


      /** static funtion that composes the TypeNth Name */
      static std::string BuildTypeName( const Type & pointerType,
                                        unsigned int mod = SCOPED | QUALIFIED );

    private:

      /**
       * pointer to the Type the Pointer points to
       * @label pointer TypeNth
       * @link aggregationByValue
       * @supplierCardinality 1
       * @clientCardinality 1
       */
      Type fPointerType;

    }; // class Pointer
  } //namespace Reflex
} //namespace ROOT


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Pointer::ToType() const {
//-------------------------------------------------------------------------------
  return fPointerType;
}

#endif // ROOT_Reflex_Pointer

