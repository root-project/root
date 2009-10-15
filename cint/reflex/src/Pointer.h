// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_Pointer
#define Reflex_Pointer

// Include files
#include "Reflex/internal/TypeBase.h"
#include "Reflex/Type.h"

namespace Reflex {
// forward declarations

/**
 * @class Pointer Pointer.h Reflex/Pointer.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class Pointer: public TypeBase {
public:
   /** default constructor */
   Pointer(const Type& pointerType,
           const std::type_info& ti);

   /** destructor */
   virtual ~Pointer() {}


   /**
    * Name will return the fully qualified Name of the pointer At
    * @param  typedefexp expand typedefs or not
    * @return fully qualified Name of pointer At
    */
   std::string Name(unsigned int mod = 0) const;


   /**
    * ToType will return a pointer to the type the pointer points to.
    * @return pointer to Type of MemberAt et. al.
    */
   Type ToType() const;


   /** static function that composes the typename */
   static std::string BuildTypeName(const Type& pointerType,
                                    unsigned int mod = SCOPED | QUALIFIED);

private:
   /**
    * The Type the Pointer points to
    * @label pointer type
    * @link aggregation
    * @supplierCardinality 1
    * @clientCardinality 1
    */
   Type fPointerType;

};    // class Pointer
} //namespace Reflex


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::Pointer::ToType() const {
//-------------------------------------------------------------------------------
   return fPointerType;
}


#endif // Reflex_Pointer
