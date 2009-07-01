// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_Enum
#define Reflex_Enum

// Include files
#include "ScopedType.h"

namespace Reflex {
// forward declarations
class DictionaryGenerator;


/**
 * @class Enum Enum.h Reflex/Enum.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class Enum: public ScopedType {
public:
   /** default constructor */
   Enum(const char* enumType,
        const std::type_info& ti,
        unsigned int modifiers);


   /** destructor */
   virtual ~Enum();


   /**
    * GenerateDict will produce the dictionary information of this type
    * @param generator a reference to the dictionary generator instance
    */
   virtual void GenerateDict(DictionaryGenerator& generator) const;

private:
};    // class Enum
} //namespace Reflex

#endif // Reflex_Enum
