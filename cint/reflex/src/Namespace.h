// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_Namespace
#define Reflex_Namespace

// Include files
#include "Reflex/Scope.h"

namespace Reflex {
// forward declarations
class Member;
class Scope;
class DictionaryGenerator;


/**
 * @class Namespace Namespace.h Reflex/Namespace.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class Namespace: public ScopeBase {
public:
   /** default constructor */
   Namespace(const char* scop);


   /** destructor */
   virtual ~Namespace();


   /**
    * GenerateDict will produce the dictionary information of this type
    * @param generator a reference to the dictionary generator instance
    */
   virtual void GenerateDict(DictionaryGenerator& generator) const;


   /**
    * function for initialisation of the global namespace
    */
   static const Scope& GlobalScope();


   /**
    * Properties will return a pointer to the PropertyNth list attached
    * to this item
    * @return pointer to PropertyNth list
    */
   virtual PropertyList Properties() const;

private:
   /** constructor for initialisation of the global namespace */
   Namespace();

private:
   /**
    * pointer to the property list
    * @label propertylist
    * @link aggregationByValue
    * @clientCardinality 1
    * @supplierCardinality 1
    */
   OwnedPropertyList fPropertyList;

};    // class Namespace
} //namespace Reflex

//-------------------------------------------------------------------------------
inline Reflex::PropertyList
Reflex::Namespace::Properties() const {
//-------------------------------------------------------------------------------
// Retrieve the namespace's properties.
   return fPropertyList;
}


#endif // Reflex_Namespace
