// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_TemplateInstance
#define Reflex_TemplateInstance

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Type.h"

namespace Reflex {
// forward declarations

/**
 * @class TemplateInstance TemplateInstance.h Reflex/TemplateInstance.h
 * @author Stefan Roiser
 * @date   2004-01-28
 * @ingroup Ref
 */
class TemplateInstance {
public:
   /** default constructor */
   TemplateInstance();


   /** constructor */
   TemplateInstance(const std::string& templateArguments);


   /** destructor */
   virtual ~TemplateInstance() {}


   /**
    * Name returns the full Name of the templated collection
    * @param  typedefexp expand typedefs or not
    * @return full Name of template collection
    */
   std::string Name(unsigned int mod = 0) const;


   /**
    * TemplateArgumentAt will return a pointer to the nth template argument
    * @param  nth nth template argument
    * @return pointer to nth template argument
    */
   Type TemplateArgumentAt(size_t nth) const;


   /**
    * templateArgSize will return the number of template arguments
    * @return number of template arguments
    */
   size_t TemplateArgumentSize() const;


   Type_Iterator TemplateArgument_Begin() const;
   Type_Iterator TemplateArgument_End() const;
   Reverse_Type_Iterator TemplateArgument_RBegin() const;
   Reverse_Type_Iterator TemplateArgument_REnd() const;

private:
   /**
    * vector of template arguments
    * @link aggregation
    * @label template arguments
    * @supplierCardinality 1
    * @clientCardinality 1..*
    */
   mutable
   std::vector<Type> fTemplateArguments;

};    // class TemplateInstance

} // namespace Reflex


//-------------------------------------------------------------------------------
inline Reflex::TemplateInstance::TemplateInstance()
//-------------------------------------------------------------------------------
   : fTemplateArguments(std::vector<Type>()) {
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::TemplateInstance::TemplateArgumentAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (nth < fTemplateArguments.size()) {
      return fTemplateArguments[nth];
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::TemplateInstance::TemplateArgumentSize() const {
//-------------------------------------------------------------------------------
   return fTemplateArguments.size();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::TemplateInstance::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
   return fTemplateArguments.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::TemplateInstance::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
   return fTemplateArguments.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::TemplateInstance::TemplateArgument_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Type> &)fTemplateArguments).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::TemplateInstance::TemplateArgument_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Type> &)fTemplateArguments).rend();
}


#endif // Reflex_TemplateInstance
