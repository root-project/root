// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_FunctionMemberTemplateInstance
#define Reflex_FunctionMemberTemplateInstance

// Include files
#include "FunctionMember.h"
#include "TemplateInstance.h"

namespace Reflex {
// forward declarations
class Type;

/**
 * @class FunctionMemberTemplateInstance FunctionMemberTemplateInstance.h Reflex/FunctionMemberTemplateInstance.h
 * @author Stefan Roiser
 * @date 13/1/2004
 * @ingroup Ref
 */
class FunctionMemberTemplateInstance: public FunctionMember,
   public TemplateInstance {
public:
   /** default constructor */
   FunctionMemberTemplateInstance(const char* nam,
                                  const Type& typ,
                                  StubFunction stubFP,
                                  void* stubCtx = 0,
                                  const char* params = 0,
                                  unsigned int modifiers = 0,
                                  const Scope& scop = Scope());


   /** destructor */
   virtual ~FunctionMemberTemplateInstance();


   /**
    * Name returns the fully qualified Name of the
    * templated function
    * @param  typedefexp expand typedefs or not
    * @return fully qualified Name of templated function
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


   virtual Type_Iterator TemplateArgument_Begin() const;
   virtual Type_Iterator TemplateArgument_End() const;
   virtual Reverse_Type_Iterator TemplateArgument_RBegin() const;
   virtual Reverse_Type_Iterator TemplateArgument_REnd() const;


   /**
    * TemplateFamily returns the corresponding MemberTemplate if any
    * @return corresponding MemberTemplate
    */
   MemberTemplate TemplateFamily() const;

private:
   /**
    * The template type (family)
    * @label template family
    * @link aggregation
    * @clientCardinality 1
    * @supplierCardinality 1
    */
   MemberTemplate fTemplateFamily;

};    // class FunctionMemberTemplateInstance
} // namespace Reflex

//-------------------------------------------------------------------------------
inline Reflex::FunctionMemberTemplateInstance::~FunctionMemberTemplateInstance() {
}

//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
inline size_t
Reflex::FunctionMemberTemplateInstance::TemplateArgumentSize() const {
//-------------------------------------------------------------------------------
   return TemplateInstance::TemplateArgumentSize();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::FunctionMemberTemplateInstance::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
   return TemplateInstance::TemplateArgument_Begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::FunctionMemberTemplateInstance::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
   return TemplateInstance::TemplateArgument_End();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::FunctionMemberTemplateInstance::TemplateArgument_RBegin() const {
//-------------------------------------------------------------------------------
   return TemplateInstance::TemplateArgument_RBegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::FunctionMemberTemplateInstance::TemplateArgument_REnd() const {
//-------------------------------------------------------------------------------
   return TemplateInstance::TemplateArgument_REnd();
}


//-------------------------------------------------------------------------------
inline Reflex::MemberTemplate
Reflex::FunctionMemberTemplateInstance::TemplateFamily() const {
//-------------------------------------------------------------------------------
   return fTemplateFamily;
}


#endif // Reflex_FunctionMemberTemplateInstance
