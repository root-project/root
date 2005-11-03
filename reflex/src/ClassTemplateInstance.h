// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_ClassTemplateInstance
#define ROOT_Reflex_ClassTemplateInstance

// Include files
#include "Class.h"
#include "TemplateInstance.h"
#include <string>


namespace ROOT {
  namespace Reflex {


    /**
     * @class ClassTemplateInstance ClassTemplateInstance.h Reflex/ClassTemplateInstance.h
     * @author Stefan Roiser
     * @date 13/1/2004
     * @ingroup Ref
     */
    class ClassTemplateInstance : public Class, public TemplateInstance {

    public:

      /** default constructor */
      ClassTemplateInstance( const char * typ, 
                             size_t size, 
                             const std::type_info & ti, 
                             unsigned int modifiers );
      

      /** destructor */
      virtual ~ClassTemplateInstance();


      /**
       * Name returns the fully qualified Name of the templated class
       * @param  typedefexp expand typedefs or not
       * @return fully qualified Name of templated class
       */
      std::string Name( unsigned int mod = 0 ) const;


      /**
       * TemplateArgumentNth will return a pointer to the nth template argument
       * @param  nth nth template argument
       * @return pointer to nth template argument
       */
      Type TemplateArgumentNth( size_t nth ) const;


      /**
       * templateArgCount will return the number of template arguments
       * @return number of template arguments
       */
      size_t TemplateArgumentCount() const;


      virtual Type_Iterator TemplateArgument_Begin() const;
      virtual Type_Iterator TemplateArgument_End() const;
      virtual Reverse_Type_Iterator TemplateArgument_Rbegin() const;
      virtual Reverse_Type_Iterator TemplateArgument_Rend() const;


      /**
       * TemplateFamily returns the corresponding TypeTemplate if any
       * @return corresponding TypeTemplate
       */
      TypeTemplate TemplateFamily() const;

    private:

      /** 
       * The template TypeNth (family)
       * @label template family
       * @link aggregationByValue
       * @clientCardinality 1
       * @supplierCardinality 0..1
       */
      TypeTemplate fTemplateFamily;      

    }; // class ClassTemplateInstance
  } // namespace Reflex
} // namespace ROOT


//-------------------------------------------------------------------------------
inline ROOT::Reflex::ClassTemplateInstance::~ClassTemplateInstance() {}
//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::ClassTemplateInstance::TemplateArgumentCount() const {
//-------------------------------------------------------------------------------
  return TemplateInstance::TemplateArgumentCount();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::ClassTemplateInstance::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::TemplateArgument_Begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::ClassTemplateInstance::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
  return ScopeBase::TemplateArgument_End();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::ClassTemplateInstance::TemplateArgument_Rbegin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::TemplateArgument_Rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::ClassTemplateInstance::TemplateArgument_Rend() const {
//-------------------------------------------------------------------------------
  return ScopeBase::TemplateArgument_Rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate ROOT::Reflex::ClassTemplateInstance::TemplateFamily() const {
//-------------------------------------------------------------------------------
  return fTemplateFamily;
}

#endif // ROOT_Reflex_ClassTemplateInstance
