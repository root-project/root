// @(#)root/reflex:$Name:  $:$Id: ClassTemplateInstance.h,v 1.10 2006/10/30 12:51:33 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
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
#include "Reflex/TypeTemplate.h"
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
          * SimpleName returns the name of the type as a reference. It provides a 
          * simplified but faster generation of a type name. Attention currently it
          * is not guaranteed that Name() and SimpleName() return the same character 
          * layout of a name (ie. spacing, commas, etc. )
          * @param pos will indicate where in the returned reference the requested name starts
          * @param mod The only 'mod' support is SCOPED
          * @return name of type
          */
         virtual const std::string & SimpleName( size_t & pos, 
                                                 unsigned int mod = 0 ) const;

 
         /**
          * TemplateArgumentAt will return a pointer to the nth template argument
          * @param  nth nth template argument
          * @return pointer to nth template argument
          */
         Type TemplateArgumentAt( size_t nth ) const;


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
          * TemplateFamily returns the corresponding TypeTemplate if any
          * @return corresponding TypeTemplate
          */
         TypeTemplate TemplateFamily() const;

      private:

         /** 
          * The template type (family)
          * @label template type
          * @link aggregation
          * @clientCardinality 1
          * @supplierCardinality 1
          */
         TypeTemplate fTemplateFamily;      

      }; // class ClassTemplateInstance
   } // namespace Reflex
} // namespace ROOT


//-------------------------------------------------------------------------------
inline ROOT::Reflex::ClassTemplateInstance::~ClassTemplateInstance() {}
//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::ClassTemplateInstance::TemplateArgumentSize() const {
//-------------------------------------------------------------------------------
   return TemplateInstance::TemplateArgumentSize();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::ClassTemplateInstance::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
   return TemplateInstance::TemplateArgument_Begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::ClassTemplateInstance::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
   return TemplateInstance::TemplateArgument_End();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::ClassTemplateInstance::TemplateArgument_RBegin() const {
//-------------------------------------------------------------------------------
   return TemplateInstance::TemplateArgument_RBegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::ClassTemplateInstance::TemplateArgument_REnd() const {
//-------------------------------------------------------------------------------
   return TemplateInstance::TemplateArgument_REnd();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate ROOT::Reflex::ClassTemplateInstance::TemplateFamily() const {
//-------------------------------------------------------------------------------
   return fTemplateFamily;
}

#endif // ROOT_Reflex_ClassTemplateInstance
