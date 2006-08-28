// @(#)root/reflex:$Name:  $:$Id: TypeTemplateImpl.h,v 1.2 2006/08/11 06:31:59 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_TypeTemplateImpl
#define ROOT_Reflex_TypeTemplateImpl

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Scope.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations
      class Type;
      class TypeTemplate;
      class TypeTemplateName;
      class ClassTemplateInstance;

      /** 
       * @class TypeTemplateImpl TypeTemplateImpl.h Reflex/TypeTemplateImpl.h
       * @author Stefan Roiser
       * @date 2005-02-03
       * @ingroup Ref
       */
      class RFLX_API TypeTemplateImpl {

      public:

         /** default constructor */
         TypeTemplateImpl( const char * templateName,
                           const Scope & scop,
                           std::vector < std::string > parameterNames, 
                           std::vector < std::string > parameterDefaults = std::vector<std::string>());


         /** destructor */
         virtual ~TypeTemplateImpl();


         /** 
          * operator == will return true if two At templates are the same
          * @return true if At templates match
          */
         bool operator == ( const TypeTemplateImpl & rh ) const;


         /**
          * TemplateInstance_Begin returns the begin iterator of the instance container
          * @return the begin iterator of the instance container
          */
         Type_Iterator TemplateInstance_Begin() const;


         /**
          * TemplateInstance_End returns the end iterator of the instance container
          * @return the end iterator of the instance container
          */
         Type_Iterator TemplateInstance_End() const;


         /**
          * TemplateInstance_RBegin returns the rbegin iterator of the instance container
          * @return the rbegin iterator of the instance container
          */
         Reverse_Type_Iterator TemplateInstance_RBegin() const;


         /**
          * TemplateInstance_Rend returns the rend iterator of the instance container
          * @return the rend iterator of the instance container
          */
         Reverse_Type_Iterator TemplateInstance_REnd() const;


         /**
          * instantion will return a pointer to the nth template instantion
          * @param  nth template instantion
          * @return pointer to nth template instantion
          */
         const Type & TemplateInstanceAt( size_t nth ) const;


         /**
          * instantionSize will return the number of template instantions for
          * this template family
          * @return number of template instantions
          */
         size_t TemplateInstanceSize() const;


         /**
          * TemplateParameterSize will return the number of template parameters
          * @return number of template parameters
          */
         size_t TemplateParameterSize() const;


         /**
          * TemplateParameterDefaultAt will return the nth FunctionParameterAt default value as string
          * @param nth template FunctionParameterAt
          * @return default value of nth template FunctionParameterAt
          */
         std::string TemplateParameterDefaultAt( size_t nth ) const;

 
         StdString_Iterator TemplateParameterDefault_Begin() const;
         StdString_Iterator TemplateParameterDefault_End() const;
         Reverse_StdString_Iterator TemplateParameterDefault_RBegin() const;
         Reverse_StdString_Iterator TemplateParameterDefault_REnd() const;


         /**
          * TemplateParameterNameAt will the Name of the nth FunctionParameterAt
          * @param  nth template FunctionParameterAt
          * @return Name of nth template FunctionParameterAt
          */
         std::string TemplateParameterNameAt( size_t nth ) const;


         StdString_Iterator TemplateParameterName_Begin() const;
         StdString_Iterator TemplateParameterName_End() const;
         Reverse_StdString_Iterator TemplateParameterName_RBegin() const;
         Reverse_StdString_Iterator TemplateParameterName_REnd() const;


         /**
          * Return the member template API class corresponding to this member template impl
          * @return corresponding member template
          */
         const TypeTemplate & ThisTypeTemplate() const;

      public:

         /** 
          * AddTemplateInstance adds one TemplateInstanceAt of the template to the local container
          * @param templateInstance the template TemplateInstanceAt
          */
         void AddTemplateInstance( const Type & templateInstance ) const;

      private:

         /**
          * pointer back to the corresponding At
          * @label At template At
          * @clientCardinality 0..*
          * @supplierCardinality 1
          */
         Scope fScope;


         /** 
          * pointer to the class template instances
          * @supplierCardinality 1..*
          * @clientCardinality 0..1
          * @label template instances
          */
         mutable
            std::vector < Type > fTemplateInstances;


         /**
          * container of FunctionParameterAt names
          */
         mutable
            std::vector < std::string > fParameterNames;


         /**
          * FunctionParameterAt default values
          */
         mutable
            std::vector < std::string > fParameterDefaults;

      
         /**
          * number of required template parameters
          */
         size_t fReqParameters;


         /**
          * pointer back to the template name
          */
         TypeTemplateName * fTypeTemplateName;
      
      }; // class TypeTemplateImpl

   } // namespace ROOT
} // namespace Reflex


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeTemplateImpl::TemplateParameterSize() const {
//-------------------------------------------------------------------------------
   return fParameterNames.size();
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::TypeTemplateImpl::TemplateParameterDefaultAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( nth < fParameterDefaults.size() ) return fParameterDefaults[ nth ];
   return "";
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::TypeTemplateImpl::TemplateParameterDefault_Begin() const {
//-------------------------------------------------------------------------------
   return fParameterDefaults.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::TypeTemplateImpl::TemplateParameterDefault_End() const {
//-------------------------------------------------------------------------------
   return fParameterDefaults.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::TypeTemplateImpl::TemplateParameterDefault_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string>&)fParameterDefaults).rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::TypeTemplateImpl::TemplateParameterDefault_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string>&)fParameterDefaults).rend();
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::TypeTemplateImpl::TemplateParameterNameAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( nth < fParameterNames.size() ) return fParameterNames[ nth ];
   return "";
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::TypeTemplateImpl::TemplateParameterName_Begin() const {
//-------------------------------------------------------------------------------
   return fParameterNames.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::TypeTemplateImpl::TemplateParameterName_End() const {
//-------------------------------------------------------------------------------
   return fParameterNames.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::TypeTemplateImpl::TemplateParameterName_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string>&)fParameterNames).rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::TypeTemplateImpl::TemplateParameterName_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string>&)fParameterNames).rend();
}

#endif // ROOT_Reflex_TypeTemplateImpl
