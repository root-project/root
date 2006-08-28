// @(#)root/reflex:$Name:  $:$Id: MemberTemplateImpl.h,v 1.3 2006/08/11 06:31:59 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_MemberTemplateImpl
#define ROOT_Reflex_MemberTemplateImpl  

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Scope.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations
      class Member;
      class Scope;
      class MemberTemplate;
      class MemberTemplateName;
      class FunctionMemberTemplateInstance;

      /** 
       * @class MemberTemplateImpl MemberTemplateImpl.h Reflex/MemberTemplateImpl.h
       * @author Stefan Roiser
       * @date 2005-02-03
       * @ingroup Ref
       */
      class RFLX_API MemberTemplateImpl {

      public:

         /** default constructor */
         MemberTemplateImpl( const char * templateName,
                             const Scope & scope,
                             const std::vector < std::string > & parameterNames, 
                             const std::vector < std::string > & parameterDefaults = std::vector<std::string>());


         /** destructor */
         virtual ~MemberTemplateImpl();


         /** 
          * operator == will return true if two At templates are the same
          * @return true if At templates match
          */
         bool operator == ( const MemberTemplateImpl & rh ) const;


         /**
          * TemplateInstance_Begin returns the begin iterator of the instance container
          * @return the begin iterator of the instance container
          */
         Member_Iterator TemplateInstance_Begin() const;


         /**
          * TemplateInstance_End returns the end iterator of the instance container
          * @return the end iterator of the instance container
          */
         Member_Iterator TemplateInstance_End() const;


         /**
          * TemplateInstance_RBegin returns the rbegin iterator of the instance container
          * @return the rbegin iterator of the instance container
          */
         Reverse_Member_Iterator TemplateInstance_RBegin() const;


         /**
          * TemplateInstance_Rend returns the rend iterator of the instance container
          * @return the rend iterator of the instance container
          */
         Reverse_Member_Iterator TemplateInstance_REnd() const;


         /**
          * instantion will return a pointer to the nth template instantion
          * @param  nth template instantion
          * @return pointer to nth template instantion
          */
         const Member & TemplateInstanceAt( size_t nth ) const;


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
         const MemberTemplate & ThisMemberTemplate() const;

      public:

         /** 
          * AddTemplateInstance adds one TemplateInstanceAt of the template to the local container
          * @param templateInstance the template TemplateInstanceAt
          */
         void AddTemplateInstance( const Member & templateInstance ) const;

      private:

         /**
          * declaring scope of this member template
          * @label member template scope
          * @clientCardinality 0..*
          * @supplierCardinality 1
          */
         Scope fScope;


         /** 
          * pointer to the class template instances
          * @clientCardinality 0..1
          * @label template instances
          */
         mutable
            std::vector < Member > fTemplateInstances;


         /**
          * container of function parameter template names
          */
         mutable
            std::vector < std::string > fParameterNames;


         /**
          * function  parameter template default values
          */
         mutable
            std::vector < std::string > fParameterDefaults;

      
         /**
          * number of required template parameters
          */
         size_t fReqParameters;

         
         /**
          * pointer back to the member template name
          */
         MemberTemplateName * fMemberTemplateName;


      }; // class MemberTemplateImpl

   } // namespace ROOT
} // namespace Reflex


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::MemberTemplateImpl::TemplateParameterSize() const {
//-------------------------------------------------------------------------------
   return fParameterNames.size();
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::MemberTemplateImpl::TemplateParameterDefaultAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( nth < fParameterDefaults.size() ) return fParameterDefaults[ nth ];
   return "";
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberTemplateImpl::TemplateParameterDefault_Begin() const {
//-------------------------------------------------------------------------------
   return fParameterDefaults.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberTemplateImpl::TemplateParameterDefault_End() const {
//-------------------------------------------------------------------------------
   return fParameterDefaults.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberTemplateImpl::TemplateParameterDefault_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string>&)fParameterDefaults).rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberTemplateImpl::TemplateParameterDefault_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string>&)fParameterDefaults).rend();
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::MemberTemplateImpl::TemplateParameterNameAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( nth < fParameterNames.size() ) return fParameterNames[ nth ];
   return "";
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberTemplateImpl::TemplateParameterName_Begin() const {
//-------------------------------------------------------------------------------
   return fParameterNames.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberTemplateImpl::TemplateParameterName_End() const {
//-------------------------------------------------------------------------------
   return fParameterNames.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberTemplateImpl::TemplateParameterName_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string>&)fParameterNames).rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberTemplateImpl::TemplateParameterName_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string>&)fParameterNames).rend();
}

#endif // ROOT_Reflex_MemberTemplateImpl
