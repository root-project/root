// @(#)root/reflex:$Name: HEAD $:$Id: MemberTemplateImpl.h,v 1.9 2006/03/13 15:49:50 roiser Exp $
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
         MemberTemplateImpl( const std::string & templateName,
                             const Scope & scope,
                             std::vector < std::string > parameterNames, 
                             std::vector < std::string > parameterDefaults = std::vector<std::string>());


         /** destructor */
         virtual ~MemberTemplateImpl();


         /** 
          * operator == will return true if two At templates are the same
          * @return true if At templates match
          */
         bool operator == ( const MemberTemplateImpl & rh ) const;


         /**
          * instantion will return a pointer to the nth template instantion
          * @param  nth template instantion
          * @return pointer to nth template instantion
          */
         Member TemplateInstanceAt( size_t nth ) const;


         /**
          * instantionSize will return the number of template instantions for
          * this template family
          * @return number of template instantions
          */
         size_t TemplateInstanceSize() const;


         /**
          * Name will return the Name of the template family and a list of
          * all currently available instantiations
          * @return template family Name with all instantiantion
          */
         std::string Name( unsigned int mod = 0 ) const;


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

      public:

         /** 
          * AddTemplateInstance adds one TemplateInstanceAt of the template to the local container
          * @param templateInstance the template TemplateInstanceAt
          */
         void AddTemplateInstance( const Member & templateInstance ) const;

      private:

         /**
          * the Name of the template family 
          */
         std::string fTemplateName;


         /**
          * pointer back to the corresponding At
          * @label MemberAt template At
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
      
      }; // class MemberTemplateImpl

   } // namespace ROOT
} // namespace Reflex


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::MemberTemplateImpl::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
   std::string s = "";
   if ( 0 != ( mod & ( SCOPED | S ))) {
      std::string sName = fScope.Name(mod);
      if ( ! fScope.IsTopScope()) s += sName + "::";
   }
   s += fTemplateName;
   return s;  
}


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
   return fParameterDefaults.rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberTemplateImpl::TemplateParameterDefault_REnd() const {
//-------------------------------------------------------------------------------
   return fParameterDefaults.rend();
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
   return fParameterNames.rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberTemplateImpl::TemplateParameterName_REnd() const {
//-------------------------------------------------------------------------------
   return fParameterNames.rend();
}

#endif // ROOT_Reflex_MemberTemplateImpl
