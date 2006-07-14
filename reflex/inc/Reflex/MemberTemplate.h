// @(#)root/reflex:$Name:  $:$Id: MemberTemplate.h,v 1.9 2006/07/03 17:02:38 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_MemberTemplate
#define ROOT_Reflex_MemberTemplate

// Include files
#include "Reflex/Kernel.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations
      class MemberTemplateImpl;
      class Member;

      /** 
       * @class MemberTemplate MemberTemplate.h Reflex/MemberTemplate.h
       * @author Stefan Roiser
       * @date 2005-02-03
       * @ingroup Ref
       */
      class RFLX_API MemberTemplate {

      public:

         /** default constructor */
         MemberTemplate( MemberTemplateImpl * = 0 );


         /** destructor */
         ~MemberTemplate();


         /** 
          * operator bool will return true if the member template is resolved
          * @return true if member template is resolved
          */
         operator bool () const;


         /**
          * TemplateInstanceAt will return the nth template instantion
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
          * Name will return the name of the template family and a list of
          * all currently available instantiations
          * @return template family name with all instantiantion
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


         /**
          * TemplateParameterDefault_Begin returns the begin of the container of template parameter default names
          * @return begin of container of template parameter default names
          */
         StdString_Iterator TemplateParameterDefault_Begin() const;

      
         /**
          * TemplateParameterDefault_End returns the end of the container of template parameter default names
          * @return end of container of template parameter default names
          */
         StdString_Iterator TemplateParameterDefault_End() const;

      
         /**
          * TemplateParameterDefault_RBegin returns the reverse begin of the container of template parameter default names
          * @return reverse begin of container of template parameter default names
          */
         Reverse_StdString_Iterator TemplateParameterDefault_RBegin() const;


         /**
          * TemplateParameterDefault_REnd returns the reverse end of the container of template parameter default names
          * @return reverse end of container of template parameter default names
          */
         Reverse_StdString_Iterator TemplateParameterDefault_REnd() const;


         /**
          * TemplateParameterNameAt will the Name of the nth FunctionParameterAt
          * @param  nth template FunctionParameterAt
          * @return Name of nth template FunctionParameterAt
          */
         std::string TemplateParameterNameAt( size_t nth ) const;


         /**
          * TemplateParameterName_Begin returns the begin of the container of template parameter names
          * @return begin of container of template parameter names
          */
         StdString_Iterator TemplateParameterName_Begin() const;

      
         /**
          * TemplateParameterName_End returns the end of the container of template parameter names
          * @return end of container of template parameter names
          */
         StdString_Iterator TemplateParameterName_End() const;

      
         /**
          * TemplateParameterName_RBegin returns the reverse begin of the container of template parameter names
          * @return reverse begin of container of template parameter names
          */
         Reverse_StdString_Iterator TemplateParameterName_RBegin() const;


         /**
          * TemplateParameterName_REnd returns the reverse end of the container of template parameter names
          * @return reverse end of container of template parameter names
          */
         Reverse_StdString_Iterator TemplateParameterName_REnd() const;

      public:

         /** 
          * AddTemplateInstance adds one TemplateInstanceAt of the template to the local container
          * @param templateInstance the template TemplateInstanceAt
          */
         void AddTemplateInstance( const Member & templateInstance ) const;

      private:

         /** 
          * pointer to the member template implementation
          * @link aggregation
          * @clientCardinality 1
          * @supplierCardinality 1
          */
         MemberTemplateImpl * fMemberTemplateImpl;
      
      }; // class MemberTemplate

   } // namespace ROOT
} // namespace Reflex

#include "Reflex/MemberTemplateImpl.h"

//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate::MemberTemplate( MemberTemplateImpl * memberTemplateImpl )
//------------------------------------------------------------------------------- 
   : fMemberTemplateImpl( memberTemplateImpl ) {}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate::~MemberTemplate() {}
//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate::operator bool () const {
//-------------------------------------------------------------------------------
   if ( fMemberTemplateImpl ) return true;
   return false;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::MemberTemplate::TemplateInstanceSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fMemberTemplateImpl->TemplateInstanceSize();
   return 0;
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::MemberTemplate::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fMemberTemplateImpl->Name( mod );
   return "";
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::MemberTemplate::TemplateParameterSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fMemberTemplateImpl->TemplateParameterSize();
   return 0;
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::MemberTemplate::TemplateParameterDefaultAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fMemberTemplateImpl->TemplateParameterDefaultAt( nth );
   return "";
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberTemplate::TemplateParameterDefault_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fMemberTemplateImpl->TemplateParameterDefault_Begin();
   return Dummy::StdStringCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberTemplate::TemplateParameterDefault_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fMemberTemplateImpl->TemplateParameterDefault_End();
   return Dummy::StdStringCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberTemplate::TemplateParameterDefault_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fMemberTemplateImpl->TemplateParameterDefault_RBegin();
   return Dummy::StdStringCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberTemplate::TemplateParameterDefault_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fMemberTemplateImpl->TemplateParameterDefault_REnd();
   return Dummy::StdStringCont().rend();
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::MemberTemplate::TemplateParameterNameAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fMemberTemplateImpl->TemplateParameterNameAt( nth );
   return "";
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberTemplate::TemplateParameterName_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fMemberTemplateImpl->TemplateParameterName_Begin();
   return Dummy::StdStringCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberTemplate::TemplateParameterName_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fMemberTemplateImpl->TemplateParameterName_End();
   return Dummy::StdStringCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberTemplate::TemplateParameterName_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fMemberTemplateImpl->TemplateParameterName_RBegin();
   return Dummy::StdStringCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberTemplate::TemplateParameterName_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fMemberTemplateImpl->TemplateParameterName_REnd();
   return Dummy::StdStringCont().rend();
}

#endif // ROOT_Reflex_MemberTemplate
