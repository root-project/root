// @(#)root/reflex:$Name:  $:$Id: TypeTemplate.h,v 1.6 2006/03/13 15:49:50 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_TypeTemplate
#define ROOT_Reflex_TypeTemplate

// Include files
#include "Reflex/Kernel.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations
      class Type;
      class TypeTemplateImpl;
      class ClassTemplateInstance;

      /** 
       * @class TypeTemplate TypeTemplate.h Reflex/TypeTemplate.h
       * @author Stefan Roiser
       * @date 2005-02-03
       * @ingroup Ref
       */
      class RFLX_API TypeTemplate {

      public:

         /** default constructor */
         TypeTemplate( TypeTemplateImpl * tti = 0 );


         /** destructor */
         ~TypeTemplate();


         /** 
          * operator bool will return true if the type template is resolved
          * @return true if type template is resolved
          */
         operator bool () const;


         /** 
          * operator == will return true if two type templates are the same
          * @return true if type templates match
          */
         bool operator == ( const TypeTemplate & rh ) const;


         /**
          * TemplateInstanceAt will return a pointer to the nth template instantion
          * @param  nth template instantion
          * @return pointer to nth template instantion
          */
         Type TemplateInstanceAt( size_t nth ) const;


         /**
          * TemplateInstanceSize will return the number of template instantions for
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
          * TemplateParameterDefaultAt will return the nth TemplateParameterAt default value as string
          * @param nth template TemplateParameterAt
          * @return default value of nth template TemplateParameterAt
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
          * TemplateParameterNameAt will the Name of the nth TemplateParameterAt
          * @param  nth template TemplateParameterAt
          * @return Name of nth template TemplateParameterAt
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
         void AddTemplateInstance( const Type & templateInstance ) const;

      private:

         /** 
          * pointer to the type template implementation
          * @link aggregation
          * @supplierCardinality 0..1
          * @clientCardinality 1
          * @label type template impl
          */
         TypeTemplateImpl * fTypeTemplateImpl;
      
      }; // class TypeTemplate

   } // namespace ROOT
} // namespace Reflex

#include "Reflex/TypeTemplateImpl.h"

//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate::TypeTemplate( TypeTemplateImpl * tti )
//------------------------------------------------------------------------------- 
   : fTypeTemplateImpl( tti ) {}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate::~TypeTemplate() {}
//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate::operator bool () const {
//-------------------------------------------------------------------------------
   if ( fTypeTemplateImpl ) return true;
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeTemplate::operator == ( const TypeTemplate & rh ) const {
//-------------------------------------------------------------------------------
   if ((*this) && (rh)) return ( fTypeTemplateImpl == rh.fTypeTemplateImpl );
   return false;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeTemplate::TemplateInstanceSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateImpl->TemplateInstanceSize();
   return 0;
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::TypeTemplate::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateImpl->Name( mod );
   return "";
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeTemplate::TemplateParameterSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateImpl->TemplateParameterSize();
   return 0;
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::TypeTemplate::TemplateParameterDefaultAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateImpl->TemplateParameterDefaultAt( nth );
   return "";
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::TypeTemplate::TemplateParameterDefault_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateImpl->TemplateParameterDefault_Begin();
   return Dummy::sStdStringCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::TypeTemplate::TemplateParameterDefault_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateImpl->TemplateParameterDefault_End();
   return Dummy::sStdStringCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::TypeTemplate::TemplateParameterDefault_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateImpl->TemplateParameterDefault_RBegin();
   return Dummy::sStdStringCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::TypeTemplate::TemplateParameterDefault_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateImpl->TemplateParameterDefault_REnd();
   return Dummy::sStdStringCont().rend();
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::TypeTemplate::TemplateParameterNameAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateImpl->TemplateParameterNameAt( nth );
   return "";
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::TypeTemplate::TemplateParameterName_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateImpl->TemplateParameterName_Begin();
   return Dummy::sStdStringCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::TypeTemplate::TemplateParameterName_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateImpl->TemplateParameterName_End();
   return Dummy::sStdStringCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::TypeTemplate::TemplateParameterName_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateImpl->TemplateParameterName_RBegin();
   return Dummy::sStdStringCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::TypeTemplate::TemplateParameterName_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeTemplateImpl->TemplateParameterName_REnd();
   return Dummy::sStdStringCont().rend();
}

#endif // ROOT_Reflex_TypeTemplate
