// @(#)root/reflex:$Name:  $:$Id: TemplateInstance.h,v 1.7 2006/08/01 09:14:33 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_TemplateInstance
#define ROOT_Reflex_TemplateInstance

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Type.h"

namespace ROOT {
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
         TemplateInstance( const std::string & templateArguments );


         /** destructor */
         virtual ~TemplateInstance() {}


         /**
          * Name returns the full Name of the templated collection
          * @param  typedefexp expand typedefs or not
          * @return full Name of template collection
          */
         std::string Name( unsigned int mod = 0 ) const;


         /**
          * TemplateArgumentAt will return a pointer to the nth template argument
          * @param  nth nth template argument
          * @return pointer to nth template argument
          */
         const Type & TemplateArgumentAt( size_t nth ) const;


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

         /** vector of template arguments */
         mutable
            std::vector < Type > fTemplateArguments;

      }; // class TemplateInstance

   } // namespace Reflex
} // namespace ROOT


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TemplateInstance::TemplateInstance() 
//-------------------------------------------------------------------------------
   : fTemplateArguments( std::vector<Type>()) {}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Type &
ROOT::Reflex::TemplateInstance::TemplateArgumentAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( nth < fTemplateArguments.size() ) { return fTemplateArguments[ nth ]; }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TemplateInstance::TemplateArgumentSize() const {
//-------------------------------------------------------------------------------
   return fTemplateArguments.size();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::TemplateInstance::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
   return fTemplateArguments.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::TemplateInstance::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
   return fTemplateArguments.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TemplateInstance::TemplateArgument_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Type>&)fTemplateArguments).rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TemplateInstance::TemplateArgument_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Type>&)fTemplateArguments).rend();
}




#endif // ROOT_Reflex_TemplateInstance
