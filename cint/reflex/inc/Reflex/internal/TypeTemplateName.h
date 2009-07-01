// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_TypeTemplateName
#define Reflex_TypeTemplateName

// Include files
#include "Reflex/Kernel.h"


namespace Reflex {
// forward declarations
class TypeTemplate;
class TypeTemplateImpl;

/**
 * @class TypeTemplateName TypeTemplateName.h Reflex/internal/TypeTemplateName.h
 * @author Stefan Roiser
 * @date 8/8/2006
 * @ingroup Ref
 */
class RFLX_API TypeTemplateName {
   friend class TypeTemplate;
   friend class TypeTemplateImpl;

public:
   /** constructor */
   TypeTemplateName(const char* name,
                    TypeTemplateImpl * typeTemplImpl);


   /**
    * ByName will return a type template corresponding to the argument name
    * @param type template name to lookup
    * @param nTemplateParams looks up the template family with this number of template parameters
    *        if it is set to 0, the first occurence of the template family name will be returned
    * @return corresponding type template to name
    */
   static TypeTemplate ByName(const std::string& name,
                              size_t nTemplateParams = 0);


   /**
    * CleanUp is called at the end of the process
    */
   static void CleanUp();


   /*
    * DeleteTypeTemplate will remove the dictionary information
    * of one type template from memory
    */
   void DeleteTypeTemplate() const;


   /**
    * Name will return the name of the type template
    * @return name of type template
    */
   std::string Name(unsigned int mod) const;


   /**
    * Name_c_str will return a char * pointer to the type template name
    * @return type template name as char *
    */
   const char* Name_c_str() const;


   /**
    * ThisTypeTemplate will return the TypeTemplate API class of this type template
    * @return API type template class
    */
   TypeTemplate ThisTypeTemplate() const;


   /**
    * TypeTemplateAt will return the nth type template defined
    * @param nth type template
    * @return nth type template
    */
   static TypeTemplate TypeTemplateAt(size_t nth);


   /**
    * TypeTemplateSize will return the number of type templates defined
    * @return number of defined type templates
    */
   static size_t TypeTemplateSize();


   /**
    * TypeTemplate_Begin returns the begin iterator of the type template container
    * @return begin iterator of type template container
    */
   static TypeTemplate_Iterator TypeTemplate_Begin();


   /**
    * TypeTemplate_End returns the end iterator of the type template container
    * @return end iterator of type template container
    */
   static TypeTemplate_Iterator TypeTemplate_End();


   /**
    * TypeTemplate_Rbegin returns the rbegin iterator of the type template container
    * @return rbegin iterator of type template container
    */
   static Reverse_TypeTemplate_Iterator TypeTemplate_RBegin();


   /**
    * TypeTemplate_Rend returns the rend iterator of the type template container
    * @return rend iterator of type template container
    */
   static Reverse_TypeTemplate_Iterator TypeTemplate_REnd();

private:
   /** destructor */
   ~TypeTemplateName();

private:
   /**
    * The name of the type template
    */
   std::string fName;


   /**
    * Pointer to the implementation of the type template
    * @label type template impl
    * @link aggregation
    * @supplierCardinality 0..1
    * @clientCardinality 1
    */
   mutable
   TypeTemplateImpl * fTypeTemplateImpl;


   /**
    * This type template
    * @label this type template
    * @link aggregation
    * @supplierCardinality 1
    * @clientCardinality 1
    */
   TypeTemplate* fThisTypeTemplate;

};    // class TypeTemplate
} // namespace Reflex


//-------------------------------------------------------------------------------
inline const char*
Reflex::TypeTemplateName::Name_c_str() const {
//-------------------------------------------------------------------------------
   return fName.c_str();
}


#endif // Reflex_TypeTemplateName
