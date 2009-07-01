// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_TypeTemplate
#define Reflex_TypeTemplate

// Include files
#include "Reflex/Kernel.h"

namespace Reflex {
// forward declarations
class Type;
class TypeTemplateName;

/**
 * @class TypeTemplate TypeTemplate.h Reflex/TypeTemplate.h
 * @author Stefan Roiser
 * @date 2005-02-03
 * @ingroup Ref
 */
class RFLX_API TypeTemplate {
public:
   /** default constructor */
   TypeTemplate(const TypeTemplateName * typeTemplateName = 0);


   /** copy constructor */
   TypeTemplate(const TypeTemplate &rh);


   /** destructor */
   ~TypeTemplate();


   /**
    * operator bool will return true if the type template is resolved
    * @return true if type template is resolved
    */
   operator bool() const;


   /**
    * operator == will return true if two type templates are the same
    * @return true if type templates match
    */
   bool operator ==(const TypeTemplate& rh) const;


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
    * Id will return a memory address which is a unique id for this type template
    * @return unique id of this type template
    */
   void* Id() const;


   /**
    * Name will return the Name of the template family and a list of
    * all currently available instantiations
    * @return template family Name with all instantiantion
    */
   std::string Name(unsigned int mod = 0) const;


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
    * TemplateInstanceAt will return a pointer to the nth template instantion
    * @param  nth template instantion
    * @return pointer to nth template instantion
    */
   Type TemplateInstanceAt(size_t nth) const;


   /**
    * TemplateInstanceSize will return the number of template instantions for
    * this template family
    * @return number of template instantions
    */
   size_t TemplateInstanceSize() const;


   /**
    * TemplateParameterDefaultAt will return the nth TemplateParameterAt default value as string
    * @param nth template TemplateParameterAt
    * @return default value of nth template TemplateParameterAt
    */
   std::string TemplateParameterDefaultAt(size_t nth) const;


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
   std::string TemplateParameterNameAt(size_t nth) const;


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


   /**
    * TemplateParameterSize will return the number of template parameters
    * @return number of template parameters
    */
   size_t TemplateParameterSize() const;


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


   /**
    * Unload will unload the dictionary information of a type template
    */
   void Unload() const;

public:
   /**
    * AddTemplateInstance adds one TemplateInstanceAt of the template to the local container
    * @param templateInstance the template TemplateInstanceAt
    */
   void AddTemplateInstance(const Type& templateInstance) const;

private:
   /**
    * pointer to the type template implementation
    * @link aggregation
    * @supplierCardinality 1
    * @clientCardinality 1
    * @label type template impl
    */
   const TypeTemplateName* fTypeTemplateName;

};    // class TypeTemplate

} // namespace Reflex

#include "Reflex/internal/TypeTemplateName.h"
#include "Reflex/internal/TypeTemplateImpl.h"

//-------------------------------------------------------------------------------
inline Reflex::TypeTemplate::TypeTemplate(const TypeTemplateName* typeTemplateName)
//-------------------------------------------------------------------------------
   : fTypeTemplateName(typeTemplateName) {
}


//-------------------------------------------------------------------------------
inline Reflex::TypeTemplate::TypeTemplate(const TypeTemplate& rh)
//-------------------------------------------------------------------------------
   : fTypeTemplateName(rh.fTypeTemplateName) {
}


//-------------------------------------------------------------------------------
inline Reflex::TypeTemplate::~TypeTemplate() {
}

//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
inline
Reflex::TypeTemplate::operator bool() const {
//-------------------------------------------------------------------------------
   if (this->fTypeTemplateName && this->fTypeTemplateName->fTypeTemplateImpl) {
      return true;
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeTemplate::operator ==(const TypeTemplate& rh) const {
//-------------------------------------------------------------------------------
   return fTypeTemplateName == rh.fTypeTemplateName;
}


//-------------------------------------------------------------------------------
inline void*
Reflex::TypeTemplate::Id() const {
//-------------------------------------------------------------------------------
   return (void*) fTypeTemplateName;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::TypeTemplate::TemplateInstanceSize() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeTemplateName->fTypeTemplateImpl->TemplateInstanceSize();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::TypeTemplate::TemplateParameterSize() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeTemplateName->fTypeTemplateImpl->TemplateParameterSize();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::TypeTemplate::TemplateParameterDefaultAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeTemplateName->fTypeTemplateImpl->TemplateParameterDefaultAt(nth);
   }
   return "";
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::TypeTemplate::TemplateParameterDefault_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeTemplateName->fTypeTemplateImpl->TemplateParameterDefault_Begin();
   }
   return Dummy::StdStringCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::TypeTemplate::TemplateParameterDefault_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeTemplateName->fTypeTemplateImpl->TemplateParameterDefault_End();
   }
   return Dummy::StdStringCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::TypeTemplate::TemplateParameterDefault_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeTemplateName->fTypeTemplateImpl->TemplateParameterDefault_RBegin();
   }
   return Dummy::StdStringCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::TypeTemplate::TemplateParameterDefault_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeTemplateName->fTypeTemplateImpl->TemplateParameterDefault_REnd();
   }
   return Dummy::StdStringCont().rend();
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::TypeTemplate::TemplateParameterNameAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeTemplateName->fTypeTemplateImpl->TemplateParameterNameAt(nth);
   }
   return "";
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::TypeTemplate::TemplateParameterName_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeTemplateName->fTypeTemplateImpl->TemplateParameterName_Begin();
   }
   return Dummy::StdStringCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::TypeTemplate::TemplateParameterName_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeTemplateName->fTypeTemplateImpl->TemplateParameterName_End();
   }
   return Dummy::StdStringCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::TypeTemplate::TemplateParameterName_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeTemplateName->fTypeTemplateImpl->TemplateParameterName_RBegin();
   }
   return Dummy::StdStringCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::TypeTemplate::TemplateParameterName_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeTemplateName->fTypeTemplateImpl->TemplateParameterName_REnd();
   }
   return Dummy::StdStringCont().rend();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::TypeTemplate::TypeTemplateSize() {
//-------------------------------------------------------------------------------
   return TypeTemplateName::TypeTemplateSize();
}


#endif // Reflex_TypeTemplate
