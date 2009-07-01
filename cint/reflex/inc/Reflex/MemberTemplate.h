// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_MemberTemplate
#define Reflex_MemberTemplate

// Include files
#include "Reflex/Kernel.h"


namespace Reflex {
// forward declarations
class MemberTemplateName;
class Member;

/**
 * @class MemberTemplate MemberTemplate.h Reflex/MemberTemplate.h
 * @author Stefan Roiser
 * @date 2005-02-03
 * @ingroup Ref
 */
class RFLX_API MemberTemplate {
   friend class OwnedMemberTemplate;

public:
   /** default constructor */
   MemberTemplate(const MemberTemplateName * memberTemplateName = 0);


   /** copy constructor */
   MemberTemplate(const MemberTemplate &rh);


   /** destructor */
   ~MemberTemplate();


   /**
    * operator bool will return true if the member template is resolved
    * @return true if member template is resolved
    */
   operator bool() const;


   /**
    * operator == will return true if two member templates are the same
    * @return true if member templates match
    */
   bool operator ==(const MemberTemplate& rh) const;


   /**
    * ByName will return a member template corresponding to the argument name
    * @param member template name to lookup
    * @param nTemplateParams looks up the template family with this number of template parameters
    *        if it is set to 0, the first occurence of the template family name will be returned
    * @return corresponding member template to name
    */
   static MemberTemplate ByName(const std::string& name,
                                size_t nTemplateParams = 0);


   /**
    * Id will return a memory address which is a unique id for this member template
    * @return unique id of this member template
    */
   void* Id() const;


   /**
    * MemberTemplateAt will return the nth member template defined
    * @param nth member template
    * @return nth member template
    */
   static MemberTemplate MemberTemplateAt(size_t nth);


   /**
    * MemberTemplateSize will return the number of member templates defined
    * @return number of defined member templates
    */
   static size_t MemberTemplateSize();


   /**
    * MemberTemplate_Begin returns the begin iterator of the member template container
    * @return begin iterator of member template container
    */
   static MemberTemplate_Iterator MemberTemplate_Begin();


   /**
    * MemberTemplate_End returns the end iterator of the member template container
    * @return end iterator of member template container
    */
   static MemberTemplate_Iterator MemberTemplate_End();


   /**
    * MemberTemplate_Rbegin returns the rbegin iterator of the member template container
    * @return rbegin iterator of member template container
    */
   static Reverse_MemberTemplate_Iterator MemberTemplate_RBegin();


   /**
    * MemberTemplate_Rend returns the rend iterator of the member template container
    * @return rend iterator of member template container
    */
   static Reverse_MemberTemplate_Iterator MemberTemplate_REnd();


   /**
    * Name will return the name of the template family and a list of
    * all currently available instantiations
    * @return template family name with all instantiantion
    */
   std::string Name(unsigned int mod = 0) const;


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
    * TemplateInstanceAt will return the nth template instantion
    * @param  nth template instantion
    * @return pointer to nth template instantion
    */
   Member TemplateInstanceAt(size_t nth) const;


   /**
    * instantionSize will return the number of template instantions for
    * this template family
    * @return number of template instantions
    */
   size_t TemplateInstanceSize() const;


   /**
    * TemplateParameterDefaultAt will return the nth FunctionParameterAt default value as string
    * @param nth template FunctionParameterAt
    * @return default value of nth template FunctionParameterAt
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
    * TemplateParameterNameAt will the Name of the nth FunctionParameterAt
    * @param  nth template FunctionParameterAt
    * @return Name of nth template FunctionParameterAt
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

public:
   /**
    * AddTemplateInstance adds one TemplateInstanceAt of the template to the local container
    * @param templateInstance the template TemplateInstanceAt
    */
   void AddTemplateInstance(const Member& templateInstance) const;

private:
   /**
    * pointer to the member template implementation
    * @label member template name
    * @link aggregation
    * @clientCardinality 1
    * @supplierCardinality 1
    */
   const MemberTemplateName* fMemberTemplateName;

};    // class MemberTemplate

} // namespace Reflex

#include "Reflex/internal/MemberTemplateName.h"
#include "Reflex/internal/MemberTemplateImpl.h"

//-------------------------------------------------------------------------------
inline Reflex::MemberTemplate::MemberTemplate(const MemberTemplateName* memberTemplateName)
//-------------------------------------------------------------------------------
   : fMemberTemplateName(memberTemplateName) {
}


//-------------------------------------------------------------------------------
inline Reflex::MemberTemplate::MemberTemplate(const MemberTemplate& rh)
//-------------------------------------------------------------------------------
   : fMemberTemplateName(rh.fMemberTemplateName) {
}


//-------------------------------------------------------------------------------
inline Reflex::MemberTemplate::~MemberTemplate() {
}

//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
inline
Reflex::MemberTemplate::operator bool() const {
//-------------------------------------------------------------------------------
   if (this->fMemberTemplateName && this->fMemberTemplateName->fMemberTemplateImpl) {
      return true;
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberTemplate::operator ==(const MemberTemplate& rh) const {
//-------------------------------------------------------------------------------
   return fMemberTemplateName == rh.fMemberTemplateName;
}


//-------------------------------------------------------------------------------
inline void*
Reflex::MemberTemplate::Id() const {
//-------------------------------------------------------------------------------
   return (void*) fMemberTemplateName;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::MemberTemplate::MemberTemplateSize() {
//-------------------------------------------------------------------------------
   return MemberTemplateName::MemberTemplateSize();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::MemberTemplate::TemplateInstanceSize() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateInstanceSize();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::MemberTemplate::TemplateParameterSize() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateParameterSize();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::MemberTemplate::TemplateParameterDefaultAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateParameterDefaultAt(nth);
   }
   return "";
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::MemberTemplate::TemplateParameterDefault_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateParameterDefault_Begin();
   }
   return Dummy::StdStringCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::MemberTemplate::TemplateParameterDefault_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateParameterDefault_End();
   }
   return Dummy::StdStringCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::MemberTemplate::TemplateParameterDefault_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateParameterDefault_RBegin();
   }
   return Dummy::StdStringCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::MemberTemplate::TemplateParameterDefault_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateParameterDefault_REnd();
   }
   return Dummy::StdStringCont().rend();
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::MemberTemplate::TemplateParameterNameAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateParameterNameAt(nth);
   }
   return "";
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::MemberTemplate::TemplateParameterName_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateParameterName_Begin();
   }
   return Dummy::StdStringCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::MemberTemplate::TemplateParameterName_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateParameterName_End();
   }
   return Dummy::StdStringCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::MemberTemplate::TemplateParameterName_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateParameterName_RBegin();
   }
   return Dummy::StdStringCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::MemberTemplate::TemplateParameterName_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberTemplateName->fMemberTemplateImpl->TemplateParameterName_REnd();
   }
   return Dummy::StdStringCont().rend();
}


#endif // Reflex_MemberTemplate
