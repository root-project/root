// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_MemberTemplateImpl
#define Reflex_MemberTemplateImpl

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Scope.h"

#ifdef _WIN32
# pragma warning( push )
# pragma warning( disable : 4251 )
#endif

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
   MemberTemplateImpl(const char* templateName,
                      const Scope &scope,
                      const std::vector<std::string> &parameterNames,
                      const std::vector<std::string>& parameterDefaults = std::vector<std::string>());


   /** destructor */
   virtual ~MemberTemplateImpl();


   /**
    * operator == will return true if two At templates are the same
    * @return true if At templates match
    */
   bool operator ==(const MemberTemplateImpl& rh) const;


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
   Member TemplateInstanceAt(size_t nth) const;


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
   std::string TemplateParameterDefaultAt(size_t nth) const;


   StdString_Iterator TemplateParameterDefault_Begin() const;
   StdString_Iterator TemplateParameterDefault_End() const;
   Reverse_StdString_Iterator TemplateParameterDefault_RBegin() const;
   Reverse_StdString_Iterator TemplateParameterDefault_REnd() const;


   /**
    * TemplateParameterNameAt will the Name of the nth FunctionParameterAt
    * @param  nth template FunctionParameterAt
    * @return Name of nth template FunctionParameterAt
    */
   std::string TemplateParameterNameAt(size_t nth) const;


   StdString_Iterator TemplateParameterName_Begin() const;
   StdString_Iterator TemplateParameterName_End() const;
   Reverse_StdString_Iterator TemplateParameterName_RBegin() const;
   Reverse_StdString_Iterator TemplateParameterName_REnd() const;


   /**
    * Return the member template API class corresponding to this member template impl
    * @return corresponding member template
    */
   MemberTemplate ThisMemberTemplate() const;

public:
   /**
    * AddTemplateInstance adds one TemplateInstanceAt of the template to the local container
    * @param templateInstance the template TemplateInstanceAt
    */
   void AddTemplateInstance(const Member& templateInstance) const;

private:
   /**
    * declaring scope of this member template
    * @link aggregation
    * @label member template scope
    * @clientCardinality 1
    * @supplierCardinality 1
    */
   Scope fScope;


   /**
    * the class template instances
    * @link aggregation
    * @supplierCardinality 1..*
    * @clientCardinality 1
    * @label template instances
    */
   mutable
   std::vector<Member> fTemplateInstances;


   /**
    * container of function parameter template names
    */
   mutable
   std::vector<std::string> fParameterNames;


   /**
    * function  parameter template default values
    */
   mutable
   std::vector<std::string> fParameterDefaults;


   /**
    * number of required template parameters
    */
   size_t fReqParameters;


   /**
    * pointer back to the member template name
    * @link aggregation
    * @label member template name
    * @supplierCardinality 1
    * @clientCardinality 1
    */
   MemberTemplateName* fMemberTemplateName;


};    // class MemberTemplateImpl

} // namespace Reflex


//-------------------------------------------------------------------------------
inline size_t
Reflex::MemberTemplateImpl::TemplateParameterSize() const {
//-------------------------------------------------------------------------------
   return fParameterNames.size();
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::MemberTemplateImpl::TemplateParameterDefaultAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (nth < fParameterDefaults.size()) {
      return fParameterDefaults[nth];
   }
   return "";
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::MemberTemplateImpl::TemplateParameterDefault_Begin() const {
//-------------------------------------------------------------------------------
   return fParameterDefaults.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::MemberTemplateImpl::TemplateParameterDefault_End() const {
//-------------------------------------------------------------------------------
   return fParameterDefaults.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::MemberTemplateImpl::TemplateParameterDefault_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string> &)fParameterDefaults).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::MemberTemplateImpl::TemplateParameterDefault_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string> &)fParameterDefaults).rend();
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::MemberTemplateImpl::TemplateParameterNameAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (nth < fParameterNames.size()) {
      return fParameterNames[nth];
   }
   return "";
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::MemberTemplateImpl::TemplateParameterName_Begin() const {
//-------------------------------------------------------------------------------
   return fParameterNames.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::MemberTemplateImpl::TemplateParameterName_End() const {
//-------------------------------------------------------------------------------
   return fParameterNames.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::MemberTemplateImpl::TemplateParameterName_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string> &)fParameterNames).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::MemberTemplateImpl::TemplateParameterName_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<std::string> &)fParameterNames).rend();
}


#ifdef _WIN32
# pragma warning( pop )
#endif

#endif // Reflex_MemberTemplateImpl
