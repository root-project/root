// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_MemberTemplateName
#define Reflex_MemberTemplateName

// Include files
#include "Reflex/Kernel.h"


namespace Reflex {
// forward declarations
class MemberTemplate;
class MemberTemplateImpl;

/**
 * @class MemberTemplateName MemberTemplateName.h Reflex/internal/MemberTemplateName.h
 * @author Stefan Roiser
 * @date 8/8/2006
 * @ingroup Ref
 */
class RFLX_API MemberTemplateName {
   friend class MemberTemplate;
   friend class MemberTemplateImpl;

public:
   /** constructor */
   MemberTemplateName(const char* name,
                      MemberTemplateImpl * memberTemplImpl);


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
    * CleanUp is called at the end of the process
    */
   static void CleanUp();


   /*
    * DeleteMemberTemplate will remove the dictionary information
    * of one member template from memory
    */
   void DeleteMemberTemplate() const;


   /**
    * Name will return the name of the member template
    * @return name of member template
    */
   std::string Name(unsigned int mod) const;


   /**
    * Name_c_str will return a char * pointer to the member template name
    * @return member template name as char *
    */
   const char* Name_c_str() const;


   /**
    * ThisMemberTemplate will return the MemberTemplate API class of this member template
    * @return API member template class
    */
   MemberTemplate ThisMemberTemplate() const;


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

private:
   /** destructor */
   ~MemberTemplateName();

private:
   /**
    * The name of the member template
    */
   std::string fName;


   /**
    * Pointer to the implementation of the member template
    * @link aggregation
    * @supplierCardinality 1
    * @clientCardinality 0..1
    * @label member template impl
    */
   mutable
   MemberTemplateImpl * fMemberTemplateImpl;


   /**
    * pointer back to the member temlate
    * @label this member template
    * @link aggregation
    * @clientCardinality 1
    * @supplierCardinality 1
    */
   MemberTemplate* fThisMemberTemplate;

};    // class MemberTemplate
} // namespace Reflex


//-------------------------------------------------------------------------------
inline const char*
Reflex::MemberTemplateName::Name_c_str() const {
//-------------------------------------------------------------------------------
   return fName.c_str();
}


#endif // Reflex_MemberTemplateName
