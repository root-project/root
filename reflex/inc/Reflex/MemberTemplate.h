// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
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
    class MemberTemplate {

    public:

      /** default constructor */
      MemberTemplate( MemberTemplateImpl * = 0 );


      /** destructor */
      ~MemberTemplate();


      /** 
       * operator bool will return true if the MemberNth template is resolved
       * @return true if MemberNth template is resolved
       */
      operator bool () const;


      /**
       * instantion will return a pointer to the nth template instantion
       * @param  nth template instantion
       * @return pointer to nth template instantion
       */
      Member InstantiationNth( size_t nth ) const;


      /**
       * instantionCount will return the number of template instantions for
       * this template family
       * @return number of template instantions
       */
      size_t InstantiationCount() const;


      /**
       * Name will return the Name of the template family and a list of
       * all currently available instantiations
       * @return template family Name with all instantiantion
       */
      std::string Name( unsigned int mod = 0 ) const;


      /**
       * ParameterCount will return the number of template parameters
       * @return number of template parameters
       */
      size_t ParameterCount() const;


      /**
       * ParameterDefault will return the nth ParameterNth default value as string
       * @param nth template ParameterNth
       * @return default value of nth template ParameterNth
       */
      std::string ParameterDefault( size_t nth ) const;


      /**
       * ParameterName will the Name of the nth ParameterNth
       * @param  nth template ParameterNth
       * @return Name of nth template ParameterNth
       */
      std::string ParameterName( size_t nth ) const;

    public:

      /** 
       * AddTemplateInstance adds one InstantiationNth of the template to the local container
       * @param templateInstance the template InstantiationNth
       */
      void AddTemplateInstance( const Member & templateInstance ) const;

    private:

      /** 
       * pointer to the MemberNth template implementation
       * @link aggregation
       * @clientCardinality 1
       * @supplierCardinality 1
       */
      MemberTemplateImpl * fMemberTemplateImpl;
      
    }; // class MemberTemplate

  } // namespace ROOT
} // namespace Reflex

#include "Reflex/Member.h"
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
inline ROOT::Reflex::Member ROOT::Reflex::MemberTemplate::InstantiationNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fMemberTemplateImpl->InstantiationNth( nth );
  return Member();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::MemberTemplate::InstantiationCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fMemberTemplateImpl->InstantiationCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::MemberTemplate::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fMemberTemplateImpl->Name( mod );
  return "";
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::MemberTemplate::ParameterCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fMemberTemplateImpl->ParameterCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::MemberTemplate::ParameterDefault( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fMemberTemplateImpl->ParameterDefault( nth );
  return "";
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::MemberTemplate::ParameterName( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fMemberTemplateImpl->ParameterName( nth );
  return "";
}

#endif // ROOT_Reflex_MemberTemplate
