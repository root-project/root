// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
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
    class TypeTemplate {

    public:

      /** default constructor */
      TypeTemplate( TypeTemplateImpl * tti = 0 );


      /** destructor */
      ~TypeTemplate();


      /** 
       * operator bool will return true if the TypeNth template is resolved
       * @return true if TypeNth template is resolved
       */
      operator bool () const;


      /** 
       * operator == will return true if two TypeNth templates are the same
       * @return true if TypeNth templates match
       */
      bool operator == ( const TypeTemplate & rh ) const;


      /**
       * instantion will return a pointer to the nth template instantion
       * @param  nth template instantion
       * @return pointer to nth template instantion
       */
      Type InstantiationNth( size_t nth ) const;


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
      void AddTemplateInstance( const Type & templateInstance ) const;

    private:

      /** 
       * pointer to the TypeNth template implementation
       * @link aggregation
       * @supplierCardinality 0..1
       * @clientCardinality 1
       * @label TypeNth template impl
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
inline size_t ROOT::Reflex::TypeTemplate::InstantiationCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeTemplateImpl->InstantiationCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::TypeTemplate::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeTemplateImpl->Name( mod );
  return "";
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeTemplate::ParameterCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeTemplateImpl->ParameterCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::TypeTemplate::ParameterDefault( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeTemplateImpl->ParameterDefault( nth );
  return "";
}


//-------------------------------------------------------------------------------s
inline std::string ROOT::Reflex::TypeTemplate::ParameterName( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeTemplateImpl->ParameterName( nth );
  return "";
}

#endif // ROOT_Reflex_TypeTemplate
