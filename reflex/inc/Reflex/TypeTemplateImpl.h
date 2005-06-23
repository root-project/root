// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_TypeTemplateImpl
#define ROOT_Reflex_TypeTemplateImpl

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Scope.h"

namespace ROOT {
  namespace Reflex {

    // forward declarations
    class Type;
    class ClassTemplateInstance;

    /** 
     * @class TypeTemplateImpl TypeTemplateImpl.h Reflex/TypeTemplateImpl.h
     * @author Stefan Roiser
     * @date 2005-02-03
     * @ingroup Ref
     */
    class TypeTemplateImpl {

    public:

      /** default constructor */
      TypeTemplateImpl( const std::string & templateName,
                        const Scope & ScopeNth,
                        std::vector < std::string > parameterNames, 
                        std::vector < std::string > parameterDefaults = std::vector<std::string>());


      /** destructor */
      virtual ~TypeTemplateImpl();


      /** 
       * operator == will return true if two TypeNth templates are the same
       * @return true if TypeNth templates match
       */
      bool operator == ( const TypeTemplateImpl & rh ) const;


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
       * the Name of the template family 
       */
      std::string fTemplateName;


      /**
       * pointer back to the corresponding ScopeNth
       * @label TypeNth template ScopeNth
       * @clientCardinality 0..*
       * @supplierCardinality 1
       */
      Scope fScope;


      /** 
       * pointer to the class template instances
       * @supplierCardinality 1..*
       * @clientCardinality 0..1
       * @label template instances
       */
      mutable
      std::vector < Type > fTemplateInstances;


      /**
       * container of ParameterNth names
       */
      std::vector < std::string > fParameterNames;


      /**
       * ParameterNth default values
       */
      std::vector < std::string > fParameterDefaults;

      
      /**
       * number of required template parameters
       */
      size_t fReqParameters;
      
    }; // class TypeTemplateImpl

  } // namespace ROOT
} // namespace Reflex


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeTemplateImpl::operator == ( const TypeTemplateImpl & tt ) const {
//-------------------------------------------------------------------------------
  return ( ( fTemplateName == tt.fTemplateName ) && 
           ( fParameterNames.size() == tt.fParameterNames.size() ) );
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::TypeTemplateImpl::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
  std::string s = "";
  if ( 0 != ( mod & ( SCOPED | S ))) {
    std::string sName = fScope.Name(mod);
    if (! fScope.IsTopScope()) s += sName + "::";
  }
  s += fTemplateName;
  return s;  
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeTemplateImpl::ParameterCount() const {
//-------------------------------------------------------------------------------
  return fParameterNames.size();
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::TypeTemplateImpl::ParameterDefault( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( nth < fParameterDefaults.size() ) return fParameterDefaults[ nth ];
  return "";
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::TypeTemplateImpl::ParameterName( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( nth < fParameterNames.size() ) return fParameterNames[ nth ];
  return "";
}


#endif // ROOT_Reflex_TypeTemplateImpl
