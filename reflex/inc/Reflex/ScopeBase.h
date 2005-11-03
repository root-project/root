// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_ScopeBase
#define ROOT_Reflex_ScopeBase

// Include files
#include "Reflex/Kernel.h"

#include "Reflex/Scope.h"
#include "Reflex/PropertyList.h"
#include <vector>

namespace ROOT {
  namespace Reflex {

    // forward declarations
    class Scope;
    class ScopeName;
    class Namespace;
    class Class;
    class Member;
    class TypeTemplate;
    class MemberTemplate;
    class Type;

    /**
     * @class ScopeBase ScopeBase.h Reflex/ScopeBase.h
     * @author Stefan Roiser
     * @date 24/11/2003
     * @ingroup Ref
     */
    class ScopeBase {

    public:

      /** constructor within a ScopeNth*/
      ScopeBase( const char * scope, 
                 TYPE scopeType );


      /** destructor */
      virtual ~ScopeBase();


      /** 
       * operator Scope will return the corresponding Scope object
       * @return Scope corresponding to this ScopeBase
       */
      operator Scope () const;


      /** 
       * the operator Type will return a corresponding Type object to the ScopeNth if
       * applicable (i.e. if the Scope is also a Type e.g. Class, Union, Enum)
       */
      operator Type () const;


      /**
       * nthBase will return the nth BaseNth class information
       * @param  nth nth BaseNth class
       * @return pointer to BaseNth class information
       */
      virtual Base BaseNth( size_t nth ) const;


      /**
       * BaseCount will return the number of BaseNth classes
       * @return number of BaseNth classes
       */
      virtual size_t BaseCount() const;


      virtual Base_Iterator Base_Begin() const;
      virtual Base_Iterator Base_End() const;
      virtual Reverse_Base_Iterator Base_Rbegin() const;
      virtual Reverse_Base_Iterator Base_Rend() const;


      /**
       * nthDataMember will return the nth data MemberNth of the ScopeNth
       * @param  nth data MemberNth
       * @return pointer to data MemberNth
       */
      Member DataMemberNth( size_t nth ) const;


      /**
       * DataMemberNth will return the MemberNth with Name
       * @param  Name of data MemberNth
       * @return data MemberNth
       */
      Member DataMemberNth( const std::string & nam ) const;


      /**
       * DataMemberCount will return the number of data members of this ScopeNth
       * @return number of data members
       */
      size_t DataMemberCount() const;

      
      Member_Iterator DataMember_Begin() const;
      Member_Iterator DataMember_End() const;
      Reverse_Member_Iterator DataMember_Rbegin() const;
      Reverse_Member_Iterator DataMember_Rend() const;


      /**
       * DeclaringScope will return a pointer to the ScopeNth of this one
       * @return pointer to declaring ScopeNth
       */
      Scope DeclaringScope() const;


      /**
       * nthFunctionMember will return the nth function MemberNth of the ScopeNth
       * @param  nth function MemberNth
       * @return pointer to function MemberNth
       */
      Member FunctionMemberNth( size_t nth ) const;

 
      /**
       * FunctionMemberNth will return the MemberNth with the Name, 
       * optionally the signature of the function may be given
       * @param  Name of function MemberNth
       * @param  signature of the MemberNth function 
       * @return function MemberNth
       */
      Member FunctionMemberNth( const std::string & name,
                                const Type & signature ) const;


      /**
       * FunctionMemberCount will return the number of function members of
       * this ScopeNth
       * @return number of function members
       */
      size_t FunctionMemberCount() const;


      Member_Iterator FunctionMember_Begin() const;
      Member_Iterator FunctionMember_End() const;
      Reverse_Member_Iterator FunctionMember_Rbegin() const;
      Reverse_Member_Iterator FunctionMember_Rend() const;


      /** 
       * IsClass returns true if the ScopeNth represents a Class
       * @return true if ScopeNth represents a Class
       */
      bool IsClass() const;

 
      /** 
       * IsEnum returns true if the TypeNth represents a Enum
       * @return true if TypeNth represents a Enum
       */
      bool IsEnum() const;

      
      /** 
       * IsNamespace returns true if the ScopeNth represents a Namespace
       * @return true if ScopeNth represents a Namespace
       */
      bool IsNamespace() const;


      /** 
       * IsTemplateInstance returns true if the ScopeNth represents a 
       * ClassTemplateInstance
       * @return true if ScopeNth represents a InstantiatedTemplateClass
       */
      bool IsTemplateInstance() const;


      /**
       * IsTopScope will return true if the current ScopeNth is the top
       * (Empty) namespace
       * @return true if current sope is top namespace
       */
      bool IsTopScope() const;


      /** 
       * IsUnion returns true if the TypeNth represents a Union
       * @return true if TypeNth represents a 
       */
      bool IsUnion() const;


      /**
       * MemberNth will return the first MemberNth with a given Name
       * @param Name  MemberNth Name
       * @return pointer to MemberNth
       */
      Member MemberNth( const std::string & name,
                        const Type & signature ) const;


      /**
       * MemberNth will return the nth MemberNth of the ScopeNth
       * @param  nth MemberNth
       * @return pointer to nth MemberNth
       */
      Member MemberNth( size_t nth ) const;


      Member_Iterator Member_Begin() const;
      Member_Iterator Member_End() const;
      Reverse_Member_Iterator Member_Rbegin() const;
      Reverse_Member_Iterator Member_Rend() const;


      /**
       * MemberCount will return the number of members
       * @return number of members
       */
      size_t MemberCount() const;


      /** 
       * MemberTemplateNth will return the nth MemberNth template of this ScopeNth
       * @param nth MemberNth template
       * @return nth MemberNth template
       */
      MemberTemplate MemberTemplateNth( size_t nth ) const;


      /** 
       * MemberTemplateCount will return the number of MemberNth templates in this socpe
       * @return number of defined MemberNth templates
       */
      size_t MemberTemplateCount() const;


      MemberTemplate_Iterator MemberTemplate_Begin() const;
      MemberTemplate_Iterator MemberTemplate_End() const;
      Reverse_MemberTemplate_Iterator MemberTemplate_Rbegin() const;
      Reverse_MemberTemplate_Iterator MemberTemplate_Rend() const;


      /**
       * Name will return the Name of the ScopeNth
       * @return Name of ScopeNth
       */
      virtual std::string Name( unsigned int mod = 0 ) const;


      /**
       * PropertyListGet will return a pointer to the PropertyNth list attached
       * to this item
       * @return pointer to PropertyNth list
       */
      virtual PropertyList PropertyListGet() const;

      
      /** 
       * ScopeNth will return the ScopeNth Object of this ScopeBase
       * @return corresponding Scope
       */
      Scope ScopeGet() const;

      
      /**
       * ScopeType will return which kind of ScopeNth is represented
       * @return TypeNth of ScopeNth
       */
      TYPE ScopeType() const;


      /**
       * ScopeTypeAsString will return the string representation of the enum
       * representing the current Scope (e.g. "CLASS")
       * @return string representation of enum for Scope
       */
      std::string ScopeTypeAsString() const;


      /**
       * SubScopeNth will return a pointer to a sub-scopes
       * @param  nth sub-ScopeNth
       * @return pointer to nth sub-ScopeNth
       */
      Scope SubScopeNth( size_t nth ) const;


      /**
       * ScopeCount will return the number of sub-scopes
       * @return number of sub-scopes
       */
      size_t SubScopeCount() const;


      Scope_Iterator SubScope_Begin() const;
      Scope_Iterator SubScope_End() const;
      Reverse_Scope_Iterator SubScope_Rbegin() const;
      Reverse_Scope_Iterator SubScope_Rend() const;


      /**
       * TypeNth will return a pointer to the nth sub-TypeNth
       * @param  nth sub-TypeNth
       * @return pointer to nth sub-TypeNth
       */
      Type SubTypeNth( size_t nth ) const;


      /**
       * TypeCount will returnt he number of sub-types
       * @return number of sub-types
       */
      size_t SubTypeCount() const;


      Type_Iterator SubType_Begin() const;
      Type_Iterator SubType_End() const;
      Reverse_Type_Iterator SubType_Rbegin() const;
      Reverse_Type_Iterator SubType_Rend() const;


      /**
       * TemplateArgumentNth will return a pointer to the nth template argument
       * @param  nth nth template argument
       * @return pointer to nth template argument
       */
      virtual Type TemplateArgumentNth( size_t nth ) const;


      /**
       * templateArgCount will return the number of template arguments
       * @return number of template arguments
       */
      virtual size_t TemplateArgumentCount() const;


      virtual Type_Iterator TemplateArgument_Begin() const;
      virtual Type_Iterator TemplateArgument_End() const;
      virtual Reverse_Type_Iterator TemplateArgument_Rbegin() const;
      virtual Reverse_Type_Iterator TemplateArgument_Rend() const;


      /**
       * TypeTemplateNth returns the corresponding TypeTemplate if any
       * @return corresponding TypeTemplate
       */
      virtual TypeTemplate TemplateFamily() const;


      /** 
       * TypeTemplateNth will return the nth TypeNth template of this ScopeNth
       * @param nth TypeNth template
       * @return nth TypeNth template
       */
      TypeTemplate TypeTemplateNth( size_t nth ) const;


      /** 
       * TypeTemplateCount will return the number of TypeNth templates in this socpe
       * @return number of defined TypeNth templates
       */
      size_t TypeTemplateCount() const;


      TypeTemplate_Iterator TypeTemplate_Begin() const;
      TypeTemplate_Iterator TypeTemplate_End() const;
      Reverse_TypeTemplate_Iterator TypeTemplate_Rbegin() const;
      Reverse_TypeTemplate_Iterator TypeTemplate_Rend() const;

    protected:

      /** protected constructor for initialisation of the global namespace */
      ScopeBase();

    public:

      /**
       * AddDataMember will add the information about a data MemberNth
       * @param dm pointer to data MemberNth
       */
      virtual void AddDataMember( const Member & dm ) const;
      virtual void AddDataMember( const char * name,
                                  const Type & type,
                                  size_t offset,
                                  unsigned int modifiers = 0 ) const;


      /**
       * AddFunctionMember will add the information about a function MemberNth
       * @param fm pointer to function MemberNth
       */
      virtual void AddFunctionMember( const Member & fm ) const;
      virtual void AddFunctionMember( const char * name,
                                      const Type & type,
                                      StubFunction stubFP,
                                      void * stubCtx = 0,
                                      const char * params = 0,
                                      unsigned int modifiers = 0 ) const;


      virtual void AddMemberTemplate( const MemberTemplate & mt ) const ;


      /**
       * AddSubScope will add a sub-ScopeNth to this one
       * @param sc pointer to Scope
       */
      virtual void AddSubScope( const Scope & sc ) const;
      virtual void AddSubScope( const char * scope,
                                TYPE scopeType ) const;


      /**
       * AddSubType will add a sub-TypeNth to this ScopeNth
       * @param sc pointer to Type
       */
      virtual void AddSubType( const Type & ty ) const;
      virtual void AddSubType( const char * type,
                               size_t size,
                               TYPE typeType,
                               const std::type_info & ti,
                               unsigned int modifiers = 0 ) const;


      void AddTypeTemplate( const TypeTemplate & tt ) const;


      /**
       * RemoveDataMember will remove the information about a data MemberNth
       * @param dm pointer to data MemberNth
       */
      virtual void RemoveDataMember( const Member & dm ) const;


      /**
       * RemoveFunctionMember will remove the information about a function MemberNth
       * @param fm pointer to function MemberNth
       */
      virtual void RemoveFunctionMember( const Member & fm ) const;


      virtual void RemoveMemberTemplate( const MemberTemplate & mt ) const;


      /**
       * RemoveSubScope will remove a sub-ScopeNth to this one
       * @param sc pointer to Scope
       */
      virtual void RemoveSubScope( const Scope & sc ) const;


      /**
       * RemoveSubType will remove a sub-TypeNth to this ScopeNth
       * @param sc pointer to Type
       */
      virtual void RemoveSubType( const Type & ty ) const;


      virtual void RemoveTypeTemplate( const TypeTemplate & tt ) const;

    protected:
      
      /** container for all members of the ScopeNth */
      typedef std::vector < Member > Members;

      /**
       * pointers to members
       * @label ScopeNth members
       * @link aggregationByValue
       * @supplierCardinality 0..*
       * @clientCardinality 1
       */
      mutable
      std::vector< Member > fMembers;

      /**
       * container with pointers to all data members in this ScopeNth
       * @label ScopeNth datamembers
       * @link aggregationByValue
       * @clientCardinality 1
       * @supplierCardinality 0..*
       */
      mutable
      std::vector< Member > fDataMembers;

      /**
       * container with pointers to all function members in this ScopeNth
       * @label ScopeNth functionmembers
       * @link aggregationByValue
       * @supplierCardinality 0..*
       * @clientCardinality 1
       */
      mutable
      std::vector< Member > fFunctionMembers;

    private:

      /**
       * pointer to the ScopeNth Name
       * @label ScopeNth Name
       * @link aggregation
       * @clientCardinality 1
       * @supplierCardinality 1
       */
      ScopeName * fScopeName;


      /**
       * TypeNth of ScopeNth
       * @link aggregationByValue
       * @label ScopeNth TypeNth
       * @clientCardinality 1
       * @supplierCardinality 1
       */
      TYPE fScopeType;


      /**
       * pointer to declaring ScopeNth
       * @label declaring ScopeNth
       * @link aggregationByValue
       * @clientCardinality 1
       * @supplierCardinality 1
       */
      Scope fDeclaringScope;


      /**
       * pointers to sub-scopes
       * @label sub scopes
       * @link aggregationByValue
       * @supplierCardinality 0..*
       * @clientCardinality 1
       */
      mutable
      std::vector< Scope > fSubScopes;
 

      /**
       * pointer to types
       * @label ScopeNth types
       * @link aggregationByValue
       * @supplierCardinality 0..*
       * @clientCardinality 1
       */
      mutable
      std::vector < Type > fSubTypes;


      /**
       * container for TypeNth templates defined in this ScopeNth
       * @label TypeNth templates
       * @link aggregationByValue
       * @supplierCardinality 0..*
       * @clientCardinality 1
       */
      mutable
      std::vector < TypeTemplate > fTypeTemplates;
 
 
      /**
       * container for TypeNth templates defined in this ScopeNth
       * @label TypeNth templates
       * @link aggregationByValue
       * @supplierCardinality 0..*
       * @clientCardinality 1
       */
      mutable
      std::vector < MemberTemplate > fMemberTemplates;


     /**
       * pointer to the PropertyNth list
       * @label propertylist
       * @link aggregationByValue
       * @clientCardinality 1
       * @supplierCardinality 1
       */
      PropertyList fPropertyList;


      /** 
       * The position where the unscoped Name starts in the scopename
       */
      size_t fBasePosition;

    }; // class ScopeBase
  } //namespace Reflex
} //namespace ROOT


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::ScopeBase::BaseCount() const {
//-------------------------------------------------------------------------------
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Base_Iterator ROOT::Reflex::ScopeBase::Base_Begin() const {
//-------------------------------------------------------------------------------
  return Base_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Base_Iterator ROOT::Reflex::ScopeBase::Base_End() const {
//-------------------------------------------------------------------------------
  return Base_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Base_Iterator ROOT::Reflex::ScopeBase::Base_Rbegin() const {
//-------------------------------------------------------------------------------
  return Reverse_Base_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Base_Iterator ROOT::Reflex::ScopeBase::Base_Rend() const {
//-------------------------------------------------------------------------------
  return Reverse_Base_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope ROOT::Reflex::ScopeBase::DeclaringScope() const {
//-------------------------------------------------------------------------------
  return fDeclaringScope;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::ScopeBase::DataMember_Begin() const {
//-------------------------------------------------------------------------------
  return fDataMembers.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::ScopeBase::DataMember_End() const {
//-------------------------------------------------------------------------------
  return fDataMembers.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::ScopeBase::DataMember_Rbegin() const {
//-------------------------------------------------------------------------------
  return fDataMembers.rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::ScopeBase::DataMember_Rend() const {
//-------------------------------------------------------------------------------
  return fDataMembers.rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::ScopeBase::FunctionMember_Begin() const {
//-------------------------------------------------------------------------------
  return fFunctionMembers.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::ScopeBase::FunctionMember_End() const {
//-------------------------------------------------------------------------------
  return fFunctionMembers.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::ScopeBase::FunctionMember_Rbegin() const {
//-------------------------------------------------------------------------------
  return fFunctionMembers.rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::ScopeBase::FunctionMember_Rend() const {
//-------------------------------------------------------------------------------
  return fFunctionMembers.rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::ScopeBase::Member_Begin() const {
//-------------------------------------------------------------------------------
  return fMembers.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::ScopeBase::Member_End() const {
//-------------------------------------------------------------------------------
  return fMembers.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::ScopeBase::Member_Rbegin() const {
//-------------------------------------------------------------------------------
  return fMembers.rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::ScopeBase::Member_Rend() const {
//-------------------------------------------------------------------------------
  return fMembers.rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate_Iterator ROOT::Reflex::ScopeBase::MemberTemplate_Begin() const {
//-------------------------------------------------------------------------------
  return fMemberTemplates.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate_Iterator ROOT::Reflex::ScopeBase::MemberTemplate_End() const {
//-------------------------------------------------------------------------------
  return fMemberTemplates.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::ScopeBase::MemberTemplate_Rbegin() const {
//-------------------------------------------------------------------------------
  return fMemberTemplates.rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::ScopeBase::MemberTemplate_Rend() const {
//-------------------------------------------------------------------------------
  return fMemberTemplates.rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope_Iterator ROOT::Reflex::ScopeBase::SubScope_Begin() const {
//-------------------------------------------------------------------------------
  return fSubScopes.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope_Iterator ROOT::Reflex::ScopeBase::SubScope_End() const {
//-------------------------------------------------------------------------------
  return fSubScopes.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::ScopeBase::SubScope_Rbegin() const {
//-------------------------------------------------------------------------------
  return fSubScopes.rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::ScopeBase::SubScope_Rend() const {
//-------------------------------------------------------------------------------
  return fSubScopes.rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::ScopeBase::SubType_Begin() const {
//-------------------------------------------------------------------------------
  return fSubTypes.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::ScopeBase::SubType_End() const {
//-------------------------------------------------------------------------------
  return fSubTypes.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::ScopeBase::SubType_Rbegin() const {
//-------------------------------------------------------------------------------
  return fSubTypes.rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::ScopeBase::SubType_Rend() const {
//-------------------------------------------------------------------------------
  return fSubTypes.rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::ScopeBase::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
  return Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::ScopeBase::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
  return Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::ScopeBase::TemplateArgument_Rbegin() const {
//-------------------------------------------------------------------------------
  return Reverse_Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::ScopeBase::TemplateArgument_Rend() const {
//-------------------------------------------------------------------------------
  return Reverse_Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::ScopeBase::TypeTemplate_Begin() const {
//-------------------------------------------------------------------------------
  return fTypeTemplates.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::ScopeBase::TypeTemplate_End() const {
//-------------------------------------------------------------------------------
  return fTypeTemplates.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::ScopeBase::TypeTemplate_Rbegin() const {
//-------------------------------------------------------------------------------
  return fTypeTemplates.rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::ScopeBase::TypeTemplate_Rend() const {
//-------------------------------------------------------------------------------
  return fTypeTemplates.rend();
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::ScopeBase::IsClass() const {
//-------------------------------------------------------------------------------
  return ( fScopeType == CLASS || fScopeType == TYPETEMPLATEINSTANCE );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::ScopeBase::IsEnum() const {
//-------------------------------------------------------------------------------
  return ( fScopeType == ENUM );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::ScopeBase::IsNamespace() const {
//-------------------------------------------------------------------------------
  return ( fScopeType == NAMESPACE );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::ScopeBase::IsTemplateInstance() const {
//-------------------------------------------------------------------------------
  return ( fScopeType == TYPETEMPLATEINSTANCE );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::ScopeBase::IsUnion() const {
//-------------------------------------------------------------------------------
  return ( fScopeType == UNION );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TYPE ROOT::Reflex::ScopeBase::ScopeType() const {
//-------------------------------------------------------------------------------
  return fScopeType;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope ROOT::Reflex::ScopeBase::SubScopeNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( nth < fSubScopes.size() ) { return fSubScopes[ nth ]; }
  return Scope( 0 );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::ScopeBase::SubScopeCount() const {
//-------------------------------------------------------------------------------
  return fSubScopes.size();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::ScopeBase::TemplateArgumentCount() const {
//-------------------------------------------------------------------------------
  return 0;
}

#endif // ROOT_Reflex_ScopeBase
