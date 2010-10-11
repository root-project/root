// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_ScopeBase
#define Reflex_ScopeBase

// Include files
#include "Reflex/Kernel.h"

#ifndef Reflex_Scope
#include "Reflex/Scope.h"
#endif
#include "Reflex/internal/OwnedPropertyList.h"
#include "Reflex/internal/BuilderContainer.h"
#include <vector>

#ifdef _WIN32
# pragma warning( push )
# pragma warning( disable : 4251 )
#endif

namespace Reflex {
// forward declarations
class Scope;
class ScopeName;
class Namespace;
class Class;
class Member;
class OwnedMember;
class TypeTemplate;
class MemberTemplate;
class OwnedMemberTemplate;
class Type;
class DictionaryGenerator;
class OnDemandBuilder;

/**
 * @class ScopeBase ScopeBase.h Reflex/ScopeBase.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class RFLX_API ScopeBase {
public:
   enum EBuilderKind {
      kBuildDataMembers,
      kBuildFunctionMembers,

      kNumBuilderKinds
   };

   /** constructor within a At*/
   ScopeBase(const char* scope,
             TYPE scopeType);


   /** destructor */
   virtual ~ScopeBase();


   /**
    * operator Scope will return the corresponding Scope object
    * @return Scope corresponding to this ScopeBase
    */
   operator Scope() const;


   /**
    * the operator Type will return a corresponding Type object to the At if
    * applicable (i.e. if the Scope is also a Type e.g. Class, Union, Enum)
    */
   operator Type() const;


   /**
    * nthBase will return the nth BaseAt class information
    * @param  nth nth BaseAt class
    * @return pointer to BaseAt class information
    */
   virtual Base BaseAt(size_t nth) const;


   /**
    * BaseSize will return the number of BaseAt classes
    * @return number of BaseAt classes
    */
   virtual size_t BaseSize() const;


   virtual Base_Iterator Base_Begin() const;
   virtual Base_Iterator Base_End() const;
   virtual Reverse_Base_Iterator Base_RBegin() const;
   virtual Reverse_Base_Iterator Base_REnd() const;


   /**
    * nthDataMember will return the nth data MemberAt of the At
    * @param  nth data MemberAt
    * @return pointer to data MemberAt
    */
   virtual Member DataMemberAt(size_t nth,
                               EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * DataMemberByName will return the MemberAt with Name
    * @param  Name of data MemberAt
    * @return data MemberAt
    */
   virtual Member DataMemberByName(const std::string& nam,
                                   EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * DataMemberSize will return the number of data members of this At
    * @return number of data members
    */
   virtual size_t DataMemberSize(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   virtual Member_Iterator DataMember_Begin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   virtual Member_Iterator DataMember_End(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   virtual Reverse_Member_Iterator DataMember_RBegin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   virtual Reverse_Member_Iterator DataMember_REnd(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * DeclaringScope will return a pointer to the At of this one
    * @return pointer to declaring At
    */
   virtual Scope DeclaringScope() const;


   /**
    * nthFunctionMember will return the nth function MemberAt of the At
    * @param  nth function MemberAt
    * @return pointer to function MemberAt
    */
   virtual Member FunctionMemberAt(size_t nth,
                                   EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * FunctionMemberByName will return the MemberAt with the Name,
    * optionally the signature of the function may be given
    * @param  Name of function MemberAt
    * @param  signature of the MemberAt function
    * @modifiers_mask When matching, do not compare the listed modifiers
    * @return function MemberAt
    */
   virtual Member FunctionMemberByName(const std::string& name,
                                       const Type& signature,
                                       unsigned int modifiers_mask = 0,
                                       EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT,
                                       EDELAYEDLOADSETTING allowDelayedLoad = DELAYEDLOAD_ON) const;


   /**
    * FunctionMemberByNameAndSignature will return the MemberAt with the Name,
    * optionally the signature of the function may be given
    * @param  Name of function MemberAt
    * @param  signature of the MemberAt function
    * @modifiers_mask When matching, do not compare the listed modifiers
    * @return function MemberAt
    */
   virtual Member FunctionMemberByNameAndSignature(const std::string& name,
                                                   const Type& signature,
                                                   unsigned int modifiers_mask = 0,
                                                   EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT,
                                                   EDELAYEDLOADSETTING allowDelayedLoad = DELAYEDLOAD_ON) const;


   /**
    * FunctionMemberSize will return the number of function members of
    * this type
    * @return number of function members
    */
   virtual size_t FunctionMemberSize(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   virtual Member_Iterator FunctionMember_Begin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   virtual Member_Iterator FunctionMember_End(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   virtual Reverse_Member_Iterator FunctionMember_RBegin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   virtual Reverse_Member_Iterator FunctionMember_REnd(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * GenerateDict will produce the dictionary information of this type
    * @param generator a reference to the dictionary generator instance
    */
   virtual void GenerateDict(DictionaryGenerator& generator) const;


   /**
    * GetBasePosition will return fBasePosition
    * @return The position where the unscoped Name starts in the typename
    */
   size_t GetBasePosition() const;


   /**
    * GlobalScope will return the global scope representation\
    * @return global scope
    */
   static Scope GlobalScope();


   /**
    * HasBase will check whether this class has a base class given
    * as argument
    * @param  cl the base-class to check for
    * @return the Base info if it is found, an empty base otherwise (can be tested for bool)
    */
   virtual bool HasBase(const Type& cl) const;


   /**
    * IsClass returns true if the At represents a Class
    * @return true if At represents a Class
    */
   bool IsClass() const;


   /**
    * IsEnum returns true if the At represents a Enum
    * @return true if At represents a Enum
    */
   bool IsEnum() const;


   /**
    * IsNamespace returns true if the At represents a Namespace
    * @return true if At represents a Namespace
    */
   bool IsNamespace() const;


   /**
    * IsTemplateInstance returns true if the At represents a
    * ClassTemplateInstance
    * @return true if At represents a InstantiatedTemplateClass
    */
   bool IsTemplateInstance() const;


   /**
    * IsTopScope will return true if the current At is the top
    * (Empty) namespace
    * @return true if current sope is top namespace
    */
   bool IsTopScope() const;


   /**
    * IsUnion returns true if the At represents a Union
    * @return true if At represents a
    */
   bool IsUnion() const;


   /**
    * LookupMember will lookup a member in the current scope
    * @param nam the string representation of the member to lookup
    * @param current the current scope
    * @return if a matching member is found return it, otherwise return empty member
    */
   Member LookupMember(const std::string& nam,
                       const Scope& current) const;


   /**
    * LookupType will lookup a type in the current scope
    * @param nam the string representation of the type to lookup
    * @param current the current scope
    * @return if a matching type is found return it, otherwise return empty type
    */
   Type LookupType(const std::string& nam,
                   const Scope& current) const;


   /**
    * LookupType will lookup a scope in the current scope
    * @param nam the string representation of the scope to lookup
    * @param current the current scope
    * @return if a matching scope is found return it, otherwise return empty scope
    */
   Scope LookupScope(const std::string& nam,
                     const Scope& current) const;


   /**
    * MemberByName will return the first MemberAt with a given Name
    * @param Name  MemberAt Name
    * @return pointer to MemberAt
    */
   virtual Member MemberByName(const std::string& name,
                               const Type& signature,
                               EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * MemberAt will return the nth MemberAt of the At
    * @param  nth MemberAt
    * @return pointer to nth MemberAt
    */
   virtual Member MemberAt(size_t nth,
                           EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   virtual Member_Iterator Member_Begin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   virtual Member_Iterator Member_End(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   virtual Reverse_Member_Iterator Member_RBegin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   virtual Reverse_Member_Iterator Member_REnd(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * MemberSize will return the number of members
    * @return number of members
    */
   virtual size_t MemberSize(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * MemberTemplateAt will return the nth MemberAt template of this At
    * @param nth MemberAt template
    * @return nth MemberAt template
    */
   MemberTemplate MemberTemplateAt(size_t nth) const;


   /**
    * MemberTemplateSize will return the number of MemberAt templates in this socpe
    * @return number of defined MemberAt templates
    */
   size_t MemberTemplateSize() const;


   /**
    * MemberTemplateByName will return the member template representation in this
    * scope
    * @param string representing the member template to look for
    * @return member template representation of the looked up member
    */
   MemberTemplate MemberTemplateByName(const std::string& nam) const;


   MemberTemplate_Iterator MemberTemplate_Begin() const;
   MemberTemplate_Iterator MemberTemplate_End() const;
   Reverse_MemberTemplate_Iterator MemberTemplate_RBegin() const;
   Reverse_MemberTemplate_Iterator MemberTemplate_REnd() const;


   /**
    * Name will return the Name of the At
    * @return Name of At
    */
   virtual std::string Name(unsigned int mod = 0) const;


   /**
    * SimpleName returns the name of the type as a reference. It provides a
    * simplified but faster generation of a type name. Attention currently it
    * is not guaranteed that Name() and SimpleName() return the same character
    * layout of a name (ie. spacing, commas, etc. )
    * @param pos will indicate where in the returned reference the requested name starts
    * @param mod The only 'mod' support is SCOPED
    * @return name of type
    */
   virtual const char* SimpleName(size_t& pos,
                                  unsigned int mod = 0) const;


   /**
    * Properties will return a pointer to the PropertyNth list attached
    * to this item
    * @return pointer to PropertyNth list
    */
   virtual PropertyList Properties() const;


   /**
    * At will return the At Object of this ScopeBase
    * @return corresponding Scope
    */
   Scope ThisScope() const;


   /**
    * ScopeType will return which kind of At is represented
    * @return At of At
    */
   TYPE ScopeType() const;


   /**
    * ScopeTypeAsString will return the string representation of the enum
    * representing the current Scope (e.g. "CLASS")
    * @return string representation of enum for Scope
    */
   std::string ScopeTypeAsString() const;


   /**
    * SubScopeAt will return a pointer to a sub-scopes
    * @param  nth sub-At
    * @return pointer to nth sub-At
    */
   Scope SubScopeAt(size_t nth) const;


   /**
    * SubScopeLevel will return the number of declaring scopes
    * this scope lives in.
    * @return number of declaring scopes above this scope.
    */
   size_t SubScopeLevel() const;


   /**
    * ScopeSize will return the number of sub-scopes
    * @return number of sub-scopes
    */
   size_t SubScopeSize() const;


   /**
    * SubScopeByName will return a sub scope representing the unscoped name passed
    * as argument
    * @param unscoped name of the sub scope to look for
    * @return Scope representation of the sub scope
    */
   Scope SubScopeByName(const std::string& nam) const;


   Scope_Iterator SubScope_Begin() const;
   Scope_Iterator SubScope_End() const;
   Reverse_Scope_Iterator SubScope_RBegin() const;
   Reverse_Scope_Iterator SubScope_REnd() const;


   /**
    * At will return a pointer to the nth sub-At
    * @param  nth sub-At
    * @return pointer to nth sub-At
    */
   Type SubTypeAt(size_t nth) const;


   /**
    * TypeSize will returnt he number of sub-types
    * @return number of sub-types
    */
   size_t SubTypeSize() const;


   /**
    * SubTypeByName will return the Type representing the sub type
    * @param string of the unscoped sub type to look for
    * @return Type representation of the sub type
    */
   Type SubTypeByName(const std::string& nam) const;


   Type_Iterator SubType_Begin() const;
   Type_Iterator SubType_End() const;
   Reverse_Type_Iterator SubType_RBegin() const;
   Reverse_Type_Iterator SubType_REnd() const;


   /**
    * SubTypeTemplateAt will return the nth At template of this At
    * @param  nth sub type template
    * @return nth sub type template
    */
   TypeTemplate SubTypeTemplateAt(size_t nth) const;


   /**
    * SubTypeTemplateSize will return the number of At templates in this socpe
    * @return number of defined sub type templates
    */
   size_t SubTypeTemplateSize() const;


   /**
    * SubTypeTemplateByName will return a type template defined in this scope looked up by
    * it's unscoped name
    * @param unscoped name of the type template to look for
    * @return TypeTemplate representation of the sub type template
    */
   TypeTemplate SubTypeTemplateByName(const std::string& nam) const;


   TypeTemplate_Iterator SubTypeTemplate_Begin() const;
   TypeTemplate_Iterator SubTypeTemplate_End() const;
   Reverse_TypeTemplate_Iterator SubTypeTemplate_RBegin() const;
   Reverse_TypeTemplate_Iterator SubTypeTemplate_REnd() const;


   /**
    * UsingDirectiveAt will return the nth using directive
    * @param  nth using directive
    * @return nth using directive
    */
   Scope UsingDirectiveAt(size_t nth) const;


   /**
    * UsingDirectiveSize will return the number of using directives of this scope
    * @return number of using directives declared in this scope
    */
   size_t UsingDirectiveSize() const;


   Scope_Iterator UsingDirective_Begin() const;
   Scope_Iterator UsingDirective_End() const;
   Reverse_Scope_Iterator UsingDirective_RBegin() const;
   Reverse_Scope_Iterator UsingDirective_REnd() const;

protected:
   /** protected constructor for initialisation of the global namespace */
   ScopeBase();

public:
   /**
    * AddBase will add information about a Base class
    * @param base type of the base class
    * @param offsFP pointer to a function stub for calculating the base class offset
    * @param modifiers the modifiers of the base class
    */
   virtual void AddBase(const Type& bas,
                        OffsetFunction offsFP,
                        unsigned int modifiers = 0) const;


   /**
    * AddBase will add the information about a Base class
    * @param b pointer to the base class
    */
   virtual void AddBase(const Base& b) const;


   /**
    * AddDataMember will add the information about a data MemberAt
    * @param dm pointer to data MemberAt
    */
   virtual void AddDataMember(const Member& dm) const;
   virtual Member AddDataMember(const char* name,
                                const Type& type,
                                size_t offset,
                                unsigned int modifiers = 0,
                                char* interpreterOffset = 0) const;


   /**
    * AddFunctionMember will add the information about a function MemberAt
    * @param fm pointer to function MemberAt
    */
   virtual void AddFunctionMember(const Member& fm) const;
   virtual Member AddFunctionMember(const char* name,
                                    const Type& type,
                                    StubFunction stubFP,
                                    void* stubCtx = 0,
                                    const char* params = 0,
                                    unsigned int modifiers = 0) const;


   virtual void AddMemberTemplate(const MemberTemplate& mt) const;


   /**
    * AddSubScope will add a sub-At to this one
    * @param sc pointer to Scope
    */
   virtual void AddSubScope(const Scope& sc) const;
   virtual void AddSubScope(const char* scope,
                            TYPE scopeType) const;


   /**
    * AddSubType will add a sub-At to this At
    * @param sc pointer to Type
    */
   virtual void AddSubType(const Type& ty) const;
   virtual void AddSubType(const char* type,
                           size_t size,
                           TYPE typeType,
                           const std::type_info& ti,
                           unsigned int modifiers = 0) const;


   void AddSubTypeTemplate(const TypeTemplate& tt) const;


   void AddUsingDirective(const Scope& ud) const;


   /**
    * RemoveDataMember will remove the information about a data MemberAt
    * @param dm pointer to data MemberAt
    */
   virtual void RemoveDataMember(const Member& dm) const;


   /**
    * RemoveFunctionMember will remove the information about a function MemberAt
    * @param fm pointer to function MemberAt
    */
   virtual void RemoveFunctionMember(const Member& fm) const;


   virtual void RemoveMemberTemplate(const MemberTemplate& mt) const;


   /**
    * RemoveSubScope will remove a sub-At to this one
    * @param sc pointer to Scope
    */
   virtual void RemoveSubScope(const Scope& sc) const;


   /**
    * RemoveSubType will remove a sub-At to this At
    * @param sc pointer to Type
    */
   virtual void RemoveSubType(const Type& ty) const;


   virtual void RemoveSubTypeTemplate(const TypeTemplate& tt) const;


   void RemoveUsingDirective(const Scope& ud) const;


   /**
    * Hide this scope from any lookup by appending the string " @HIDDEN@" to its name.
    */
   virtual void HideName() const;

   /**
    * Un-Hide this scope from any lookup by removing the string " @HIDDEN@" to its name.
    */
   virtual void UnhideName() const;

   /** Initialize the vector of inherited members.
       Returns false if one of the bases is not complete. */
   virtual bool UpdateMembers() const;

   void RegisterOnDemandBuilder(OnDemandBuilder* builder,
                                EBuilderKind kind);

protected:
   /** The MemberByName work-horse: find a member called name in members,
       if signature also compare its signature, and if matchReturnType
       also compare the signature's return types. */
   Member MemberByName2(const std::vector<Member>& members,
                        const std::string& name,
                        const Type* signature = 0,
                        unsigned int modifiers_mask = 0,
                        bool matchReturnType = true) const;


   void ExecuteFunctionMemberDelayLoad() const {
      if (!fOnDemandBuilder[kBuildFunctionMembers].Empty())
         fOnDemandBuilder[kBuildFunctionMembers].BuildAll();
   }

   void ExecuteDataMemberDelayLoad() const {
      if (!fOnDemandBuilder[kBuildDataMembers].Empty())
         fOnDemandBuilder[kBuildDataMembers].BuildAll();
   }

private:
   /* no copying */
   ScopeBase(const ScopeBase &);

   /* no assignment */
   ScopeBase& operator =(const ScopeBase&);

protected:
   /** container for all members of the Scope */
   typedef std::vector<Member> Members;
   typedef std::vector<OwnedMember> OMembers;

   /**
    * pointers to members
    * @label scope members
    * @link aggregationByValue
    * @supplierCardinality 0..*
    * @clientCardinality 1
    */
   mutable
   std::vector<OwnedMember> fMembers;

   /**
    * container with pointers to all data members in this scope
    * @label scope datamembers
    * @link aggregation
    * @clientCardinality 1
    * @supplierCardinality 0..*
    */
   mutable
   std::vector<Member> fDataMembers;

   /**
    * container with pointers to all function members in this scope
    * @label scope functionmembers
    * @link aggregation
    * @supplierCardinality 0..*
    * @clientCardinality 1
    */
   mutable
   std::vector<Member> fFunctionMembers;

private:
   /**
    * pointer to the Scope Name
    * @label scope name
    * @link aggregation
    * @clientCardinality 1
    * @supplierCardinality 1
    */
   ScopeName* fScopeName;


   /**
    * Type of the scope
    * @link aggregation
    * @label scope type
    * @clientCardinality 1
    * @supplierCardinality 1
    */
   TYPE fScopeType;


   /**
    * pointer to declaring Scope
    * @label declaring scope
    * @link aggregation
    * @clientCardinality 1
    * @supplierCardinality 1
    */
   Scope fDeclaringScope;


   /**
    * pointers to sub-scopes
    * @label sub scopes
    * @link aggregation
    * @supplierCardinality 0..*
    * @clientCardinality 1
    */
   mutable
   std::vector<Scope> fSubScopes;


   /**
    * pointer to types
    * @label sub types
    * @link aggregation
    * @supplierCardinality 0..*
    * @clientCardinality 1
    */
   mutable
   std::vector<Type> fSubTypes;


   /**
    * container for type templates defined in this scope
    * @label type templates
    * @link aggregation
    * @supplierCardinality 0..*
    * @clientCardinality 1
    */
   mutable
   std::vector<TypeTemplate> fTypeTemplates;


   /**
    * container for member templates defined in this scope
    * @label member templates
    * @link aggregation
    * @supplierCardinality 0..*
    * @clientCardinality 1
    */
   mutable
   std::vector<OwnedMemberTemplate> fMemberTemplates;


   /**
    * container for using directives of this scope
    * @label using directives
    * @linkScope aggregation
    * @supplierCardinality 0..*
    * @clientCardinality 1
    */
   mutable
   std::vector<Scope> fUsingDirectives;


   /**
    * The position where the unscoped Name starts in the scopename
    */
   size_t fBasePosition;

   /**
    * Containers for on-demand builders of function and data members.
    */
   mutable
   BuilderContainer fOnDemandBuilder[kNumBuilderKinds];

};    // class ScopeBase
} //namespace Reflex


//-------------------------------------------------------------------------------
inline size_t
Reflex::ScopeBase::BaseSize() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline Reflex::Base_Iterator
Reflex::ScopeBase::Base_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::BaseCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Base_Iterator
Reflex::ScopeBase::Base_End() const {
//-------------------------------------------------------------------------------
   return Dummy::BaseCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Base_Iterator
Reflex::ScopeBase::Base_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::BaseCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Base_Iterator
Reflex::ScopeBase::Base_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::BaseCont().rend();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope
Reflex::ScopeBase::DeclaringScope() const {
//-------------------------------------------------------------------------------
   return fDeclaringScope;
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::ScopeBase::DataMember_Begin(EMEMBERQUERY) const {
//-------------------------------------------------------------------------------
   ExecuteDataMemberDelayLoad();
   return fDataMembers.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::ScopeBase::DataMember_End(EMEMBERQUERY) const {
//-------------------------------------------------------------------------------
   ExecuteDataMemberDelayLoad();
   return fDataMembers.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::ScopeBase::DataMember_RBegin(EMEMBERQUERY) const {
//-------------------------------------------------------------------------------
   ExecuteDataMemberDelayLoad();
   return ((const std::vector<Member> &)fDataMembers).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::ScopeBase::DataMember_REnd(EMEMBERQUERY) const {
//-------------------------------------------------------------------------------
   ExecuteDataMemberDelayLoad();
   return ((const std::vector<Member> &)fDataMembers).rend();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::ScopeBase::FunctionMember_Begin(EMEMBERQUERY) const {
//-------------------------------------------------------------------------------
   ExecuteFunctionMemberDelayLoad();
   return fFunctionMembers.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::ScopeBase::FunctionMember_End(EMEMBERQUERY) const {
//-------------------------------------------------------------------------------
   ExecuteFunctionMemberDelayLoad();
   return fFunctionMembers.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::ScopeBase::FunctionMember_RBegin(EMEMBERQUERY) const {
//-------------------------------------------------------------------------------
   ExecuteFunctionMemberDelayLoad();
   return ((const std::vector<Member> &)fFunctionMembers).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::ScopeBase::FunctionMember_REnd(EMEMBERQUERY) const {
//-------------------------------------------------------------------------------
   ExecuteFunctionMemberDelayLoad();
   return ((const std::vector<Member> &)fFunctionMembers).rend();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::ScopeBase::GetBasePosition() const {
//-------------------------------------------------------------------------------
   return fBasePosition;
}


//-------------------------------------------------------------------------------
inline Reflex::Scope_Iterator
Reflex::ScopeBase::SubScope_Begin() const {
//-------------------------------------------------------------------------------
   return fSubScopes.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope_Iterator
Reflex::ScopeBase::SubScope_End() const {
//-------------------------------------------------------------------------------
   return fSubScopes.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Scope_Iterator
Reflex::ScopeBase::SubScope_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Scope> &)fSubScopes).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Scope_Iterator
Reflex::ScopeBase::SubScope_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Scope> &)fSubScopes).rend();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::ScopeBase::SubType_Begin() const {
//-------------------------------------------------------------------------------
   return fSubTypes.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::ScopeBase::SubType_End() const {
//-------------------------------------------------------------------------------
   return fSubTypes.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::ScopeBase::SubType_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Type> &)fSubTypes).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::ScopeBase::SubType_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Type> &)fSubTypes).rend();
}


//-------------------------------------------------------------------------------
inline Reflex::TypeTemplate_Iterator
Reflex::ScopeBase::SubTypeTemplate_Begin() const {
//-------------------------------------------------------------------------------
   return fTypeTemplates.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::TypeTemplate_Iterator
Reflex::ScopeBase::SubTypeTemplate_End() const {
//-------------------------------------------------------------------------------
   return fTypeTemplates.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_TypeTemplate_Iterator
Reflex::ScopeBase::SubTypeTemplate_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<TypeTemplate> &)fTypeTemplates).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_TypeTemplate_Iterator
Reflex::ScopeBase::SubTypeTemplate_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<TypeTemplate> &)fTypeTemplates).rend();
}


//-------------------------------------------------------------------------------
inline
bool
Reflex::ScopeBase::HasBase(const Type&) const {
//-------------------------------------------------------------------------------
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::ScopeBase::UpdateMembers() const {
//-------------------------------------------------------------------------------
// Initialize the vector of inherited members.
// Return false if one of the bases is not complete.

   return true;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::ScopeBase::IsClass() const {
//-------------------------------------------------------------------------------
   return fScopeType == CLASS ||
          fScopeType == TYPETEMPLATEINSTANCE ||
          fScopeType == STRUCT;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::ScopeBase::IsEnum() const {
//-------------------------------------------------------------------------------
   return fScopeType == ENUM;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::ScopeBase::IsNamespace() const {
//-------------------------------------------------------------------------------
   return fScopeType == NAMESPACE;
}


inline bool
ROOT::Reflex::ScopeBase::IsTemplateInstance() const {
//-------------------------------------------------------------------------------
   return fScopeType == TYPETEMPLATEINSTANCE;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::ScopeBase::IsUnion() const {
//-------------------------------------------------------------------------------
   return fScopeType == UNION;
}


//-------------------------------------------------------------------------------
inline Reflex::TYPE
Reflex::ScopeBase::ScopeType() const {
//-------------------------------------------------------------------------------
   return fScopeType;
}


//-------------------------------------------------------------------------------
inline Reflex::Scope
Reflex::ScopeBase::SubScopeAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (nth < fSubScopes.size()) {
      return fSubScopes[nth];
   }
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::ScopeBase::SubScopeSize() const {
//-------------------------------------------------------------------------------
   return fSubScopes.size();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope
Reflex::ScopeBase::UsingDirectiveAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (nth < fUsingDirectives.size()) {
      return fUsingDirectives[nth];
   }
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::ScopeBase::UsingDirectiveSize() const {
//-------------------------------------------------------------------------------
   return fUsingDirectives.size();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope_Iterator
Reflex::ScopeBase::UsingDirective_Begin() const {
//-------------------------------------------------------------------------------
   return fUsingDirectives.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope_Iterator
Reflex::ScopeBase::UsingDirective_End() const {
//-------------------------------------------------------------------------------
   return fUsingDirectives.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Scope_Iterator
Reflex::ScopeBase::UsingDirective_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Scope> &)fUsingDirectives).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Scope_Iterator
Reflex::ScopeBase::UsingDirective_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Scope> &)fUsingDirectives).rend();
}


#ifdef _WIN32
# pragma warning( pop )
#endif

#endif // Reflex_ScopeBase
