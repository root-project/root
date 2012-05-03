// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_Scope
#define Reflex_Scope

// Include files
#include "Reflex/Kernel.h"
#include <string>
#include <typeinfo>

namespace Reflex {
// forward declarations
class Base;
class Member;
class PropertyList;
class Type;
class ScopeBase;
class ScopeName;
class TypeTemplate;
class MemberTemplate;
class DictionaryGenerator;


/**
 * @class Scope Scope.h Reflex/Scope.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class RFLX_API Scope {
public:
   /** constructor */
   Scope(const ScopeName * scopeName = 0);


   /** copy constructor */
   Scope(const Scope &rh);


   /** destructor */
   ~Scope();


   /** assignment op */
   Scope& operator=(const Scope &rh);


   /**
    * inequal operator
    */
   bool operator !=(const Scope& rh) const;


   /**
    * the bool operator will return true if the Scope is resolved (implemented)
    * @return true if Scope is implemented
    */
   operator bool() const;

#if defined(REFLEX_CINT_MERGE)
   // To prevent any un-authorized use as the old type
   bool
   operator !() const { return !operator bool(); }

   bool
   operator &&(bool right) const { return operator bool() && right; }

   bool
   operator &&(int right) const { return operator bool() && right; }

   bool
   operator &&(long right) const { return operator bool() && right; }

   bool operator &&(const Scope& right) const;
   bool operator &&(const Type& right) const;
   bool operator &&(const Member& right) const;
   bool
   operator ||(bool right) const { return operator bool() || right; }

   bool
   operator ||(int right) const { return operator bool() || right; }

   bool
   operator ||(long right) const { return operator bool() || right; }

   bool operator ||(const Scope& right) const;
   bool operator ||(const Type& right) const;
   bool operator ||(const Member& right) const;

private:
   operator int() const;

public:
#endif


   /**
    * the operator Type will return a corresponding type object to the scope if
    * applicable (i.e. if the Scope is also a Type e.g. Class, Union, Enum)
    */
   operator Type() const;


   /**
    * BaseAt will return the nth base class information
    * @param  nth base class
    * @return pointer to base class information
    */
   Base BaseAt(size_t nth) const;


   /**
    * BaseSize will return the number of base classes
    * @return number of base classes
    */
   size_t BaseSize() const;


   /**
    * Base_Begin returns the begin of the container of bases
    * @return begin of container of bases
    */
   Base_Iterator Base_Begin() const;


   /**
    * Base_End returns the end of the container of bases
    * @return end of container of bases
    */
   Base_Iterator Base_End() const;


   /**
    * Base_RBegin returns the reverse begin of the container of bases
    * @return reverse begin of container of bases
    */
   Reverse_Base_Iterator Base_RBegin() const;


   /**
    * Base_REnd returns the reverse end of the container of bases
    * @return reverse end of container of bases
    */
   Reverse_Base_Iterator Base_REnd() const;


   /**
    * ByName will return reflection information of the scope passed as argument
    * @param  name fully qualified name of the scope
    * @return reflection information of the scope
    */
   static Scope ByName(const std::string& name);


   /**
    * DataMemberAt will return the nth data member of the type
    * @param  nth the nth data member
    * @return nth data member
    */
   Member DataMemberAt(size_t nth,
                       EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * DataMemberByName will lookup a data member by name
    * @param  name of data member
    * @return data member
    */
   Member DataMemberByName(const std::string& name,
                           EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * DataMemberSize will return the number of data members of this type
    * @return number of data members
    */
   size_t DataMemberSize(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * Member_Begin returns the begin of the container of members
    * @return begin of container of members
    */
   Member_Iterator DataMember_Begin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * Member_End returns the end of the container of members
    * @return end of container of members
    */
   Member_Iterator DataMember_End(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * Member_RBegin returns the reverse begin of the container of members
    * @return reverse begin of container of members
    */
   Reverse_Member_Iterator DataMember_RBegin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * Member_REnd returns the reverse end of the container of members
    * @return reverse end of container of members
    */
   Reverse_Member_Iterator DataMember_REnd(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * DeclaringScope will return the declaring socpe of this type
    * @return declaring scope of this type
    */
   Scope DeclaringScope() const;


   /**
    * FunctionMemberAt will return the nth function member of the type
    * @param  nth function member
    * @return reflection information of nth function member
    */
   Member FunctionMemberAt(size_t nth,
                           EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * FunctionMemberByName will return the member with the name,
    * optionally the signature of the function may be given as a type
    * @param  name of function member
    * @return reflection information of the function member
    */
   Member FunctionMemberByName(const std::string& name,
                               EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT,
                               EDELAYEDLOADSETTING allowDelayedLoad = DELAYEDLOAD_ON) const;


   /**
    * FunctionMemberByName will return the member with the name,
    * optionally the signature of the function may be given as a type
    * @param  name of function member
    * @param  signature of the member function
    * @modifiers_mask When matching, do not compare the listed modifiers
    * @return reflection information of the function member
    */
   // this overloading is unfortunate but I can't include Type.h here
   Member FunctionMemberByName(const std::string& name,
                               const Type& signature,
                               unsigned int modifers_mask = 0,
                               EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT,
                               EDELAYEDLOADSETTING allowDelayedLoad = DELAYEDLOAD_ON) const;


   /**
    * FunctionMemberByNameAndSignature will return the member with the name,
    * optionally the signature of the function may be given as a type
    * @param  name of function member
    * @param  signature of the member function
    * @modifiers_mask When matching, do not compare the listed modifiers
    * @return reflection information of the function member
    */
   // this overloading is unfortunate but I can't include Type.h here
   Member FunctionMemberByNameAndSignature(const std::string& name,
                                           const Type& signature,
                                           unsigned int modifers_mask = 0,
                                           EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT,
                                           EDELAYEDLOADSETTING allowDelayedLoad = DELAYEDLOAD_ON) const;


   /**
    * FunctionMemberSize will return the number of function members of
    * this type
    * @return number of function members
    */
   size_t FunctionMemberSize(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * FunctionMember_Begin returns the begin of the container of function members
    * @return begin of container of function members
    */
   Member_Iterator FunctionMember_Begin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * FunctionMember_End returns the end of the container of function members
    * @return end of container of function members
    */
   Member_Iterator FunctionMember_End(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * FunctionMember_RBegin returns the reverse begin of the container of function members
    * @return reverse begin of container of function members
    */
   Reverse_Member_Iterator FunctionMember_RBegin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * FunctionMember_RBegin returns the reverse begin of the container of function members
    * @return reverse begin of container of function members
    */
   Reverse_Member_Iterator FunctionMember_REnd(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * GenerateDict will produce the dictionary information of this type
    * @param generator a reference to the dictionary generator instance
    */
   void GenerateDict(DictionaryGenerator& generator) const;


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
   bool HasBase(const Type& cl) const;


   /**
    * Id returns a unique identifier of the type in the system
    * @return unique identifier
    */
   void* Id() const;


   /**
    * IsClass returns true if the type represents a class
    * @return true if type represents a class
    */
   bool IsClass() const;


   /**
    * IsEnum returns true if the type represents a enum
    * @return true if type represents a enum
    */
   bool IsEnum() const;


   /**
    * IsNamespace returns true if the scope represents a namespace
    * @return true if scope represents a namespace
    */
   bool IsNamespace() const;


   /**
    * IsPrivate will check if the scope access is private
    * @return true if scope access is private
    */
   bool IsPrivate() const;


   /**
    * IsProtected will check if the scope access is protected
    * @return true if scope access is protected
    */
   bool IsProtected() const;


   /**
    * IsPublic will check if the scope access is public
    * @return true if scope access is public
    */
   bool IsPublic() const;


   /**
    * IsTemplateInstance will return true if the the class is templated
    * @return true if the class is templated
    */
   bool IsTemplateInstance() const;


   /**
    * IsTopScope returns true if this scope is the top scope
    * @return true if this scope is the top scope
    */
   bool IsTopScope() const;


   /**
    * IsUnion returns true if the type represents a union
    * @return true if type represents a union
    */
   bool IsUnion() const;


   /**
    * LookupMember will lookup a member in the current scope
    * @param nam the string representation of the member to lookup
    * @return if a matching member is found return it, otherwise return empty member
    */
   Member LookupMember(const std::string& nam) const;


   /**
    * LookupType will lookup a type in the current scope
    * @param nam the string representation of the type to lookup
    * @return if a matching type is found return it, otherwise return empty type
    */
   Type LookupType(const std::string& nam) const;


   /**
    * LookupScope will lookup a scope in the current scope
    * @param nam the string representation of the scope to lookup
    * @return if a matching scope is found return it, otherwise return empty scope
    */
   Scope LookupScope(const std::string& nam) const;


   /**
    * MemberAt will return the nth member of the type
    * @param  nth member
    * @return reflection information nth member
    */
   Member MemberAt(size_t nth,
                   EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * MemberByName will return the first member with a given Name
    * @param  member name
    * @return reflection information of the member
    */
   Member MemberByName(const std::string& name,
                       EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * MemberByName will return the first member with a given Name
    * @param  member name
    * @param  signature of the (function) member
    * @return reflection information of the member
    */
   // this overloading is unfortunate but I can't include Type.h here
   Member MemberByName(const std::string& name,
                       const Type& signature,
                       EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * MemberSize will return the number of members
    * @return number of members
    */
   size_t MemberSize(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * Member_Begin returns the begin of the container of members
    * @return begin of container of members
    */
   Member_Iterator Member_Begin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * Member_End returns the end of the container of members
    * @return end of container of members
    */
   Member_Iterator Member_End(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * Member_RBegin returns the reverse begin of the container of members
    * @return reverse begin of container of members
    */
   Reverse_Member_Iterator Member_RBegin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * Member_REnd returns the reverse end of the container of members
    * @return reverse end of container of members
    */
   Reverse_Member_Iterator Member_REnd(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * MemberTemplateAt will return the nth member template of this type
    * @param nth member template
    * @return nth member template
    */
   MemberTemplate MemberTemplateAt(size_t nth) const;


   /**
    * MemberTemplateSize will return the number of member templates in this scope
    * @return number of defined member templates
    */
   size_t MemberTemplateSize() const;


   /**
    * MemberTemplateByName will return the member template representation in this
    * scope
    * @param string representing the member template to look for
    * @return member template representation of the looked up member
    */
   MemberTemplate MemberTemplateByName(const std::string& nam) const;


   /**
    * MemberTemplate_Begin returns the begin of the container of member templates
    * @return begin of container of member templates
    */
   MemberTemplate_Iterator MemberTemplate_Begin() const;


   /**
    * MemberTemplate_End returns the end of the container of member templates
    * @return end of container of member templates
    */
   MemberTemplate_Iterator MemberTemplate_End() const;


   /**
    * MemberTemplate_End returns the end of the container of member templates
    * @return end of container of member templates
    */
   Reverse_MemberTemplate_Iterator MemberTemplate_RBegin() const;


   /**
    * MemberTemplate_REnd returns the reverse end of the container of member templates
    * @return reverse end of container of member templates
    */
   Reverse_MemberTemplate_Iterator MemberTemplate_REnd() const;


   /**
    * Name returns the name of the type
    * @param  mod qualifiers can be or'ed
    *   FINAL     - resolve typedefs
    *   SCOPED    - fully scoped name
    *   QUALIFIED - cv, reference qualification
    * @return name of the type
    */
   std::string Name(unsigned int mod = 0) const;


   /**
    * Name_c_str returns a char* pointer to the qualified type name
    * @return c string to unqualified type name
    */
   const char* Name_c_str() const;

   /**
    * Properties will return a PropertyList attached to this item
    * @return PropertyList of this type
    */
   PropertyList Properties() const;


   /**
    * ScopeAt will return the nth scope defined in the system
    * @param  nth scope defined in the system
    * @return nth scope defined in the system
    */
   static Scope ScopeAt(size_t nth);


   /**
    * ScopeSize will return the number of currently defined scopes
    * @return number of currently defined scopes
    */
   static size_t ScopeSize();


   /**
    * Scope_Begin returns the begin of the container of scopes defined in the systems
    * @return begin of container of scopes defined in the systems
    */
   static Scope_Iterator Scope_Begin();


   /**
    * Scope_End returns the end of the container of scopes defined in the systems
    * @return end of container of scopes defined in the systems
    */
   static Scope_Iterator Scope_End();


   /**
    * Scope_RBegin returns the reverse begin of the container of scopes defined in the systems
    * @return reverse begin of container of scopes defined in the systems
    */
   static Reverse_Scope_Iterator Scope_RBegin();


   /**
    * Scope_REnd returns the reverse end of the container of scopes defined in the systems
    * @return reverse end of container of scopes defined in the systems
    */
   static Reverse_Scope_Iterator Scope_REnd();


   /**
    * ScopeType will return the enum information about this scope
    * @return enum information of this scope
    */
   TYPE ScopeType() const;


   /**
    * ScopeTypeAsString will return the string representation of the ENUM
    * representing the real type of the scope (e.g. "CLASS")
    * @return string representation of the TYPE enum of the scope
    */
   std::string ScopeTypeAsString() const;


   /**
    * SubScopeAt will return a pointer to a sub scopes
    * @param  nth sub scope
    * @return reflection information of nth sub scope
    */
   Scope SubScopeAt(size_t nth) const;


   /**
    * SubScopeLevel will return the number of declaring scopes
    * this scope lives in.
    * @return number of declaring scopes above this scope.
    */
   size_t SubScopeLevel() const;


   /**
    * SubScopeSize will return the number of sub scopes
    * @return number of sub scopes
    */
   size_t SubScopeSize() const;


   /**
    * SubScopeByName will return a sub scope representing the unscoped name passed
    * as argument
    * @param unscoped name of the sub scope to look for
    * @return Scope representation of the sub scope
    */
   Scope SubScopeByName(const std::string& nam) const;


   /**
    * SubScope_Begin returns the begin of the container of sub scopes
    * @return begin of container of sub scopes
    */
   Scope_Iterator SubScope_Begin() const;


   /**
    * SubScope_End returns the end of the container of sub scopes
    * @return end of container of sub scopes
    */
   Scope_Iterator SubScope_End() const;


   /**
    * SubScope_RBegin returns the reverse begin of the container of sub scopes
    * @return reverse begin of container of sub scopes
    */
   Reverse_Scope_Iterator SubScope_RBegin() const;


   /**
    * SubScope_REnd returns the reverse end of the container of sub scopes
    * @return reverse end of container of sub scopes
    */
   Reverse_Scope_Iterator SubScope_REnd() const;


   /**
    * SubTypeAt will return the nth sub type
    * @param  nth sub type
    * @return reflection information of nth sub type
    */
   Type SubTypeAt(size_t nth) const;


   /**
    * SubTypeSize will return he number of sub types
    * @return number of sub types
    */
   size_t SubTypeSize() const;


   /**
    * SubTypeByName will return the Type representing the sub type
    * @param string of the unscoped sub type to look for
    * @return Type representation of the sub type
    */
   Type SubTypeByName(const std::string& nam) const;


   /**
    * SubType_Begin returns the begin of the container of sub types
    * @return begin of container of sub types
    */
   Type_Iterator SubType_Begin() const;


   /**
    * SubType_End returns the end of the container of sub types
    * @return end of container of sub types
    */
   Type_Iterator SubType_End() const;


   /**
    * SubType_RBegin returns the reverse begin of the container of sub types
    * @return reverse begin of container of sub types
    */
   Reverse_Type_Iterator SubType_RBegin() const;


   /**
    * SubType_REnd returns the reverse end of the container of sub types
    * @return reverse end of container of sub types
    */
   Reverse_Type_Iterator SubType_REnd() const;


   /**
    * SubTypeTemplateAt will return the nth type template of this type
    * @param nth type template
    * @return nth type template
    */
   TypeTemplate SubTypeTemplateAt(size_t nth) const;


   /**
    * SubTypeTemplateSize will return the number of type templates in this scope
    * @return number of defined type templates
    */
   size_t SubTypeTemplateSize() const;


   /**
    * SubTypeTemplateByName will return a type template defined in this scope looked up by
    * it's unscoped name
    * @param unscoped name of the type template to look for
    * @return TypeTemplate representation of the sub type template
    */
   TypeTemplate SubTypeTemplateByName(const std::string& nam) const;


   /**
    * SubTypeTemplate_Begin returns the begin of the container of sub type templates
    * @return begin of container of sub type templates
    */
   TypeTemplate_Iterator SubTypeTemplate_Begin() const;


   /**
    * SubTypeTemplate_End returns the end of the container of sub type templates
    * @return end of container of sub type templates
    */
   TypeTemplate_Iterator SubTypeTemplate_End() const;


   /**
    * SubTypeTemplate_RBegin returns the reverse begin of the container of sub type templates
    * @return reverse begin of container of sub type templates
    */
   Reverse_TypeTemplate_Iterator SubTypeTemplate_RBegin() const;


   /**
    * SubTypeTemplate_REnd returns the reverse end of the container of sub type templates
    * @return reverse end of container of sub type templates
    */
   Reverse_TypeTemplate_Iterator SubTypeTemplate_REnd() const;


   /**
    * TemplateArgumentAt will return a pointer to the nth template argument
    * @param  nth nth template argument
    * @return reflection information of nth template argument
    */
   Type TemplateArgumentAt(size_t nth) const;


   /**
    * TemplateArgumentSize will return the number of template arguments
    * @return number of template arguments
    */
   size_t TemplateArgumentSize() const;


   /**
    * TemplateArgument_Begin returns the begin of the container of template arguments
    * @return begin of container of template arguments
    */
   Type_Iterator TemplateArgument_Begin() const;


   /**
    * TemplateArgument_End returns the end of the container of template arguments
    * @return end of container of template arguments
    */
   Type_Iterator TemplateArgument_End() const;


   /**
    * TemplateArgument_RBegin returns the reverse begin of the container of template arguments
    * @return reverse begin of container of template arguments
    */
   Reverse_Type_Iterator TemplateArgument_RBegin() const;


   /**
    * TemplateArgument_REnd returns the reverse end of the container of template arguments
    * @return reverse end of container of template arguments
    */
   Reverse_Type_Iterator TemplateArgument_REnd() const;


   /**
    * TemplateFamily returns the corresponding TypeTemplate if any
    * @return corresponding TypeTemplate
    */
   TypeTemplate TemplateFamily() const;


   /**
    * Unload will unload the dictionary information of a scope
    */
   void Unload() const;


   /**
    * UpdateMembers will update the list of Function/Data/Members with all
    * members of base classes currently availabe in the system, switching
    * INHERITEDMEMBERS_DEFAULT to INHERITEDMEMBERS_ALSO.
    */
   void UpdateMembers() const;

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

public:
   /**
    * AddBase will add information about a Base class
    * @param base type of the base class
    * @param offsFP pointer to a function stub for calculating the base class offset
    * @param modifiers the modifiers of the base class
    */
   void AddBase(const Type& bas,
                OffsetFunction offsFP,
                unsigned int modifiers = 0) const;


   /**
    * AddBase will add the information about a Base class
    * @param b pointer to the base class
    */
   void AddBase(const Base& b) const;


   /**
    * AddDataMember will add the information about a data member
    * @param dm data member to add
    */
   void AddDataMember(const Member& dm) const;


   /**
    * AddDataMember will add the information about a data member
    * @param nam the name of the data member
    * @param typ the type of the data member
    * @param offs the offset of the data member relative to the beginning of the scope
    * @param modifiers of the data member
    */
   Member AddDataMember(const char* name,
                        const Type& type,
                        size_t offset,
                        unsigned int modifiers = 0,
                        char* interpreterOffset = 0) const;

   /**
    * AddFunctionMember will add the information about a function member
    * @param fm function member to add
    */
   void AddFunctionMember(const Member& fm) const;


   /**
    * AddFunctionMember will add the information about a function member
    * @param nam the name of the function member
    * @param typ the type of the function member
    * @param stubFP a pointer to the stub function
    * @param stubCtx a pointer to the context of the function member
    * @param params a semi colon separated list of parameters
    * @param modifiers of the function member
    */
   Member AddFunctionMember(const char* name,
                            const Type& type,
                            StubFunction stubFP,
                            void* stubCtx = 0,
                            const char* params = 0,
                            unsigned int modifiers = 0) const;


   /**
    * AddMemberTemplate will add a member template to this scope
    * @param mt member template to add
    */
   void AddMemberTemplate(const MemberTemplate& mt) const;


   /**
    * AddSubScope will add a sub scope to this one
    * @param sc sub scope to add
    */
   void AddSubScope(const Scope& sc) const;


   /**
    * AddSubScope will add a sub scope to this one
    * @param scop the name of the sub scope
    * @param scopeType enum value of the scope type
    */
   void AddSubScope(const char* scope,
                    TYPE scopeType = NAMESPACE) const;


   /**
    * AddSubType will add a sub type to this type
    * @param ty sub type to add
    */
   void AddSubType(const Type& ty) const;


   /**
    * AddSubType will add a sub type to this type
    * @param typ the name of the sub type
    * @param size the sizeof of the sub type
    * @param typeType the enum specifying the sub type
    * @param ti the type_info of the sub type
    * @param modifiers of the sub type
    */
   void AddSubType(const char* type,
                   size_t size,
                   TYPE typeType,
                   const std::type_info& typeInfo,
                   unsigned int modifiers = 0) const;


   /**
    * AddTypeTemplate will add a sub type template to this scope
    * @param tt type template to add
    */
   void AddSubTypeTemplate(const TypeTemplate& mt) const;


   /**
    * AddUsingDirective will add a using namespace directive to this scope
    * @param ud using directive to add
    */
   void AddUsingDirective(const Scope& ud) const;


   /**
    * RemoveDataMember will remove the information about a data member
    * @param dm data member to remove
    */
   void RemoveDataMember(const Member& dm) const;


   /**
    * RemoveFunctionMember will remove the information about a function member
    * @param fm function member to remove
    */
   void RemoveFunctionMember(const Member& fm) const;


   /**
    * RemoveMemberTemplate will remove a member template from this scope
    * @param mt member template to remove
    */
   void RemoveMemberTemplate(const MemberTemplate& mt) const;


   /**
    * RemoveSubScope will remove a sub scope from this type
    * @param sc sub scope to remove
    */
   void RemoveSubScope(const Scope& sc) const;


   /**
    * RemoveSubType will remove a sub type from this type
    * @param sc sub type to remove
    */
   void RemoveSubType(const Type& ty) const;


   /**
    * RemoveSubTypeTemplate will remove a sub type template from this scope
    * @param tt sub type template to remove
    */
   void RemoveSubTypeTemplate(const TypeTemplate& tt) const;


   /**
    * RemoveUsingDirective will remove a using namespace directive from this scope
    * @param ud using namespace directive to remove
    */
   void RemoveUsingDirective(const Scope& ud) const;


   /** */
   const ScopeBase* ToScopeBase() const;

public:
   /**
    * @label __NIRVANA__
    * @link association
    */
   static Scope& __NIRVANA__();

private:
   /**
    * pointer to the resolved scope
    * @label scope name
    * @link aggregation
    * @supplierCardinality 1
    * @clientCardinality 1
    */
   const ScopeName* fScopeName;

};    // class Scope

bool operator <(const Scope& lh,
                const Scope& rh);

bool operator ==(const Scope& lh,
                 const Scope& rh);

} // namespace Reflex

#include "Reflex/internal/ScopeBase.h"
#include "Reflex/internal/ScopeName.h"
#include "Reflex/PropertyList.h"


//-------------------------------------------------------------------------------
inline bool
Reflex::Scope::operator !=(const Scope& rh) const {
//-------------------------------------------------------------------------------
   return fScopeName != rh.fScopeName;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::operator <(const Scope& lh,
                   const Scope& rh) {
//-------------------------------------------------------------------------------
   return const_cast<Scope*>(&lh)->Id() < const_cast<Scope*>(&rh)->Id();
}


//-------------------------------------------------------------------------------
inline bool
Reflex::operator ==(const Scope& lh,
                    const Scope& rh) {
//-------------------------------------------------------------------------------
   return const_cast<Scope*>(&lh)->Id() == const_cast<Scope*>(&rh)->Id();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope::Scope(const ScopeName* scopeName)
//-------------------------------------------------------------------------------
   : fScopeName(scopeName) {
}


//-------------------------------------------------------------------------------
inline Reflex::Scope::Scope(const Scope& rh)
//-------------------------------------------------------------------------------
   : fScopeName(rh.fScopeName) {
}


//-------------------------------------------------------------------------------
inline Reflex::Scope::~Scope() {
//-------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
inline
Reflex::Scope&
Reflex::Scope::operator=(const Scope& rh) {
//-------------------------------------------------------------------------------
   if (&rh != this) {
      fScopeName = rh.fScopeName;
   }
   return *this;
}


//-------------------------------------------------------------------------------
inline
Reflex::Scope::operator bool() const {
//-------------------------------------------------------------------------------
   if (this->fScopeName && this->fScopeName->fScopeBase) {
      return true;
   }
   //throw RuntimeError("Scope is not implemented");
   return false;
}


//-------------------------------------------------------------------------------
inline Reflex::Base_Iterator
Reflex::Scope::Base_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->Base_Begin();
   }
   return Dummy::BaseCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Base_Iterator
Reflex::Scope::Base_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->Base_End();
   }
   return Dummy::BaseCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Base_Iterator
Reflex::Scope::Base_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->Base_RBegin();
   }
   return Dummy::BaseCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Base_Iterator
Reflex::Scope::Base_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->Base_REnd();
   }
   return Dummy::BaseCont().rend();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Scope::DataMember_Begin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->DataMember_Begin(inh);
   }
   return Dummy::MemberCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Scope::DataMember_End(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->DataMember_End(inh);
   }
   return Dummy::MemberCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Scope::DataMember_RBegin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->DataMember_RBegin(inh);
   }
   return Dummy::MemberCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Scope::DataMember_REnd(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->DataMember_REnd(inh);
   }
   return Dummy::MemberCont().rend();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope
Reflex::Scope::DeclaringScope() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->DeclaringScope();
   }
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Scope::FunctionMember_Begin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->FunctionMember_Begin(inh);
   }
   return Dummy::MemberCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Scope::FunctionMember_End(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->FunctionMember_End(inh);
   }
   return Dummy::MemberCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Scope::FunctionMember_RBegin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->FunctionMember_RBegin(inh);
   }
   return Dummy::MemberCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Scope::FunctionMember_REnd(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->FunctionMember_REnd(inh);
   }
   return Dummy::MemberCont().rend();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope
Reflex::Scope::GlobalScope() {
//-------------------------------------------------------------------------------
   return ScopeBase::GlobalScope();
}


//-------------------------------------------------------------------------------
inline void*
Reflex::Scope::Id() const {
//-------------------------------------------------------------------------------
   return (void*) fScopeName;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Scope::IsClass() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->IsClass();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Scope::IsEnum() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->IsEnum();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Scope::IsNamespace() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->IsNamespace();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Scope::IsTemplateInstance() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->IsTemplateInstance();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Scope::IsTopScope() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->IsTopScope();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Scope::IsUnion() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->IsUnion();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Scope::MemberSize(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->MemberSize(inh);
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline Reflex::MemberTemplate_Iterator
Reflex::Scope::MemberTemplate_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->MemberTemplate_Begin();
   }
   return Dummy::MemberTemplateCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::MemberTemplate_Iterator
Reflex::Scope::MemberTemplate_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->MemberTemplate_End();
   }
   return Dummy::MemberTemplateCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_MemberTemplate_Iterator
Reflex::Scope::MemberTemplate_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->MemberTemplate_RBegin();
   }
   return Dummy::MemberTemplateCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_MemberTemplate_Iterator
Reflex::Scope::MemberTemplate_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->MemberTemplate_REnd();
   }
   return Dummy::MemberTemplateCont().rend();
}


//-------------------------------------------------------------------------------
inline const char*
Reflex::Scope::Name_c_str() const {
//-------------------------------------------------------------------------------
   if (fScopeName) {
      return fScopeName->Name();
   }
   return "";
}


//-------------------------------------------------------------------------------
inline Reflex::PropertyList
Reflex::Scope::Properties() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->Properties();
   }
   return Dummy::PropertyList();
}


//-------------------------------------------------------------------------------
inline Reflex::TYPE
Reflex::Scope::ScopeType() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->ScopeType();
   }
   return UNRESOLVED;
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::Scope::ScopeTypeAsString() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->ScopeTypeAsString();
   }
   return "UNRESOLVED";
}


//-------------------------------------------------------------------------------
inline Reflex::Scope_Iterator
Reflex::Scope::Scope_Begin() {
//-------------------------------------------------------------------------------
   return ScopeName::Scope_Begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope_Iterator
Reflex::Scope::Scope_End() {
//-------------------------------------------------------------------------------
   return ScopeName::Scope_End();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Scope_Iterator
Reflex::Scope::Scope_RBegin() {
//-------------------------------------------------------------------------------
   return ScopeName::Scope_RBegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Scope_Iterator
Reflex::Scope::Scope_REnd() {
//-------------------------------------------------------------------------------
   return ScopeName::Scope_REnd();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope
Reflex::Scope::SubScopeAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubScopeAt(nth);
   }
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Scope::SubScopeLevel() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubScopeLevel();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Scope::SubScopeSize() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubScopeSize();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline Reflex::Scope
Reflex::Scope::SubScopeByName(const std::string& nam) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubScopeByName(nam);
   }
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope_Iterator
Reflex::Scope::SubScope_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubScope_Begin();
   }
   return Dummy::ScopeCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope_Iterator
Reflex::Scope::SubScope_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubScope_End();
   }
   return Dummy::ScopeCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Scope_Iterator
Reflex::Scope::SubScope_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubScope_RBegin();
   }
   return Dummy::ScopeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Scope_Iterator
Reflex::Scope::SubScope_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubScope_REnd();
   }
   return Dummy::ScopeCont().rend();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::Scope::SubType_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubType_Begin();
   }
   return Dummy::TypeCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::Scope::SubType_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubType_End();
   }
   return Dummy::TypeCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::Scope::SubType_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubType_RBegin();
   }
   return Dummy::TypeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::Scope::SubType_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubType_REnd();
   }
   return Dummy::TypeCont().rend();
}


//-------------------------------------------------------------------------------
inline const Reflex::ScopeBase*
Reflex::Scope::ToScopeBase() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase;
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline Reflex::TypeTemplate_Iterator
Reflex::Scope::SubTypeTemplate_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubTypeTemplate_Begin();
   }
   return Dummy::TypeTemplateCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::TypeTemplate_Iterator
Reflex::Scope::SubTypeTemplate_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubTypeTemplate_End();
   }
   return Dummy::TypeTemplateCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_TypeTemplate_Iterator
Reflex::Scope::SubTypeTemplate_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubTypeTemplate_RBegin();
   }
   return Dummy::TypeTemplateCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_TypeTemplate_Iterator
Reflex::Scope::SubTypeTemplate_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->SubTypeTemplate_REnd();
   }
   return Dummy::TypeTemplateCont().rend();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope
Reflex::Scope::UsingDirectiveAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->UsingDirectiveAt(nth);
   }
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Scope::UsingDirectiveSize() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->UsingDirectiveSize();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline Reflex::Scope_Iterator
Reflex::Scope::UsingDirective_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->UsingDirective_Begin();
   }
   return Dummy::ScopeCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope_Iterator
Reflex::Scope::UsingDirective_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->UsingDirective_End();
   }
   return Dummy::ScopeCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Scope_Iterator
Reflex::Scope::UsingDirective_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->UsingDirective_RBegin();
   }
   return Dummy::ScopeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Scope_Iterator
Reflex::Scope::UsingDirective_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fScopeName->fScopeBase->UsingDirective_REnd();
   }
   return Dummy::ScopeCont().rend();
}


//-------------------------------------------------------------------------------
inline void
Reflex::Scope::AddBase(const Type& bas,
                       OffsetFunction offsFP,
                       unsigned int modifiers /* = 0 */) const {
//-------------------------------------------------------------------------------
   if (*this) {
      fScopeName->fScopeBase->AddBase(bas, offsFP, modifiers);
   }
}


//-------------------------------------------------------------------------------
inline void
Reflex::Scope::AddBase(const Base& b) const {
//-------------------------------------------------------------------------------
   if (*this) {
      fScopeName->fScopeBase->AddBase(b);
   }
}


//-------------------------------------------------------------------------------
inline void
Reflex::Scope::AddSubScope(const Scope& sc) const {
//-------------------------------------------------------------------------------
   if (*this) {
      fScopeName->fScopeBase->AddSubScope(sc);
   }
}


//-------------------------------------------------------------------------------
inline void
Reflex::Scope::AddSubScope(const char* scope,
                           TYPE scopeType) const {
//-------------------------------------------------------------------------------
   if (*this) {
      fScopeName->fScopeBase->AddSubScope(scope, scopeType);
   }
}


//-------------------------------------------------------------------------------
inline void
Reflex::Scope::RemoveSubScope(const Scope& sc) const {
//-------------------------------------------------------------------------------
   if (*this) {
      fScopeName->fScopeBase->RemoveSubScope(sc);
   }
}


//-------------------------------------------------------------------------------
inline void
Reflex::Scope::AddUsingDirective(const Scope& ud) const {
//-------------------------------------------------------------------------------
   if (*this) {
      fScopeName->fScopeBase->AddUsingDirective(ud);
   }
}


//-------------------------------------------------------------------------------
inline void
Reflex::Scope::RemoveUsingDirective(const Scope& ud) const {
//-------------------------------------------------------------------------------
   if (*this) {
      fScopeName->fScopeBase->RemoveUsingDirective(ud);
   }
}


#ifdef REFLEX_CINT_MERGE
inline bool
operator &&(bool b,
            const Reflex::Scope& rh) {
   return b && rh.operator bool();
}


inline bool
operator &&(int i,
            const Reflex::Scope& rh) {
   return i && rh.operator bool();
}


inline bool
operator ||(bool b,
            const Reflex::Scope& rh) {
   return b || rh.operator bool();
}


inline bool
operator ||(int i,
            const Reflex::Scope& rh) {
   return i || rh.operator bool();
}


inline bool
operator &&(char* c,
            const Reflex::Scope& rh) {
   return c && rh.operator bool();
}


inline bool
operator ||(char* c,
            const Reflex::Scope& rh) {
   return c || rh.operator bool();
}

#endif

#endif // Reflex_Scope
