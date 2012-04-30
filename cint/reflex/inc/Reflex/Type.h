// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_Type
#define Reflex_Type

// Include files
#include "Reflex/Kernel.h"
#include <vector>
#include <string>
#include <typeinfo>
#include <utility>


namespace Reflex {
// forward declarations
class Base;
class Member;
class Object;
class PropertyList;
class Scope;
class TypeBase;
class TypeName;
class MemberTemplate;
class TypeTemplate;
class DictionaryGenerator;

/**
 * @class Type Type.h Reflex/Type.h
 * @author Stefan Roiser
 * @date 05/11/2004
 * @ingroup Ref
 */
class RFLX_API Type {
public:
   enum TYPE_MODIFICATION {
      REPLACE = 0,
      APPEND = 1,
      MASK = 2
   };

   /** default constructor */
   Type(const TypeName * typName = 0,
        unsigned int modifiers = 0);


   /** copy constructor */
   Type(const Type &rh);


   /**
    * copy constructor applying qualification
    * @param rh the right hand type
    * @param modifiers to be applied
    * @param operation, the default is to replace the modifiers, you can also APPEND or MASK them.  MASK removes the modifiers specified in the 2nd argument
    */
   Type(const Type &rh,
        unsigned int modifiers,
        TYPE_MODIFICATION operation = REPLACE);


   /** destructor */
   ~Type();


   /**
    * assignment operator
    */
   Type& operator =(const Type& rh);


   /**
    * equal operator
    */
   bool operator ==(const Type& rh) const;


   /**
    * not equal operator
    */
   bool operator !=(const Type& rh) const;


   /**
    * lesser than operator
    */
   bool operator <(const Type& rh) const;


   /**
    * operator Scope will return the corresponding scope of this type if
    * applicable (i.e. if the Type is also a Scope e.g. Class, Union, Enum)
    */
   operator Scope() const;


   /**
    * the bool operator returns true if the Type is resolved (implemented)
    * @return true if Type is implemented
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

   bool
   operator &&(void* right) const { return operator bool() && right; }

   bool operator &&(const Scope& right) const;
   bool operator &&(const Type& right) const;
   bool operator &&(const Member& right) const;
   bool
   operator ||(bool right) const { return operator bool() || right; }

   bool
   operator ||(int right) const { return operator bool() || right; }

   bool
   operator ||(long right) const { return operator bool() || right; }

   bool
   operator ||(void* right) const { return operator bool() || right; }

   bool operator ||(const Scope& right) const;
   bool operator ||(const Type& right) const;
   bool operator ||(const Member& right) const;

private:
   operator int() const;

public:
#endif


   /**
    * Allocate will reserve memory for the size of the object
    * @return pointer to allocated memory
    */
   void* Allocate() const;


   /**
    * ArrayLength returns the size of the array (if the type represents one)
    * @return size of array
    */
   size_t ArrayLength() const;


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
    * ByName will look for a type given as a string and return it's
    * reflection type information
    * @param  key fully qualified name of the type as string
    * @return reflection type information
    */
   static Type ByName(const std::string& key);


   /**
    * ByTypeInfo will look for a type given as a
    * std::type_info and return its reflection information
    * @param  tid std::type_info to look for
    * @return reflection information of type
    */
   static Type ByTypeInfo(const std::type_info& tid);


   /**
    * CastObject an object from this class type to another one
    * @param  to is the class type to cast into
    * @param  obj the memory address of the object to be casted
    */
   Object CastObject(const Type& to,
                     const Object& obj) const;


   /**
    * Construct will call the constructor of a given type and allocate
    * the memory for it
    * @param  signature of the constructor
    * @param  values for parameters of the constructor
    * @param  mem place in memory for implicit construction
    * @return new object
    */

   /*
      Object Construct( const Type & signature,
      const std::vector < Object > & values,
      void * mem = 0 ) const;
    */
   Object Construct(const Type& signature = Type(0, 0),
                    const std::vector<void*>& values = std::vector<void*>(),
                    void* mem = 0) const;


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
   Member DataMemberByName(const std::string& nam,
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
    * Deallocate will deallocate the memory for a given object
    * @param  instance of the type in memory
    */
   void Deallocate(void* instance) const;


   /**
    * DeclaringScope will return the declaring socpe of this type
    * @return declaring scope of this type
    */
   Scope DeclaringScope() const;


   /**
    * Destruct will call the destructor of a type and remove its memory
    * allocation if desired
    * @param  instance of the type in memory
    * @param  dealloc for also deallacoting the memory
    */
   void Destruct(void* instance,
                 bool dealloc = true) const;


   /**
    * DynamicType is used to discover the dynamic type (useful in
    * case of polymorphism)
    * @param  mem is the memory address of the object to checked
    * @return the actual class of the object
    */
   Type DynamicType(const Object& obj) const;


   /**
    * FinalType will return the type without typedefs
    * @return type with all typedef info removed
    */
   Type FinalType() const;


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
    * @param  signature of the member function
    * @return reflection information of the function member
    */
   Member FunctionMemberByName(const std::string& nam,
                               const Type& signature = Type(0, 0),
                               unsigned int modifiers_mask = 0,
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
    * FunctionMember_REnd returns the reverse end of the container of function members
    * @return reverse end of container of function members
    */
   Reverse_Member_Iterator FunctionMember_REnd(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * FunctionParameterAt returns the nth function parameter
    * @param  nth function parameter
    * @return reflection information of nth function parameter
    */
   Type FunctionParameterAt(size_t nth) const;


   /**
    * FunctionParameterSize will return the number of parameters of this function
    * @return number of parameters
    */
   size_t FunctionParameterSize() const;


   /**
    * FunctionParameter_Begin returns the begin of the container of function parameters
    * @return begin of container of function parameters
    */
   Type_Iterator FunctionParameter_Begin() const;


   /**
    * FunctionParameter_End returns the end of the container of function parameters
    * @return end of container of function parameters
    */
   Type_Iterator FunctionParameter_End() const;


   /**
    * FunctionParameter_RBegin returns the reverse begin of the container of function parameters
    * @return reverse begin of container of function parameters
    */
   Reverse_Type_Iterator FunctionParameter_RBegin() const;


   /**
    * FunctionParameter_REnd returns the reverse end of the container of function parameters
    * @return reverse end of container of function parameters
    */
   Reverse_Type_Iterator FunctionParameter_REnd() const;


   /**
    * GenerateDict will produce the dictionary information of this type
    * @param generator a reference to the dictionary generator instance
    */
   void GenerateDict(DictionaryGenerator& generator) const;


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
    * IsAbstract will return true if the the class is abstract
    * @return true if the class is abstract
    */
   bool IsAbstract() const;


   /**
    * IsArray returns true if the type represents a array
    * @return true if type represents a array
    */
   bool IsArray() const;


   /**
    * IsClass returns true if the type represents a class
    * @return true if type represents a class
    */
   bool IsClass() const;


   /**
    * IsComplete will return true if all classes and base classes of this
    * class are resolved and fully known in the system
    */
   bool IsComplete() const;


   /**
    * IsConst returns true if the type represents a const type
    * @return true if type represents a const type
    */
   bool IsConst() const;


   /**
    * IsConstVolatile returns true if the type represents a const volatile type
    * @return true if type represents a const volatile type
    */
   bool IsConstVolatile() const;


   /**
    * IsEnum returns true if the type represents a enum
    * @return true if type represents a enum
    */
   bool IsEnum() const;


   /**
    * IsEquivalentTo returns true if the two types are equivalent
    * @param type to compare to
    * @modifiers_mask do not compare the listed modifiers
    * @return true if two types are equivalent
    */
   bool IsEquivalentTo(const Type& typ,
                       unsigned int modifiers_mask = 0) const;


   /**
    * IsSignatureEquivalentTo returns true if the two types are equivalent,
    * ignoring the return type for functions
    * @param type to compare to
    * @modifiers_mask do not compare the listed modifiers
    * @return true if two types are equivalent
    */
   bool IsSignatureEquivalentTo(const Type& typ,
                                unsigned int modifiers_mask = 0) const;


   /**
    * IsFunction returns true if the type represents a function
    * @return true if type represents a function
    */
   bool IsFunction() const;


   /**
    * IsFundamental returns true if the type represents a fundamental
    * @return true if type represents a fundamental
    */
   bool IsFundamental() const;


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
    * IsPointer returns true if the type represents a pointer
    * @return true if type represents a pointer
    */
   bool IsPointer() const;


   /**
    * IsPointerToMember returns true if the type represents a pointer to member
    * @return true if type represents a pointer to member
    */
   bool IsPointerToMember() const;


   /**
    * IsReference returns true if the type represents a reference
    * @return true if type represents a reference
    */
   bool IsReference() const;


   /**
    * IsStruct will return true if the type represents a struct (not a class)
    * @return true if type represents a struct
    */
   bool IsStruct() const;


   /**
    * IsTemplateInstance will return true if the the class is templated
    * @return true if the class is templated
    */
   bool IsTemplateInstance() const;


   /**
    * IsTypedef returns true if the type represents a typedef
    * @return true if type represents a typedef
    */
   bool IsTypedef() const;


   /**
    * IsUnion returns true if the type represents a union
    * @return true if type represents a union
    */
   bool IsUnion() const;


   /**
    * IsUnqualified returns true if the type represents an unqualified type
    * @return true if type represents an unqualified type
    */
   bool IsUnqualified() const;


   /**
    * IsVirtual will return true if the class contains a virtual table
    * @return true if the class contains a virtual table
    */
   bool IsVirtual() const;


   /**
    * IsVolatile returns true if the type represents a volatile type
    * @return true if type represents a volatile type
    */
   bool IsVolatile() const;


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
    * @param  signature of the (function) member
    * @return reflection information of the member
    */
   Member MemberByName(const std::string& nam,
                       const Type& signature = Type(0, 0),
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
    * MemberTemplate_RBegin returns the reverse begin of the container of member templates
    * @return reverse begin of container of member templates
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
    * Name_c_str returns a char* pointer to the unqualified type name
    * @return c string to unqualified type name
    */
   const char* Name_c_str() const;


   /**
    * PointerToMemberScope will return the scope of the pointer to member type
    * @return scope of the pointer to member type
    */
   Scope PointerToMemberScope() const;


   /**
    * Properties will return a PropertyList attached to this item
    * @return PropertyList of this type
    */
   PropertyList Properties() const;


   /**
    * RawType will return the underlying type of a type removing all information
    * of pointers, arrays, typedefs
    * @return the raw type representation
    */
   Type RawType() const;


   /**
    * ReturnType will return the type of the return type
    * @return reflection information of the return type
    */
   Type ReturnType() const;


   /**
    * sizeof will return the size of the type
    * @return size of the type as int
    */
   size_t SizeOf() const;


   /**
    * SubScopeAt will return a pointer to a sub scopes
    * @param  nth sub scope
    * @return reflection information of nth sub scope
    */
   Scope SubScopeAt(size_t nth) const;


   /**
    * SubScopeSize will return the number of sub scopes
    * @return number of sub scopes
    */
   size_t SubScopeSize() const;


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
    * ToType will return an underlying type if possible (e.g. typedef, pointer..)
    * @return reflection information of underlying type
    */
   Type ToType() const;


   /**
    * TypeAt will return the nth Type in the system
    * @param  nth number of type to return
    * @return reflection information of nth type in the system
    */
   static Type TypeAt(size_t nth);


   /**
    * TypeSize will return the number of currently defined types in
    * the system
    * @return number of currently defined types
    */
   static size_t TypeSize();


   /**
    * Type_Begin returns the begin of the container of types in the system
    * @return begin of container of types in the system
    */
   static Type_Iterator Type_Begin();


   /**
    * Type_End returns the end of the container of types in the system
    * @return end of container of types in the system
    */
   static Type_Iterator Type_End();


   /**
    * Type_RBegin returns the reverse begin of the container of types in the system
    * @return reverse begin of container of types in the system
    */
   static Reverse_Type_Iterator Type_RBegin();


   /**
    * Type_REnd returns the reverse end of the container of types in the system
    * @return reverse end of container of types in the system
    */
   static Reverse_Type_Iterator Type_REnd();


   /**
    * TypeInfo will return the c++ type_info object of this type
    * @return type_info object of this type
    */
   const std::type_info& TypeInfo() const;


   /**
    * TypeType will return the enum information about this type
    * @return enum information of this type
    */
   TYPE TypeType() const;


   /**
    * TypeTypeAsString will return the string representation of the ENUM
    * representing the real type of the Type (e.g. "CLASS")
    * @return string representation of the TYPE enum of the Type
    */
   std::string TypeTypeAsString() const;


   /**
    * Unload will unload the dictionary information of a type from the system
    */
   void Unload() const;


   /**
    * UpdateMembers will update the list of Function/Data/Members with all
    * members of base classes currently availabe in the system
    */
   void UpdateMembers() const;

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
   Member AddDataMember(const char* nam,
                        const Type& typ,
                        size_t offs,
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
   Member AddFunctionMember(const char* nam,
                            const Type& typ,
                            StubFunction stubFP,
                            void* stubCtx = 0,
                            const char* params = 0,
                            unsigned int modifiers = 0) const;


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
   void AddSubScope(const char* scop,
                    TYPE scopeTyp = NAMESPACE) const;


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
   void AddSubType(const char* typ,
                   size_t size,
                   TYPE typeTyp,
                   const std::type_info& ti,
                   unsigned int modifiers = 0) const;


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
    * SetSize will set the size of the type. This function shall
    * be used with care. It will change the reflection information
    * of this type.
    */
   void SetSize(size_t s) const;


   /**
    * SetTypeInfo will set the type_info object of this type.
    * Attention: This will change the reflection information
    * of this type.
    */
   void SetTypeInfo(const std::type_info& ti) const;


   /** */
   const TypeBase* ToTypeBase() const;

   REPRESTYPE RepresType() const;

private:
   /**
    * pointer to the TypeName
    * @link aggregation
    * @supplierCardinality 1
    * @clientCardinality 1..
    **/
   const TypeName* fTypeName;


   /** modifiers */
   unsigned int fModifiers;

};    // class Type

} //namespace Reflex

#include "Reflex/internal/TypeName.h"
#include "Reflex/internal/TypeBase.h"
#include "Reflex/PropertyList.h"

inline Reflex::REPRESTYPE
Reflex::Type::RepresType() const {
   if (*this) {
      return fTypeName->fTypeBase->RepresType();
   }
   return REPRES_NOTYPE;
}


//-------------------------------------------------------------------------------
inline Reflex::Type&
Reflex::Type::operator =(const Type& rh) {
//-------------------------------------------------------------------------------
   if (&rh != this) {
      fTypeName = rh.fTypeName;
      fModifiers = rh.fModifiers;
   }
   return *this;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::operator ==(const Type& rh) const {
//-------------------------------------------------------------------------------
   return fTypeName == rh.fTypeName && fModifiers == rh.fModifiers;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::operator !=(const Type& rh) const {
//-------------------------------------------------------------------------------
   return fTypeName != rh.fTypeName || fModifiers != rh.fModifiers;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::operator <(const Type& rh) const {
//-------------------------------------------------------------------------------
   return Id() < rh.Id();
}


//-------------------------------------------------------------------------------
inline
Reflex::Type::operator bool() const {
//-------------------------------------------------------------------------------
   if (this->fTypeName && this->fTypeName->fTypeBase) {
      return true;
   }
   //throw RuntimeError("Type is not implemented");
   return false;
}


//-------------------------------------------------------------------------------
inline Reflex::Type::Type(const TypeName* typName,
                          unsigned int modifiers)
//-------------------------------------------------------------------------------
   : fTypeName(typName),
   fModifiers(modifiers) {
}


//-------------------------------------------------------------------------------
inline Reflex::Type::Type(const Type& rh)
//-------------------------------------------------------------------------------
   : fTypeName(rh.fTypeName),
   fModifiers(rh.fModifiers) {
}


//-------------------------------------------------------------------------------
inline Reflex::Type::Type(const Type& rh,
                          unsigned int modifiers,
                          TYPE_MODIFICATION operation)
//-------------------------------------------------------------------------------
   : fTypeName(rh.fTypeName),
   fModifiers(operation == APPEND ? rh.fModifiers | modifiers:
                 (operation == MASK ? rh.fModifiers & (~modifiers): modifiers)) {
}


//-------------------------------------------------------------------------------
inline Reflex::Type::~Type() {
//-------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
inline void*
Reflex::Type::Allocate() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->Allocate();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Type::BaseSize() const {
//-------------------------------------------------------------------------------
   return operator Scope().BaseSize();
}


//-------------------------------------------------------------------------------
inline Reflex::Base_Iterator
Reflex::Type::Base_Begin() const {
//-------------------------------------------------------------------------------
   return operator Scope().Base_Begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Base_Iterator
Reflex::Type::Base_End() const {
//-------------------------------------------------------------------------------
   return operator Scope().Base_End();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Base_Iterator
Reflex::Type::Base_RBegin() const {
//-------------------------------------------------------------------------------
   return operator Scope().Base_RBegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Base_Iterator
Reflex::Type::Base_REnd() const {
//-------------------------------------------------------------------------------
   return operator Scope().Base_REnd();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Type::DataMemberSize(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().DataMemberSize(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Type::DataMember_Begin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().DataMember_Begin(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Type::DataMember_End(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().DataMember_End(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Type::DataMember_RBegin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().DataMember_RBegin(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Type::DataMember_REnd(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().DataMember_REnd(inh);
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::Deallocate(void* instance) const {
//-------------------------------------------------------------------------------
   if (*this) {
      fTypeName->fTypeBase->Deallocate(instance);
   }
}


//-------------------------------------------------------------------------------
inline Reflex::Scope
Reflex::Type::DeclaringScope() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->DeclaringScope();
   }
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::Destruct(void* instance,
                       bool dealloc) const {
//-------------------------------------------------------------------------------
   if (*this) {
      fTypeName->fTypeBase->Destruct(instance, dealloc);
   }
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Type::FunctionMemberSize(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().FunctionMemberSize(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Type::FunctionMember_Begin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().FunctionMember_Begin(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Type::FunctionMember_End(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().FunctionMember_End(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Type::FunctionMember_RBegin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().FunctionMember_RBegin(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Type::FunctionMember_REnd(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().FunctionMember_REnd(inh);
}


//-------------------------------------------------------------------------------
inline void*
Reflex::Type::Id() const {
//-------------------------------------------------------------------------------
   return (void*) fTypeName;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsAbstract() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsAbstract();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsArray() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsArray();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsClass() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsClass();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsComplete() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsComplete();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsConst() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & CONST);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsConstVolatile() const {
//-------------------------------------------------------------------------------
   return (fModifiers & CONST) && (fModifiers & VOLATILE);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsEnum() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsEnum();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsFunction() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsFunction();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsFundamental() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsFundamental();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsPointer() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsPointer();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsPointerToMember() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsPointerToMember();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsPrivate() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsPrivate();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsProtected() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsProtected();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsPublic() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsPublic();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsReference() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & REFERENCE);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsStruct() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsStruct();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsTemplateInstance() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsTemplateInstance();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsTypedef() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsTypedef();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsUnion() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsUnion();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsUnqualified() const {
//-------------------------------------------------------------------------------
   return 0 == fModifiers;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsVirtual() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->IsVirtual();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Type::IsVolatile() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & VOLATILE);
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Type::ArrayLength() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->ArrayLength();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::Type::FinalType() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return Reflex::Type(fTypeName->fTypeBase->FinalType(), fModifiers, APPEND);
   }
   return *this;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Type::MemberTemplateSize() const {
//-------------------------------------------------------------------------------
   return operator Scope().MemberTemplateSize();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Type::MemberSize(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().MemberSize(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Type::Member_Begin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().Member_Begin(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Type::Member_End(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().Member_End(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Type::Member_RBegin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().Member_RBegin(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Type::Member_REnd(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   return operator Scope().Member_REnd(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::MemberTemplate_Iterator
Reflex::Type::MemberTemplate_Begin() const {
//-------------------------------------------------------------------------------
   return operator Scope().MemberTemplate_Begin();
}


//-------------------------------------------------------------------------------
inline Reflex::MemberTemplate_Iterator
Reflex::Type::MemberTemplate_End() const {
//-------------------------------------------------------------------------------
   return operator Scope().MemberTemplate_End();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_MemberTemplate_Iterator
Reflex::Type::MemberTemplate_RBegin() const {
//-------------------------------------------------------------------------------
   return operator Scope().MemberTemplate_RBegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_MemberTemplate_Iterator
Reflex::Type::MemberTemplate_REnd() const {
//-------------------------------------------------------------------------------
   return operator Scope().MemberTemplate_REnd();
}


//-------------------------------------------------------------------------------
inline const char*
Reflex::Type::Name_c_str() const {
//-------------------------------------------------------------------------------
   if (fTypeName) {
      return fTypeName->Name();
   }
   return "";
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::Type::FunctionParameterAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->FunctionParameterAt(nth);
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Type::FunctionParameterSize() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->FunctionParameterSize();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::Type::FunctionParameter_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->FunctionParameter_Begin();
   }
   return Dummy::TypeCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::Type::FunctionParameter_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->FunctionParameter_End();
   }
   return Dummy::TypeCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::Type::FunctionParameter_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->FunctionParameter_RBegin();
   }
   return Dummy::TypeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::Type::FunctionParameter_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->FunctionParameter_REnd();
   }
   return Dummy::TypeCont().rend();
}


//-------------------------------------------------------------------------------
inline Reflex::PropertyList
Reflex::Type::Properties() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->Properties();
   }
   return Dummy::PropertyList();
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::Type::RawType() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->RawType();
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::Type::ReturnType() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->ReturnType();
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope_Iterator
Reflex::Type::SubScope_Begin() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubScope_Begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope_Iterator
Reflex::Type::SubScope_End() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubScope_End();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Scope_Iterator
Reflex::Type::SubScope_RBegin() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubScope_RBegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Scope_Iterator
Reflex::Type::SubScope_REnd() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubScope_REnd();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Type::SizeOf() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->SizeOf();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Type::SubScopeSize() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubScopeSize();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Type::SubTypeSize() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubTypeSize();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::Type::SubType_Begin() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubType_Begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::Type::SubType_End() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubType_End();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::Type::SubType_RBegin() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubType_RBegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::Type::SubType_REnd() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubType_REnd();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::Type::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->TemplateArgument_Begin();
   }
   return Dummy::TypeCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::Type::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->TemplateArgument_End();
   }
   return Dummy::TypeCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::Type::TemplateArgument_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->TemplateArgument_RBegin();
   }
   return Dummy::TypeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::Type::TemplateArgument_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->TemplateArgument_REnd();
   }
   return Dummy::TypeCont().rend();
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::Type::ToType() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->ToType();
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::Type::Type_Begin() {
//-------------------------------------------------------------------------------
   return TypeName::Type_Begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::Type::Type_End() {
//-------------------------------------------------------------------------------
   return TypeName::Type_End();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::Type::Type_RBegin() {
//-------------------------------------------------------------------------------
   return TypeName::Type_RBegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::Type::Type_REnd() {
//-------------------------------------------------------------------------------
   return TypeName::Type_REnd();
}


//-------------------------------------------------------------------------------
inline const std::type_info&
Reflex::Type::TypeInfo() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->TypeInfo();
   }
   return typeid(void);
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Type::SubTypeTemplateSize() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubTypeTemplateSize();
}


//-------------------------------------------------------------------------------
inline Reflex::TypeTemplate_Iterator
Reflex::Type::SubTypeTemplate_Begin() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubTypeTemplate_Begin();
}


//-------------------------------------------------------------------------------
inline Reflex::TypeTemplate_Iterator
Reflex::Type::SubTypeTemplate_End() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubTypeTemplate_End();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_TypeTemplate_Iterator
Reflex::Type::SubTypeTemplate_RBegin() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubTypeTemplate_RBegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_TypeTemplate_Iterator
Reflex::Type::SubTypeTemplate_REnd() const {
//-------------------------------------------------------------------------------
   return operator Scope().SubTypeTemplate_REnd();
}


//-------------------------------------------------------------------------------
inline Reflex::TYPE
Reflex::Type::TypeType() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->TypeType();
   }
   return UNRESOLVED;
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::Type::TypeTypeAsString() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->TypeTypeAsString();
   }
   return "UNRESOLVED";
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::UpdateMembers() const {
//-------------------------------------------------------------------------------
   operator Scope().UpdateMembers();
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::Type::TemplateArgumentAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->TemplateArgumentAt(nth);
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Type::TemplateArgumentSize() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->TemplateArgumentSize();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline Reflex::TypeTemplate
Reflex::Type::TemplateFamily() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase->TemplateFamily();
   }
   return Dummy::TypeTemplate();
}


//-------------------------------------------------------------------------------
inline const Reflex::TypeBase*
Reflex::Type::ToTypeBase() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fTypeName->fTypeBase;
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::AddBase(const Type& bas,
                      OffsetFunction offsFP,
                      unsigned int modifiers /* = 0 */) const {
//-------------------------------------------------------------------------------
   operator Scope().AddBase(bas, offsFP, modifiers);
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::AddBase(const Base& b) const {
//-------------------------------------------------------------------------------
   operator Scope().AddBase(b);
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::AddDataMember(const Member& dm) const {
//-------------------------------------------------------------------------------
   operator Scope().AddDataMember(dm);
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::AddFunctionMember(const Member& fm) const {
//-------------------------------------------------------------------------------
   operator Scope().AddFunctionMember(fm);
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::AddSubScope(const Scope& sc) const {
//-------------------------------------------------------------------------------
   return operator Scope().AddSubScope(sc);
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::AddSubScope(const char* scop,
                          TYPE scopeTyp) const {
//-------------------------------------------------------------------------------
   return operator Scope().AddSubScope(scop, scopeTyp);
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::AddSubType(const Type& ty) const {
//-------------------------------------------------------------------------------
   return operator Scope().AddSubType(ty);
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::AddSubType(const char* typ,
                         size_t size,
                         TYPE typeTyp,
                         const std::type_info& ti,
                         unsigned int modifiers) const {
//-------------------------------------------------------------------------------
   return operator Scope().AddSubType(typ, size, typeTyp, ti, modifiers);
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::RemoveDataMember(const Member& dm) const {
//-------------------------------------------------------------------------------
   return operator Scope().RemoveDataMember(dm);
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::RemoveFunctionMember(const Member& fm) const {
//-------------------------------------------------------------------------------
   return operator Scope().RemoveFunctionMember(fm);
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::RemoveSubScope(const Scope& sc) const {
//-------------------------------------------------------------------------------
   return operator Scope().RemoveSubScope(sc);
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::RemoveSubType(const Type& ty) const {
//-------------------------------------------------------------------------------
   return operator Scope().RemoveSubType(ty);
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::SetSize(size_t s) const {
//-------------------------------------------------------------------------------
   if (*this) {
      fTypeName->fTypeBase->SetSize(s);
   }
}


//-------------------------------------------------------------------------------
inline void
Reflex::Type::SetTypeInfo(const std::type_info& ti) const {
//-------------------------------------------------------------------------------
   if (*this) {
      fTypeName->fTypeBase->SetTypeInfo(ti);
   }
}


#ifdef REFLEX_CINT_MERGE
inline bool
operator &&(bool b,
            const Reflex::Type& rh) {
   return b && rh.operator bool();
}


inline bool
operator &&(int i,
            const Reflex::Type& rh) {
   return i && rh.operator bool();
}


inline bool
operator ||(bool b,
            const Reflex::Type& rh) {
   return b || rh.operator bool();
}


inline bool
operator ||(int i,
            const Reflex::Type& rh) {
   return i || rh.operator bool();
}


inline bool
operator &&(const char* c,
            const Reflex::Type& rh) {
   return c && rh.operator bool();
}


inline bool
operator ||(const char* c,
            const Reflex::Type& rh) {
   return c || rh.operator bool();
}


inline bool
operator &&(char* c,
            const Reflex::Type& rh) {
   return c && rh.operator bool();
}


inline bool
operator ||(char* c,
            const Reflex::Type& rh) {
   return c || rh.operator bool();
}


#endif
#endif // Reflex_Type
