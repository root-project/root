// @(#)root/reflex:$Name:  $:$Id: Type.h,v 1.18 2006/08/11 06:31:59 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Type
#define ROOT_Reflex_Type

// Include files
#include "Reflex/Kernel.h"
#include <vector>
#include <string>
#include <typeinfo>
#include <utility>

namespace ROOT {
   namespace Reflex {

      // forward declarations
      class Array;
      class Base;
      class Class;
      class Fundamental;
      class Function;
      class Enum;
      class InstantiatedTemplateFunction;
      class InstantiatedTemplateClass;
      class Member;
      class Object;
      class Pointer;
      class PropertyList;
      class Scope;
      class TypeBase;
      class Typedef;
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

         /** default constructor */
         Type( const TypeName * typName = 0,
               unsigned int modifiers = 0 );

      
         /** copy constructor */
         Type( const Type & rh );


         /** 
          * copy constructor applying qualification 
          * @param rh the right hand type
          * @param modifiers to be applied
          * @param append if set to true modifiers will be appended
          */
         Type( const Type & rh,
               unsigned int modifiers,
               bool append = false );
      

         /** destructor */
         ~Type();

      
         /** 
          * assignment operator
          */
         Type & operator = ( const Type & rh );


         /** 
          * equal operator 
          */
         bool operator == ( const Type & rh ) const;


         /** 
          * not equal operator
          */
         bool operator != ( const Type & rh ) const;

      
         /**
          * lesser than operator
          */
         bool operator < ( const Type & rh) const; 


         /**
          * operator Scope will return the corresponding scope of this type if
          * applicable (i.e. if the Type is also a Scope e.g. Class, Union, Enum)
          */                                       
         operator const Scope &() const;


         /**
          * the bool operator returns true if the Type is resolved (implemented)
          * @return true if Type is implemented 
          */
         operator bool () const;


         /**
          * Allocate will reserve memory for the size of the object
          * @return pointer to allocated memory
          */
         void * Allocate() const;
      

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
         const Base & BaseAt( size_t nth ) const;


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
         static const Type & ByName( const std::string & key );
      
      
         /**
          * ByTypeInfo will look for a type given as a 
          * std::type_info and return its reflection information
          * @param  tid std::type_info to look for
          * @return reflection information of type
          */
         static const Type & ByTypeInfo( const std::type_info & tid );


         /**
          * CastObject an object from this class type to another one
          * @param  to is the class type to cast into
          * @param  obj the memory address of the object to be casted
          */
         Object CastObject( const Type & to, 
                            const Object & obj ) const;
      

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
         Object Construct( const Type & signature = Type(0,0),
                           const std::vector < void * > & values = std::vector < void * > (),
                           void * mem = 0 ) const;


         /**
          * DataMemberAt will return the nth data member of the type
          * @param  nth the nth data member
          * @return nth data member 
          */
         const Member & DataMemberAt( size_t nth ) const;


         /**
          * DataMemberByName will lookup a data member by name
          * @param  name of data member
          * @return data member
          */
         const Member & DataMemberByName( const std::string & nam ) const;


         /**
          * DataMemberSize will return the number of data members of this type
          * @return number of data members
          */
         size_t DataMemberSize() const;


         /**
          * Member_Begin returns the begin of the container of members
          * @return begin of container of members
          */
         Member_Iterator DataMember_Begin() const;


         /**
          * Member_End returns the end of the container of members
          * @return end of container of members
          */
         Member_Iterator DataMember_End() const;


         /**
          * Member_RBegin returns the reverse begin of the container of members
          * @return reverse begin of container of members
          */
         Reverse_Member_Iterator DataMember_RBegin() const;


         /**
          * Member_REnd returns the reverse end of the container of members
          * @return reverse end of container of members
          */
         Reverse_Member_Iterator DataMember_REnd() const;


         /**
          * Deallocate will deallocate the memory for a given object
          * @param  instance of the type in memory
          */
         void Deallocate( void * instance ) const;


         /**
          * DeclaringScope will return the declaring socpe of this type
          * @return declaring scope of this type
          */
         const Scope & DeclaringScope() const;


         /**
          * Destruct will call the destructor of a type and remove its memory
          * allocation if desired
          * @param  instance of the type in memory
          * @param  dealloc for also deallacoting the memory
          */
         void Destruct( void * instance, 
                        bool dealloc = true ) const;


         /**
          * DynamicType is used to discover the dynamic type (useful in 
          * case of polymorphism)
          * @param  mem is the memory address of the object to checked
          * @return the actual class of the object
          */
         const Type & DynamicType( const Object & obj ) const;

         
         /**
          * FinalType will return the type without typedefs 
          * @return type with all typedef info removed
          */
         const Type & FinalType() const;


         /**
          * FunctionMemberAt will return the nth function member of the type
          * @param  nth function member
          * @return reflection information of nth function member
          */
         const Member & FunctionMemberAt( size_t nth ) const;


         /**
          * FunctionMemberByName will return the member with the name, 
          * optionally the signature of the function may be given as a type
          * @param  name of function member
          * @param  signature of the member function 
          * @return reflection information of the function member
          */
         const Member & FunctionMemberByName( const std::string & nam,
                                              const Type & signature = Type(0,0) ) const;


         /**
          * FunctionMemberSize will return the number of function members of
          * this type
          * @return number of function members
          */
         size_t FunctionMemberSize() const;

 
         /**
          * FunctionMember_Begin returns the begin of the container of function members
          * @return begin of container of function members
          */
         Member_Iterator FunctionMember_Begin() const;


         /**
          * FunctionMember_End returns the end of the container of function members
          * @return end of container of function members
          */
         Member_Iterator FunctionMember_End() const;


         /**
          * FunctionMember_RBegin returns the reverse begin of the container of function members
          * @return reverse begin of container of function members
          */
         Reverse_Member_Iterator FunctionMember_RBegin() const;


         /**
          * FunctionMember_REnd returns the reverse end of the container of function members
          * @return reverse end of container of function members
          */
         Reverse_Member_Iterator FunctionMember_REnd() const;


         /**
          * FunctionParameterAt returns the nth function parameter
          * @param  nth function parameter
          * @return reflection information of nth function parameter
          */
         const Type & FunctionParameterAt( size_t nth ) const;


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
         void GenerateDict( DictionaryGenerator & generator) const;


         /**
          * HasBase will check whether this class has a base class given
          * as argument
          * @param  cl the base-class to check for
          * @return true if this class has a base-class cl, false otherwise
          */
         bool HasBase( const Type & cl ) const;


         /**
          * Id returns a unique identifier of the type in the system
          * @return unique identifier
          */
         void * Id() const;


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
          * @return true if two types are equivalent
          */
         bool IsEquivalentTo( const Type & typ ) const;


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
         const Member & MemberAt( size_t nth ) const;


         /**
          * MemberByName will return the first member with a given Name
          * @param  member name
          * @param  signature of the (function) member 
          * @return reflection information of the member
          */
         const Member & MemberByName( const std::string & nam,
                                      const Type & signature = Type(0,0)) const;


         /**
          * MemberSize will return the number of members
          * @return number of members
          */
         size_t MemberSize() const;


         /**
          * Member_Begin returns the begin of the container of members
          * @return begin of container of members
          */
         Member_Iterator Member_Begin() const;

      
         /**
          * Member_End returns the end of the container of members
          * @return end of container of members
          */
         Member_Iterator Member_End() const;

      
         /**
          * Member_RBegin returns the reverse begin of the container of members
          * @return reverse begin of container of members
          */
         Reverse_Member_Iterator Member_RBegin() const;


         /**
          * Member_REnd returns the reverse end of the container of members
          * @return reverse end of container of members
          */
         Reverse_Member_Iterator Member_REnd() const;


         /** 
          * MemberTemplateAt will return the nth member template of this type
          * @param nth member template
          * @return nth member template
          */
         const MemberTemplate & MemberTemplateAt( size_t nth ) const;


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
         std::string Name( unsigned int mod = 0 ) const;
      
      
         /**
          * Name_c_str returns a char* pointer to the unqualified type name
          * @return c string to unqualified type name
          */
         const char * Name_c_str() const;


         /**
          * PointerToMemberScope will return the scope of the pointer to member type
          * @return scope of the pointer to member type
          */
         const Scope & PointerToMemberScope() const;


         /**
          * Properties will return a PropertyList attached to this item
          * @return PropertyList of this type
          */
         const PropertyList & Properties() const;


         /**
          * RawType will return the underlying type of a type removing all information
          * of pointers, arrays, typedefs
          * @return the raw type representation
          */
         const Type & RawType() const;


         /**
          * ReturnType will return the type of the return type
          * @return reflection information of the return type
          */
         const Type & ReturnType() const;
      

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
         const Scope & SubScopeAt( size_t nth ) const;


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
         const Type & SubTypeAt( size_t nth ) const;


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
         const TypeTemplate & SubTypeTemplateAt( size_t nth ) const;


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
         const Type & TemplateArgumentAt( size_t nth ) const;


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
         const TypeTemplate & TemplateFamily() const;


         /**
          * ToType will return an underlying type if possible (e.g. typedef, pointer..)
          * @return reflection information of underlying type
          */
         const Type & ToType() const;


         /**
          * TypeAt will return the nth Type in the system
          * @param  nth number of type to return
          * @return reflection information of nth type in the system
          */
         static const Type & TypeAt( size_t nth );


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
         const std::type_info & TypeInfo() const;


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
          * AddDataMember will add the information about a data member
          * @param dm data member to add
          */
         void AddDataMember( const Member & dm ) const;


         /**
          * AddDataMember will add the information about a data member
          * @param nam the name of the data member
          * @param typ the type of the data member
          * @param offs the offset of the data member relative to the beginning of the scope
          * @param modifiers of the data member
          */
         void AddDataMember( const char * nam,
                             const Type & typ,
                             size_t offs,
                             unsigned int modifiers = 0 ) const;


         /**
          * AddFunctionMember will add the information about a function member
          * @param fm function member to add
          */
         void AddFunctionMember( const Member & fm ) const;


         /**
          * AddFunctionMember will add the information about a function member
          * @param nam the name of the function member
          * @param typ the type of the function member
          * @param stubFP a pointer to the stub function
          * @param stubCtx a pointer to the context of the function member
          * @param params a semi colon separated list of parameters 
          * @param modifiers of the function member
          */ 
         void AddFunctionMember( const char * nam,
                                 const Type & typ,
                                 StubFunction stubFP,
                                 void * stubCtx = 0,
                                 const char * params = 0,
                                 unsigned int modifiers = 0 ) const;


         /**
          * AddSubScope will add a sub scope to this one
          * @param sc sub scope to add
          */
         void AddSubScope( const Scope & sc ) const;


         /**
          * AddSubScope will add a sub scope to this one
          * @param scop the name of the sub scope
          * @param scopeType enum value of the scope type
          */
         void AddSubScope( const char * scop,
                           TYPE scopeTyp = NAMESPACE ) const;


         /**
          * AddSubType will add a sub type to this type
          * @param ty sub type to add
          */
         void AddSubType( const Type & ty ) const;


         /**
          * AddSubType will add a sub type to this type
          * @param typ the name of the sub type
          * @param size the sizeof of the sub type
          * @param typeType the enum specifying the sub type
          * @param ti the type_info of the sub type
          * @param modifiers of the sub type
          */
         void AddSubType( const char * typ,
                          size_t size,
                          TYPE typeTyp,
                          const std::type_info & ti,
                          unsigned int modifiers = 0 ) const;


         /**
          * RemoveDataMember will remove the information about a data member
          * @param dm data member to remove
          */
         void RemoveDataMember( const Member & dm ) const;


         /**
          * RemoveFunctionMember will remove the information about a function member
          * @param fm function member to remove
          */
         void RemoveFunctionMember( const Member & fm ) const;


         /**
          * RemoveSubScope will remove a sub scope from this type
          * @param sc sub scope to remove
          */
         void RemoveSubScope( const Scope & sc ) const;


         /**
          * RemoveSubType will remove a sub type from this type
          * @param sc sub type to remove
          */
         void RemoveSubType( const Type & ty ) const;


         /** */
         const TypeBase * ToTypeBase() const;

      private:

         /** 
          * pointer to the TypeName 
          * @link aggregation
          * @supplierCardinality 1
          * @clientCardinality 1..
          **/
         const TypeName * fTypeName;


         /** modifiers */
         unsigned int fModifiers;
        
      }; // class Type

   } //namespace Reflex
} // namespace ROOT

#include "Reflex/internal/TypeName.h"
#include "Reflex/internal/TypeBase.h"
#include "Reflex/PropertyList.h"

//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type & ROOT::Reflex::Type::operator = ( const Type & rh ) {
//-------------------------------------------------------------------------------
   fTypeName = rh.fTypeName;
   fModifiers = rh.fModifiers;
   return * this;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::operator == ( const Type & rh ) const {
//-------------------------------------------------------------------------------
   return ( fTypeName == rh.fTypeName );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::operator != ( const Type & rh ) const {
//-------------------------------------------------------------------------------
   return ( fTypeName != rh.fTypeName );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::operator < ( const Type & rh ) const {
//-------------------------------------------------------------------------------
   return Id() < rh.Id();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type::operator bool () const {
//-------------------------------------------------------------------------------
   if ( this->fTypeName && this->fTypeName->fTypeBase ) return true;
   //throw RuntimeError("Type is not implemented");
   return false;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type::Type( const TypeName * typName,
                                 unsigned int modifiers ) 
//-------------------------------------------------------------------------------
   : fTypeName( typName ),
     fModifiers( modifiers ) {}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type::Type( const Type & rh )
//-------------------------------------------------------------------------------
   : fTypeName ( rh.fTypeName ),
     fModifiers ( rh.fModifiers ) {}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type::Type( const Type & rh, 
                                 unsigned int modifiers,
                                 bool append ) 
//-------------------------------------------------------------------------------
   : fTypeName( rh.fTypeName ),
     fModifiers( append ? rh.fModifiers | modifiers : modifiers ) {}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type::~Type() {
//-------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
inline void * ROOT::Reflex::Type::Allocate() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->Allocate();
   return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::BaseSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->BaseSize();
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Base_Iterator ROOT::Reflex::Type::Base_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->Base_Begin();
   return Dummy::BaseCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Base_Iterator ROOT::Reflex::Type::Base_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->Base_End();
   return Dummy::BaseCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Base_Iterator ROOT::Reflex::Type::Base_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->Base_RBegin();
   return Dummy::BaseCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Base_Iterator ROOT::Reflex::Type::Base_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->Base_REnd();
   return Dummy::BaseCont().rend();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::DataMemberSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->DataMemberSize();
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Type::DataMember_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->DataMember_Begin();
   return Dummy::MemberCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Type::DataMember_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->DataMember_End();
   return Dummy::MemberCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Type::DataMember_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->DataMember_RBegin();
   return Dummy::MemberCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Type::DataMember_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->DataMember_REnd();
   return Dummy::MemberCont().rend();
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::Deallocate( void * instance ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->Deallocate( instance ); 
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Scope & ROOT::Reflex::Type::DeclaringScope() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->DeclaringScope();
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::Destruct( void * instance, 
                                          bool dealloc ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->Destruct( instance, dealloc ); 
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::FunctionMemberSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->FunctionMemberSize();
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Type::FunctionMember_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->FunctionMember_Begin();
   return Dummy::MemberCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Type::FunctionMember_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->FunctionMember_End();
   return Dummy::MemberCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Type::FunctionMember_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->FunctionMember_RBegin();
   return Dummy::MemberCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Type::FunctionMember_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->FunctionMember_REnd();
   return Dummy::MemberCont().rend();
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::HasBase( const Type & cl ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->HasBase( cl );
   return false;
}


//-------------------------------------------------------------------------------
inline void * ROOT::Reflex::Type::Id() const {
//-------------------------------------------------------------------------------
   return (void*)fTypeName;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsAbstract() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsAbstract();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsArray() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsArray();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsClass() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsClass();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsComplete() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsComplete();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsConst() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & CONST);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsConstVolatile() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & CONST & VOLATILE);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsEnum() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsEnum();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsFunction() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsFunction();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsFundamental() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsFundamental();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsPointer() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsPointer();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsPointerToMember() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsPointerToMember();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsPrivate() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsPrivate();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsProtected() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsProtected();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsPublic() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsPublic();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsReference() const {
//-------------------------------------------------------------------------------
   return 0 != ( fModifiers & REFERENCE );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsStruct() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsStruct();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsTemplateInstance() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsTemplateInstance();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsTypedef() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsTypedef();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsUnion() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsUnion();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsUnqualified() const {
//-------------------------------------------------------------------------------
   return 0 == fModifiers;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsVirtual() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->IsVirtual();
   return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Type::IsVolatile() const {
//-------------------------------------------------------------------------------
   return 0 != ( fModifiers & VOLATILE );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::ArrayLength() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->ArrayLength();
   return 0;
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Type & ROOT::Reflex::Type::FinalType() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->FinalType();
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::MemberTemplateSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->MemberTemplateSize();
   return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::MemberSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->MemberSize();
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Type::Member_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->Member_Begin();
   return Dummy::MemberCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Type::Member_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->Member_End();
   return Dummy::MemberCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Type::Member_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->Member_RBegin();
   return Dummy::MemberCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Type::Member_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->Member_REnd();
   return Dummy::MemberCont().rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate_Iterator ROOT::Reflex::Type::MemberTemplate_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->MemberTemplate_Begin();
   return Dummy::MemberTemplateCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate_Iterator ROOT::Reflex::Type::MemberTemplate_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->MemberTemplate_End();
   return Dummy::MemberTemplateCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::Type::MemberTemplate_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->MemberTemplate_RBegin();
   return Dummy::MemberTemplateCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::Type::MemberTemplate_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->MemberTemplate_REnd();
   return Dummy::MemberTemplateCont().rend();
}


//-------------------------------------------------------------------------------
inline const char * ROOT::Reflex::Type::Name_c_str() const {
//-------------------------------------------------------------------------------
   if ( fTypeName ) return fTypeName->Name_c_str();
   return "";
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Type & ROOT::Reflex::Type::FunctionParameterAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->FunctionParameterAt( nth );
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::FunctionParameterSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->FunctionParameterSize();
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Type::FunctionParameter_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->FunctionParameter_Begin();
   return Dummy::TypeCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Type::FunctionParameter_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->FunctionParameter_End();
   return Dummy::TypeCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::FunctionParameter_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->FunctionParameter_RBegin();
   return Dummy::TypeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::FunctionParameter_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->FunctionParameter_REnd();
   return Dummy::TypeCont().rend();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::PropertyList & ROOT::Reflex::Type::Properties() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->Properties();
   return Dummy::PropertyList();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Type & ROOT::Reflex::Type::RawType() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->RawType();
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Type & ROOT::Reflex::Type::ReturnType() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->ReturnType();
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope_Iterator ROOT::Reflex::Type::SubScope_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubScope_Begin();
   return Dummy::ScopeCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope_Iterator ROOT::Reflex::Type::SubScope_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubScope_End();
   return Dummy::ScopeCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::Type::SubScope_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubScope_RBegin();
   return Dummy::ScopeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::Type::SubScope_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubScope_REnd();
   return Dummy::ScopeCont().rend();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::SizeOf() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SizeOf();
   return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::SubScopeSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubScopeSize();
   return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::SubTypeSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubTypeSize();
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Type::SubType_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubType_Begin();
   return Dummy::TypeCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Type::SubType_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubType_End();
   return Dummy::TypeCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::SubType_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubType_RBegin();
   return Dummy::TypeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::SubType_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubType_REnd();
   return Dummy::TypeCont().rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Type::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->TemplateArgument_Begin();
   return Dummy::TypeCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Type::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->TemplateArgument_End();
   return Dummy::TypeCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::TemplateArgument_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->TemplateArgument_RBegin();
   return Dummy::TypeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::TemplateArgument_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->TemplateArgument_REnd();
   return Dummy::TypeCont().rend();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Type & ROOT::Reflex::Type::ToType() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->ToType();
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Type::Type_Begin() {
//-------------------------------------------------------------------------------
   return TypeName::Type_Begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Type::Type_End() {
//-------------------------------------------------------------------------------
   return TypeName::Type_End();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::Type_RBegin() {
//-------------------------------------------------------------------------------
   return TypeName::Type_RBegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::Type_REnd() {
//-------------------------------------------------------------------------------
   return TypeName::Type_REnd();
}


//-------------------------------------------------------------------------------
inline const std::type_info & ROOT::Reflex::Type::TypeInfo() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->TypeInfo(); 
   return typeid(void);
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::SubTypeTemplateSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubTypeTemplateSize();
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::Type::SubTypeTemplate_Begin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubTypeTemplate_Begin();
   return Dummy::TypeTemplateCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::Type::SubTypeTemplate_End() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubTypeTemplate_End();
   return Dummy::TypeTemplateCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::Type::SubTypeTemplate_RBegin() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubTypeTemplate_RBegin();
   return Dummy::TypeTemplateCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::Type::SubTypeTemplate_REnd() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubTypeTemplate_REnd();
   return Dummy::TypeTemplateCont().rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TYPE ROOT::Reflex::Type::TypeType() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->TypeType();
   return UNRESOLVED;
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::Type::TypeTypeAsString() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->TypeTypeAsString(); 
   return "UNRESOLVED";
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::UpdateMembers() const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->UpdateMembers();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Type & ROOT::Reflex::Type::TemplateArgumentAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->TemplateArgumentAt( nth );
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::TemplateArgumentSize() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->TemplateArgumentSize();
   return 0;
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::TypeTemplate & ROOT::Reflex::Type::TemplateFamily() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->TemplateFamily();
   return Dummy::TypeTemplate();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::TypeBase * ROOT::Reflex::Type::ToTypeBase() const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase;
   return 0;
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::Unload() const {
//-------------------------------------------------------------------------------
  if ( * this ) delete fTypeName->fTypeBase;
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::AddDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->AddDataMember( dm );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::AddDataMember( const char * nam,
                                               const Type & typ,
                                               size_t offs,
                                               unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->AddDataMember( nam, typ, offs, modifiers );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::AddFunctionMember( const Member & fm ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->AddFunctionMember( fm );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::AddFunctionMember( const char * nam,
                                                   const Type & typ,
                                                   StubFunction stubFP,
                                                   void * stubCtx,
                                                   const char * params,
                                                   unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->AddFunctionMember( nam, typ, stubFP, stubCtx, params, modifiers );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::AddSubScope( const Scope & sc ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->AddSubScope( sc );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::AddSubScope( const char * scop,
                                             TYPE scopeTyp ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->AddSubScope( scop, scopeTyp );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::AddSubType( const Type & ty ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->AddSubType( ty );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::AddSubType( const char * typ,
                                            size_t size,
                                            TYPE typeTyp,
                                            const std::type_info & ti,
                                            unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->AddSubType( typ, size, typeTyp, ti, modifiers );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::RemoveDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->RemoveDataMember( dm );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::RemoveFunctionMember( const Member & fm ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->RemoveFunctionMember( fm );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::RemoveSubScope( const Scope & sc ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->RemoveSubScope( sc );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::RemoveSubType( const Type & ty ) const {
//-------------------------------------------------------------------------------
   if ( * this ) fTypeName->fTypeBase->RemoveSubType( ty );
}

#endif // ROOT_Reflex_Type
