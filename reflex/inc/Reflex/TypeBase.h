// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_TypeBase
#define ROOT_Reflex_TypeBase

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Scope.h"
#include "Reflex/PropertyList.h"
#include <vector>
#include <typeinfo>

namespace ROOT {
  namespace Reflex {

    // forward declarations
    class Fundamental;
    class Function;
    class Array;
    class Class;
    class Base;
    class Enum;
    class Typedef;
    class Pointer;
    class ClassTemplateInstance;
    class FunctionMemberTemplateInstance;
    class Type;
    class TypeName;
    class Object;
    class Member;
    class MemberTemplate;
    class TypeTemplate;

    /**
     * @class TypeBase TypeBase.h Reflex/TypeBase.h
     * @author Stefan Roiser
     * @date 24/11/2003
     * @ingroup Ref
     */
    class TypeBase {

    public:

      /** default constructor */
      TypeBase( const char *           TypeNth, 
                size_t                 size, 
                TYPE                   TypeType,
                const std::type_info & ti);


      /** destructor */
      virtual ~TypeBase();


      /**
       * operator Type will return the corresponding Type object
       * @return Type corresponding to this TypeBase
       */
      operator Type() const;


      /**
       * Allocate will reserve memory for the size of the object
       * @return pointer to allocated memory
       */
      void * Allocate() const;


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


      /**
       * CastObject an object from this class TypeNth to another one
       * @param  to is the class TypeNth to cast into
       * @param  obj the memory AddressGet of the object to be casted
       */
      virtual Object CastObject( const Type & to, 
                                 const Object & obj ) const;


      /**
       * Construct will call the constructor of a given TypeNth and Allocate the
       * memory for it
       * @param  signature of the constructor
       * @param  values for parameters of the constructor
       * @param  mem place in memory for implicit construction
       * @return pointer to new instance
       */
      /*
      virtual Object Construct( const Type & signature,
                                std::vector < Object > values,
                                void * mem) const;*/
      virtual Object Construct( const Type & signature,
                                std::vector < void * > values,
                                void * mem) const;
      
      /**
       * DataMemberNth will return the nth data MemberNth of the ScopeNth
       * @param  nth data MemberNth
       * @return pointer to data MemberNth
       */
      virtual Member DataMemberNth( size_t nth ) const;


      /**
       * DataMemberCount will return the number of data members of this ScopeNth
       * @return number of data members
       */
      virtual size_t DataMemberCount() const;


      /**
       * Deallocate will Deallocate the memory for a given object
       * @param  instance of the TypeNth in memory
       */
      void Deallocate( void * instance ) const;


      /**
       * Destruct will call the destructor of a TypeNth and remove its memory
       * allocation if desired
       * @param  instance of the TypeNth in memory
       * @param  dealloc for also deallacoting the memory
       */
      virtual void Destruct( void * instance, 
                             bool dealloc = true ) const;


      /**
       * DynamicType is used to discover whether an object represents the
       * current class TypeNth or not
       * @param  mem is the memory AddressGet of the object to checked
       * @return the actual class of the object
       */
      virtual Type DynamicType( const Object & obj ) const;


      /**
       * FunctionMemberNth will return the nth function MemberNth of the ScopeNth
       * @param  nth function MemberNth
       * @return pointer to function MemberNth
       */
      virtual Member FunctionMemberNth( size_t nth ) const;


      /**
       * FunctionMemberCount will return the number of function members of
       * this ScopeNth
       * @return number of function members
       */
      virtual size_t FunctionMemberCount() const;


      /**
       * HasBase will check whether this class has a BaseNth class given
       * as argument
       * @param  cl the BaseNth-class to check for
       * @return true if this class has a BaseNth-class cl, false otherwise
       */
      virtual bool HasBase( const Type & cl ) const;


      /**
       * IsAbstract will return true if the the class is abstract
       * @return true if the class is abstract
       */
      virtual bool IsAbstract() const;


      /** 
       * IsArray returns true if the TypeNth represents a Array
       * @return true if TypeNth represents a Array
       */
      bool IsArray() const;


      /** 
       * IsClass returns true if the TypeNth represents a Class
       * @return true if TypeNth represents a Class
       */
      bool IsClass() const;


      /** 
       * IsComplete will return true if all classes and BaseNth classes of this 
       * class are resolved and fully known in the system
       */
      virtual bool IsComplete() const;


      /** 
       * IsEnum returns true if the TypeNth represents a Enum
       * @return true if TypeNth represents a Enum
       */
      bool IsEnum() const;


      /** 
       * IsFunction returns true if the TypeNth represents a Function
       * @return true if TypeNth represents a Function
       */
      bool IsFunction() const;


      /** 
       * IsFundamental returns true if the TypeNth represents a Fundamental
       * @return true if TypeNth represents a Fundamental
       */
      bool IsFundamental() const;


      /** 
       * IsPointer returns true if the TypeNth represents a Pointer
       * @return true if TypeNth represents a Pointer
       */
      bool IsPointer() const;


      /** 
       * IsPointerToMember returns true if the TypeNth represents a PointerToMember
       * @return true if TypeNth represents a PointerToMember
       */
      bool IsPointerToMember() const;


      /**
       * IsStruct will return true if the TypeNth represents a struct (not a class)
       * @return true if TypeNth represents a struct
       */
      bool IsStruct() const;


      /**
       * IsTemplateInstance will return true if the the class is templated
       * @return true if the class is templated
       */
      bool IsTemplateInstance() const;


      /** 
       * IsTypedef returns true if the TypeNth represents a Typedef
       * @return true if TypeNth represents a Typedef
       */
      bool IsTypedef() const;


      /** 
       * IsUnion returns true if the TypeNth represents a Union
       * @return true if TypeNth represents a 
       */
      bool IsUnion() const;


      /**
       * IsVirtual will return true if the class contains a virtual table
       * @return true if the class contains a virtual table
       */
      virtual bool IsVirtual() const;


      /** Array
       * size returns the size of the array
       * @return size of array
       */
      virtual size_t Length() const;

      
      /**
       * MemberNth will return the first MemberNth with a given Name
       * @param  MemberNth Name
       * @return pointer to MemberNth
       */
      virtual Member MemberNth( const std::string & Name ) const;


      /**
       * MemberNth will return the nth MemberNth of the ScopeNth
       * @param  nth MemberNth
       * @return pointer to nth MemberNth
       */
      virtual Member MemberNth( size_t nth ) const;


      /**
       * MemberCount will return the number of members
       * @return number of members
       */
      virtual size_t MemberCount() const;


      /** 
       * MemberTemplateNth will return the nth MemberNth template of this ScopeNth
       * @param nth MemberNth template
       * @return nth MemberNth template
       */
      virtual MemberTemplate MemberTemplateNth( size_t nth ) const;


      /** 
       * MemberTemplateCount will return the number of MemberNth templates in this socpe
       * @return number of defined MemberNth templates
       */
      virtual size_t MemberTemplateCount() const;


      /**
       * Name returns the Name of the TypeNth
       * @return Name of TypeNth
       */
      virtual std::string Name( unsigned int mod = 0 ) const;

 
      /**
       * ParameterNth returns the nth ParameterNth
       * @param  nth nth ParameterNth
       * @return pointer to nth ParameterNth TypeNth
       */
      virtual Type ParameterNth( size_t nth ) const;


      /**
       * ParameterCount will return the number of parameters of this function
       * @return number of parameters
       */
      virtual size_t ParameterCount() const;


      /**
       * PropertyListGet will return a pointer to the PropertyNth list attached
       * to this item
       * @return pointer to PropertyNth list
       */
      virtual PropertyList PropertyListGet() const;


      /**
       * ReturnType will return a pointer to the TypeNth of the return TypeNth.
       * @return pointer to Type of return TypeNth
       */
      virtual Type ReturnType() const;
      

      /**
       * ScopeNth will return the ScopeNth of the Type if any 
       * @return pointer to ScopeNth of TypeNth
       */
      Scope ScopeGet() const;


      /**
       * sizeof will return the size of the TypeNth
       * @return size of the TypeNth as int
       */
      size_t SizeOf() const;


      /**
       * SubScopeNth will return a pointer to a sub-scopes
       * @param  nth sub-ScopeNth
       * @return pointer to nth sub-ScopeNth
       */
      virtual Scope SubScopeNth( size_t nth ) const;


      /**
       * ScopeCount will return the number of sub-scopes
       * @return number of sub-scopes
       */
      virtual size_t SubScopeCount() const;


      /**
       * nthType will return a pointer to the nth sub-TypeNth
       * @param  nth sub-TypeNth
       * @return pointer to nth sub-TypeNth
       */
      virtual Type SubTypeNth( size_t nth ) const;


      /**
       * TypeCount will returnt he number of sub-types
       * @return number of sub-types
       */
      virtual size_t SubTypeCount() const;


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


      /**
       * TemplateFamily returns the corresponding TypeTemplate if any
       * @return corresponding TypeTemplate
       */
      virtual TypeTemplate TemplateFamily() const;


      /**
       * arrayType will return a pointer to the TypeNth of the array.
       * @return pointer to Type of MemberNth et. al.
       */
      virtual Type ToType() const;


      /** 
       * TypeNth returns the corresponding unqualified Type object
       * @return corresponding TypeNth object
       */
      Type TypeGet() const;


      /** 
       * TypeTemplateNth will return the nth TypeNth template of this ScopeNth
       * @param nth TypeNth template
       * @return nth TypeNth template
       */
      virtual TypeTemplate TypeTemplateNth( size_t nth ) const;


      /** 
       * TypeTemplateCount will return the number of TypeNth templates in this socpe
       * @return number of defined TypeNth templates
       */
      virtual size_t TypeTemplateCount() const;


      /**
       * TypeInfo will return the c++ type_info object of the TypeNth
       * @return type_info object of TypeNth
       */
      virtual const std::type_info & TypeInfo() const;


      /**
       * TypeType will return the real TypeNth
       * @return real TypeNth
       */
      TYPE TypeType() const;


      /**
       * TypeTypeAsString will return the string representation of the TYPE TypeNth
       * @return string representation of TYPE TypeNth
       */
      std::string TypeTypeAsString() const;


      /** 
       * UpdateMembers2 will update the list of Function/Data/Members with all
       * MemberNth of BaseNth classes currently availabe in the system
       */
      virtual void UpdateMembers() const;

    public:

      /**
       * AddDataMember will add the information about a data MemberNth
       * @param dm pointer to data MemberNth
       */
      virtual void AddDataMember( const Member & dm ) const;
      virtual void AddDataMember( const char * Name,
                                  const Type & TypeNth,
                                  size_t Offset,
                                  unsigned int modifiers = 0 ) const;


      /**
       * AddFunctionMember will add the information about a function MemberNth
       * @param fm pointer to function MemberNth
       */
      virtual void AddFunctionMember( const Member & fm ) const;
      virtual void AddFunctionMember( const char * Name,
                                      const Type & TypeNth,
                                      StubFunction stubFP,
                                      void * stubCtx = 0,
                                      const char * params = 0,
                                      unsigned int modifiers = 0 ) const;


      /**
       * AddSubScope will add a sub-ScopeNth to this one
       * @param sc pointer to Scope
       */
      virtual void AddSubScope( const Scope & sc ) const;
      virtual void AddSubScope( const char * ScopeNth,
                                TYPE ScopeType ) const;


      /**
       * AddSubType will add a sub-TypeNth to this ScopeNth
       * @param sc pointer to Type
       */
      virtual void AddSubType( const Type & ty ) const;
      virtual void AddSubType( const char * TypeNth,
                               size_t size,
                               TYPE TypeType,
                               const std::type_info & TypeInfo,
                               unsigned int modifiers ) const;


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

    protected:

      /**
       * Pointer to the TypeName 
       * @label TypeNth Name
       * @ling aggregation
       * @link aggregation
       * @supplierCardinality 1
       * @clientCardinality 1
       */
      TypeName * fTypeName;

    private:

      /**
       * The ScopeNth of the TypeNth
       * @label ScopeNth TypeNth
       * @link aggregationByValue
       * @clientCardinality 0..1
       * @supplierCardinality 1
       */
      Scope fScope;


      /** size of the TypeNth in int */
      size_t fSize;


      /** C++ type_info object */
      const std::type_info & fTypeInfo;


      /**
       * TypeNth of the TypeNth
       * @link aggregationByValue
       * @label TypeType
       * @clientCardinality 1
       * @supplierCardinality 1
       */
      TYPE fTypeType;


      /**
       * pointer to the PropertyNth list
       * @label propertylist
       * @link aggregationByValue
       * @clientCardinality 1
       * @supplierCardinality 1
       */
      PropertyList fPropertyList;


      /**
       * The position where the unscoped Name starts in the typename
       */
      size_t fBasePosition;

    }; // class TypeBase
  } //namespace Reflex
} //namespace ROOT

#include "Reflex/TypeTemplate.h"

//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::DataMemberCount() const {
//-------------------------------------------------------------------------------
  return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::FunctionMemberCount() const {
//-------------------------------------------------------------------------------
  return 0;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::HasBase( const Type & /* cl */ ) const {
//-------------------------------------------------------------------------------
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::IsAbstract() const {
//-------------------------------------------------------------------------------
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::IsArray() const {
//-------------------------------------------------------------------------------
  return ( fTypeType == ARRAY );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::IsClass() const {
//-------------------------------------------------------------------------------
  return ( fTypeType == CLASS || 
           fTypeType == TYPETEMPLATEINSTANCE || 
           fTypeType == STRUCT );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::IsComplete() const {
//-------------------------------------------------------------------------------
  return true;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::IsEnum() const {
//-------------------------------------------------------------------------------
  return ( fTypeType == ENUM );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::IsFunction() const {
//-------------------------------------------------------------------------------
  return ( fTypeType == FUNCTION );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::IsFundamental() const {
//-------------------------------------------------------------------------------
  return ( fTypeType == FUNDAMENTAL );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::IsPointer() const {
//-------------------------------------------------------------------------------
  return ( fTypeType == POINTER );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::IsStruct() const {
//-------------------------------------------------------------------------------
  return ( fTypeType == STRUCT );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::IsPointerToMember() const {
//-------------------------------------------------------------------------------
  return ( fTypeType == POINTERTOMEMBER );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::IsTemplateInstance() const {
//-------------------------------------------------------------------------------
  return ( fTypeType == TYPETEMPLATEINSTANCE || 
           fTypeType == MEMBERTEMPLATEINSTANCE );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::IsTypedef() const {
//-------------------------------------------------------------------------------
  return ( fTypeType == TYPEDEF );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::IsUnion() const {
//-------------------------------------------------------------------------------
  return ( fTypeType == UNION );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::TypeBase::IsVirtual() const {
//-------------------------------------------------------------------------------
  return false;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::MemberCount() const {
//-------------------------------------------------------------------------------
  return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::MemberTemplateCount() const {
//-------------------------------------------------------------------------------
  return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::SizeOf() const { 
//-------------------------------------------------------------------------------
  return fSize; 
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::SubScopeCount() const {
//-------------------------------------------------------------------------------
  return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::SubTypeCount() const {
//-------------------------------------------------------------------------------
  return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::TemplateArgumentCount() const {
//-------------------------------------------------------------------------------
  return 0;
}


//-------------------------------------------------------------------------------
inline const std::type_info & ROOT::Reflex::TypeBase::TypeInfo() const {
//-------------------------------------------------------------------------------
  return fTypeInfo;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate ROOT::Reflex::TypeBase::TemplateFamily() const {
//-------------------------------------------------------------------------------
  return TypeTemplate();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::TypeTemplateCount() const {
//-------------------------------------------------------------------------------
  return 0;
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::UpdateMembers() const {
//-------------------------------------------------------------------------------
  throw RuntimeError("This function can only be called on Class/Struct");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::AddDataMember( const Member & /* dm */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("Function AddDataMember  not callable on this object");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::AddDataMember( const char * /* Name */,
                                                   const Type & /* TypeNth */,
                                                   size_t /* Offset */,
                                                   unsigned int /* modifiers */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("Function AddDataMember  not callable on this object");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::AddFunctionMember( const Member & /* fm */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("Function AddFunctionMember not callable on this object");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::AddFunctionMember( const char * /* Name */,
                                                       const Type & /* TypeNth */,
                                                       StubFunction /* stubFP */,
                                                       void * /* stubCtx */,
                                                       const char * /* params */,
                                                       unsigned int /* modifiers */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("Function AddFunctionMember not callable on this object");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::AddSubScope( const Scope & /* sc */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("Function AddSubScope not callable on this object");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::AddSubScope( const char * /* ScopeNth */,
                                                 TYPE /* ScopeType */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("Function AddSubScope not callable on this object");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::AddSubType( const Type & /* ty */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("Function AddSubType not callable on this object");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::AddSubType( const char * /* TypeNth */,
                                                size_t /* size */,
                                                TYPE /* TypeType */,
                                                const std::type_info & /* TypeInfo */,
                                                unsigned int /* modifiers */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("Function AddSubType not callable on this object");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::RemoveDataMember( const Member & /* dm */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("Function RemoveDataMember not callable on this object");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::RemoveFunctionMember( const Member & /* fm */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("Function RemoveFunctionMember not callable on this object");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::RemoveSubScope( const Scope & /* sc */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("Function RemoveSubScope not callable on this object");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::RemoveSubType( const Type & /* ty */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("Function RemoveSubType not callable on this object");
}

#endif // ROOT_Reflex_TypeBase



