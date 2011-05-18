// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_TypeBase
#define Reflex_TypeBase

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Scope.h"
#include "Reflex/internal/OwnedPropertyList.h"
#include <vector>
#include <typeinfo>

namespace Reflex {
// forward declarations
class Type;
class Object;
class TypeTemplate;
class TypeName;
class DictionaryGenerator;

/**
 * @class TypeBase TypeBase.h Reflex/TypeBase.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class RFLX_API TypeBase {
public:
   /** default constructor */
   TypeBase(const char* nam,
            size_t size,
            TYPE typeTyp,
            const std::type_info & ti,
            const Type& finalType = Dummy::Type(),
            REPRESTYPE represType = REPRES_NOTYPE);


   /** destructor */
   virtual ~TypeBase();


   /**
    * operator Scope will return the corresponding scope of this type if
    * applicable (i.e. if the Type is also a Scope e.g. Class, Union, Enum)
    */
   operator Scope() const;


   /**
    * operator Type will return the corresponding Type object
    * @return Type corresponding to this TypeBase
    */
   operator Type() const;


   /**
    * Allocate will reserve memory for the size of the object
    * @return pointer to allocated memory
    */
   void* Allocate() const;


   /**
    * CastObject an object from this class At to another one
    * @param  to is the class At to cast into
    * @param  obj the memory AddressGet of the object to be casted
    */
   virtual Object CastObject(const Type& to,
                             const Object& obj) const;


   /**
    * Construct will call the constructor of a given At and Allocate the
    * memory for it
    * @param  signature of the constructor
    * @param  values for parameters of the constructor
    * @param  mem place in memory for implicit construction
    * @return pointer to new instance
    */

   /*
      virtual Object Construct( const Type & signature,
      const std::vector < Object > & values,
      void * mem ) const;*/
   virtual Object Construct(const Type& signature,
                            const std::vector<void*>& values,
                            void* mem) const;


   /**
    * Deallocate will Deallocate the memory for a given object
    * @param  instance of the At in memory
    */
   void Deallocate(void* instance) const;


   /**
    * Destruct will call the destructor of a At and remove its memory
    * allocation if desired
    * @param  instance of the At in memory
    * @param  dealloc for also deallacoting the memory
    */
   virtual void Destruct(void* instance,
                         bool dealloc = true) const;


   /**
    * DeclaringScope will return a pointer to the At of this one
    * @return pointer to declaring At
    */
   Scope DeclaringScope() const;


   /**
    * DynamicType is used to discover whether an object represents the
    * current class At or not
    * @param  mem is the memory AddressGet of the object to checked
    * @return the actual class of the object
    */
   virtual Type DynamicType(const Object& obj) const;


   /**
    * FinalType will return the type without typedefs
    * @return type with all typedef info removed
    */
   Type FinalType() const;


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
    * IsAbstract will return true if the the class is abstract
    * @return true if the class is abstract
    */
   virtual bool IsAbstract() const;


   /**
    * IsArray returns true if the At represents a Array
    * @return true if At represents a Array
    */
   bool IsArray() const;


   /**
    * IsClass returns true if the At represents a Class
    * @return true if At represents a Class
    */
   bool IsClass() const;


   /**
    * IsComplete will return true if all classes and BaseAt classes of this
    * class are resolved and fully known in the system
    */
   virtual bool IsComplete() const;


   /**
    * IsEnum returns true if the At represents a Enum
    * @return true if At represents a Enum
    */
   bool IsEnum() const;


   /**
    * IsFunction returns true if the At represents a Function
    * @return true if At represents a Function
    */
   bool IsFunction() const;


   /**
    * IsFundamental returns true if the At represents a Fundamental
    * @return true if At represents a Fundamental
    */
   bool IsFundamental() const;


   /**
    * IsPointer returns true if the At represents a Pointer
    * @return true if At represents a Pointer
    */
   bool IsPointer() const;


   /**
    * IsPointerToMember returns true if the At represents a PointerToMember
    * @return true if At represents a PointerToMember
    */
   bool IsPointerToMember() const;


   /**
    * IsPrivate will check if the scope access is private
    * @return true if scope access is private
    */
   virtual bool IsPrivate() const;


   /**
    * IsProtected will check if the scope access is protected
    * @return true if scope access is protected
    */
   virtual bool IsProtected() const;


   /**
    * IsPublic will check if the scope access is public
    * @return true if scope access is public
    */
   virtual bool IsPublic() const;


   /**
    * IsStruct will return true if the At represents a struct (not a class)
    * @return true if At represents a struct
    */
   bool IsStruct() const;


   /**
    * IsTemplateInstance will return true if the the class is templated
    * @return true if the class is templated
    */
   bool IsTemplateInstance() const;


   /**
    * IsTypedef returns true if the At represents a Typedef
    * @return true if At represents a Typedef
    */
   bool IsTypedef() const;


   /**
    * IsUnion returns true if the At represents a Union
    * @return true if At represents a
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
   virtual size_t ArrayLength() const;


   /**
    * Name returns the name of the type
    * @return name of type
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
    * FunctionParameterAt returns the nth FunctionParameterAt
    * @param  nth nth FunctionParameterAt
    * @return pointer to nth FunctionParameterAt At
    */
   virtual Type FunctionParameterAt(size_t nth) const;


   /**
    * FunctionParameterSize will return the number of parameters of this function
    * @return number of parameters
    */
   virtual size_t FunctionParameterSize() const;


   virtual Type_Iterator FunctionParameter_Begin() const;
   virtual Type_Iterator FunctionParameter_End() const;
   virtual Reverse_Type_Iterator FunctionParameter_RBegin() const;
   virtual Reverse_Type_Iterator FunctionParameter_REnd() const;


   /**
    * PointerToMemberScope will return the scope of the pointer to member type
    * @return scope of the pointer to member type
    */
   virtual Scope PointerToMemberScope() const;


   /**
    * Properties will return a pointer to the PropertyNth list attached
    * to this item
    * @return pointer to PropertyNth list
    */
   virtual PropertyList Properties() const;


   /**
    * RawType will return the underlying type of a type removing all information
    * of pointers, arrays, typedefs
    * @return the raw type representation
    */
   Type RawType() const;


   /**
    * ReturnType will return a pointer to the At of the return At.
    * @return pointer to Type of return At
    */
   virtual Type ReturnType() const;


   /**
    * sizeof will return the size of the At
    * @return size of the At as int
    */
   size_t SizeOf() const;


   /**
    * TemplateArgumentAt will return a pointer to the nth template argument
    * @param  nth nth template argument
    * @return pointer to nth template argument
    */
   virtual Type TemplateArgumentAt(size_t nth) const;


   /**
    * templateArgSize will return the number of template arguments
    * @return number of template arguments
    */
   virtual size_t TemplateArgumentSize() const;


   virtual Type_Iterator TemplateArgument_Begin() const;
   virtual Type_Iterator TemplateArgument_End() const;
   virtual Reverse_Type_Iterator TemplateArgument_RBegin() const;
   virtual Reverse_Type_Iterator TemplateArgument_REnd() const;


   /**
    * TemplateFamily returns the corresponding TypeTemplate if any
    * @return corresponding TypeTemplate
    */
   virtual TypeTemplate TemplateFamily() const;


   /**
    * arrayType will return a pointer to the At of the array.
    * @return pointer to Type of MemberAt et. al.
    */
   virtual Type ToType() const;


   /**
    * At returns the corresponding unqualified Type object
    * @return corresponding At object
    */
   Type ThisType() const;


   /**
    * TypeInfo will return the c++ type_info object of the At
    * @return type_info object of At
    */
   virtual const std::type_info& TypeInfo() const;


   /**
    * TypeType will return the real At
    * @return real At
    */
   TYPE TypeType() const;


   /**
    * TypeTypeAsString will return the string representation of the TYPE At
    * @return string representation of TYPE At
    */
   std::string TypeTypeAsString() const;

public:
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


   /**
    * Hide this type from any lookup by appending the string " @HIDDEN@" to its name.
    */
   virtual void HideName() const;

   /**
    * Un-Hide this type from any lookup by removing the string " @HIDDEN@" to its name.
    */
   virtual void UnhideName() const;

   REPRESTYPE
   RepresType() const { return fRepresType; }

protected:
   /**
    * DetermineFinalType will return the t without typedefs
    * @return type with all typedef info removed
    */
   Type DetermineFinalType(const Type& t) const;

   /**
    * Calculate the size for types based on other types,
    * if the other type was not yet available to calculate the
    * size at construction time.
    * @return The calculated size, 0 if the underlying size is unknown.
    */
   virtual size_t CalculateSize() const;

   /**
    * Pointer to the TypeName
    * @label At Name
    * @ling aggregation
    * @link aggregation
    * @supplierCardinality 1
    * @clientCardinality 1
    */
   TypeName* fTypeName;


   /** C++ type_info object */
   mutable
   const std::type_info * fTypeInfo;

private:
   REPRESTYPE fRepresType;

   /**
    * The Scope of the Type
    * @label type scope
    * @link aggregation
    * @clientCardinality 1
    * @supplierCardinality 1
    */
   Scope fScope;


   /** size of the type in int */
   mutable
   size_t fSize;


   /**
    * TYPE (kind) of the Type
    * @link aggregation
    * @label type type
    * @clientCardinality 1
    * @supplierCardinality 1
    */
   TYPE fTypeType;


   /**
    * Property list attached to this type
    * @label propertylist
    * @link aggregationByValue
    * @clientCardinality 1
    * @supplierCardinality 1
    */
   OwnedPropertyList fPropertyList;


   /**
    * The position where the unscoped Name starts in the typename
    */
   size_t fBasePosition;


   /**
    * the final type excluding typedefs
    * @label final typedef type
    * @link aggregation
    * @supplierCardinality 0..1
    * @clientCardinality 1
    */
   mutable
   Type * fFinalType;


   /**
    * the raw type excluding pointers, typedefs and arrays
    * @label raw type
    * @link aggregation
    * @supplierCardinality 0..1
    * @clientCardinality 1
    */
   mutable
   Type * fRawType;

};    // class TypeBase
} //namespace Reflex

#include "Reflex/TypeTemplate.h"


//-------------------------------------------------------------------------------
inline size_t
Reflex::TypeBase::CalculateSize() const {
//-------------------------------------------------------------------------------
   return fSize;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::TypeBase::GetBasePosition() const {
//-------------------------------------------------------------------------------
   return fBasePosition;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsAbstract() const {
//-------------------------------------------------------------------------------
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsArray() const {
//-------------------------------------------------------------------------------
   return fTypeType == ARRAY;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsClass() const {
//-------------------------------------------------------------------------------
   return fTypeType == CLASS ||
          fTypeType == TYPETEMPLATEINSTANCE ||
          fTypeType == STRUCT;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsComplete() const {
//-------------------------------------------------------------------------------
   return true;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsEnum() const {
//-------------------------------------------------------------------------------
   return fTypeType == ENUM;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsFunction() const {
//-------------------------------------------------------------------------------
   return fTypeType == FUNCTION;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsFundamental() const {
//-------------------------------------------------------------------------------
   return fTypeType == FUNDAMENTAL;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsPointer() const {
//-------------------------------------------------------------------------------
   return fTypeType == POINTER;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsStruct() const {
//-------------------------------------------------------------------------------
   return fTypeType == STRUCT;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsPointerToMember() const {
//-------------------------------------------------------------------------------
   return fTypeType == POINTERTOMEMBER;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsTemplateInstance() const {
//-------------------------------------------------------------------------------
   return fTypeType == TYPETEMPLATEINSTANCE ||
          fTypeType == MEMBERTEMPLATEINSTANCE;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsTypedef() const {
//-------------------------------------------------------------------------------
   return fTypeType == TYPEDEF;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsUnion() const {
//-------------------------------------------------------------------------------
   return fTypeType == UNION;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsPrivate() const {
//-------------------------------------------------------------------------------
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsProtected() const {
//-------------------------------------------------------------------------------
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsPublic() const {
//-------------------------------------------------------------------------------
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::TypeBase::IsVirtual() const {
//-------------------------------------------------------------------------------
   return false;
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::TypeBase::FunctionParameter_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::TypeBase::FunctionParameter_End() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::TypeBase::FunctionParameter_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::TypeBase::FunctionParameter_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().rend();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::TypeBase::SizeOf() const {
//-------------------------------------------------------------------------------
   if (!fSize) {
      fSize = CalculateSize();
   }
   return fSize;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::TypeBase::TemplateArgumentSize() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::TypeBase::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::TypeBase::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::TypeBase::TemplateArgument_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::TypeBase::TemplateArgument_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().rend();
}


//-------------------------------------------------------------------------------
inline Reflex::TypeTemplate
Reflex::TypeBase::TemplateFamily() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeTemplate();
}


//-------------------------------------------------------------------------------
inline const std::type_info&
Reflex::TypeBase::TypeInfo() const {
//-------------------------------------------------------------------------------
   return *fTypeInfo;
}


//-------------------------------------------------------------------------------
inline void
Reflex::TypeBase::SetSize(size_t s) const {
//-------------------------------------------------------------------------------
   fSize = s;
}


//-------------------------------------------------------------------------------
inline void
Reflex::TypeBase::SetTypeInfo(const std::type_info& ti) const {
//-------------------------------------------------------------------------------
   fTypeInfo = &ti;
}


#endif // Reflex_TypeBase
