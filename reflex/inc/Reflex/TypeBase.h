// @(#)root/reflex:$Name:  $:$Id: TypeBase.h,v 1.10 2006/06/08 16:05:14 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
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
      class RFLX_API TypeBase {

      public:

         /** default constructor */
         TypeBase( const char *           nam, 
                   size_t                 size, 
                   TYPE                   typeTyp,
                   const std::type_info & ti);


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
         void * Allocate() const;


         /**
          * nthBase will return the nth BaseAt class information
          * @param  nth nth BaseAt class
          * @return pointer to BaseAt class information
          */
         virtual Base BaseAt( size_t nth ) const;


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
          * CastObject an object from this class At to another one
          * @param  to is the class At to cast into
          * @param  obj the memory AddressGet of the object to be casted
          */
         virtual Object CastObject( const Type & to, 
                                    const Object & obj ) const;


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
         virtual Object Construct( const Type & signature,
                                   const std::vector < void * > & values,
                                   void * mem) const;
      

         /**
          * DataMemberAt will return the nth data MemberAt of the At
          * @param  nth data MemberAt
          * @return pointer to data MemberAt
          */
         virtual Member DataMemberAt( size_t nth ) const;


         /**
          * DataMemberByName will return the MemberAt with Name
          * @param  Name of data MemberAt
          * @return data MemberAt
          */
         virtual Member DataMemberByName( const std::string & nam ) const;


         /**
          * DataMemberSize will return the number of data members of this At
          * @return number of data members
          */
         virtual size_t DataMemberSize() const;


         virtual Member_Iterator DataMember_Begin() const;
         virtual Member_Iterator DataMember_End() const;
         virtual Reverse_Member_Iterator DataMember_RBegin() const;
         virtual Reverse_Member_Iterator DataMember_REnd() const;


         /**
          * Deallocate will Deallocate the memory for a given object
          * @param  instance of the At in memory
          */
         void Deallocate( void * instance ) const;


         /**
          * Destruct will call the destructor of a At and remove its memory
          * allocation if desired
          * @param  instance of the At in memory
          * @param  dealloc for also deallacoting the memory
          */
         virtual void Destruct( void * instance, 
                                bool dealloc = true ) const;


         /**
          * DeclaringScope will return a pointer to the At of this one
          * @return pointer to declaring At
          */
         virtual Scope DeclaringScope() const;


         /**
          * DynamicType is used to discover whether an object represents the
          * current class At or not
          * @param  mem is the memory AddressGet of the object to checked
          * @return the actual class of the object
          */
         virtual Type DynamicType( const Object & obj ) const;


         /**
          * FunctionMemberAt will return the nth function MemberAt of the At
          * @param  nth function MemberAt
          * @return pointer to function MemberAt
          */
         virtual Member FunctionMemberAt( size_t nth ) const;


         /**
          * FunctionMemberByName will return the MemberAt with the Name, 
          * optionally the signature of the function may be given
          * @param  Name of function MemberAt
          * @param  signature of the MemberAt function 
          * @return function MemberAt
          */
         virtual Member FunctionMemberByName( const std::string & nam,
                                              const Type & signature ) const;


         /**
          * FunctionMemberSize will return the number of function members of
          * this At
          * @return number of function members
          */
         virtual size_t FunctionMemberSize() const;


         virtual Member_Iterator FunctionMember_Begin() const;
         virtual Member_Iterator FunctionMember_End() const;
         virtual Reverse_Member_Iterator FunctionMember_RBegin() const;
         virtual Reverse_Member_Iterator FunctionMember_REnd() const;


         /**
          * HasBase will check whether this class has a BaseAt class given
          * as argument
          * @param  cl the BaseAt-class to check for
          * @return true if this class has a BaseAt-class cl, false otherwise
          */
         virtual bool HasBase( const Type & cl ) const;


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
          * MemberByName will return the first MemberAt with a given Name
          * @param  MemberAt Name
          * @return pointer to MemberAt
          */
         virtual Member MemberByName( const std::string & nam,
                                      const Type & signature ) const;


         /**
          * MemberAt will return the nth MemberAt of the At
          * @param  nth MemberAt
          * @return pointer to nth MemberAt
          */
         virtual Member MemberAt( size_t nth ) const;


         /**
          * MemberSize will return the number of members
          * @return number of members
          */
         virtual size_t MemberSize() const;


         virtual Member_Iterator Member_Begin() const;
         virtual Member_Iterator Member_End() const;
         virtual Reverse_Member_Iterator Member_RBegin() const;
         virtual Reverse_Member_Iterator Member_REnd() const;


         /** 
          * MemberTemplateAt will return the nth MemberAt template of this At
          * @param nth MemberAt template
          * @return nth MemberAt template
          */
         virtual MemberTemplate MemberTemplateAt( size_t nth ) const;


         /** 
          * MemberTemplateSize will return the number of MemberAt templates in this socpe
          * @return number of defined MemberAt templates
          */
         virtual size_t MemberTemplateSize() const;


         virtual MemberTemplate_Iterator MemberTemplate_Begin() const;
         virtual MemberTemplate_Iterator MemberTemplate_End() const;
         virtual Reverse_MemberTemplate_Iterator MemberTemplate_RBegin() const;
         virtual Reverse_MemberTemplate_Iterator MemberTemplate_REnd() const;


         /**
          * Name returns the Name of the At
          * @return Name of At
          */
         virtual std::string Name( unsigned int mod = 0 ) const;

 
         /**
          * FunctionParameterAt returns the nth FunctionParameterAt
          * @param  nth nth FunctionParameterAt
          * @return pointer to nth FunctionParameterAt At
          */
         virtual Type FunctionParameterAt( size_t nth ) const;


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
          * Properties will return a pointer to the PropertyNth list attached
          * to this item
          * @return pointer to PropertyNth list
          */
         virtual PropertyList Properties() const;


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
          * SubScopeAt will return a pointer to a sub-scopes
          * @param  nth sub-At
          * @return pointer to nth sub-At
          */
         virtual Scope SubScopeAt( size_t nth ) const;


         /**
          * ScopeSize will return the number of sub-scopes
          * @return number of sub-scopes
          */
         virtual size_t SubScopeSize() const;


         virtual Scope_Iterator SubScope_Begin() const;
         virtual Scope_Iterator SubScope_End() const;
         virtual Reverse_Scope_Iterator SubScope_RBegin() const;
         virtual Reverse_Scope_Iterator SubScope_REnd() const;


         /**
          * nthType will return a pointer to the nth sub-At
          * @param  nth sub-At
          * @return pointer to nth sub-At
          */
         virtual Type SubTypeAt( size_t nth ) const;


         /**
          * TypeSize will returnt he number of sub-types
          * @return number of sub-types
          */
         virtual size_t SubTypeSize() const;


         virtual Type_Iterator SubType_Begin() const;
         virtual Type_Iterator SubType_End() const;
         virtual Reverse_Type_Iterator SubType_RBegin() const;
         virtual Reverse_Type_Iterator SubType_REnd() const;


         /**
          * TemplateArgumentAt will return a pointer to the nth template argument
          * @param  nth nth template argument
          * @return pointer to nth template argument
          */
         virtual Type TemplateArgumentAt( size_t nth ) const;


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
         virtual Type ToType( unsigned int mod = 0 ) const;


         /** 
          * At returns the corresponding unqualified Type object
          * @return corresponding At object
          */
         Type ThisType() const;


         /** 
          * SubTypeTemplateAt will return the nth At template of this At
          * @param nth At template
          * @return nth At template
          */
         virtual TypeTemplate SubTypeTemplateAt( size_t nth ) const;


         /** 
          * SubTypeTemplateSize will return the number of At templates in this socpe
          * @return number of defined At templates
          */
         virtual size_t SubTypeTemplateSize() const;


         virtual TypeTemplate_Iterator SubTypeTemplate_Begin() const;
         virtual TypeTemplate_Iterator SubTypeTemplate_End() const;
         virtual Reverse_TypeTemplate_Iterator SubTypeTemplate_RBegin() const;
         virtual Reverse_TypeTemplate_Iterator SubTypeTemplate_REnd() const;


         /**
          * TypeInfo will return the c++ type_info object of the At
          * @return type_info object of At
          */
         virtual const std::type_info & TypeInfo() const;


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


         /** 
          * UpdateMembers2 will update the list of Function/Data/Members with all
          * MemberAt of BaseAt classes currently availabe in the system
          */
         virtual void UpdateMembers() const;

      public:

         /**
          * AddDataMember will add the information about a data MemberAt
          * @param dm pointer to data MemberAt
          */
         virtual void AddDataMember( const Member & dm ) const;
         virtual void AddDataMember( const char * nam,
                                     const Type & typ,
                                     size_t offs,
                                     unsigned int modifiers = 0 ) const;


         /**
          * AddFunctionMember will add the information about a function MemberAt
          * @param fm pointer to function MemberAt
          */
         virtual void AddFunctionMember( const Member & fm ) const;
         virtual void AddFunctionMember( const char * nam,
                                         const Type & typ,
                                         StubFunction stubFP,
                                         void * stubCtx = 0,
                                         const char * params = 0,
                                         unsigned int modifiers = 0 ) const;


         /**
          * AddSubScope will add a sub-At to this one
          * @param sc pointer to Scope
          */
         virtual void AddSubScope( const Scope & sc ) const;
         virtual void AddSubScope( const char * scop,
                                   TYPE scopeTyp ) const;


         /**
          * AddSubType will add a sub-At to this At
          * @param sc pointer to Type
          */
         virtual void AddSubType( const Type & ty ) const;
         virtual void AddSubType( const char * typ,
                                  size_t size,
                                  TYPE typeTyp,
                                  const std::type_info & ti,
                                  unsigned int modifiers ) const;


         /**
          * RemoveDataMember will remove the information about a data MemberAt
          * @param dm pointer to data MemberAt
          */
         virtual void RemoveDataMember( const Member & dm ) const;


         /**
          * RemoveFunctionMember will remove the information about a function MemberAt
          * @param fm pointer to function MemberAt
          */
         virtual void RemoveFunctionMember( const Member & fm ) const;


         /**
          * RemoveSubScope will remove a sub-At to this one
          * @param sc pointer to Scope
          */
         virtual void RemoveSubScope( const Scope & sc ) const;


         /**
          * RemoveSubType will remove a sub-At to this At
          * @param sc pointer to Type
          */
         virtual void RemoveSubType( const Type & ty ) const;     

      protected:

         /**
          * Pointer to the TypeName 
          * @label At Name
          * @ling aggregation
          * @link aggregation
          * @supplierCardinality 1
          * @clientCardinality 1
          */
         TypeName * fTypeName;

      private:

         /**
          * The At of the At
          * @label At At
          * @link aggregationByValue
          * @clientCardinality 0..1
          * @supplierCardinality 1
          */
         Scope fScope;


         /** size of the At in int */
         size_t fSize;


         /** C++ type_info object */
         const std::type_info & fTypeInfo;


         /**
          * At of the At
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
inline ROOT::Reflex::Base_Iterator ROOT::Reflex::TypeBase::Base_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::sBaseCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Base_Iterator ROOT::Reflex::TypeBase::Base_End() const {
//-------------------------------------------------------------------------------
   return Dummy::sBaseCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Base_Iterator ROOT::Reflex::TypeBase::Base_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::sBaseCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Base_Iterator ROOT::Reflex::TypeBase::Base_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::sBaseCont().rend();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::DataMemberSize() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::TypeBase::DataMember_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::TypeBase::DataMember_End() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::TypeBase::DataMember_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::TypeBase::DataMember_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberCont().rend();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::FunctionMemberSize() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::TypeBase::FunctionMember_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::TypeBase::FunctionMember_End() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::TypeBase::FunctionMember_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::TypeBase::FunctionMember_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberCont().rend();
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
inline size_t ROOT::Reflex::TypeBase::MemberSize() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::TypeBase::Member_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::TypeBase::Member_End() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::TypeBase::Member_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::TypeBase::Member_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberCont().rend();  
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::MemberTemplateSize() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate_Iterator ROOT::Reflex::TypeBase::MemberTemplate_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberTemplateCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate_Iterator ROOT::Reflex::TypeBase::MemberTemplate_End() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberTemplateCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::TypeBase::MemberTemplate_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberTemplateCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::TypeBase::MemberTemplate_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::sMemberTemplateCont().rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::TypeBase::FunctionParameter_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::TypeBase::FunctionParameter_End() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TypeBase::FunctionParameter_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TypeBase::FunctionParameter_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeCont().rend();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::SizeOf() const { 
//-------------------------------------------------------------------------------
   return fSize; 
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::SubScopeSize() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope_Iterator ROOT::Reflex::TypeBase::SubScope_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::sScopeCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope_Iterator ROOT::Reflex::TypeBase::SubScope_End() const {
//-------------------------------------------------------------------------------
   return Dummy::sScopeCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::TypeBase::SubScope_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::sScopeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::TypeBase::SubScope_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::sScopeCont().rend();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::SubTypeSize() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::TypeBase::SubType_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::TypeBase::SubType_End() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TypeBase::SubType_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TypeBase::SubType_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeCont().rend();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::TemplateArgumentSize() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::TypeBase::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::TypeBase::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TypeBase::TemplateArgument_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TypeBase::TemplateArgument_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeCont().rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate ROOT::Reflex::TypeBase::TemplateFamily() const {
//-------------------------------------------------------------------------------
   return TypeTemplate();
}


//-------------------------------------------------------------------------------
inline const std::type_info & ROOT::Reflex::TypeBase::TypeInfo() const {
//-------------------------------------------------------------------------------
   return fTypeInfo;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TypeBase::SubTypeTemplateSize() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::TypeBase::SubTypeTemplate_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeTemplateCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::TypeBase::SubTypeTemplate_End() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeTemplateCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::TypeBase::SubTypeTemplate_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeTemplateCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::TypeBase::SubTypeTemplate_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::sTypeTemplateCont().rend();
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::UpdateMembers() const {
//-------------------------------------------------------------------------------
   throw RuntimeError("Function UpdateMembers can only be called on Class/Struct");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::AddDataMember( const Member & /* dm */ ) const {
//-------------------------------------------------------------------------------
   throw RuntimeError("Function AddDataMember  not callable on this object");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::AddDataMember( const char * /* nam */,
                                                   const Type & /* typ */,
                                                   size_t /* offs */,
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
inline void ROOT::Reflex::TypeBase::AddFunctionMember( const char * /* nam */,
                                                       const Type & /* typ */,
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
inline void ROOT::Reflex::TypeBase::AddSubScope( const char * /* scop */,
                                                 TYPE /* scopeTyp */ ) const {
//-------------------------------------------------------------------------------
   throw RuntimeError("Function AddSubScope not callable on this object");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::AddSubType( const Type & /* ty */ ) const {
//-------------------------------------------------------------------------------
   throw RuntimeError("Function AddSubType not callable on this object");
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::TypeBase::AddSubType( const char * /* typ */,
                                                size_t /* size */,
                                                TYPE /* typeTyp */,
                                                const std::type_info & /* ti */,
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



