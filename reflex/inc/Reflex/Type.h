// @(#)root/reflex:$Name:  $:$Id: Type.h,v 1.3 2005/11/07 09:22:20 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
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

    /**
     * @class Type Type.h Reflex/Type.h
     * @author Stefan Roiser
     * @date 05/11/2004
     * @ingroup Ref
     */
    class Type {

    public:

      /** default constructor */
      Type( const TypeName * typName = 0,
            unsigned int modifiers = 0 );

      
      /** copy constructor */
      Type( const Type & rh );


      /** copy constructor applying qualification */
      Type( const Type & rh,
            unsigned int modifiers );
      

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
       * inequal operator
       */
      bool operator != ( const Type & rh ) const;

      
      /**
       * lesser than operator
       */
      bool operator < ( const Type & rh) const; 


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
       * nthBase will return the nth BaseNth class information
       * @param  nth nth BaseNth class
       * @return pointer to BaseNth class information
       */
      Base BaseNth( size_t nth ) const;


      /**
       * BaseCount will return the number of BaseNth classes
       * @return number of BaseNth classes
       */
      size_t BaseCount() const;


      Base_Iterator Base_Begin() const;
      Base_Iterator Base_End() const;
      Reverse_Base_Iterator Base_Rbegin() const;
      Reverse_Base_Iterator Base_Rend() const;


      /**
       * ByName will look for a TypeNth given as a string and return a pointer to
       * its reflexion TypeNth
       * @param  key fully qualified Name of the TypeNth as string
       * @return pointer to TypeNth or 0 if none is found
       */
      static Type ByName( const std::string & key );
      
      
      /**
       * byTypeId will look for a TypeNth given as a string representation of a
       * type_info and return a pointer to its reflexion TypeNth
       * @param  tid string representation of the type_info TypeNth
       * @return pointer to TypeNth or 0 if none is found
       */
      static Type ByTypeInfo( const std::type_info & tid );


      /**
       * CastObject an object from this class TypeNth to another one
       * @param  to is the class TypeNth to cast into
       * @param  obj the memory AddressGet of the object to be casted
       */
      Object CastObject( const Type & to, 
                         const Object & obj ) const;
      

      /**
       * Construct will call the constructor of a given TypeNth and Allocate
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
       * DataMemberNth will return the nth data MemberNth of the ScopeNth
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
       * Deallocate will Deallocate the memory for a given object
       * @param  instance of the TypeNth in memory
       */
      void Deallocate( void * instance ) const;


      /**
       * DeclaringScope will return a pointer to the ScopeNth of this one
       * @return pointer to declaring ScopeNth
       */
      Scope DeclaringScope() const;


      /**
       * Destruct will call the destructor of a TypeNth and remove its memory
       * allocation if desired
       * @param  instance of the TypeNth in memory
       * @param  dealloc for also deallacoting the memory
       */
      void Destruct( void * instance, 
                     bool dealloc = true ) const;


      /**
       * DynamicType is used to discover whether an object represents the
       * current class TypeNth or not
       * @param  mem is the memory AddressGet of the object to checked
       * @return the actual class of the object
       */
      Type DynamicType( const Object & obj ) const;


      /**
       * FunctionMemberNth will return the nth function MemberNth of the ScopeNth
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
      Member FunctionMemberNth( const std::string & nam,
                                const Type & signature = Type(0,0) ) const;


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
       * HasBase will check whether this class has a BaseNth class given
       * as argument
       * @param  cl the BaseNth-class to check for
       * @return true if this class has a BaseNth-class cl, false otherwise
       */
      bool HasBase( const Type & cl ) const;


      /**
       * Id returns a unique identifier of the TypeNth in the system
       * @return unique identifier
       */
      void * Id() const;


      /**
       * IsAbstract will return true if the the class is abstract
       * @return true if the class is abstract
       */
      bool IsAbstract() const;


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
      bool IsComplete() const;


      /** 
       * IsConst returns true if the TypeNth represents a const TypeNth
       * @return true if TypeNth represents a const TypeNth
       */
      bool IsConst() const;


      /** 
       * IsConstVolatile returns true if the TypeNth represents a const volatile TypeNth
       * @return true if TypeNth represents a const volatile TypeNth
       */
      bool IsConstVolatile() const;


      /** 
       * IsEnum returns true if the TypeNth represents a Enum
       * @return true if TypeNth represents a Enum
       */
      bool IsEnum() const;

      
      /** 
       * IsEquivalentTo returns true if the two types are equivalent
       * @param TypeNth to compare to
       * @return true if two types are equivalent
       */
      bool IsEquivalentTo( const Type & typ ) const;


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
       * IsReference returns true if the TypeNth represents a Reference
       * @return true if TypeNth represents a Reference
       */
      bool IsReference() const;


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
       * IsUnqualified returns true if the TypeNth represents an unqualified TypeNth
       * @return true if TypeNth represents an unqualified TypeNth
       */
      bool IsUnqualified() const;


      /**
       * IsVirtual will return true if the class contains a virtual table
       * @return true if the class contains a virtual table
       */
      bool IsVirtual() const;


      /** 
       * IsVolatile returns true if the TypeNth represents a volatile TypeNth
       * @return true if TypeNth represents a volatile TypeNth
       */
      bool IsVolatile() const;

 
      /** 
       * size returns the size of the array
       * @return size of array
       */
      size_t Length() const;

 
      /**
       * MemberNth will return the first MemberNth with a given Name
       * @param  MemberNth Name
       * @return pointer to MemberNth
       */
      Member MemberNth( const std::string & nam,
                        const Type & signature = Type(0,0)) const;


      /**
       * MemberNth will return the nth MemberNth of the ScopeNth
       * @param  nth MemberNth
       * @return pointer to nth MemberNth
       */
      Member MemberNth( size_t nth ) const;


      /**
       * MemberCount will return the number of members
       * @return number of members
       */
      size_t MemberCount() const;


      Member_Iterator Member_Begin() const;
      Member_Iterator Member_End() const;
      Reverse_Member_Iterator Member_Rbegin() const;
      Reverse_Member_Iterator Member_Rend() const;


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
       * Name returns the Name of the TypeNth
       * @return Name of the TypeNth
       */
      std::string Name( unsigned int mod = 0 ) const;
      
      
      /**
       * Name_c_str returns a char* pointer to the unqualified TypeNth Name
       * @ return c string to unqualified TypeNth Name
       */
      const char * Name_c_str() const;


      /**
       * ParameterNth returns the nth ParameterNth
       * @param  nth nth ParameterNth
       * @return pointer to nth ParameterNth TypeNth
       */
      Type ParameterNth( size_t nth ) const;


      /**
       * ParameterCount will return the number of parameters of this function
       * @return number of parameters
       */
      size_t ParameterCount() const;


      Type_Iterator Parameter_Begin() const;
      Type_Iterator Parameter_End() const;
      Reverse_Type_Iterator Parameter_Rbegin() const;
      Reverse_Type_Iterator Parameter_Rend() const;


      /**
       * PropertyListGet will return a pointer to the PropertyNth list attached
       * to this item
       * @return pointer to PropertyNth list
       */
      PropertyList PropertyListGet() const;


      /**
       * ReturnType will return a pointer to the TypeNth of the return TypeNth.
       * @return pointer to Type of return TypeNth
       */
      Type ReturnType() const;
      

      /** ScopeNth will return the ScopeNth of the Type if any 
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
       * nthType will return a pointer to the nth sub-TypeNth
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
      Type TemplateArgumentNth( size_t nth ) const;


      /**
       * templateArgCount will return the number of template arguments
       * @return number of template arguments
       */
      size_t TemplateArgumentCount() const;

 
      Type_Iterator TemplateArgument_Begin() const;
      Type_Iterator TemplateArgument_End() const;
      Reverse_Type_Iterator TemplateArgument_Rbegin() const;
      Reverse_Type_Iterator TemplateArgument_Rend() const;


      /**
       * TemplateFamily returns the corresponding TypeTemplate if any
       * @return corresponding TypeTemplate
       */
      TypeTemplate TemplateFamily() const;


      /**
       * arrayType will return a pointer to the TypeNth of the array.
       * @return pointer to Type of MemberNth et. al.
       */
      Type ToType() const;


      /**
       * TypeNth will return a pointer to the nth Type in the system
       * @param  nth number of TypeNth to return
       * @return pointer to nth Type in the system
       */
      static Type TypeNth( size_t nth );


      /**
       * TypeCount will return the number of currently defined types in
       * the system
       * @return number of currently defined types
       */
      static size_t TypeCount();


      static Type_Iterator Type_Begin();
      static Type_Iterator Type_End();
      static Reverse_Type_Iterator Type_Rbegin();
      static Reverse_Type_Iterator Type_Rend();


      /**
       * typeId will return the c++ type_info object of the TypeNth
       * @return type_info object of TypeNth
       */
      const std::type_info & TypeInfo() const;


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


      /**
       * TypeType will return the real TypeNth
       * @return real TypeNth
       */
      TYPE TypeType() const;


      /**
       * TypeTypeAsString will return the string representation of the ENUM
       * representing the real TypeNth of the Type (e.g. "CLASS")
       * @return string representation of the TYPE enum of the Type
       */
      std::string TypeTypeAsString() const;


      /**
       * Unload will unload the dictionary information of a type from the system
       * (FIXME - not implemented yet)
       */
      void Unload() const;


      /** 
       * UpdateMembers will update the list of Function/Data/Members with all
       * MemberNth of BaseNth classes currently availabe in the system
       */
      void UpdateMembers() const;

    public:

      /** */
      const TypeBase * TypeBaseNth() const;

      /**
       * AddDataMember will add the information about a data MemberNth
       * @param dm pointer to data MemberNth
       */
      void AddDataMember( const Member & dm ) const;
      void AddDataMember( const char * nam,
                          const Type & typ,
                          size_t offs,
                          unsigned int modifiers = 0 ) const;


      /**
       * AddFunctionMember will add the information about a function MemberNth
       * @param fm pointer to function MemberNth
       */
      void AddFunctionMember( const Member & fm ) const;
      void AddFunctionMember( const char * nam,
                              const Type & typ,
                              StubFunction stubFP,
                              void * stubCtx = 0,
                              const char * params = 0,
                              unsigned int modifiers = 0 ) const;


      /**
       * AddSubScope will add a sub-ScopeNth to this one
       * @param sc pointer to Scope
       */
      void AddSubScope( const Scope & sc ) const;
      void AddSubScope( const char * scop,
                        TYPE scopeTyp = NAMESPACE ) const;


      /**
       * AddSubType will add a sub-TypeNth to this ScopeNth
       * @param sc pointer to Type
       */
      void AddSubType( const Type & ty ) const;
      void AddSubType( const char * typ,
                       size_t size,
                       TYPE typeTyp,
                       const std::type_info & ti,
                       unsigned int modifiers = 0 ) const;


      /**
       * RemoveDataMember will remove the information about a data MemberNth
       * @param dm pointer to data MemberNth
       */
      void RemoveDataMember( const Member & dm ) const;


      /**
       * RemoveFunctionMember will remove the information about a function MemberNth
       * @param fm pointer to function MemberNth
       */
      void RemoveFunctionMember( const Member & fm ) const;


      /**
       * RemoveSubScope will remove a sub-ScopeNth to this one
       * @param sc pointer to Scope
       */
      void RemoveSubScope( const Scope & sc ) const;


      /**
       * RemoveSubType will remove a sub-TypeNth to this ScopeNth
       * @param sc pointer to Type
       */
      void RemoveSubType( const Type & ty ) const;

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

#include "Reflex/TypeName.h"
#include "Reflex/TypeBase.h"
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
                                 unsigned int modifiers ) 
//-------------------------------------------------------------------------------
  : fTypeName( rh.fTypeName ),
    fModifiers( modifiers ) {}


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
inline size_t ROOT::Reflex::Type::BaseCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->BaseCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Base_Iterator ROOT::Reflex::Type::Base_Begin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Base_Begin();
  return Base_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Base_Iterator ROOT::Reflex::Type::Base_End() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Base_End();
  return Base_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Base_Iterator ROOT::Reflex::Type::Base_Rbegin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Base_Rbegin();
  return Reverse_Base_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Base_Iterator ROOT::Reflex::Type::Base_Rend() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Base_Rend();
  return Reverse_Base_Iterator();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::DataMemberCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->DataMemberCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Type::DataMember_Begin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->DataMember_Begin();
  return Member_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Type::DataMember_End() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->DataMember_End();
  return Member_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Type::DataMember_Rbegin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->DataMember_Rbegin();
  return Reverse_Member_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Type::DataMember_Rend() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->DataMember_Rend();
  return Reverse_Member_Iterator();
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::Deallocate( void * instance ) const {
//-------------------------------------------------------------------------------
  if ( * this ) fTypeName->fTypeBase->Deallocate( instance ); 
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope ROOT::Reflex::Type::DeclaringScope() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->DeclaringScope();
  return Scope();
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::Destruct( void * instance, 
                                          bool dealloc ) const {
//-------------------------------------------------------------------------------
  if ( * this ) fTypeName->fTypeBase->Destruct( instance, dealloc ); 
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::FunctionMemberCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->FunctionMemberCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Type::FunctionMember_Begin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->FunctionMember_Begin();
  return Member_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Type::FunctionMember_End() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->FunctionMember_End();
  return Member_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Type::FunctionMember_Rbegin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->FunctionMember_Rbegin();
  return Reverse_Member_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Type::FunctionMember_Rend() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->FunctionMember_Rend();
  return Reverse_Member_Iterator();
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
inline size_t ROOT::Reflex::Type::Length() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Length();
  return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::MemberTemplateCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->MemberTemplateCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::MemberCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->MemberCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Type::Member_Begin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Member_Begin();
  return Member_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Type::Member_End() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Member_End();
  return Member_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Type::Member_Rbegin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Member_Rbegin();
  return Reverse_Member_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Type::Member_Rend() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Member_Rend();
  return Reverse_Member_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate_Iterator ROOT::Reflex::Type::MemberTemplate_Begin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->MemberTemplate_Begin();
  return MemberTemplate_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate_Iterator ROOT::Reflex::Type::MemberTemplate_End() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->MemberTemplate_End();
  return MemberTemplate_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::Type::MemberTemplate_Rbegin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->MemberTemplate_Rbegin();
  return Reverse_MemberTemplate_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::Type::MemberTemplate_Rend() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->MemberTemplate_Rend();
  return Reverse_MemberTemplate_Iterator();
}


//-------------------------------------------------------------------------------
inline const char * ROOT::Reflex::Type::Name_c_str() const {
//-------------------------------------------------------------------------------
  if ( fTypeName ) return fTypeName->Name_c_str();
  return "";
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Type::ParameterNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->ParameterNth( nth );
  return Type();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::ParameterCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->ParameterCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Type::Parameter_Begin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Parameter_Begin();
  return Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Type::Parameter_End() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Parameter_End();
  return Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::Parameter_Rbegin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Parameter_Rbegin();
  return Reverse_Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::Parameter_Rend() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Parameter_Rend();
  return Reverse_Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::PropertyList ROOT::Reflex::Type::PropertyListGet() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->PropertyListGet();
  return PropertyList();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Type::ReturnType() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->ReturnType();
  return Type();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope_Iterator ROOT::Reflex::Type::SubScope_Begin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->SubScope_Begin();
  return Scope_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope_Iterator ROOT::Reflex::Type::SubScope_End() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->SubScope_End();
  return Scope_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::Type::SubScope_Rbegin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->SubScope_Rbegin();
  return Reverse_Scope_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::Type::SubScope_Rend() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->SubScope_Rend();
  return Reverse_Scope_Iterator();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::SizeOf() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->SizeOf();
  return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::SubScopeCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->SubScopeCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::SubTypeCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->SubTypeCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Type::SubType_Begin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->SubType_Begin();
  return Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Type::SubType_End() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->SubType_End();
  return Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::SubType_Rbegin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->SubType_Rbegin();
  return Reverse_Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::SubType_Rend() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->SubType_Rend();
  return Reverse_Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Type::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->TemplateArgument_Begin();
  return Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Type::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->TemplateArgument_End();
  return Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::TemplateArgument_Rbegin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->TemplateArgument_Rbegin();
  return Reverse_Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::TemplateArgument_Rend() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->TemplateArgument_Rend();
  return Reverse_Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Type::ToType() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->ToType();
  return Type();
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
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::Type_Rbegin() {
//-------------------------------------------------------------------------------
  return TypeName::Type_Rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Type::Type_Rend() {
//-------------------------------------------------------------------------------
  return TypeName::Type_Rend();
}


//-------------------------------------------------------------------------------
inline const std::type_info & ROOT::Reflex::Type::TypeInfo() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->TypeInfo(); 
  return typeid(void);
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::TypeTemplateCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->TypeTemplateCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::Type::TypeTemplate_Begin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->TypeTemplate_Begin();
  return TypeTemplate_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::Type::TypeTemplate_End() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->TypeTemplate_End();
  return TypeTemplate_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::Type::TypeTemplate_Rbegin() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->TypeTemplate_Rbegin();
  return Reverse_TypeTemplate_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::Type::TypeTemplate_Rend() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->TypeTemplate_Rend();
  return Reverse_TypeTemplate_Iterator();
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
inline ROOT::Reflex::Type ROOT::Reflex::Type::TemplateArgumentNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->TemplateArgumentNth( nth );
  return Type();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Type::TemplateArgumentCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->TemplateArgumentCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate ROOT::Reflex::Type::TemplateFamily() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->TemplateFamily();
  return TypeTemplate();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::TypeBase * ROOT::Reflex::Type::TypeBaseNth() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase;
  return 0;
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Type::Unload() const {
//-------------------------------------------------------------------------------
  // FIXME 
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
