// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_MemberBase
#define Reflex_MemberBase

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/PropertyList.h"
#include "Reflex/Type.h"
#include "Reflex/Scope.h"
#include "Reflex/internal/LiteralString.h"

namespace Reflex {
// forward declarations
class Object;
class DictionaryGenerator;

/**
 * @class MemberBase MemberBase.h Reflex/internal/MemberBase.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class RFLX_API MemberBase {
public:
   /** default constructor */
   MemberBase(const char* name,
              const Type &type,
              TYPE memberType,
              unsigned int modifiers);


   /** destructor */
   virtual ~MemberBase();


   /**
    * operator member will return the member object of this MemberBase
    */
   operator Member() const;


   /**
    * DeclaringScope will return the scope which the member lives in
    * @return the declaring scope of the member
    */
   Scope DeclaringScope() const;


   /**
    * DeclaringType will return the type which the member lives in
    * @return the declaring type of the member
    */
   Type DeclaringType() const;


   /**
    * GenerateDict will produce the dictionary information of this type
    * @param generator a reference to the dictionary generator instance
    */
   virtual void GenerateDict(DictionaryGenerator& generator) const;


   /** Get the member value */
   virtual Object Get(const Object& obj) const;


   /** Invoke the member function */

   /*virtual Object Invoke( const Object & obj,
      const std::vector < Object > & paramList ) const;*/
   virtual void Invoke(const Object& obj,
                       Object* ret,
                       const std::vector<void*>& paramList =
                          std::vector<void*>()) const;


   /** Invoke the function (for static functions) */
   //virtual Object Invoke( const std::vector < Object > & paramList ) const;
   virtual void Invoke(Object* ret,
                       const std::vector<void*>& paramList =
                          std::vector<void*>()) const;


   /** check whether abstract is set for the data member */
   bool IsAbstract() const;


   /** check whether artificial is set for the data member */
   bool IsArtificial() const;

   /** check whether auto is set for the data member */
   bool IsAuto() const;


   /** check whether the function member is a constructor */
   bool IsConstructor() const;


   /** check whether a member is const qualified */
   bool IsConst() const;


   /** check whether the function member is a user defined conversion function */
   bool IsConverter() const;


   /** check whether the function member is a copy constructor */
   bool IsCopyConstructor() const;


   /** return true if this is a data member */
   bool IsDataMember() const;


   /** check whether the function member is a destructor */
   bool IsDestructor() const;


   /** check whether explicit is set for the function member */
   bool IsExplicit() const;


   /** check whether extern is set for the data member */
   bool IsExtern() const;


   /** return true if this is a function member */
   bool IsFunctionMember() const;


   /** check whether inline is set for the function member */
   bool IsInline() const;


   /** check whether mutable is set for the data member */
   bool IsMutable() const;


   /** check whether the function member is an operator */
   bool IsOperator() const;


   /** check whether the function member is private */
   bool IsPrivate() const;


   /** check whether the function member is protected */
   bool IsProtected() const;


   /** check whether the function member is public */
   bool IsPublic() const;


   /** check whether register is set for the data member */
   bool IsRegister() const;


   /** check whether static is set for the data member */
   bool IsStatic() const;


   /**
    * IsTemplateInstance returns true if the type represents a
    * ClassTemplateInstance
    * @return true if type represents a InstantiatedTemplateClass
    */
   bool IsTemplateInstance() const;


   /** check whether transient is set for the data member */
   bool IsTransient() const;


   /** check whether virtual is set for the function member */
   bool IsVirtual() const;


   /** check whether a member is volatile qualified */
   bool IsVolatile() const;


   /** return the type of the member (function or data member) */
   TYPE MemberType() const;


   /** returns the string representation of the member species */
   std::string MemberTypeAsString() const;


   /** return the name of the member */
   virtual std::string Name(unsigned int mod = 0) const;

   /**
    * Name_c_str returns a char* pointer to the unqualified Name
    * @ return c string to unqualified Name
    */
   const char* Name_c_str() const;


   /** return the offset of the member */
   virtual size_t Offset() const;
   virtual void InterpreterOffset(char*);
   virtual char*& InterpreterOffset() const;


   /** number of parameters */
   virtual size_t FunctionParameterSize(bool required = false) const;


   /** FunctionParameterDefaultAt returns the nth default value if declared*/
   virtual std::string FunctionParameterDefaultAt(size_t nth) const;


   virtual StdString_Iterator FunctionParameterDefault_Begin() const;
   virtual StdString_Iterator FunctionParameterDefault_End() const;
   virtual Reverse_StdString_Iterator FunctionParameterDefault_RBegin() const;
   virtual Reverse_StdString_Iterator FunctionParameterDefault_REnd() const;


   /** FunctionParameterNameAt returns the nth name if declared*/
   virtual std::string FunctionParameterNameAt(size_t nth) const;


   virtual StdString_Iterator FunctionParameterName_Begin() const;
   virtual StdString_Iterator FunctionParameterName_End() const;
   virtual Reverse_StdString_Iterator FunctionParameterName_RBegin() const;
   virtual Reverse_StdString_Iterator FunctionParameterName_REnd() const;


   /**
    * Properties will return a pointer to the property list attached
    * to this item
    * @return pointer to property list
    */
   PropertyList Properties() const;


   /** Set the member value */

   /*virtual void Set( const Object & instance,
      const Object & value ) const;*/
   virtual void Set(const Object& instance,
                    const void* value) const;


   /** Set the type of the member */
   void SetScope(const Scope& scope) const;


   /** return the context of the member */
   virtual void* Stubcontext() const;


   /** return the pointer to the stub function */
   virtual StubFunction Stubfunction() const;


   /**
    * TemplateArgumentAt will return a pointer to the nth template argument
    * @param  nth nth template argument
    * @return pointer to nth template argument
    */
   virtual Type TemplateArgumentAt(size_t nth) const;


   /**
    * TemplateArgumentSize will return the number of template arguments
    * @return number of template arguments
    */
   virtual size_t TemplateArgumentSize() const;


   virtual Type_Iterator TemplateArgument_Begin() const;
   virtual Type_Iterator TemplateArgument_End() const;
   virtual Reverse_Type_Iterator TemplateArgument_RBegin() const;
   virtual Reverse_Type_Iterator TemplateArgument_REnd() const;


   /**
    * TemplateFamily returns the corresponding MemberTemplate if any
    * @return corresponding MemberTemplate
    */
   virtual MemberTemplate TemplateFamily() const;


   /** return pointer to member type */
   Type TypeOf() const;


   /**
    * UpdateFunctionParameterNames updates the names of parameters
    * @param  parameters new list of ';' separated parameter names, must not specify default values
    */
   virtual void UpdateFunctionParameterNames(const char* parameters);

protected:
   /**
    * CalculateBaseObject will calculate the inheritance between an object
    * and the local type if necessary
    * @param obj the object from which the calculation should start
    * @return memory address of new local object relative to obj
    */
   void* CalculateBaseObject(const Object& obj) const;

protected:
   /**
    * characteristics of the Member
    * @label Member
    * @supplierCardinality 1
    * @link aggregation
    * @clientCardinality 1
    */
   Type fType;


   /** all modifiers of the member */
   unsigned int fModifiers;

private:
   /** name of member */
   LiteralString fName;


   /**
    * scope of the member
    * @label member scope
    * @link aggregation
    * @supplierCardinality 1
    * @clientCardinality 1
    */
   mutable
   Scope fScope;


   /**
    * the kind of member ( data/function-member)
    * @label member type
    * @link aggregation
    * @clientCardinality 1
    * @supplierCardinality 1
    */
   TYPE fMemberType;


   /**
    * property list
    * @label propertylist
    * @link aggregationByValue
    * @clientCardinality 1
    * @supplierCardinality 0..1
    */
   OwnedPropertyList fPropertyList;


   /**
    * pointer back to the member object
    * @label this member
    * @link aggregation
    * @supplierCardinality 1
    * @clientCardinality 1
    */
   Member* fThisMember;

};    // class Member

} //namespace Reflex

#ifndef Reflex_Object
#include "Reflex/Object.h"
#endif
#ifndef Reflex_MemberTemplate
#include "Reflex/MemberTemplate.h"
#endif


//-------------------------------------------------------------------------------
inline Reflex::Object
Reflex::MemberBase::Get(const Object& /* obj */) const {
//-------------------------------------------------------------------------------
   return Object();
}


//-------------------------------------------------------------------------------
//inline Reflex::Object
//Reflex::MemberBase::Invoke( const Object & /* obj */ ,
//                                  const std::vector < Object > & /* paramList */ ) const {
//-------------------------------------------------------------------------------
//  return Object();
//}


//-------------------------------------------------------------------------------
inline void
Reflex::MemberBase::Invoke(const Object& /* obj */,
                           Object* /*ret*/,
                           const std::vector<void*>& /* paramList */) const {
//-------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
//inline Reflex::Object
//Reflex::MemberBase::Invoke( const std::vector < Object > & /* paramList */ ) const {
//-------------------------------------------------------------------------------
//  return Object();
//}


//-------------------------------------------------------------------------------
inline void
Reflex::MemberBase::Invoke(Object* /*ret*/,
                           const std::vector<void*>& /* paramList */) const {
//-------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsAbstract() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & ABSTRACT);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsArtificial() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & ARTIFICIAL);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsAuto() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & AUTO);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsConstructor() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & CONSTRUCTOR);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsConst() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & CONST);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsConverter() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & CONVERTER);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsCopyConstructor() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & COPYCONSTRUCTOR);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsDataMember() const {
//-------------------------------------------------------------------------------
   return fMemberType == DATAMEMBER;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsDestructor() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & DESTRUCTOR);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsExplicit() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & EXPLICIT);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsExtern() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & EXTERN);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsFunctionMember() const {
//-------------------------------------------------------------------------------
   return fMemberType == FUNCTIONMEMBER;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsInline() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & INLINE);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsMutable() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & MUTABLE);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsOperator() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & OPERATOR);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsPrivate() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & PRIVATE);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsProtected() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & PROTECTED);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsPublic() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & PUBLIC);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsRegister() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & REGISTER);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsStatic() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & STATIC);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsTemplateInstance() const {
//-------------------------------------------------------------------------------
   return fMemberType == MEMBERTEMPLATEINSTANCE;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsTransient() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & TRANSIENT);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsVirtual() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & VIRTUAL);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::MemberBase::IsVolatile() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & VOLATILE);
}


//-------------------------------------------------------------------------------
inline Reflex::TYPE
Reflex::MemberBase::MemberType() const {
//-------------------------------------------------------------------------------
   return fMemberType;
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::MemberBase::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
   if (0 != (mod & (SCOPED | S))) {
      std::string s(DeclaringScope().Name(mod));

      if (!DeclaringScope().IsTopScope()) {
         s += "::";
      }
      s += fName.c_str();
      return s;
   }
   return fName.c_str();
}


//-------------------------------------------------------------------------------
inline const char*
Reflex::MemberBase::Name_c_str() const {
//-------------------------------------------------------------------------------
   return fName.c_str();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::MemberBase::Offset() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline void
Reflex::MemberBase::InterpreterOffset(char*) {
//-------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
inline char*&
Reflex::MemberBase::InterpreterOffset() const {
//-------------------------------------------------------------------------------
   static char* p = 0;
   return p;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::MemberBase::FunctionParameterSize(bool /* required */) const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::MemberBase::FunctionParameterDefaultAt(size_t /* nth */) const {
//-------------------------------------------------------------------------------
   return "";
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::MemberBase::FunctionParameterDefault_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::MemberBase::FunctionParameterDefault_End() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::MemberBase::FunctionParameterDefault_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::MemberBase::FunctionParameterDefault_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().rend();
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::MemberBase::FunctionParameterNameAt(size_t /* nth */) const {
//-------------------------------------------------------------------------------
   return "";
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::MemberBase::FunctionParameterName_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::MemberBase::FunctionParameterName_End() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::MemberBase::FunctionParameterName_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::MemberBase::FunctionParameterName_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().rend();
}


//-------------------------------------------------------------------------------
//inline void Reflex::MemberBase::Set( const Object & /* instance */,
//                                           const Object & /* value */ ) const {}
//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
inline void
Reflex::MemberBase::Set(const Object& /* instance */,
                        const void* /* value */) const {}

//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
inline void
Reflex::MemberBase::SetScope(const Scope& scope) const {
//-------------------------------------------------------------------------------
   fScope = scope;
}


//-------------------------------------------------------------------------------
inline void*
Reflex::MemberBase::Stubcontext() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline Reflex::StubFunction
Reflex::MemberBase::Stubfunction() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::MemberBase::TemplateArgumentSize() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::MemberBase::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::MemberBase::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::MemberBase::TemplateArgument_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::MemberBase::TemplateArgument_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().rend();
}


//-------------------------------------------------------------------------------
inline Reflex::MemberTemplate
Reflex::MemberBase::TemplateFamily() const {
//-------------------------------------------------------------------------------
   return Dummy::MemberTemplate();
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::MemberBase::TypeOf() const {
//-------------------------------------------------------------------------------
   return fType;
}


//-------------------------------------------------------------------------------
inline void
Reflex::MemberBase::UpdateFunctionParameterNames(const char* /*parameters*/) {}

//-------------------------------------------------------------------------------


#endif // Reflex_MemberBase
