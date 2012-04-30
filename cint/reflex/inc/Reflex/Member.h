// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_Member
#define Reflex_Member

// Include files
#include "Reflex/Kernel.h"


namespace Reflex {
// forward declarations
class MemberBase;
class Type;
class Scope;
class PropertyList;
class Object;
class MemberTemplate;
class DictionaryGenerator;


/**
 * @class Member Member.h Reflex/Member.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class RFLX_API Member {
   friend class OwnedMember;

public:
   /** default constructor */
   Member(const MemberBase * memberBase = 0);


   /** copy constructor */
   Member(const Member &rh);


   /** destructor */
   ~Member();


   /**
    * lesser than operator
    */
   bool operator <(const Member& rh) const;


   /**
    * equal operator
    */
   bool operator ==(const Member& rh) const;


   /**
    * not equal operator
    */
   bool operator !=(const Member& rh) const;


   /**
    * assignment operator
    */
   Member& operator =(const Member& rh);


   /**
    * operator bool will return true if the member is valid
    * @return true if member is implemented
    */
   operator bool() const;

#ifdef REFLEX_CINT_MERGE
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
    * DeclaringScope will return the scope which the member lives in
    * @return the declaring scope of the member
    */
   Scope DeclaringScope() const;


   /**
    * DeclaringType will return the type which the member lives in
    * (i.e. the same as the Scope)
    * @return the declaring type of the member
    */
   Type DeclaringType() const;


   /**
    * GenerateDict will produce the dictionary information of this type
    * @param generator a reference to the dictionary generator instance
    */
   void GenerateDict(DictionaryGenerator& generator) const;


   /**
    * Get a static data member value
    * @return member value as object
    */
   Object Get() const;


   /**
    * Get the data member value
    * @return member value as object
    */
   Object Get(const Object& obj) const;


   /**
    * Id returns a unique identifier of the member in the system
    * @return unique identifier
    */
   void* Id() const;


   /**
    * Invoke a member function
    * @param obj the object which owns the member function
    * @param paramList a vector of addresses to paramter values
    * @return the return value of the function as object
    */
   void Invoke(const Object& obj,
               Object* ret,
               const std::vector<void*>& paramList = std::vector<void*>()) const;

   /**
    * Invoke a member function
    * @param obj the object which owns the member function
    * @param paramList a vector of addresses to paramter values
    * @return the return value of the function as object
    */
   template <typename T>
   void Invoke(const Object& obj,
               T& ret,
               const std::vector<void*>& paramList = std::vector<void*>()) const;

   /**
    * Invoke a static function
    * @param paramList a vector of addresses to parameter values
    * @return the return value of the function as object
    */
   void Invoke(Object* ret,
               const std::vector<void*>& paramList = std::vector<void*>()) const;


   /**
    * Invoke a static function
    * @param paramList a vector of addresses to parameter values
    * @return the return value of the function as object
    */
   template <typename T>
   void Invoke(T& ret,
               const std::vector<void*>& paramList = std::vector<void*>()) const;


   /**
    * IsAbstract checks whether abstract is set for the data member,
    * or a function member is pure virtual
    * @return true if abstract modifier is set for this member
    */
   bool IsAbstract() const;


   /**
    * IsArtificial checks whether artificial is set for the data member
    * @return true if artificial modifier is set for this member
    */
   bool IsArtificial() const;

   /**
    * IsAuto checks whether auto is set for the data member
    * @return true if auto modifier is set for this member
    */
   bool IsAuto() const;


   /**
    * IsConstructor checks whether the function member is a constructor
    * @return true if member is a constructor
    */
   bool IsConstructor() const;


   /**
    * IsConst will check whether this member is const qualified.
    * @return true if the member is const qualified
    */
   bool IsConst() const;


   /**
    * IsConverter checks whether the function member is a user defined conversion function
    * @return true if member is a conversion operator
    */
   bool IsConverter() const;


   /**
    *IsCopyConstructor checks whether the function member is a copy constructor
    * @return true if member is a copy constructor
    */
   bool IsCopyConstructor() const;


   /**
    * IsDataMember returns true if this is a data member
    * @return true if this member is a data member
    */
   bool IsDataMember() const;


   /**
    * check whether the function member is a destructor
    * @return true if this member is a destructor
    */
   bool IsDestructor() const;


   /**
    * IsExplicit checks whether explicit is set for the function member
    * @return true if explicit modifier is set for this member
    */
   bool IsExplicit() const;


   /**
    * IsExtern checks whether extern is set for the data member
    * @return true if extern modifier is set for this member
    */
   bool IsExtern() const;


   /**
    * IsFunctionMember returns true if this is a function member
    * @return true if this member is a function member
    */
   bool IsFunctionMember() const;


   /**
    * IsInline checks whether inline is set for the function member
    * @return true if inline modifier is set for this member
    */
   bool IsInline() const;


   /**
    * IsMutable check whether mutable is set for the data member
    * @return true if mutable modifier is set for this member
    */
   bool IsMutable() const;


   /**
    * IsOperator check whether the function member is an operator
    * @return true if this member is an operator function
    */
   bool IsOperator() const;


   /**
    * IsPrivate checks whether the function member is private
    * @return true if access to this member is private
    */
   bool IsPrivate() const;


   /**
    * IsProtected checks whether the function member is protected
    * @return true if access to this member is protected
    */
   bool IsProtected() const;


   /**
    * IsPublic checks whether the function member is public
    * @return true if access to this member is public
    */
   bool IsPublic() const;


   /**
    * IsPureVirtual checks whether the Member is a pure virtual
    * function.
    * @return true if function and abstract modifier is set
    */
   bool IsPureVirtual() const;


   /**
    * IsRegister checks whether register is set for the data member
    * @return true if register modifier is set for this member
    */
   bool IsRegister() const;


   /*
    * IsStatic checks whether static is set for the data member
    * @return true is static modifier is set for this member
    */
   bool IsStatic() const;


   /**
    * IsTemplateInstance returns true if the member represents a
    * templated member function
    * @return true if member represents a templated member function
    */
   bool IsTemplateInstance() const;


   /**
    * IsTransient checks whether the function member is transient
    * @return true if transient modifier is set for this member (not a C++ modifier)
    */
   bool IsTransient() const;


   /**
    * IsVirtual checks whether virtual is set for the function member
    * @return true if virtual modifier is set for this member
    */
   bool IsVirtual() const;


   /**
    * IsVolatile will check whether this member is volatile qualified.
    * @return true if the member is volatile qualified
    */
   bool IsVolatile() const;


   /**
    * MemberType return the type of the member as enum value (function or data member)
    * @return member type as enum
    */
   TYPE MemberType() const;


   /**
    * MemberTypeAsString returns the string representation of the member species
    * @return member type as string representation
    */
   std::string MemberTypeAsString() const;


   /**
    * Name returns the Name of the member
    * @param mod modifiers can be or'ed as argument
    * SCOPED - fully scoped name
    * FINAL  - resolve all typedefs
    * QUALIFIED - cv and reference qualification
    * @return name of the member
    */
   std::string Name(unsigned int mod = 0) const;

   /**
    * Name_c_str returns a char* pointer to the unqualified member name
    * @return c string to unqualified member name
    */
   const char* Name_c_str() const;

   /**
    * Offset returns the offset of the data member relative to the start of the scope
    * @return offset of member as int
    */
   size_t Offset() const;
   void InterpreterOffset(char*);
   char*& InterpreterOffset() const;


   /**
    * FunctionParameterSize returns the number of parameters
    * @param required if true only returns the number of required parameters
    * @return number of parameters
    */
   size_t FunctionParameterSize(bool required = false) const;


   /** FunctionParameterAt nth default value if declared*/
   std::string FunctionParameterDefaultAt(size_t nth) const;


   StdString_Iterator FunctionParameterDefault_Begin() const;
   StdString_Iterator FunctionParameterDefault_End() const;
   Reverse_StdString_Iterator FunctionParameterDefault_RBegin() const;
   Reverse_StdString_Iterator FunctionParameterDefault_REnd() const;


   /**
    * FunctionParametertNameAt returns the nth parameter name
    * @param nth parameter name
    * @return nth parameter name
    */
   std::string FunctionParameterNameAt(size_t nth) const;


   StdString_Iterator FunctionParameterName_Begin() const;
   StdString_Iterator FunctionParameterName_End() const;
   Reverse_StdString_Iterator FunctionParameterName_RBegin() const;
   Reverse_StdString_Iterator FunctionParameterName_REnd() const;


   /**
    * Properties will return the properties attached to this item
    * @return properties of this member
    */
   PropertyList Properties() const;


   /*void Set( const Object & instance,
      const Object & value ) const;*/

   /**
    * Set will set the value of a data member
    * @param instance of the object owning the data member
    * @param value the memory address of the value to set
    */
   void Set(const Object& instance,
            const void* value) const;


   /**
    * SetScope will set the Scope of the member
    * @param sc scope to set
    */
   void SetScope(const Scope& sc) const;


   /**
    * Stubcontext returns a pointer to the context of the member
    * @return pointer to member context
    */
   void* Stubcontext() const;


   /**
    * Stubfunction returns the pointer to the stub function
    * @return function pointer to stub function
    */
   StubFunction Stubfunction() const;


   /**
    * TemplateArgumentAt will return the nth template argument
    * @param  nth template argument
    * @return nth template argument
    */
   Type TemplateArgumentAt(size_t nth) const;


   /**
    * TemplateArgumentSize will return the number of template arguments
    * @return number of template arguments
    */
   size_t TemplateArgumentSize() const;


   Type_Iterator TemplateArgument_Begin() const;
   Type_Iterator TemplateArgument_End() const;
   Reverse_Type_Iterator TemplateArgument_RBegin() const;
   Reverse_Type_Iterator TemplateArgument_REnd() const;


   /**
    * TemplateFamily returns the corresponding MemberTemplate if any
    * @return corresponding MemberTemplate
    */
   MemberTemplate TemplateFamily() const;


   /**
    * ToMemberBase returns the underlying, internal MemberBase
    * @return memberbase pointer
    */
   MemberBase* ToMemberBase() const;


   /**
    * TypeOf returns the member type
    * @return member type
    */
   Type TypeOf() const;

   /**
    * UpdateFunctionParameterNames updates the names of parameters
    * @param  parameters new list of ';' separated parameter names, must not specify default values
    */
   void UpdateFunctionParameterNames(const char* parameters);

private:
   void Delete();

   /**
    * the pointer to the member implementation
    * @link aggregation
    * @supplierCardinality 1
    * @clientCardinality 0..1
    * @label member base
    */
   MemberBase* fMemberBase;

};    // class Member

} //namespace Reflex

#include "Reflex/internal/MemberBase.h"
#include "Reflex/Scope.h"
#include "Reflex/PropertyList.h"
#include "Reflex/Type.h"
#include "Reflex/MemberTemplate.h"


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::operator <(const Member& rh) const {
//-------------------------------------------------------------------------------
   if ((*this) && (bool) rh) {
      return TypeOf() < rh.TypeOf() && Name() < rh.Name();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::operator ==(const Member& rh) const {
//-------------------------------------------------------------------------------
   if ((*this) && (bool) rh) {
      return TypeOf() == rh.TypeOf() && 0 == strcmp(Name_c_str(), rh.Name_c_str());
   }
   // both invalid is equal, too
   return (!(*this)) && (!rh);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::operator !=(const Member& rh) const {
//-------------------------------------------------------------------------------
   return !(*this == rh);
}


//-------------------------------------------------------------------------------
inline Reflex::Member&
Reflex::Member::operator =(const Member& rh) {
//-------------------------------------------------------------------------------
   if (&rh != this) {
      fMemberBase = rh.fMemberBase;
   }
   return *this;
}


//-------------------------------------------------------------------------------
inline
Reflex::Member::operator bool() const {
//-------------------------------------------------------------------------------
   return 0 != fMemberBase;
}


//-------------------------------------------------------------------------------
inline Reflex::Scope
Reflex::Member::DeclaringScope() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->DeclaringScope();
   }
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::Member::DeclaringType() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->DeclaringScope();
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline void*
Reflex::Member::Id() const {
//-------------------------------------------------------------------------------
   return (void*) fMemberBase;
}


//-------------------------------------------------------------------------------
template <typename T>
inline void
Reflex::Member::Invoke(const Object& obj,
                       T& ret,
                       const std::vector<void*>& paramList) const {
//-------------------------------------------------------------------------------
   Object retO(Type::ByTypeInfo(typeid(T)), &ret);
   Invoke(obj, &retO, paramList);
}


//-------------------------------------------------------------------------------
template <typename T>
inline void
Reflex::Member::Invoke(T& ret,
                       const std::vector<void*>& paramList) const {
//-------------------------------------------------------------------------------
   Object retO(Type::ByTypeInfo(typeid(T)), &ret);
   Invoke(&retO, paramList);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsAbstract() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsAbstract();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsArtificial() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsArtificial();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsAuto() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsAuto();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsConstructor() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsConstructor();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsConst() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsConst();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsConverter() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsConverter();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsCopyConstructor() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsCopyConstructor();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsDataMember() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsDataMember();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsDestructor() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsDestructor();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsExplicit() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsExplicit();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsExtern() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsExtern();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsFunctionMember() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsFunctionMember();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsInline() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsInline();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsMutable() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsMutable();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsOperator() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsOperator();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsPrivate() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsPrivate();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsProtected() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsProtected();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsPublic() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsPublic();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsPureVirtual() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return IsFunctionMember() && IsAbstract();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsRegister() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsRegister();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsStatic() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsStatic();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsTemplateInstance() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsTemplateInstance();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsTransient() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsTransient();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsVirtual() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsVirtual();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Member::IsVolatile() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->IsVolatile();
   }
   return false;
}


//-------------------------------------------------------------------------------
inline Reflex::TYPE
Reflex::Member::MemberType() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->MemberType();
   }
   return UNRESOLVED;
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::Member::MemberTypeAsString() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->MemberTypeAsString();
   }
   return "";
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::Member::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->Name(mod);
   }
   return "";
}


//-------------------------------------------------------------------------------
inline const char*
Reflex::Member::Name_c_str() const {
   //-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->Name_c_str();
   }
   return "";
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Member::Offset() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->Offset();
   }
   return 0;
}


inline void
Reflex::Member::InterpreterOffset(char* offset) {
   if (*this) {
      fMemberBase->InterpreterOffset(offset);
   }
}


inline char*&
Reflex::Member::InterpreterOffset() const {
   return fMemberBase->InterpreterOffset();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Member::FunctionParameterSize(bool required) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->FunctionParameterSize(required);
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::Member::FunctionParameterDefaultAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->FunctionParameterDefaultAt(nth);
   }
   return "";
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::Member::FunctionParameterDefault_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->FunctionParameterDefault_Begin();
   }
   return Dummy::StdStringCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::Member::FunctionParameterDefault_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->FunctionParameterDefault_End();
   }
   return Dummy::StdStringCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::Member::FunctionParameterDefault_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->FunctionParameterDefault_RBegin();
   }
   return Dummy::StdStringCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::Member::FunctionParameterDefault_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->FunctionParameterDefault_REnd();
   }
   return Dummy::StdStringCont().rend();
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::Member::FunctionParameterNameAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->FunctionParameterNameAt(nth);
   }
   return "";
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::Member::FunctionParameterName_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->FunctionParameterName_Begin();
   }
   return Dummy::StdStringCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::StdString_Iterator
Reflex::Member::FunctionParameterName_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->FunctionParameterName_End();
   }
   return Dummy::StdStringCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::Member::FunctionParameterName_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->FunctionParameterName_RBegin();
   }
   return Dummy::StdStringCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_StdString_Iterator
Reflex::Member::FunctionParameterName_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->FunctionParameterName_REnd();
   }
   return Dummy::StdStringCont().rend();
}


//-------------------------------------------------------------------------------
inline Reflex::PropertyList
Reflex::Member::Properties() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->Properties();
   }
   return Dummy::PropertyList();
}


//-------------------------------------------------------------------------------
inline void
Reflex::Member::SetScope(const Scope& sc) const {
//-------------------------------------------------------------------------------
   if (*this) {
      fMemberBase->SetScope(sc);
   }
}


//-------------------------------------------------------------------------------
inline void*
Reflex::Member::Stubcontext() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->Stubcontext();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline Reflex::StubFunction
Reflex::Member::Stubfunction() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->Stubfunction();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::Member::TemplateArgumentAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->TemplateArgumentAt(nth);
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Member::TemplateArgumentSize() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->TemplateArgumentSize();
   }
   return 0;
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::Member::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->TemplateArgument_Begin();
   }
   return Dummy::TypeCont().begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Type_Iterator
Reflex::Member::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->TemplateArgument_End();
   }
   return Dummy::TypeCont().end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::Member::TemplateArgument_RBegin() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->TemplateArgument_RBegin();
   }
   return Dummy::TypeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Type_Iterator
Reflex::Member::TemplateArgument_REnd() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->TemplateArgument_REnd();
   }
   return Dummy::TypeCont().rend();
}


//-------------------------------------------------------------------------------
inline Reflex::MemberTemplate
Reflex::Member::TemplateFamily() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->TemplateFamily();
   }
   return Dummy::MemberTemplate();
}


//-------------------------------------------------------------------------------
inline Reflex::MemberBase*
Reflex::Member::ToMemberBase() const {
//-------------------------------------------------------------------------------
   return fMemberBase;
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::Member::TypeOf() const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->TypeOf();
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
inline void
Reflex::Member::UpdateFunctionParameterNames(const char* parameters) {
//-------------------------------------------------------------------------------
   if (*this) {
      return fMemberBase->UpdateFunctionParameterNames(parameters);
   }
}


#ifdef REFLEX_CINT_MERGE
inline bool
operator &&(bool b,
            const Reflex::Member& rh) {
   return b && rh.operator bool();
}


inline bool
operator &&(int i,
            const Reflex::Member& rh) {
   return i && rh.operator bool();
}


inline bool
operator &&(short s,
            const Reflex::Member& rh) {
   return s && rh.operator bool();
}


inline bool
operator ||(short s,
            const Reflex::Member& rh) {
   return s || rh.operator bool();
}


inline bool
operator ||(bool b,
            const Reflex::Member& rh) {
   return b || rh.operator bool();
}


inline bool
operator ||(int i,
            const Reflex::Member& rh) {
   return i || rh.operator bool();
}


#endif


#endif // Reflex_Member
