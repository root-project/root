// @(#)root/reflex:$Name:  $:$Id: MemberBase.h,v 1.4 2006/10/30 12:51:33 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_MemberBase
#define ROOT_Reflex_MemberBase

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/PropertyList.h"
#include "Reflex/Type.h"
#include "Reflex/Scope.h"

namespace ROOT {
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
         MemberBase( const char *   name,
                     const Type &   type,
                     TYPE           memberType,
                     unsigned int   modifiers );


         /** destructor */
         virtual ~MemberBase();


         /**
          * operator member will return the member object of this MemberBase
          */
         operator Member () const;


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
         virtual void GenerateDict(DictionaryGenerator & generator) const;
	  
      
         /** Get the member value */
         virtual Object Get( const Object & obj ) const;


         /** Invoke the member function */
         /*virtual Object Invoke( const Object & obj, 
           const std::vector < Object > & paramList ) const;*/
         virtual Object Invoke( const Object & obj, 
                                const std::vector < void * > & paramList = 
                                std::vector<void*>()) const;


         /** Invoke the function (for static functions) */
         //virtual Object Invoke( const std::vector < Object > & paramList ) const;
         virtual Object Invoke( const std::vector < void * > & paramList = 
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
         virtual std::string Name( unsigned int mod = 0 ) const;


         /** return the offset of the member */
         virtual size_t Offset() const;


         /** number of parameters */
         virtual size_t FunctionParameterSize( bool required = false ) const;


         /** FunctionParameterDefaultAt returns the nth default value if declared*/
         virtual std::string FunctionParameterDefaultAt( size_t nth ) const;


         virtual StdString_Iterator FunctionParameterDefault_Begin() const;
         virtual StdString_Iterator FunctionParameterDefault_End() const;
         virtual Reverse_StdString_Iterator FunctionParameterDefault_RBegin() const;
         virtual Reverse_StdString_Iterator FunctionParameterDefault_REnd() const;


         /** FunctionParameterNameAt returns the nth name if declared*/
         virtual std::string FunctionParameterNameAt( size_t nth ) const;


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
         virtual void Set( const Object & instance,
                           const void * value ) const;


         /** Set the type of the member */
         void SetScope( const Scope & scope ) const;


         /** return the context of the member */
         virtual void * Stubcontext() const;


         /** return the pointer to the stub function */
         virtual StubFunction Stubfunction() const;


         /**
          * TemplateArgumentAt will return a pointer to the nth template argument
          * @param  nth nth template argument
          * @return pointer to nth template argument
          */
         virtual Type TemplateArgumentAt( size_t nth ) const;


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

      protected:

         /** 
          * CalculateBaseObject will calculate the inheritance between an object
          * and the local type if necessary
          * @param obj the object from which the calculation should start
          * @return memory address of new local object relative to obj
          */
         void * CalculateBaseObject( const Object & obj ) const;

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
         std::string fName;


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
         Member * fThisMember;

      }; // class Member

   } //namespace Reflex
} //namespace ROOT

#include "Reflex/Object.h"
#include "Reflex/MemberTemplate.h"


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Object ROOT::Reflex::MemberBase::Get( const Object & /* obj */ ) const {
//-------------------------------------------------------------------------------
   return Object();
}


//-------------------------------------------------------------------------------
//inline ROOT::Reflex::Object 
//ROOT::Reflex::MemberBase::Invoke( const Object & /* obj */ ,
//                                  const std::vector < Object > & /* paramList */ ) const {
//-------------------------------------------------------------------------------
//  return Object();
//}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Object 
ROOT::Reflex::MemberBase::Invoke( const Object & /* obj */ ,
                                  const std::vector < void * > & /* paramList */ ) const {
//-------------------------------------------------------------------------------
   return Object();
}


//-------------------------------------------------------------------------------
//inline ROOT::Reflex::Object 
//ROOT::Reflex::MemberBase::Invoke( const std::vector < Object > & /* paramList */ ) const {
//-------------------------------------------------------------------------------
//  return Object();
//}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Object 
ROOT::Reflex::MemberBase::Invoke( const std::vector < void * > & /* paramList */ ) const {
//-------------------------------------------------------------------------------
   return Object();
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsAbstract() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & ABSTRACT);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsArtificial() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & ARTIFICIAL);
}

//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsAuto() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & AUTO);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsConstructor() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & CONSTRUCTOR);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsConst() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & CONST);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsConverter() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & CONVERTER);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsCopyConstructor() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & COPYCONSTRUCTOR);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsDataMember() const {
//-------------------------------------------------------------------------------
   return ( fMemberType == DATAMEMBER );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsDestructor() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & DESTRUCTOR);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsExplicit() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & EXPLICIT);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsExtern() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & EXTERN);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsFunctionMember() const {
//-------------------------------------------------------------------------------
   return ( fMemberType == FUNCTIONMEMBER );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsInline() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & INLINE);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsMutable() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & MUTABLE);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsOperator() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & OPERATOR);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsPrivate() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & PRIVATE);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsProtected() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & PROTECTED);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsPublic() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & PUBLIC);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsRegister() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & REGISTER);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsStatic() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & STATIC);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsTemplateInstance() const {
//-------------------------------------------------------------------------------
   return ( fMemberType == MEMBERTEMPLATEINSTANCE );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsTransient() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & TRANSIENT);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsVirtual() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & VIRTUAL);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::MemberBase::IsVolatile() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & VOLATILE);
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TYPE ROOT::Reflex::MemberBase::MemberType() const {
//-------------------------------------------------------------------------------
   return fMemberType;
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::MemberBase::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
   std::string s = "";
   if ( 0 != ( mod & ( SCOPED | S ))) {
      s += DeclaringScope().Name( mod );
      if ( ! DeclaringScope().IsTopScope()) s += "::";
   }
   s += fName;
   return s;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::MemberBase::Offset() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::MemberBase::FunctionParameterSize( bool /* required */ ) const {
//-------------------------------------------------------------------------------
   return 0; 
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::MemberBase::FunctionParameterDefaultAt( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
   return "";
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterDefault_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterDefault_End() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterDefault_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterDefault_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().rend();
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::MemberBase::FunctionParameterNameAt( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
   return "";
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterName_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterName_End() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterName_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterName_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::StdStringCont().rend();
}


//-------------------------------------------------------------------------------
//inline void ROOT::Reflex::MemberBase::Set( const Object & /* instance */,
//                                           const Object & /* value */ ) const {}
//-------------------------------------------------------------------------------



//-------------------------------------------------------------------------------
inline void ROOT::Reflex::MemberBase::Set( const Object & /* instance */,
                                           const void * /* value */ ) const {}
//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::MemberBase::SetScope( const Scope & scope ) const {
//-------------------------------------------------------------------------------
   fScope = scope;
}


//-------------------------------------------------------------------------------
inline void * ROOT::Reflex::MemberBase::Stubcontext() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StubFunction ROOT::Reflex::MemberBase::Stubfunction() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::MemberBase::TemplateArgumentSize() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::MemberBase::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::MemberBase::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::MemberBase::TemplateArgument_RBegin() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::MemberBase::TemplateArgument_REnd() const {
//-------------------------------------------------------------------------------
   return Dummy::TypeCont().rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate ROOT::Reflex::MemberBase::TemplateFamily() const {
//-------------------------------------------------------------------------------
   return Dummy::MemberTemplate();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::MemberBase::TypeOf() const {
//-------------------------------------------------------------------------------
   return fType;
}



#endif // ROOT_Reflex_MemberBase



