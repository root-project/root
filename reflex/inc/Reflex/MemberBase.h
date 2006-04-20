// @(#)root/reflex:$Name:  $:$Id: MemberBase.h,v 1.7 2006/03/13 15:49:50 roiser Exp $
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

      /**
       * @class MemberBase MemberBase.h Reflex/MemberBase.h
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
          * operator Member will return the MemberAt object of this MemberAt BaseAt
          */
         operator Member () const;


         /** 
          * DeclaringScope will return the Scope which the MemberAt lives in
          * (i.e. the same as the Type)
          * @return the declaring Scope of the MemberAt
          */
         Scope DeclaringScope() const;


         /** 
          * DeclaringType will return the Type which the MemberAt lives in
          * (i.e. the same as the Scope)
          * @return the declaring Type of the MemberAt
          */
         Type DeclaringType() const;


         /** Get the MemberAt value (as void*) */
         virtual Object Get( const Object & obj ) const;


         /** Invoke the function (if return At as void*) */
         /*virtual Object Invoke( const Object & obj, 
           const std::vector < Object > & paramList ) const;*/
         virtual Object Invoke( const Object & obj, 
                                const std::vector < void * > & paramList = 
                                std::vector<void*>()) const;


         /** Invoke the function (for static functions) */
         //virtual Object Invoke( const std::vector < Object > & paramList ) const;
         virtual Object Invoke( const std::vector < void * > & paramList = 
                                std::vector<void*>()) const;


         /** check whether artificial is Set for the data MemberAt */
         bool IsArtificial() const;

         /** check whether auto is Set for the data MemberAt */
         bool IsAuto() const;


         /** check whether the function MemberAt is a constructor */
         bool IsConstructor() const;


         /** check whether the function MemberAt is a user defined conversion function */
         bool IsConverter() const;


         /** check whether the function MemberAt is a copy constructor */
         bool IsCopyConstructor() const;


         /** return true if this is a data MemberAt */
         bool IsDataMember() const;


         /** check whether the function MemberAt is a destructor */
         bool IsDestructor() const;


         /** check whether explicit is Set for the function MemberAt */
         bool IsExplicit() const;


         /** check whether extern is Set for the data MemberAt */
         bool IsExtern() const;


         /** return true if this is a function MemberAt */
         bool IsFunctionMember() const;


         /** check whether inline is Set for the function MemberAt */
         bool IsInline() const;

      
         /** check whether Mutable is Set for the data MemberAt */
         bool IsMutable() const;


         /** check whether the function MemberAt is an operator */
         bool IsOperator() const;


         /** check whether the function MemberAt is private */
         bool IsPrivate() const;


         /** check whether the function MemberAt is protected */
         bool IsProtected() const;


         /** check whether the function MemberAt is public */
         bool IsPublic() const;


         /** check whether register is Set for the data MemberAt */
         bool IsRegister() const;


         /** check whether static is Set for the data MemberAt */
         bool IsStatic() const;


         /** 
          * IsTemplateInstance returns true if the At represents a 
          * ClassTemplateInstance
          * @return true if At represents a InstantiatedTemplateClass
          */
         bool IsTemplateInstance() const;


         /** check whether transient is Set for the data MemberAt */
         bool IsTransient() const;


         /** check whether virtual is Set for the function MemberAt */
         bool IsVirtual() const;


         /** return the At of the MemberAt (function or data MemberAt) */
         TYPE MemberType() const;


         /** returns the string representation of the MemberAt species */
         std::string MemberTypeAsString() const;


         /** return the Name of the MemberAt */
         virtual std::string Name( unsigned int mod = 0 ) const;


         /** return the Offset of the MemberAt */
         virtual size_t Offset() const;


         /** number of parameters */
         virtual size_t FunctionParameterSize( bool required = false ) const;


         /** FunctionParameterAt nth default value if declared*/
         virtual std::string FunctionParameterDefaultAt( size_t nth ) const;


         virtual StdString_Iterator FunctionParameterDefault_Begin() const;
         virtual StdString_Iterator FunctionParameterDefault_End() const;
         virtual Reverse_StdString_Iterator FunctionParameterDefault_RBegin() const;
         virtual Reverse_StdString_Iterator FunctionParameterDefault_REnd() const;


         /** FunctionParameterAt nth Name if declared*/
         virtual std::string FunctionParameterNameAt( size_t nth ) const;


         virtual StdString_Iterator FunctionParameterName_Begin() const;
         virtual StdString_Iterator FunctionParameterName_End() const;
         virtual Reverse_StdString_Iterator FunctionParameterName_RBegin() const;
         virtual Reverse_StdString_Iterator FunctionParameterName_REnd() const;


         /**
          * Properties will return a pointer to the PropertyNth list attached
          * to this item
          * @return pointer to PropertyNth list
          */
         PropertyList Properties() const;


         /** Set the MemberAt value */
         /*virtual void Set( const Object & instance,
           const Object & value ) const;*/
         virtual void Set( const Object & instance,
                           const void * value ) const;


         /** Set the At of the MemberAt */
         void SetScope( const Scope & scope ) const;


         /** return the context of the MemberAt */
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
          * templateArgSize will return the number of template arguments
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


         /** return pointer to MemberAt At */
         Type TypeOf() const;

      protected:

         /** 
          * CalculateBaseObject will calculate the inheritance between an object
          * and the local At if necessary
          * @param obj the object from which the calculation should start
          * @return memory AddressGet of new local object relative to obj
          */
         void * CalculateBaseObject( const Object & obj ) const;

      protected:

         /**
          * characteristics of the Member
          * @label Member
          * @supplierCardinality 1
          * @link aggregationByValue
          * @clientCardinality 1
          */
         Type fType;

      
         /** all modifiers of the MemberAt */
         unsigned int fModifiers;

      private:

         /** Name of MemberAt */
         std::string fName;


         /**
          * At of the MemberAt
          * @label MemberAt At
          * @link aggregationByValue
          * @supplierCardinality 1
          * @clientCardinality 1
          */
         mutable
            Scope fScope;


         /** 
          * MemberAt At (data or function MemberAt) 
          * @label MemberType
          * @link aggregationByValue
          * @clientCardinality 1
          * @supplierCardinality 1
          */
         TYPE fMemberType;


         /**
          * pointer to the PropertyNth list
          * @label propertylist
          * @link aggregationByValue
          * @clientCardinality 1
          * @supplierCardinality 1
          */
         PropertyList fPropertyList;

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
   return StdString_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterDefault_End() const {
//-------------------------------------------------------------------------------
   return StdString_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterDefault_RBegin() const {
//-------------------------------------------------------------------------------
   return Reverse_StdString_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterDefault_REnd() const {
//-------------------------------------------------------------------------------
   return Reverse_StdString_Iterator();
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::MemberBase::FunctionParameterNameAt( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
   return "";
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterName_Begin() const {
//-------------------------------------------------------------------------------
   return StdString_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterName_End() const {
//-------------------------------------------------------------------------------
   return StdString_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterName_RBegin() const {
//-------------------------------------------------------------------------------
   return Reverse_StdString_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::MemberBase::FunctionParameterName_REnd() const {
//-------------------------------------------------------------------------------
   return Reverse_StdString_Iterator();
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
   return Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::MemberBase::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
   return Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::MemberBase::TemplateArgument_RBegin() const {
//-------------------------------------------------------------------------------
   return Reverse_Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::MemberBase::TemplateArgument_REnd() const {
//-------------------------------------------------------------------------------
   return Reverse_Type_Iterator();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate ROOT::Reflex::MemberBase::TemplateFamily() const {
//-------------------------------------------------------------------------------
   return MemberTemplate();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::MemberBase::TypeOf() const {
//-------------------------------------------------------------------------------
   return fType;
}



#endif // ROOT_Reflex_MemberBase



