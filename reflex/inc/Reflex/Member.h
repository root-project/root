// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Member
#define ROOT_Reflex_Member

// Include files
#include "Reflex/Kernel.h"

namespace ROOT {
  namespace Reflex {

    // forward declarations
    class MemberBase;
    class Type;
    class Scope;
    class PropertyList;
    class Object;
    class MemberTemplate;

    /**
     * @class Member Member.h Reflex/Member.h
     * @author Stefan Roiser
     * @date 24/11/2003
     * @ingroup Ref
     */
    class Member {

    public:

      /** default constructor */
      Member( const MemberBase * memberBase = 0 );


      /** Copy Constructor */
      Member( const Member & rh );


      /**
       * lesser than operator 
       */
      bool operator < ( const Member & rh ) const;
      

      /** 
       * equal operator 
       */
      bool operator == ( const Member & rh ) const;


      /**
       * inequal operator 
       */
      bool operator != ( const Member & rh ) const;


      /** 
       * assignment operator 
       */
      Member & operator = ( const Member & rh );
      

      /** 
       * operator bool will return true if the MemberNth is valid
       * @return true if MemberNth is implemented
       */
      operator bool () const;


      /** destructor */
      ~Member();


      /** 
       * DeclaringScope will return the Scope which the MemberNth lives in
       * (i.e. the same as the Type)
       * @return the declaring Scope of the MemberNth
       */
      Scope DeclaringScope() const;


      /** 
       * DeclaringType will return the Type which the MemberNth lives in
       * (i.e. the same as the Scope)
       * @return the declaring Type of the MemberNth
       */
      Type DeclaringType() const;


      /** Get a static MemberNth value */
      Object Get() const;


      /** Get the MemberNth value */
      Object Get( const Object & obj) const;


      /** Invoke the function (if return TypeNth as void*) */
      /*Object Invoke( const Object & obj, 
        const std::vector < Object > & paramList ) const;*/
      Object Invoke( const Object & obj, 
                     const std::vector < void * > & paramList = 
                     std::vector<void*>()) const;


      /** Invoke the function (for static functions) */
      //Object Invoke( const std::vector < Object > & paramList ) const;
      Object Invoke( const std::vector < void * > & paramList = 
                     std::vector<void*>()) const;


      /** check whether auto is Set for the data MemberNth */
      bool IsAuto() const;


      /** check whether the function MemberNth is a constructor */
      bool IsConstructor() const;


      /** check whether the function MemberNth is a user defined conversion function */
      bool IsConverter() const;


      /** check whether the function MemberNth is a copy constructor */
      bool IsCopyConstructor() const;


      /** return true if this is a data MemberNth */
      bool IsDataMember() const;


      /** check whether the function MemberNth is a destructor */
      bool IsDestructor() const;


      /** check whether explicit is Set for the function MemberNth */
      bool IsExplicit() const;


      /** check whether extern is Set for the data MemberNth */
      bool IsExtern() const;


      /** return true if this is a function MemberNth */
      bool IsFunctionMember() const;


      /** check whether inline is Set for the function MemberNth */
      bool IsInline() const;

      
      /** check whether Mutable is Set for the data MemberNth */
      bool IsMutable() const;


      /** check whether the function MemberNth is an operator */
      bool IsOperator() const;


      /** check whether the function MemberNth is private */
      bool IsPrivate() const;


      /** check whether the function MemberNth is protected */
      bool IsProtected() const;


      /** check whether the function MemberNth is public */
      bool IsPublic() const;


      /** check whether register is Set for the data MemberNth */
      bool IsRegister() const;


      /** check whether static is Set for the data MemberNth */
      bool IsStatic() const;

 
      /** 
       * IsTemplateInstance returns true if the ScopeNth represents a 
       * ClassTemplateInstance
       * @return true if ScopeNth represents a InstantiatedTemplateClass
       */
      bool IsTemplateInstance() const;


      /** check whether the function MemberNth is transient */
      bool IsTransient() const;


      /** check whether virtual is Set for the function MemberNth */
      bool IsVirtual() const;


      /** return the TypeNth of the MemberNth (function or data MemberNth) */
      TYPE MemberType() const;


      /** returns the string representation of the MemberNth species */
      std::string MemberTypeAsString() const;


      /** return the Name of the MemberNth */
      std::string Name( unsigned int mod = 0 ) const;


      /** return the Offset of the MemberNth */
      size_t Offset() const;


      /** number of parameters */
      size_t ParameterCount( bool required = false ) const;


      /** ParameterNth nth default value if declared*/
      std::string ParameterDefault( size_t nth ) const;


      /** ParameterNth nth Name if declared*/
      std::string ParameterName( size_t nth ) const;


      /**
       * PropertyListGet will return a pointer to the PropertyNth list attached
       * to this item
       * @return pointer to PropertyNth list
       */
      PropertyList PropertyListGet() const;


      /** this function will be deprecated, use DeclaringScope instead */
      /** return the ScopeNth of the MemberNth */
      Scope ScopeGet() const;


      /** Set the MemberNth value */
      /*void Set( const Object & instance,
        const Object & value ) const;*/
      void Set( const Object & instance,
                const void * value ) const;


      /** Set the ScopeNth of the MemberNth */
      void SetScope( const Scope & sc ) const;


      /** return a pointer to the context of the MemberNth */
      void * Stubcontext() const;


      /** return the pointer to the stub function */
      StubFunction Stubfunction() const;


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


      /**
       * TemplateFamily returns the corresponding MemberTemplate if any
       * @return corresponding MemberTemplate
       */
      MemberTemplate TemplateFamily() const;


      /** return pointer to MemberNth TypeNth */
      Type TypeGet() const;

    private:

      /** the pointer to the MemberNth implementation 
       * @link aggregation
       * @supplierCardinality 1
       * @clientCardinality 1..*
       * @label MemberNth BaseNth*/
      const MemberBase * fMemberBase;

    }; // class Member

  } //namespace Reflex
} //namespace ROOT

#include "Reflex/MemberBase.h"
#include "Reflex/Scope.h"
#include "Reflex/PropertyList.h"
#include "Reflex/Type.h"
#include "Reflex/MemberTemplate.h"

//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::operator < ( const Member & rh ) const {
//-------------------------------------------------------------------------------
  if ( (*this) && rh ) 
    return ( TypeGet() < rh.TypeGet() && Name() < rh.Name());
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::operator == ( const Member & rh ) const {
//-------------------------------------------------------------------------------
  if ( (*this) && rh ) 
    return ( TypeGet() == rh.TypeGet() && Name() == rh.Name() );
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::operator != ( const Member & rh ) const {
//-------------------------------------------------------------------------------
  return ! ( *this == rh );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member & ROOT::Reflex::Member::operator = ( const Member & rh ) {
//-------------------------------------------------------------------------------
  fMemberBase = rh.fMemberBase;
  return * this;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member::operator bool () const {
//-------------------------------------------------------------------------------
  return 0 != fMemberBase;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope ROOT::Reflex::Member::DeclaringScope() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->DeclaringScope();
  return Scope();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Member::DeclaringType() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->DeclaringScope();
  return Type();
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsAuto() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsAuto();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsConstructor() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsConstructor();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsConverter() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsConverter();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsCopyConstructor() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsCopyConstructor();\
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsDataMember() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsDataMember();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsDestructor() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsDestructor();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsExplicit() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsExplicit();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsExtern() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsExtern();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsFunctionMember() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsFunctionMember();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsInline() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsInline();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsMutable() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsMutable();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsOperator() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsOperator();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsPrivate() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsPrivate();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsProtected() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsProtected();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsPublic() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsPublic();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsRegister() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsRegister();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsStatic() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsStatic();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsTemplateInstance() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsTemplateInstance();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsTransient() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsTransient();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Member::IsVirtual() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->IsVirtual();
  return false;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TYPE ROOT::Reflex::Member::MemberType() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->MemberType();
  return UNRESOLVED;
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::Member::MemberTypeAsString() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->MemberTypeAsString();
  return "";
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::Member::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->Name( mod );
  return "";
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Member::Offset() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->Offset();
  return 0;
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Member::ParameterCount( bool required ) const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->ParameterCount( required );
  return 0;
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::Member::ParameterDefault( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->ParameterDefault( nth );
  return "";
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::Member::ParameterName( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->ParameterName( nth );
  return "";
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::PropertyList ROOT::Reflex::Member::PropertyListGet() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->PropertyListGet();
  return PropertyList();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope ROOT::Reflex::Member::ScopeGet() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->ScopeGet();
  return Scope();
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Member::SetScope( const Scope & sc ) const  {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) fMemberBase->SetScope( sc );
}


//-------------------------------------------------------------------------------
inline void * ROOT::Reflex::Member::Stubcontext() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->Stubcontext();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StubFunction ROOT::Reflex::Member::Stubfunction() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->Stubfunction();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Member::TemplateArgumentNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fMemberBase->TemplateArgumentNth( nth );
  return Type();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Member::TemplateArgumentCount() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fMemberBase->TemplateArgumentCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate ROOT::Reflex::Member::TemplateFamily() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fMemberBase->TemplateFamily();
  return MemberTemplate();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Member::TypeGet() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->TypeGet();
  return Type();
}



#endif // ROOT_Reflex_Member



