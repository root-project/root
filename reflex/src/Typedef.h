// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Typedef
#define ROOT_Reflex_Typedef

// Include files
#include "Reflex/TypeBase.h"
#include "Reflex/Type.h"

namespace ROOT {
  namespace Reflex {

    // forward declarations
    class Base;
    class Object;
    class Member;
    class MemberTemplate;
    class Scope;
    class TypeTemplate;

    /**
     * @class Typedef Typedef.h Reflex/Typedef.h
     * @author Stefan Roiser
     * @date 24/11/2003
     * @ingroup Ref
     */
    class Typedef : public TypeBase {

    public:

      /** constructor */
      Typedef( const char * TypeNth,
               const Type & typedefType,
               TYPE TypeType = TYPEDEF ) ;


      /** destructor */
      virtual ~Typedef();


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
       * IsComplete will return true if all classes and BaseNth classes of this 
       * class are resolved and fully known in the system
       */
      virtual bool IsComplete() const;


      /**
       * IsVirtual will return true if the class contains a virtual table
       * @return true if the class contains a virtual table
       */
      virtual bool IsVirtual() const;


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
       * Name will return the fully qualified Name of the Typedef
       * @param  typedefexp expand typedefs or not
       * @return fully expanded Name of typedef
       */
      virtual std::string Name( unsigned int mod = 0 ) const;


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
       * typedefType will return a pointer to the TypeNth of the typedef.
       * @return pointer to Type of MemberNth et. al.
       */
      virtual Type ToType() const;

    private:  

      bool ForwardStruct() const;
      bool ForwardTemplate() const;
        
    private:

      /**
       * pointer to the TypeNth of the typedef
       * @label typedef TypeNth
       * @link aggregationByValue
       * @supplierCardinality 1
       * @clientCardinality 1
       */
      Type fTypedefType;

    }; // class Typedef
  } //namespace Reflex
} //namespace ROOT

#include "Reflex/Base.h"
#include "Reflex/Object.h"
#include "Reflex/Member.h"
#include "Reflex/MemberTemplate.h"
#include "Reflex/Scope.h"
#include "Reflex/TypeTemplate.h"

//-------------------------------------------------------------------------------
inline ROOT::Reflex::Base ROOT::Reflex::Typedef::BaseNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.BaseNth( nth );
  return Base();  
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Typedef::BaseCount() const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.BaseCount();
  return 0;  
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Object ROOT::Reflex::Typedef::CastObject( const Type & to,
							       const Object & obj ) const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.CastObject( to, obj );
  return Object();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Typedef::DataMemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.DataMemberNth( nth );
  return Member();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Typedef::DataMemberCount() const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.DataMemberCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Typedef::Destruct( void * instance,
					     bool dealloc ) const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) fTypedefType.Destruct( instance, dealloc );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Typedef::DynamicType( const Object & obj ) const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.DynamicType( obj );
  return Type();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Typedef::FunctionMemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.FunctionMemberNth( nth );
  return Member();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Typedef::FunctionMemberCount() const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.FunctionMemberCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Typedef::HasBase( const Type & cl ) const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.HasBase( cl );
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Typedef::IsAbstract() const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.IsAbstract();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Typedef::IsComplete() const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.IsComplete();
  return false;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Typedef::IsVirtual() const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.IsVirtual();
  return false;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Typedef::MemberNth( const std::string & Name ) const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.MemberNth( Name );
  return Member();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Typedef::MemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.MemberNth( nth );
  return Member();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Typedef::MemberCount() const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.MemberCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate ROOT::Reflex::Typedef::MemberTemplateNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.MemberTemplateNth( nth );
  return MemberTemplate();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Typedef::MemberTemplateCount() const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.MemberTemplateCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Typedef::ForwardStruct() const {
//-------------------------------------------------------------------------------
  switch ( fTypedefType.TypeType()) {
  case TYPEDEF:
  case CLASS:
  case STRUCT:
  case TYPETEMPLATEINSTANCE:
    return true;
  default:
    return false;
  }
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Typedef::ForwardTemplate() const {
//-------------------------------------------------------------------------------
  switch ( fTypedefType.TypeType()) {
  case TYPEDEF:
  case TYPETEMPLATEINSTANCE:
  case MEMBERTEMPLATEINSTANCE:
    return true;
  default:
    return false;
  }
}

 
//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::Typedef::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
  if ( 0 != ( mod & ( FINAL | F ))) return ToType().Name( mod );
  else                              return TypeBase::Name( mod );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope ROOT::Reflex::Typedef::SubScopeNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.SubScopeNth( nth );
  return Scope();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Typedef::SubScopeCount() const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.SubScopeCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Typedef::SubTypeNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.SubTypeNth( nth );
  return Type();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Typedef::SubTypeCount() const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.SubTypeCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Typedef::TemplateArgumentNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( ForwardTemplate()) return fTypedefType.TemplateArgumentNth( nth );
  return Type();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Typedef::TemplateArgumentCount() const {
//-------------------------------------------------------------------------------
  if ( ForwardTemplate()) return fTypedefType.TemplateArgumentCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate ROOT::Reflex::Typedef::TemplateFamily() const {
//-------------------------------------------------------------------------------
  if ( ForwardTemplate()) return fTypedefType.TemplateFamily();
  return TypeTemplate();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate ROOT::Reflex::Typedef::TypeTemplateNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.TypeTemplateNth( nth );
  return TypeTemplate();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Typedef::TypeTemplateCount() const {
//-------------------------------------------------------------------------------
  if ( ForwardStruct()) return fTypedefType.TypeTemplateCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline const std::type_info & ROOT::Reflex::Typedef::TypeInfo() const {
//-------------------------------------------------------------------------------
  return ToType().TypeInfo();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Typedef::ToType() const {
//-------------------------------------------------------------------------------
  return fTypedefType;
}


#endif // ROOT_Reflex_Typedef



