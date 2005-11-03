// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Class
#define ROOT_Reflex_Class

// Include files
#include "Reflex/TypeBase.h"
#include "Reflex/ScopeBase.h"
#include "Reflex/Member.h"
#include "Reflex/Base.h"
#include <map>
#include <vector>

namespace ROOT {
  namespace Reflex {
    
    // forward declarations
    class MemberTemplate;
    class TypeTemplate;
    
    /**
     * @class Class Class.h Reflex/Class.h
     * @author Stefan Roiser
     * @date 24/11/2003
     * @ingroup Ref
     */
    class Class : public TypeBase, public ScopeBase {
        
    public:

      /** constructor */
      Class( const char *           typ, 
             size_t                 size, 
             const std::type_info & ti, 
             unsigned int           modifiers = 0,
             TYPE                   classType = CLASS );


      /** destructor */
      virtual ~Class() {}

      
      /** 
       * the operator Type will return a corresponding Type object to the ScopeNth if
       * applicable (i.e. if the Scope is also a Type e.g. Class, Union, Enum)
       */
      operator Type () const;


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


      virtual Base_Iterator Base_Begin() const;
      virtual Base_Iterator Base_End() const;
      virtual Reverse_Base_Iterator Base_Rbegin() const;
      virtual Reverse_Base_Iterator Base_Rend() const;


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
                                void * mem = 0 ) const;
      */
      virtual Object Construct( const Type & signature = Type(),
                                std::vector < void * > values = std::vector<void*>(),
                                void * mem = 0 ) const;


      /**
       * nthDataMember will return the nth data MemberNth of the ScopeNth
       * @param  nth data MemberNth
       * @return pointer to data MemberNth
       */
      virtual Member DataMemberNth( size_t nth ) const;


      /**
       * DataMemberNth will return the MemberNth with Name
       * @param  Name of data MemberNth
       * @return data MemberNth
       */
      virtual Member DataMemberNth( const std::string & nam ) const;


      /**
       * DataMemberCount will return the number of data members of this ScopeNth
       * @return number of data members
       */
      virtual size_t DataMemberCount() const;


      virtual Member_Iterator DataMember_Begin() const;
      virtual Member_Iterator DataMember_End() const;
      virtual Reverse_Member_Iterator DataMember_Rbegin() const;
      virtual Reverse_Member_Iterator DataMember_Rend() const;


      /**
       * DeclaringScope will return a pointer to the ScopeNth of this one
       * @return pointer to declaring ScopeNth
       */
      virtual Scope DeclaringScope() const;


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
       * nthFunctionMember will return the nth function MemberNth of the ScopeNth
       * @param  nth function MemberNth
       * @return pointer to function MemberNth
       */
      virtual Member FunctionMemberNth( size_t nth ) const;

 
      /**
       * FunctionMemberNth will return the MemberNth with the Name, 
       * optionally the signature of the function may be given
       * @param  Name of function MemberNth
       * @param  signature of the MemberNth function 
       * @return function MemberNth
       */
      virtual Member FunctionMemberNth( const std::string & nam,
                                        const Type & signature ) const;


      /**
       * FunctionMemberCount will return the number of function members of
       * this ScopeNth
       * @return number of function members
       */
      virtual size_t FunctionMemberCount() const;


      virtual Member_Iterator FunctionMember_Begin() const;
      virtual Member_Iterator FunctionMember_End() const;
      virtual Reverse_Member_Iterator FunctionMember_Rbegin() const;
      virtual Reverse_Member_Iterator FunctionMember_Rend() const;


      /**
       * HasBase will check whether this class has a BaseNth class given
       * as argument
       * @param  cl the BaseNth-class to check for
       * @return true if this class has a BaseNth-class cl, false otherwise
       */
      virtual bool HasBase( const Type & cl ) const;


      /**
       * HasBase will check whether this class has a BaseNth class given
       * as argument
       * @param  cl the BaseNth-class to check for
       * @param  path optionally the path to the BaseNth can be retrieved
       * @return true if this class has a BaseNth-class cl, false otherwise
       */
      bool HasBase( const Type & cl,
                    std::vector< Base > & path ) const;


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
       * @param Name  MemberNth Name
       * @return pointer to MemberNth
       */
      virtual Member MemberNth( const std::string & nam,
                                const Type & signature ) const;


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


      virtual Member_Iterator Member_Begin() const;
      virtual Member_Iterator Member_End() const;
      virtual Reverse_Member_Iterator Member_Rbegin() const;
      virtual Reverse_Member_Iterator Member_Rend() const;


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


      virtual MemberTemplate_Iterator MemberTemplate_Begin() const;
      virtual MemberTemplate_Iterator MemberTemplate_End() const;
      virtual Reverse_MemberTemplate_Iterator MemberTemplate_Rbegin() const;
      virtual Reverse_MemberTemplate_Iterator MemberTemplate_Rend() const;


      /**
       * Name will return the Name of the class
       * @return Name of class
       */
      virtual std::string Name( unsigned int mod = 0 ) const;


      /** 
       * PathToBase will return a vector of function pointers to the BaseNth class
       * ( !!! Attention !!! the most derived class comes first )
       * @param BaseNth the BaseNth TypeNth 
       * @return vector of function pointers to calculate BaseNth Offset
       */
      const std::vector < OffsetFunction > & PathToBase( const Scope & bas ) const;


      /**
       * PropertyListGet will return a pointer to the PropertyNth list attached
       * to this item
       * @return pointer to PropertyNth list
       */
      virtual PropertyList PropertyListGet() const;


      /**
       * ScopeNth will return the ScopeNth corresponding to the class 
       * @return ScopeNth representation of this class
       */
      virtual Scope ScopeGet() const;


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


      virtual Scope_Iterator SubScope_Begin() const;
      virtual Scope_Iterator SubScope_End() const;
      virtual Reverse_Scope_Iterator SubScope_Rbegin() const;
      virtual Reverse_Scope_Iterator SubScope_Rend() const;


      /**
       * TypeNth will return a pointer to the nth sub-TypeNth
       * @param  nth sub-TypeNth
       * @return pointer to nth sub-TypeNth
       */
      virtual Type SubTypeNth( size_t nth ) const;


      /**
       * TypeCount will returnt he number of sub-types
       * @return number of sub-types
       */
      virtual size_t SubTypeCount() const;


      virtual Type_Iterator SubType_Begin() const;
      virtual Type_Iterator SubType_End() const;
      virtual Reverse_Type_Iterator SubType_Rbegin() const;
      virtual Reverse_Type_Iterator SubType_Rend() const;


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


      virtual TypeTemplate_Iterator TypeTemplate_Begin() const;
      virtual TypeTemplate_Iterator TypeTemplate_End() const;
      virtual Reverse_TypeTemplate_Iterator TypeTemplate_Rbegin() const;
      virtual Reverse_TypeTemplate_Iterator TypeTemplate_Rend() const;


      /** 
       * UpdateMembers2 will update the list of Function/Data/Members with all
       * MemberNth of BaseNth classes currently availabe in the system
       */
      virtual void UpdateMembers() const;

    public:

      /** 
       * AddBase will add the information about a BaseNth class
       * @param  BaseNth TypeNth of the BaseNth class
       * @param  OffsetFP the pointer to the stub function for calculating the Offset
       * @param  modifiers the modifiers of the BaseNth class
       * @return this
       */
      virtual void AddBase( const Type &   bas,
                            OffsetFunction offsFP,
                            unsigned int   modifiers = 0 ) const;
      

      /** 
       * AddBase will add the information about a BaseNth class
       * @param b the pointer to the BaseNth class info
       */
      virtual void AddBase( const Base & b ) const;
      

      /**
       * AddDataMember will add the information about a data MemberNth
       * @param dm pointer to data MemberNth
       */
      virtual void AddDataMember( const Member & dm ) const;
      virtual void AddDataMember( const char * nam,
                                  const Type & typ,
                                  size_t offs,
                                  unsigned int modifiers = 0 ) const;


      /**
       * AddFunctionMember will add the information about a function MemberNth
       * @param fm pointer to function MemberNth
       */
      virtual void AddFunctionMember( const Member & fm ) const;
      virtual void AddFunctionMember( const char * nam,
                                      const Type & typ,
                                      StubFunction stubFP,
                                      void * stubCtx = 0,
                                      const char * params = 0,
                                      unsigned int modifiers = 0 ) const;


      /**
       * AddSubScope will add a sub-ScopeNth to this one
       * @param sc pointer to Scope
       */
      virtual void AddSubScope( const Scope & sc ) const;
      virtual void AddSubScope( const char * scop,
                                TYPE scopeTyp ) const;


      /**
       * AddSubType will add a sub-TypeNth to this ScopeNth
       * @param sc pointer to Type
       */
      virtual void AddSubType( const Type & ty ) const;
      virtual void AddSubType( const char * typ,
                               size_t size,
                               TYPE typeTyp,
                               const std::type_info & ti,
                               unsigned int modifiers = 0 ) const;


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

    public:

      /** 
       * return the TypeNth Name 
       */
      TypeName * TypeNameGet() const;

    private:

      /** map with the class as a key and the path to it as the value
          the key (void*) is a pointer to the unique ScopeName */
      typedef std::map < void *, std::vector < OffsetFunction > * > PathsToBase;

     
      /** 
       * UpdateMembers2 will update the list of Function/Data/Members with all
       * MemberNth of BaseNth classes currently availabe in the system
       * @param members the list of members
       * @param dataMembers the list of data members
       * @param functionMembers the list of function members 
       * @param pathsToBase the cache storing pathes to all known bases
       * @param basePath the current path to the BaseNth class
       */
      void UpdateMembers2( Members & members,
                           Members & dataMembers,
                           Members & functionMembers,
                           PathsToBase & pathsToBase,
                           std::vector < OffsetFunction > & basePath ) const;


      /** 
       * NewBases will return true if new BaseNth classes have been discovered
       * since the last time it was called
       * @return true if new BaseNth classes were resolved
       */
      bool NewBases() const;


      /** 
       * internal recursive checking for completeness 
       * @return true if class is complete (all bases are resolved)
       */
      bool IsComplete2() const;


      /**
       * AllBases will return the number of all BaseNth classes 
       * (double count even in case of virtual inheritance)
       * @return number of all BaseNth classes
       */
      size_t AllBases() const;

    private:

      /**
       * pointer to BaseNth class info
       * @label class bases
       * @link aggregationByValue
       * @clientCardinality 1
       * @supplierCardinality 0..*
       */
      mutable
      std::vector < Base > fBases;


      /** modifiers of class */
      unsigned int fModifiers;


      /** caches */
      /** all currently known BaseNth classes */
      mutable
      size_t fAllBases;


      /** boolean is true if the whole object is resolved */
      mutable
      bool fCompleteType;

      
      /**
       * short cut to constructors
       * @label constructors
       * @link aggregationByValue
       */
      mutable
      std::vector < Member > fConstructors;


      /**
       * short cut to destructor
       * @label destructor
       * @link aggregationByValue
       */
      mutable
      Member fDestructor;


      /** map to all inherited datamembers and their inheritance path */
      mutable
      PathsToBase fPathsToBase;

    }; // class Class
  } //namespace Reflex
} //namespace ROOT

#include "Reflex/Base.h"
#include "Reflex/MemberTemplate.h"
#include "Reflex/TypeTemplate.h"


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Class::operator ROOT::Reflex::Type () const {
//-------------------------------------------------------------------------------
  return TypeBase::operator Type();
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Class::AddBase( const Base & b ) const {
//-------------------------------------------------------------------------------
  fBases.push_back( b );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Base ROOT::Reflex::Class::BaseNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( nth < fBases.size() ) { return fBases[ nth ]; }
  return Base();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::BaseCount() const {
//-------------------------------------------------------------------------------
  return fBases.size();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Base_Iterator ROOT::Reflex::Class::Base_Begin() const {
//-------------------------------------------------------------------------------
  return fBases.begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Base_Iterator ROOT::Reflex::Class::Base_End() const {
//-------------------------------------------------------------------------------
  return fBases.end();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Base_Iterator ROOT::Reflex::Class::Base_Rbegin() const {
//-------------------------------------------------------------------------------
  return fBases.rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Base_Iterator ROOT::Reflex::Class::Base_Rend() const {
//-------------------------------------------------------------------------------
  return fBases.rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Class::DataMemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMemberNth( nth );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Class::DataMemberNth( const std::string & nam ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMemberNth( nam );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::DataMemberCount() const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMemberCount();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Class::DataMember_Begin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMember_Begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Class::DataMember_End() const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMember_End();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Class::DataMember_Rbegin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMember_Rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Class::DataMember_Rend() const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMember_Rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope ROOT::Reflex::Class::DeclaringScope() const {
//-------------------------------------------------------------------------------
  return ScopeBase::DeclaringScope();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Class::FunctionMemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::FunctionMemberNth( nth );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Class::FunctionMemberNth( const std::string & nam,
                                                                    const Type & signature ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::FunctionMemberNth( nam, signature );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::FunctionMemberCount() const {
//-------------------------------------------------------------------------------
  return ScopeBase::FunctionMemberCount();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Class::FunctionMember_Begin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::FunctionMember_Begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Class::FunctionMember_End() const {
//-------------------------------------------------------------------------------
  return ScopeBase::FunctionMember_End();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Class::FunctionMember_Rbegin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::FunctionMember_Rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Class::FunctionMember_Rend() const {
//-------------------------------------------------------------------------------
  return ScopeBase::FunctionMember_Rend();
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Class::IsAbstract() const {
//-------------------------------------------------------------------------------
  return 0 != (fModifiers & ABSTRACT);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Class::IsVirtual() const {
//-------------------------------------------------------------------------------
  return 0 != (fModifiers & VIRTUAL);
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Class::MemberNth( const std::string & nam,
                                                            const Type & signature ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberNth( nam, signature );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Class::MemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberNth( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::MemberCount() const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberCount();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Class::Member_Begin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::Member_Begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Class::Member_End() const {
//-------------------------------------------------------------------------------
  return ScopeBase::Member_End();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Class::Member_Rbegin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::Member_Rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Class::Member_Rend() const {
//-------------------------------------------------------------------------------
  return ScopeBase::Member_Rend();  
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate ROOT::Reflex::Class::MemberTemplateNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberTemplateNth( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::MemberTemplateCount() const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberTemplateCount();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate_Iterator ROOT::Reflex::Class::MemberTemplate_Begin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberTemplate_Begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::MemberTemplate_Iterator ROOT::Reflex::Class::MemberTemplate_End() const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberTemplate_End();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::Class::MemberTemplate_Rbegin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberTemplate_Rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::Class::MemberTemplate_Rend() const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberTemplate_Rend();
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::Class::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::Name( mod );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::PropertyList ROOT::Reflex::Class::PropertyListGet() const {
//-------------------------------------------------------------------------------
  return ScopeBase::PropertyListGet();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope ROOT::Reflex::Class::ScopeGet() const {
//-------------------------------------------------------------------------------
  return ScopeBase::ScopeGet();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope ROOT::Reflex::Class::SubScopeNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::SubScopeNth( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::SubScopeCount() const {
//-------------------------------------------------------------------------------
  return ScopeBase::SubScopeCount();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope_Iterator ROOT::Reflex::Class::SubScope_Begin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::SubScope_Begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope_Iterator ROOT::Reflex::Class::SubScope_End() const {
//-------------------------------------------------------------------------------
  return ScopeBase::SubScope_End();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::Class::SubScope_Rbegin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::SubScope_Rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::Class::SubScope_Rend() const {
//-------------------------------------------------------------------------------
  return ScopeBase::SubScope_Rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Class::SubTypeNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::SubTypeNth( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::SubTypeCount() const {
//-------------------------------------------------------------------------------
  return ScopeBase::SubTypeCount();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Class::SubType_Begin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::SubType_Begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type_Iterator ROOT::Reflex::Class::SubType_End() const {
//-------------------------------------------------------------------------------
  return ScopeBase::SubType_End();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Class::SubType_Rbegin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::SubType_Rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Class::SubType_Rend() const {
//-------------------------------------------------------------------------------
  return ScopeBase::SubType_Rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate ROOT::Reflex::Class::TypeTemplateNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::TypeTemplateNth( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::TypeTemplateCount() const {
//-------------------------------------------------------------------------------
  return ScopeBase::TypeTemplateCount();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::Class::TypeTemplate_Begin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::TypeTemplate_Begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::Class::TypeTemplate_End() const {
//-------------------------------------------------------------------------------
  return ScopeBase::TypeTemplate_End();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::Class::TypeTemplate_Rbegin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::TypeTemplate_Rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::Class::TypeTemplate_Rend() const {
//-------------------------------------------------------------------------------
  return ScopeBase::TypeTemplate_Rend();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeName * ROOT::Reflex::Class::TypeNameGet() const {
//-------------------------------------------------------------------------------
  return fTypeName;
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Class::AddSubScope( const Scope & sc ) const {
//-------------------------------------------------------------------------------
  ScopeBase::AddSubScope( sc );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Class::AddSubScope( const char * scop,
                                              TYPE scopeTyp ) const {
//-------------------------------------------------------------------------------
  ScopeBase::AddSubScope( scop, scopeTyp );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Class::AddSubType( const Type & ty ) const {
//-------------------------------------------------------------------------------
  ScopeBase::AddSubType( ty );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Class::AddSubType( const char * typ,
                                             size_t size,
                                             TYPE typeTyp,
                                             const std::type_info & ti,
                                             unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
  ScopeBase::AddSubType( typ, size, typeTyp, ti, modifiers );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Class::RemoveSubScope( const Scope & sc ) const {
//-------------------------------------------------------------------------------
  ScopeBase::RemoveSubScope( sc );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Class::RemoveSubType( const Type & ty ) const {
//-------------------------------------------------------------------------------
  ScopeBase::RemoveSubType( ty );
}

#endif // ROOT_Reflex_Class

