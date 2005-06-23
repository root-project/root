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
      Class( const char *           TypeNth, 
             size_t                 size, 
             const std::type_info & TypeInfo, 
             unsigned int           modifiers = 0,
             TYPE                   classType = CLASS );


      /** destructor */
      virtual ~Class() {}

      
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


      /**
       * CastObject an object from this class TypeNth to another one
       * @param  to is the class TypeNth to cast into
       * @param  obj the memory AddressGet of the object to be casted
       */
      Object CastObject( const Type & to, 
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
      Member DataMemberNth( size_t nth ) const;


      /**
       * DataMemberCount will return the number of data members of this ScopeNth
       * @return number of data members
       */
      size_t DataMemberCount() const;


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
       * nthFunctionMember will return the nth function MemberNth of the ScopeNth
       * @param  nth function MemberNth
       * @return pointer to function MemberNth
       */
      Member FunctionMemberNth( size_t nth ) const;

 
      /**
       * FunctionMemberCount will return the number of function members of
       * this ScopeNth
       * @return number of function members
       */
      size_t FunctionMemberCount() const;


      /**
       * HasBase will check whether this class has a BaseNth class given
       * as argument
       * @param  cl the BaseNth-class to check for
       * @return true if this class has a BaseNth-class cl, false otherwise
       */
      bool HasBase( const Type & cl ) const;


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
      bool IsAbstract() const;


      /** 
       * IsComplete will return true if all classes and BaseNth classes of this 
       * class are resolved and fully known in the system
       */
      bool IsComplete() const;


      /**
       * IsVirtual will return true if the class contains a virtual table
       * @return true if the class contains a virtual table
       */
      bool IsVirtual() const;


      /**
       * MemberNth will return the first MemberNth with a given Name
       * @param Name  MemberNth Name
       * @return pointer to MemberNth
       */
      Member MemberNth( const std::string & Name ) const;


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
      const std::vector < OffsetFunction > & PathToBase( const Scope & BaseNth ) const;


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
      Scope ScopeGet() const;


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


      /**
       * TypeNth will return a pointer to the nth sub-TypeNth
       * @param  nth sub-TypeNth
       * @return pointer to nth sub-TypeNth
       */
      Type SubTypeNth( size_t nth ) const;


      /**
       * TypeCount will returnt he number of sub-types
       * @return number of sub-types
       */
      size_t SubTypeCount() const;


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


      /** 
       * UpdateMembers2 will update the list of Function/Data/Members with all
       * MemberNth of BaseNth classes currently availabe in the system
       */
      void UpdateMembers() const;

    public:

      /** 
       * AddBase will add the information about a BaseNth class
       * @param  BaseNth TypeNth of the BaseNth class
       * @param  OffsetFP the pointer to the stub function for calculating the Offset
       * @param  modifiers the modifiers of the BaseNth class
       * @return this
       */
      void AddBase( const Type &   BaseNth,
                    OffsetFunction OffsetFP,
                    unsigned int   modifiers = 0 ) const;
      

      /** 
       * AddBase will add the information about a BaseNth class
       * @param b the pointer to the BaseNth class info
       */
      void AddBase( const Base & b ) const;


      /**
       * AddDataMember will add the information about a data MemberNth
       * @param dm pointer to data MemberNth
       */
      virtual void AddDataMember( const Member & dm ) const;
      virtual void AddDataMember( const char * Name,
                                  const Type & TypeNth,
                                  size_t Offset,
                                  unsigned int modifiers = 0 ) const;


      /**
       * AddFunctionMember will add the information about a function MemberNth
       * @param fm pointer to function MemberNth
       */
      virtual void AddFunctionMember( const Member & fm ) const;
      virtual void AddFunctionMember( const char * Name,
                                      const Type & TypeNth,
                                      StubFunction stubFP,
                                      void * stubCtx = 0,
                                      const char * params = 0,
                                      unsigned int modifiers = 0 ) const;


      /**
       * AddSubScope will add a sub-ScopeNth to this one
       * @param sc pointer to Scope
       */
      virtual void AddSubScope( const Scope & sc ) const;
      virtual void AddSubScope( const char * ScopeNth,
                                TYPE ScopeType ) const;


      /**
       * AddSubType will add a sub-TypeNth to this ScopeNth
       * @param sc pointer to Type
       */
      virtual void AddSubType( const Type & ty ) const;
      virtual void AddSubType( const char * TypeNth,
                               size_t size,
                               TYPE TypeType,
                               const std::type_info & TypeInfo,
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
inline ROOT::Reflex::Member ROOT::Reflex::Class::DataMemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMemberNth( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::DataMemberCount() const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMemberCount();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Class::FunctionMemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::FunctionMemberNth( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::FunctionMemberCount() const {
//-------------------------------------------------------------------------------
  return ScopeBase::FunctionMemberCount();
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
inline ROOT::Reflex::Member ROOT::Reflex::Class::MemberNth( const std::string & Name ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberNth( Name );
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
inline void ROOT::Reflex::Class::AddSubScope( const char * ScopeNth,
                                              TYPE ScopeType ) const {
//-------------------------------------------------------------------------------
  ScopeBase::AddSubScope( ScopeNth, ScopeType );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Class::AddSubType( const Type & ty ) const {
//-------------------------------------------------------------------------------
  ScopeBase::AddSubType( ty );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Class::AddSubType( const char * TypeNth,
                                             size_t size,
                                             TYPE TypeType,
                                             const std::type_info & TypeInfo,
                                             unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
  ScopeBase::AddSubType( TypeNth, size, TypeType, TypeInfo, modifiers );
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

