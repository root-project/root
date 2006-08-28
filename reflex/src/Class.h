// @(#)root/reflex:$Name:  $:$Id: Class.h,v 1.14 2006/08/17 13:50:30 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Class
#define ROOT_Reflex_Class

// Include files
#include "Reflex/internal/TypeBase.h"
#include "Reflex/internal/ScopeBase.h"
#include "Reflex/internal/OwnedMember.h"
#include <map>
#include <vector>

namespace ROOT {
   namespace Reflex {
    
      // forward declarations
      class Base;
      class Member;
      class MemberTemplate;
      class TypeTemplate;
      class DictionaryGenerator;
      

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
         virtual ~Class();
	 

         /**
          * operator Scope will return the corresponding scope of this type if
          * applicable (i.e. if the Type is also a Scope e.g. Class, Union, Enum)
          */                                       
         operator const Scope & () const;


         /** 
          * the operator Type will return a corresponding Type object to the At if
          * applicable (i.e. if the Scope is also a Type e.g. Class, Union, Enum)
          */
         operator const Type & () const;


         /**
          * nthBase will return the nth BaseAt class information
          * @param  nth nth BaseAt class
          * @return pointer to BaseAt class information
          */
         virtual const Base & BaseAt( size_t nth ) const;


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
                                     void * mem = 0 ) const;
         */
         virtual Object Construct( const Type & signature = Type(),
                                   const std::vector < void * > & values = std::vector<void*>(),
                                   void * mem = 0 ) const;


         /**
          * GenerateDict will produce the dictionary information of this type
          * @param generator a reference to the dictionary generator instance
          */
         virtual void GenerateDict(DictionaryGenerator &generator) const;
	  
      
         /**
          * nthDataMember will return the nth data MemberAt of the At
          * @param  nth data MemberAt
          * @return pointer to data MemberAt
          */
         virtual const Member & DataMemberAt( size_t nth ) const;


         /**
          * DataMemberByName will return the MemberAt with Name
          * @param  Name of data MemberAt
          * @return data MemberAt
          */
         virtual const Member & DataMemberByName( const std::string & nam ) const;


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
          * DeclaringScope will return a pointer to the At of this one
          * @return pointer to declaring At
          */
         virtual const Scope & DeclaringScope() const;


         /**
          * Destruct will call the destructor of a At and remove its memory
          * allocation if desired
          * @param  instance of the At in memory
          * @param  dealloc for also deallacoting the memory
          */
         virtual void Destruct( void * instance, 
                                bool dealloc = true ) const;


         /**
          * DynamicType is used to discover whether an object represents the
          * current class At or not
          * @param  mem is the memory AddressGet of the object to checked
          * @return the actual class of the object
          */
         virtual const Type & DynamicType( const Object & obj ) const;


         /**
          * nthFunctionMember will return the nth function MemberAt of the At
          * @param  nth function MemberAt
          * @return pointer to function MemberAt
          */
         virtual const Member & FunctionMemberAt( size_t nth ) const;

 
         /**
          * FunctionMemberByName will return the MemberAt with the Name, 
          * optionally the signature of the function may be given
          * @param  Name of function MemberAt
          * @param  signature of the MemberAt function 
          * @return function MemberAt
          */
         virtual const Member & FunctionMemberByName( const std::string & nam,
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
          * @return the Base info if it is found, an empty base otherwise (can be tested for bool)
          */
         virtual const Base & HasBase( const Type & cl ) const;


         /**
          * HasBase will check whether this class has a BaseAt class given
          * as argument
          * @param  cl the BaseAt-class to check for
          * @param  path optionally the path to the BaseAt can be retrieved
          * @return true if this class has a BaseAt-class cl, false otherwise
          */
         bool HasBase( const Type & cl,
                       std::vector< Base > & path ) const;


         /**
          * IsAbstract will return true if the the class is abstract
          * @return true if the class is abstract
          */
         virtual bool IsAbstract() const;


         /** 
          * IsComplete will return true if all classes and BaseAt classes of this 
          * class are resolved and fully known in the system
          */
         virtual bool IsComplete() const;


         /** 
          * IsPrivate will check if the scope access is private
          * @return true if scope access is private
          */
         virtual bool IsPrivate() const;


         /** 
          * IsProtected will check if the scope access is protected
          * @return true if scope access is protected
          */
         virtual bool IsProtected() const;


         /** 
          * IsPublic will check if the scope access is public
          * @return true if scope access is public
          */
         virtual bool IsPublic() const;


         /**
          * IsVirtual will return true if the class contains a virtual table
          * @return true if the class contains a virtual table
          */
         virtual bool IsVirtual() const;


         /**
          * MemberByName will return the first MemberAt with a given Name
          * @param Name  MemberAt Name
          * @return pointer to MemberAt
          */
         virtual const Member & MemberByName( const std::string & nam,
                                              const Type & signature ) const;


         /**
          * MemberAt will return the nth MemberAt of the At
          * @param  nth MemberAt
          * @return pointer to nth MemberAt
          */
         virtual const Member & MemberAt( size_t nth ) const;


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
         virtual const MemberTemplate & MemberTemplateAt( size_t nth ) const;


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
          * Name will return the Name of the class
          * @return Name of class
          */
         virtual std::string Name( unsigned int mod = 0 ) const;


         /** 
          * PathToBase will return a vector of function pointers to the BaseAt class
          * ( !!! Attention !!! the most derived class comes first )
          * @param BaseAt the BaseAt At 
          * @return vector of function pointers to calculate BaseAt Offset
          */
         const std::vector < OffsetFunction > & PathToBase( const Scope & bas ) const;


         /**
          * Properties will return a pointer to the PropertyNth list attached
          * to this item
          * @return pointer to PropertyNth list
          */
         virtual const PropertyList & Properties() const;


         /**
          * SubScopeAt will return a pointer to a sub-scopes
          * @param  nth sub-At
          * @return pointer to nth sub-At
          */
         virtual const Scope & SubScopeAt( size_t nth ) const;


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
          * At will return a pointer to the nth sub-At
          * @param  nth sub-At
          * @return pointer to nth sub-At
          */
         virtual const Type & SubTypeAt( size_t nth ) const;


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
          * SubTypeTemplateAt will return the nth At template of this At
          * @param nth At template
          * @return nth At template
          */
         virtual const TypeTemplate & SubTypeTemplateAt( size_t nth ) const;


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
          * UpdateMembers2 will update the list of Function/Data/Members with all
          * MemberAt of BaseAt classes currently availabe in the system
          */
         virtual void UpdateMembers() const;

      public:

         /** 
          * AddBase will add the information about a BaseAt class
          * @param  BaseAt At of the BaseAt class
          * @param  OffsetFP the pointer to the stub function for calculating the Offset
          * @param  modifiers the modifiers of the BaseAt class
          * @return this
          */
         virtual void AddBase( const Type &   bas,
                               OffsetFunction offsFP,
                               unsigned int   modifiers = 0 ) const;
      

         /** 
          * AddBase will add the information about a BaseAt class
          * @param b the pointer to the BaseAt class info
          */
         virtual void AddBase( const Base & b ) const;
      

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
                                  unsigned int modifiers = 0 ) const;


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

      public:

         /** 
          * return the type name 
          */
         TypeName * TypeNameGet() const;

      private:

         /** map with the class as a key and the path to it as the value
             the key (void*) is a pointer to the unique ScopeName */
         typedef std::map < void *, std::vector < OffsetFunction > * > PathsToBase;

     
         /** 
          * UpdateMembers2 will update the list of Function/Data/Members with all
          * MemberAt of BaseAt classes currently availabe in the system
          * @param members the list of members
          * @param dataMembers the list of data members
          * @param functionMembers the list of function members 
          * @param pathsToBase the cache storing pathes to all known bases
          * @param basePath the current path to the BaseAt class
          */
         void UpdateMembers2( OMembers & members,
                              Members & dataMembers,
                              Members & functionMembers,
                              PathsToBase & pathsToBase,
                              std::vector < OffsetFunction > & basePath ) const;


         /** 
          * NewBases will return true if new BaseAt classes have been discovered
          * since the last time it was called
          * @return true if new BaseAt classes were resolved
          */
         bool NewBases() const;


         /** 
          * internal recursive checking for completeness 
          * @return true if class is complete (all bases are resolved)
          */
         bool IsComplete2() const;


         /**
          * AllBases will return the number of all BaseAt classes 
          * (double count even in case of virtual inheritance)
          * @return number of all BaseAt classes
          */
         size_t AllBases() const;

      private:

         /**
          * pointer to BaseAt class info
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
         /** all currently known BaseAt classes */
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
inline ROOT::Reflex::Class::operator const ROOT::Reflex::Scope & () const {
//-------------------------------------------------------------------------------
   return ScopeBase::operator const Scope & ();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Class::operator const ROOT::Reflex::Type & () const {
//-------------------------------------------------------------------------------
   return TypeBase::operator const Type & ();
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Class::AddBase( const Base & b ) const {
//-------------------------------------------------------------------------------
   fBases.push_back( b );
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Base & ROOT::Reflex::Class::BaseAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( nth < fBases.size() ) { return fBases[ nth ]; }
   return Dummy::Base();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::BaseSize() const {
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
inline ROOT::Reflex::Reverse_Base_Iterator ROOT::Reflex::Class::Base_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Base>&)fBases).rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Base_Iterator ROOT::Reflex::Class::Base_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Base>&)fBases).rend();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Member & ROOT::Reflex::Class::DataMemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMemberAt( nth );
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Member & ROOT::Reflex::Class::DataMemberByName( const std::string & nam ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMemberByName( nam );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::DataMemberSize() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMemberSize();
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
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Class::DataMember_RBegin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMember_RBegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Class::DataMember_REnd() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMember_REnd();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Scope & ROOT::Reflex::Class::DeclaringScope() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DeclaringScope();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Member & ROOT::Reflex::Class::FunctionMemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::FunctionMemberAt( nth );
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Member & ROOT::Reflex::Class::FunctionMemberByName( const std::string & nam,
                                                                       const Type & signature ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::FunctionMemberByName( nam, signature );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::FunctionMemberSize() const {
//-------------------------------------------------------------------------------
   return ScopeBase::FunctionMemberSize();
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
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Class::FunctionMember_RBegin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::FunctionMember_RBegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Class::FunctionMember_REnd() const {
//-------------------------------------------------------------------------------
   return ScopeBase::FunctionMember_REnd();
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Class::IsAbstract() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & ABSTRACT);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Class::IsPrivate() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & PRIVATE);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Class::IsProtected() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & PROTECTED);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Class::IsPublic() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & PUBLIC);
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Class::IsVirtual() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & VIRTUAL);
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Member & ROOT::Reflex::Class::MemberByName( const std::string & nam,
                                                                       const Type & signature ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::MemberByName( nam, signature );
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Member & ROOT::Reflex::Class::MemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::MemberAt( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::MemberSize() const {
//-------------------------------------------------------------------------------
   return ScopeBase::MemberSize();
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
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Class::Member_RBegin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::Member_RBegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Class::Member_REnd() const {
//-------------------------------------------------------------------------------
   return ScopeBase::Member_REnd();  
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::MemberTemplate & ROOT::Reflex::Class::MemberTemplateAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::MemberTemplateAt( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::MemberTemplateSize() const {
//-------------------------------------------------------------------------------
   return ScopeBase::MemberTemplateSize();
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
inline ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::Class::MemberTemplate_RBegin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::MemberTemplate_RBegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_MemberTemplate_Iterator ROOT::Reflex::Class::MemberTemplate_REnd() const {
//-------------------------------------------------------------------------------
   return ScopeBase::MemberTemplate_REnd();
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::Class::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::Name( mod );
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::PropertyList & ROOT::Reflex::Class::Properties() const {
//-------------------------------------------------------------------------------
   return ScopeBase::Properties();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Scope & ROOT::Reflex::Class::SubScopeAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::SubScopeAt( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::SubScopeSize() const {
//-------------------------------------------------------------------------------
   return ScopeBase::SubScopeSize();
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
inline ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::Class::SubScope_RBegin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::SubScope_RBegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::Class::SubScope_REnd() const {
//-------------------------------------------------------------------------------
   return ScopeBase::SubScope_REnd();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::Type & ROOT::Reflex::Class::SubTypeAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::SubTypeAt( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::SubTypeSize() const {
//-------------------------------------------------------------------------------
   return ScopeBase::SubTypeSize();
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
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Class::SubType_RBegin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::SubType_RBegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::Class::SubType_REnd() const {
//-------------------------------------------------------------------------------
   return ScopeBase::SubType_REnd();
}


//-------------------------------------------------------------------------------
inline const ROOT::Reflex::TypeTemplate & ROOT::Reflex::Class::SubTypeTemplateAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::SubTypeTemplateAt( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Class::SubTypeTemplateSize() const {
//-------------------------------------------------------------------------------
   return ScopeBase::SubTypeTemplateSize();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::Class::SubTypeTemplate_Begin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::SubTypeTemplate_Begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TypeTemplate_Iterator ROOT::Reflex::Class::SubTypeTemplate_End() const {
//-------------------------------------------------------------------------------
   return ScopeBase::SubTypeTemplate_End();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::Class::SubTypeTemplate_RBegin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::SubTypeTemplate_RBegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_TypeTemplate_Iterator ROOT::Reflex::Class::SubTypeTemplate_REnd() const {
//-------------------------------------------------------------------------------
   return ScopeBase::SubTypeTemplate_REnd();
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

