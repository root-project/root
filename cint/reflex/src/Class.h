// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_Class
#define Reflex_Class

// Include files
#include "ScopedType.h"
#include "Reflex/Member.h"
#include "Reflex/internal/OwnedMember.h"
#include <map>
#include <vector>

namespace Reflex
{

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
class Class : public ScopedType
{

public:

   /** constructor */
   Class(const char* typ, size_t size, const std::type_info& ti, unsigned int modifiers = 0, TYPE classType = CLASS);

   /** destructor */
   virtual ~Class();

   /**
   * nthBase will return the nth BaseAt class information
   * @param  nth nth BaseAt class
   * @return pointer to BaseAt class information
   */
   virtual Base BaseAt(size_t nth) const;

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
   virtual Object CastObject(const Type& to, const Object& obj) const;

   /**
   * Construct will call the constructor of a given At and Allocate the
   * memory for it
   * @param  signature of the constructor
   * @param  values for parameters of the constructor
   * @param  mem place in memory for implicit construction
   * @return pointer to new instance
   */
   //virtual Object Construct(const Type& signature, const std::vector<Object>& values, void* mem = 0) const;
   virtual Object Construct(const Type& signature = Type(), const std::vector<void*>& values = std::vector<void*>(), void* mem = 0) const;


   /**
   * GenerateDict will produce the dictionary information of this type
   * @param generator a reference to the dictionary generator instance
   */
   virtual void GenerateDict(DictionaryGenerator& generator) const;

   /**
   * Destruct will call the destructor of a At and remove its memory
   * allocation if desired
   * @param  instance of the At in memory
   * @param  dealloc for also deallacoting the memory
   */
   virtual void Destruct(void* instance, bool dealloc = true) const;

   /**
   * DynamicType is used to discover whether an object represents the
   * current class At or not
   * @param  mem is the memory AddressGet of the object to checked
   * @return the actual class of the object
   */
   virtual Type DynamicType(const Object& obj) const;

   /**
   * HasBase will check whether this class has a BaseAt class given
   * as argument
   * @param  cl the BaseAt-class to check for
   * @return the Base info if it is found, an empty base otherwise (can be tested for bool)
   */
   virtual bool HasBase(const Type& cl) const;

   /**
   * HasBase will check whether this class has a BaseAt class given
   * as argument
   * @param  cl the BaseAt-class to check for
   * @param  path optionally the path to the BaseAt can be retrieved
   * @return true if this class has a BaseAt-class cl, false otherwise
   */
   bool HasBase(const Type& cl, std::vector<Base>& path) const;

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
   * IsVirtual will return true if the class contains a virtual table
   * @return true if the class contains a virtual table
   */
   virtual bool IsVirtual() const;

   /**
   * PathToBase will return a vector of function pointers to the base class
   * ( !!! Attention !!! the most derived class comes first )
   * @param base the scope to look for
   * @return vector of function pointers to calculate base offset
   */
   const std::vector<OffsetFunction>& PathToBase(const Scope& bas) const;


   /**
   * UpdateMembers2 will update the list of Function/Data/Members with all
   * MemberAt of BaseAt classes currently availabe in the system
   */
   virtual void UpdateMembers() const;

   /**
   * AddBase will add the information about a BaseAt class
   * @param  BaseAt At of the BaseAt class
   * @param  OffsetFP the pointer to the stub function for calculating the Offset
   * @param  modifiers the modifiers of the BaseAt class
   * @return this
   */
   virtual void AddBase(const Type & bas, OffsetFunction offsFP, unsigned int modifiers = 0) const;

   /**
   * AddBase will add the information about a BaseAt class
   * @param b the pointer to the BaseAt class info
   */
   virtual void AddBase(const Base& b) const;

   /**
   * AddDataMember will add the information about a data MemberAt
   * @param dm pointer to data MemberAt
   */
   virtual void AddDataMember(const Member& dm) const;
   virtual void AddDataMember(const char* nam, const Type& typ, size_t offs, unsigned int modifiers = 0) const;
   virtual void AddDataMember(Member& output, const char* nam, const Type& typ, size_t offs, unsigned int modifiers = 0, char* interpreterOffset = 0) const;

   /**
   * AddFunctionMember will add the information about a function MemberAt
   * @param fm pointer to function MemberAt
   */
   virtual void AddFunctionMember(const Member& fm) const;
   virtual void AddFunctionMember(const char* nam, const Type& typ, StubFunction stubFP, void* stubCtx = 0, const char* params = 0, unsigned int modifiers = 0) const;

   /**
   * RemoveDataMember will remove the information about a data MemberAt
   * @param dm pointer to data MemberAt
   */
   virtual void RemoveDataMember(const Member& dm) const;

   /**
   * RemoveFunctionMember will remove the information about a function MemberAt
   * @param fm pointer to function MemberAt
   */
   virtual void RemoveFunctionMember(const Member& fm) const;

private:

   /** map with the class as a key and the path to it as the value
   the key (void*) is a pointer to the unique ScopeName */
   typedef std::map<void*, std::vector<OffsetFunction>* > PathsToBase;

   /**
   * UpdateMembers2 will update the list of Function/Data/Members with all
   * MemberAt of BaseAt classes currently availabe in the system
   * @param members the list of members
   * @param dataMembers the list of data members
   * @param functionMembers the list of function members
   * @param pathsToBase the cache storing pathes to all known bases
   * @param basePath the current path to the BaseAt class
   */
   void UpdateMembers2(OMembers& members, Members& dataMembers, Members& functionMembers, PathsToBase& pathsToBase, std::vector<OffsetFunction>& basePath) const;

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
   * container of base classes
   * @label class bases
   * @link aggregation
   * @clientCardinality 1
   * @supplierCardinality 0..*
   */
   mutable std::vector<Base> fBases;

   /** caches */
   /** all currently known BaseAt classes */
   mutable size_t fAllBases;

   /** boolean is true if the whole object is resolved */
   mutable bool fCompleteType;

   /**
   * short cut to constructors
   * @label constructors
   * @link aggregation
   * @clientCardinality 1
   * @supplierCardinality 1..*
   */
   mutable std::vector<Member> fConstructors;

   /**
   * short cut to destructor
   * @label destructor
   * @link aggregation
   * @clientCardinality 1
   * @supplierCardinality 1
   */
   mutable Member fDestructor;

   /** map to all inherited datamembers and their inheritance path */
   mutable PathsToBase fPathsToBase;

}; // class Class
} //namespace Reflex

#include "Reflex/Base.h"


//-------------------------------------------------------------------------------
inline void Reflex::Class::AddBase(const Base & b) const
{
//-------------------------------------------------------------------------------
   fBases.push_back(b);
}


//-------------------------------------------------------------------------------
inline Reflex::Base Reflex::Class::BaseAt(size_t nth) const
{
//-------------------------------------------------------------------------------
   if (nth < fBases.size()) {
      return fBases[ nth ];
   }
   return Dummy::Base();
}


//-------------------------------------------------------------------------------
inline size_t Reflex::Class::BaseSize() const
{
//-------------------------------------------------------------------------------
   return fBases.size();
}


//-------------------------------------------------------------------------------
inline Reflex::Base_Iterator Reflex::Class::Base_Begin() const
{
//-------------------------------------------------------------------------------
   return fBases.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Base_Iterator Reflex::Class::Base_End() const
{
//-------------------------------------------------------------------------------
   return fBases.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Base_Iterator Reflex::Class::Base_RBegin() const
{
//-------------------------------------------------------------------------------
   return ((const std::vector<Base>&)fBases).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Base_Iterator Reflex::Class::Base_REnd() const
{
//-------------------------------------------------------------------------------
   return ((const std::vector<Base>&)fBases).rend();
}


//-------------------------------------------------------------------------------
inline bool Reflex::Class::IsAbstract() const
{
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & ABSTRACT);
}


//-------------------------------------------------------------------------------
inline bool Reflex::Class::IsVirtual() const
{
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & VIRTUAL);
}

#endif // Reflex_Class

