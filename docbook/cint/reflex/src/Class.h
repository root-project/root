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
class Class: public ScopedType {
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
   virtual Object CastObject(const Type& to,
                             const Object& obj) const;

   /**
    * Construct will call the constructor of a given At and Allocate the
    * memory for it
    * @param  signature of the constructor
    * @param  values for parameters of the constructor
    * @param  mem place in memory for implicit construction
    * @return pointer to new instance
    */
   //virtual Object Construct(const Type& signature, const std::vector<Object>& values, void* mem = 0) const;
   virtual Object Construct(const Type& signature = Type(),
                            const std::vector<void*>& values = std::vector<void*>(),
                            void* mem = 0) const;


   /**
    * nthDataMember will return the nth data MemberAt of the At
    * @param  nth data MemberAt
    * @return pointer to data MemberAt
    */
   Member DataMemberAt(size_t nth,
                       EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * DataMemberByName will return the MemberAt with Name
    * @param  Name of data MemberAt
    * @return data MemberAt
    */
   Member DataMemberByName(const std::string& nam,
                           EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * DataMemberSize will return the number of data members of this At
    * @return number of data members
    */
   size_t DataMemberSize(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   Member_Iterator DataMember_Begin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   Member_Iterator DataMember_End(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   Reverse_Member_Iterator DataMember_RBegin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   Reverse_Member_Iterator DataMember_REnd(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;

   /**
    * nthFunctionMember will return the nth function MemberAt of the At
    * @param  nth function MemberAt
    * @return pointer to function MemberAt
    */
   Member FunctionMemberAt(size_t nth,
                           EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * FunctionMemberByName will return the MemberAt with the Name,
    * optionally the signature of the function may be given
    * @param  Name of function MemberAt
    * @param  signature of the MemberAt function
    * @modifiers_mask When matching, do not compare the listed modifiers
    * @return function MemberAt
    */
   Member FunctionMemberByName(const std::string& name,
                               const Type& signature,
                               unsigned int modifiers_mask = 0,
                               EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT,
                               EDELAYEDLOADSETTING allowDelayedLoad = DELAYEDLOAD_ON) const;


   /**
    * FunctionMemberByNameAndSignature will return the MemberAt with the Name,
    * optionally the signature of the function may be given
    * @param  Name of function MemberAt
    * @param  signature of the MemberAt function
    * @modifiers_mask When matching, do not compare the listed modifiers
    * @return function MemberAt
    */
   Member FunctionMemberByNameAndSignature(const std::string& name,
                                           const Type& signature,
                                           unsigned int modifiers_mask = 0,
                                           EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT,
                                           EDELAYEDLOADSETTING allowDelayedLoad = DELAYEDLOAD_ON) const;


   /**
    * FunctionMemberSize will return the number of function members of
    * this type
    * @return number of function members
    */
   size_t FunctionMemberSize(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   Member_Iterator FunctionMember_Begin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   Member_Iterator FunctionMember_End(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   Reverse_Member_Iterator FunctionMember_RBegin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   Reverse_Member_Iterator FunctionMember_REnd(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


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
   virtual void Destruct(void* instance,
                         bool dealloc = true) const;

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
   bool HasBase(const Type& cl,
                std::vector<Base>& path) const;

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
    * MemberByName will return the first MemberAt with a given Name
    * @param Name  MemberAt Name
    * @return pointer to MemberAt
    */
   Member MemberByName(const std::string& name,
                       const Type& signature,
                       EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * MemberAt will return the nth MemberAt of the At
    * @param  nth MemberAt
    * @return pointer to nth MemberAt
    */
   Member MemberAt(size_t nth,
                   EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   Member_Iterator Member_Begin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   Member_Iterator Member_End(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   Reverse_Member_Iterator Member_RBegin(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;
   Reverse_Member_Iterator Member_REnd(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * MemberSize will return the number of members
    * @return number of members
    */
   size_t MemberSize(EMEMBERQUERY inh = INHERITEDMEMBERS_DEFAULT) const;


   /**
    * PathToBase will return a vector of function pointers to the base class
    * ( !!! Attention !!! the most derived class comes first )
    * @param base the scope to look for
    * @return vector of function pointers to calculate base offset
    */
   const std::vector<OffsetFunction>& PathToBase(const Scope& bas) const;


   /**
    * AddBase will add the information about a BaseAt class
    * @param  BaseAt At of the BaseAt class
    * @param  OffsetFP the pointer to the stub function for calculating the Offset
    * @param  modifiers the modifiers of the BaseAt class
    * @return this
    */
   virtual void AddBase(const Type& bas,
                        OffsetFunction offsFP,
                        unsigned int modifiers = 0) const;

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
   virtual Member AddDataMember(const char* nam,
                                const Type& typ,
                                size_t offs,
                                unsigned int modifiers = 0,
                                char* interpreterOffset = 0) const;

   /**
    * AddFunctionMember will add the information about a function MemberAt
    * @param fm pointer to function MemberAt
    */
   virtual void AddFunctionMember(const Member& fm) const;
   virtual Member AddFunctionMember(const char* nam,
                                    const Type& typ,
                                    StubFunction stubFP,
                                    void* stubCtx = 0,
                                    const char* params = 0,
                                    unsigned int modifiers = 0) const;

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

   /** Initialize the vector of inherited members.
    *  Returns false if one of the bases is not complete.
    */
   virtual bool UpdateMembers() const;

private:
   /** map with the class as a key and the path to it as the value
      the key (void*) is a pointer to the unique ScopeName */
   typedef std::vector<OffsetFunction> BasePath_t;
   typedef std::map<void*, std::vector<OffsetFunction>*> PathsToBase;

   struct InheritedMembersInfo_t {
      InheritedMembersInfo_t(size_t ndata, size_t nfunc):
         fDataMembers(ndata),
         fFunctionMembers(nfunc),
         fMembers(ndata + nfunc) {
         // Resize the members, so the c'tor allocates the right amount directly.
         // Then clear, because vector does not have a "reserve only" c'tor.
         fDataMembers.clear();
         fFunctionMembers.clear();
         fMembers.clear();
         // The reserve again, just to be sure - ideally this is a no-op.
         fDataMembers.reserve(ndata);
         fFunctionMembers.reserve(nfunc);
         fMembers.reserve(ndata + nfunc);
      }


      std::vector<Member> fDataMembers;
      std::vector<Member> fFunctionMembers;
      std::vector<Member> fMembers;
   };


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

   mutable InheritedMembersInfo_t * fInherited;

}; // class Class
} //namespace Reflex

#include "Reflex/Base.h"


//-------------------------------------------------------------------------------
inline void
Reflex::Class::AddBase(const Base& b) const {
//-------------------------------------------------------------------------------
   fBases.push_back(b);
}


//-------------------------------------------------------------------------------
inline Reflex::Base
Reflex::Class::BaseAt(size_t nth) const {
//-------------------------------------------------------------------------------
   if (nth < fBases.size()) {
      return fBases[nth];
   }
   return Dummy::Base();
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Class::BaseSize() const {
//-------------------------------------------------------------------------------
   return fBases.size();
}


//-------------------------------------------------------------------------------
inline Reflex::Base_Iterator
Reflex::Class::Base_Begin() const {
//-------------------------------------------------------------------------------
   return fBases.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Base_Iterator
Reflex::Class::Base_End() const {
//-------------------------------------------------------------------------------
   return fBases.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Base_Iterator
Reflex::Class::Base_RBegin() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Base> &)fBases).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Base_Iterator
Reflex::Class::Base_REnd() const {
//-------------------------------------------------------------------------------
   return ((const std::vector<Base> &)fBases).rend();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Class::DataMember_Begin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   ExecuteDataMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return fInherited->fDataMembers.begin();
      } else { return Dummy::MemberCont().begin(); }
   }
   return fDataMembers.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Class::DataMember_End(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   ExecuteDataMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return fInherited->fDataMembers.end();
      } else { return Dummy::MemberCont().end(); }
   }
   return fDataMembers.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Class::DataMember_RBegin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   ExecuteDataMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return const_cast<const std::vector<Member>&>(fInherited->fDataMembers).rbegin();
      } else { return Dummy::MemberCont().rbegin(); }
   }
   return ((const std::vector<Member> &)fDataMembers).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Class::DataMember_REnd(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   ExecuteDataMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return const_cast<const std::vector<Member>&>(fInherited->fDataMembers).rend();
      } else { return Dummy::MemberCont().rend(); }
   }
   return ((const std::vector<Member> &)fDataMembers).rend();
}


//-------------------------------------------------------------------------------
inline Reflex::Member
Reflex::Class::DataMemberAt(size_t nth,
                            EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return nth data member info.
   ExecuteDataMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers() && nth < fInherited->fDataMembers.size()) {
         return fInherited->fDataMembers[nth];
      } else { return Dummy::Member(); }
   }

   if (nth < fDataMembers.size()) {
      return fDataMembers[nth];
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
inline Reflex::Member
Reflex::Class::DataMemberByName(const std::string& nam,
                                EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return function member by name and signature including the return type.
   ExecuteDataMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return MemberByName2(fInherited->fDataMembers, nam);
      } else { return Dummy::Member(); }
   }
   return MemberByName2(fDataMembers, nam);
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Class::DataMemberSize(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return number of data members.
   ExecuteDataMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return fInherited->fDataMembers.size();
      } else { return 0; }
   }
   return fDataMembers.size();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Class::FunctionMember_Begin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return fInherited->fFunctionMembers.begin();
      } else { return Dummy::MemberCont().begin(); }
   }
   return fFunctionMembers.begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Class::FunctionMember_End(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return fInherited->fFunctionMembers.end();
      } else { return Dummy::MemberCont().end(); }
   }
   return fFunctionMembers.end();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Class::FunctionMember_RBegin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return const_cast<const std::vector<Member>&>(fInherited->fFunctionMembers).rbegin();
      } else { return Dummy::MemberCont().rbegin(); }
   }
   return ((const std::vector<Member> &)fFunctionMembers).rbegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Class::FunctionMember_REnd(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return const_cast<const std::vector<Member>&>(fInherited->fFunctionMembers).rend();
      } else { return Dummy::MemberCont().rend(); }
   }
   return ((const std::vector<Member> &)fFunctionMembers).rend();
}


//-------------------------------------------------------------------------------
inline Reflex::Member
Reflex::Class::FunctionMemberAt(size_t nth,
                                EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return nth data member info.
   ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers() && nth < fInherited->fFunctionMembers.size()) {
         return fInherited->fFunctionMembers[nth];
      } else { return Dummy::Member(); }
   }

   if (nth < fFunctionMembers.size()) {
      return fFunctionMembers[nth];
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
inline Reflex::Member
Reflex::Class::FunctionMemberByName(const std::string& nam,
                                    const Type& signature,
                                    unsigned int modifiers_mask,
                                    EMEMBERQUERY inh,
                                    EDELAYEDLOADSETTING allowDelayedLoad) const {
//-------------------------------------------------------------------------------
// Return function member by name and signature including the return type.
   if (allowDelayedLoad == DELAYEDLOAD_ON)
      ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return MemberByName2(fInherited->fFunctionMembers, nam, &signature, modifiers_mask);
      } else { return Dummy::Member(); }
   }
   return MemberByName2(fFunctionMembers, nam, &signature, modifiers_mask);
}


//-------------------------------------------------------------------------------
inline Reflex::Member
Reflex::Class::FunctionMemberByNameAndSignature(const std::string& nam,
                                                const Type& signature,
                                                unsigned int modifiers_mask,
                                                EMEMBERQUERY inh,
                                                EDELAYEDLOADSETTING allowDelayedLoad) const {
//-------------------------------------------------------------------------------
// Return function member by name and signature excluding the return type.
   if (allowDelayedLoad == DELAYEDLOAD_ON)
      ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return MemberByName2(fInherited->fFunctionMembers, nam, &signature, modifiers_mask, false);
      } else { return Dummy::Member(); }
   }
   return MemberByName2(fFunctionMembers, nam, &signature, modifiers_mask, false);
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Class::FunctionMemberSize(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return number of data members.
   ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return fInherited->fFunctionMembers.size();
      } else { return 0; }
   }
   return fFunctionMembers.size();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Class::Member_Begin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   ExecuteDataMemberDelayLoad();
   ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return fInherited->fMembers.begin();
      } else { return Dummy::MemberCont().begin(); }
   }
   return ScopeBase::Member_Begin(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator
Reflex::Class::Member_End(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   ExecuteDataMemberDelayLoad();
   ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return fInherited->fMembers.end();
      } else { return Dummy::MemberCont().end(); }
   }
   return ScopeBase::Member_End(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Class::Member_RBegin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   ExecuteDataMemberDelayLoad();
   ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return const_cast<const std::vector<Member>&>(fInherited->fMembers).rbegin();
      } else { return Dummy::MemberCont().rbegin(); }
   }
   return ScopeBase::Member_RBegin(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator
Reflex::Class::Member_REnd(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
   ExecuteDataMemberDelayLoad();
   ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return const_cast<const std::vector<Member>&>(fInherited->fMembers).rend();
      } else { return Dummy::MemberCont().rend(); }
   }
   return ScopeBase::Member_REnd(inh);
}


//-------------------------------------------------------------------------------
inline Reflex::Member
Reflex::Class::MemberAt(size_t nth,
                        EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return nth data member info.
   ExecuteDataMemberDelayLoad();
   ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers() && nth < fInherited->fMembers.size()) {
         return fInherited->fMembers[nth];
      } else { return Dummy::Member(); }
   }

   if (nth < fMembers.size()) {
      return fMembers[nth];
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
inline Reflex::Member
Reflex::Class::MemberByName(const std::string& nam,
                            const Type& signature,
                            EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return function member by name and signature including the return type.
   ExecuteDataMemberDelayLoad();
   ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return MemberByName2(fInherited->fMembers, nam, &signature);
      } else { return Dummy::Member(); }
   }
   return ScopeBase::MemberByName(nam, signature, inh);
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::Class::MemberSize(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return number of data members.
   ExecuteDataMemberDelayLoad();
   ExecuteFunctionMemberDelayLoad();
   if (inh == INHERITEDMEMBERS_ALSO || (inh == INHERITEDMEMBERS_DEFAULT && fInherited)) {
      if (Class::UpdateMembers()) {
         return fInherited->fMembers.size();
      } else { return 0; }
   }
   return fMembers.size();
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Class::IsAbstract() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & ABSTRACT);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Class::IsVirtual() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & VIRTUAL);
}


#endif // Reflex_Class
