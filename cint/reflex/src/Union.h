// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_Union
#define Reflex_Union

// Include Files
#include "Reflex/Member.h"
#include "Reflex/internal/TypeBase.h"
#include "Reflex/internal/ScopeBase.h"

namespace Reflex
{

/**
 * @class Union Union.h Reflex/Union.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class Union : public TypeBase, public ScopeBase
{

public:

   /** constructor */
   Union(const char* typ, size_t size, const std::type_info& ti, unsigned int modifiers, TYPE unionType = UNION);

   /** destructor */
   virtual ~Union();

   /**
    * operator Scope will return the corresponding scope of this type if
    * applicable (i.e. if the Type is also a Scope e.g. Class, Union, Enum)
    */
   operator Scope() const;

   /**
    * operator Type will return the corresponding Type object
    * @return Type corresponding to this TypeBase
    */
   operator Type() const;

   /**
    * Hide this type from any lookup by appending the string " @HIDDEN@" to its name.
    */
   virtual void HideName() const;
   
   /**
    * Un-Hide this type from any lookup by removing the string " @HIDDEN@" to its name.
    */
   virtual void UnhideName() const;
      
   /**
    * DataMemberAt will return the nth data MemberAt of the At
    * @param  nth data MemberAt
    * @return pointer to data MemberAt
    */
   virtual Member DataMemberAt(size_t nth) const;

   /**
    * DataMemberByName will return the MemberAt with Name
    * @param  Name of data MemberAt
    * @return data MemberAt
    */
   virtual Member DataMemberByName(const std::string& nam) const;

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
   virtual Scope DeclaringScope() const;

   /**
   * nthFunctionMember will return the nth function MemberAt of the At
   * @param  nth function MemberAt
   * @return pointer to function MemberAt
   */
   virtual Member FunctionMemberAt(size_t nth) const; 

   /**
   * FunctionMemberByName will return the MemberAt with the Name,
   * optionally the signature of the function may be given
   * @param  Name of function MemberAt
   * @param  signature of the MemberAt function
   * @modifers_mask When matching, do not compare the listed modifiers
   * @return function MemberAt
   */
   virtual Member FunctionMemberByName(const std::string& nam, const Type& signature, unsigned int modifiers_mask = 0) const; 

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
    * MemberByName will return the first MemberAt with a given Name
    * @param  MemberAt Name
    * @return pointer to MemberAt
    */
   virtual Member MemberByName(const std::string& nam, const Type& signature) const;

   /**
    * MemberAt will return the nth MemberAt of the At
    * @param  nth MemberAt
    * @return pointer to nth MemberAt
    */
   virtual Member MemberAt(size_t nth) const;

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
   * Name will return the Name of the union
   * @return Name of union
   */
   virtual std::string Name(unsigned int mod = 0) const;

   /**
   * SimpleName returns the name of the type as a reference. It provides a
   * simplified but faster generation of a type name. Attention currently it
   * is not guaranteed that Name() and SimpleName() return the same character
   * layout of a name (ie. spacing, commas, etc. )
   * @param pos will indicate where in the returned reference the requested name starts
   * @param mod The only 'mod' support is SCOPED
   * @return name of type
   */
   virtual const std::string& SimpleName(size_t& pos, unsigned int mod = 0) const;

   /**
    * Properties will return a pointer to the PropertyNth list attached
    * to this item
    * @return pointer to PropertyNth list
    */
   virtual PropertyList Properties() const;

public:

   /**
    * AddDataMember will add the information about a data MemberAt
    * @param dm pointer to data MemberAt
    */
   virtual void AddDataMember(const Member& dm) const;
   virtual void AddDataMember(const char* nam, const Type& typ, size_t offs, unsigned int modifiers = 0) const;
   virtual void AddDataMember(Member &output, const char* nam, const Type& typ, size_t offs, unsigned int modifiers = 0) const;

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

public:

   /**
   * return the type name
   */
   TypeName* TypeNameGet() const;

private:

   /**
    * Modifiers of this union
    */
   unsigned int fModifiers;

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

}; // class Union

} // namespace Reflex

#include "Reflex/internal/OwnedMember.h"

#endif // Reflex_Union

