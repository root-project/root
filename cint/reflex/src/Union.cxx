// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
#define REFLEX_BUILD
#endif

#include "Union.h"

#include "Reflex/Tools.h"

//______________________________________________________________________________
Reflex::Union::Union(const char* typ, size_t size, const std::type_info& ti, unsigned int modifiers, TYPE unionType /*=UNION*/)
: TypeBase(typ, size, unionType, ti)
, ScopeBase(typ, unionType)
, fModifiers(modifiers)
, fCompleteType(false)
, fConstructors(std::vector<Member>())
, fDestructor(Member())
{
// Construct union info.
}

//______________________________________________________________________________
Reflex::Union::~Union()
{
// Destructor.
}

//______________________________________________________________________________
Reflex::Union::operator Reflex::Scope() const
{
   return ScopeBase::operator Scope();
}

//______________________________________________________________________________
Reflex::Union::operator Reflex::Type() const
{
   return TypeBase::operator Type();
}

//______________________________________________________________________________
void Reflex::Union::HideName() const
{
//  Hide the union from name lookup; forwards to TypeBase and ScopeBase.
   TypeBase::HideName();
   ScopeBase::HideName();
}

//______________________________________________________________________________
void Reflex::Union::UnhideName() const
{
   //  Hide the union from name lookup; forwards to TypeBase and ScopeBase.
   TypeBase::UnhideName();
   ScopeBase::UnhideName();
}

//______________________________________________________________________________
Reflex::Member Reflex::Union::DataMemberAt(size_t nth) const
{
// Return the nth datamember of the union
   return ScopeBase::DataMemberAt(nth);
}

//______________________________________________________________________________
Reflex::Member Reflex::Union::DataMemberByName(const std::string& nam) const
{
// Return the first data member of the union named nam.
   return ScopeBase::DataMemberByName(nam);
}

//______________________________________________________________________________
size_t Reflex::Union::DataMemberSize() const
{
// Return the number of data members.
   return ScopeBase::DataMemberSize();
}

//______________________________________________________________________________
Reflex::Member_Iterator Reflex::Union::DataMember_Begin() const
{
// Return an iterator to the first data member.
   return ScopeBase::DataMember_Begin();
}

//-------------------------------------------------------------------------------
Reflex::Member_Iterator Reflex::Union::DataMember_End() const
{
// Return an iterator pointing beyond the last data member.
   return ScopeBase::DataMember_End();
}

//-------------------------------------------------------------------------------
Reflex::Reverse_Member_Iterator Reflex::Union::DataMember_RBegin() const
{
// Return a reverse iterator to the last data member.
   return ScopeBase::DataMember_RBegin();
}

//______________________________________________________________________________
Reflex::Reverse_Member_Iterator Reflex::Union::DataMember_REnd() const
{
// Return a reverse iterator pointing beyond the first data member.
   return ScopeBase::DataMember_REnd();
}

//______________________________________________________________________________
Reflex::Scope Reflex::Union::DeclaringScope() const
{
// Return the scope the union is a member of.
   return ScopeBase::DeclaringScope();
}

//______________________________________________________________________________
Reflex::Member Reflex::Union::FunctionMemberAt(size_t nth) const
{
// Return the nth function member of the union.
   return ScopeBase::FunctionMemberAt(nth);
}

//______________________________________________________________________________
Reflex::Member Reflex::Union::FunctionMemberByName(const std::string& nam, const Type& signature, unsigned int modifiers_mask) const
{
// Return the first function member named nam, with a given function signature,
// where mask determines which modifiers (see ENTITY_DESCRIPTION) to ignore in
// the signature matching.
   return ScopeBase::FunctionMemberByName(nam, signature, modifiers_mask);
}

//______________________________________________________________________________
size_t Reflex::Union::FunctionMemberSize() const
{
// Return the number of function members.
   return ScopeBase::FunctionMemberSize();
}

//______________________________________________________________________________
inline Reflex::Member_Iterator Reflex::Union::FunctionMember_Begin() const
{
// Return an iterator pointing to the first function member.
   return ScopeBase::FunctionMember_Begin();
}

//______________________________________________________________________________
Reflex::Member_Iterator Reflex::Union::FunctionMember_End() const
{
// Return an iterator pointing beyond the last function member.
   return ScopeBase::FunctionMember_End();
}

//______________________________________________________________________________
Reflex::Reverse_Member_Iterator Reflex::Union::FunctionMember_RBegin() const
{
// Return a reverse iterator pointing to the last function member.
   return ScopeBase::FunctionMember_RBegin();
}

//______________________________________________________________________________
Reflex::Reverse_Member_Iterator Reflex::Union::FunctionMember_REnd() const
{
// Return a reverse iterator pointing beyond the first function member.
   return ScopeBase::FunctionMember_REnd();
}

//______________________________________________________________________________
bool Reflex::Union::IsComplete() const
{
   // Return true if this union is complete. I.e. all dictionary information for all
   // data and function member types is available.
   if (!fCompleteType) {
      fCompleteType = true;
   }
   return fCompleteType;
}

//______________________________________________________________________________
bool Reflex::Union::IsPrivate() const
{
// Return whether the union has private access.
   return fModifiers & PRIVATE;
}

//______________________________________________________________________________
bool Reflex::Union::IsProtected() const
{
// Return whether the union has protected access.
   return fModifiers & PROTECTED;
}

//______________________________________________________________________________
bool Reflex::Union::IsPublic() const
{
// Return whether the union has public access.
   return fModifiers & PUBLIC;
}

//______________________________________________________________________________
Reflex::Member Reflex::Union::MemberByName(const std::string& nam, const Type& signature) const
{
// Return the first member matching nam with a given signature.
   return ScopeBase::MemberByName(nam, signature);
}

//______________________________________________________________________________
Reflex::Member Reflex::Union::MemberAt(size_t nth) const
{
// Return nth member of this union.
   return ScopeBase::MemberAt(nth);
}

//______________________________________________________________________________
size_t Reflex::Union::MemberSize() const
{
// Return the number of members.
   return ScopeBase::MemberSize();
}

//______________________________________________________________________________
Reflex::Member_Iterator Reflex::Union::Member_Begin() const
{
// Return an iterator pointing to the first member.
   return ScopeBase::Member_Begin();
}

//______________________________________________________________________________
Reflex::Member_Iterator Reflex::Union::Member_End() const
{
// Return an iterator pointing beyond the last member.
   return ScopeBase::Member_End();
}

//______________________________________________________________________________
Reflex::Reverse_Member_Iterator Reflex::Union::Member_RBegin() const
{
// Return a reverse iterator pointing to the last member.
   return ScopeBase::Member_RBegin();
}

//______________________________________________________________________________
Reflex::Reverse_Member_Iterator Reflex::Union::Member_REnd() const
{
// Return a reverse iterator pointing beyond the first member.
   return ScopeBase::Member_REnd();
}

//______________________________________________________________________________
std::string Reflex::Union::Name(unsigned int mod /*= 0*/) const
{
// Return the name of the union; possible modifiers:
//      *   FINAL     - resolve typedefs
//      *   SCOPED    - fully scoped name 
//      *   QUALIFIED - cv, reference qualification 
   return ScopeBase::Name(mod);
}

//______________________________________________________________________________
const std::string& Reflex::Union::SimpleName(size_t& pos, unsigned int mod /*=0*/) const
{
// Return the "simple" name of the union (only the left-most part of the scoped name)
// possible modifiers:
//      *   FINAL     - resolve typedefs
//      *   SCOPED    - fully scoped name 
//      *   QUALIFIED - cv, reference qualification 
// See ScopeBase::SimpleName().
   return ScopeBase::SimpleName(pos, mod);
}

//______________________________________________________________________________
Reflex::PropertyList Reflex::Union::Properties() const
{
// Return the union's list of properties.
   return ScopeBase::Properties();
}

//______________________________________________________________________________
void Reflex::Union::AddDataMember(const Member& dm) const
{
// Add a data memebr to the union.
   ScopeBase::AddDataMember(dm);
}

//______________________________________________________________________________
void Reflex::Union::AddDataMember(const char* nam, const Type& typ, size_t offs, unsigned int modifiers /*= 0*/) const
{
// Add a data memebr to the union.
   ScopeBase::AddDataMember(nam, typ, offs, modifiers);
}

//______________________________________________________________________________
void Reflex::Union::AddDataMember(Member &output, const char* nam, const Type& typ, size_t offs, unsigned int modifiers /*= 0*/) const
{
   // Add a data memebr to the union.
   ScopeBase::AddDataMember(output, nam, typ, offs, modifiers);
}

//______________________________________________________________________________
void Reflex::Union::AddFunctionMember(const Member & fm) const
{
// Add function member fm to this union
   ScopeBase::AddFunctionMember(fm);
   if (fm.IsConstructor()) {
      fConstructors.push_back(fm);
   }
   else if (fm.IsDestructor()) {
      fDestructor = fm;
   }
}

//______________________________________________________________________________
void Reflex::Union::AddFunctionMember(const char* nam, const Type& typ, StubFunction stubFP, void* stubCtx, const char* params, unsigned int modifiers) const
{
// Add function member to this union.
   ScopeBase::AddFunctionMember(nam, typ, stubFP, stubCtx, params, modifiers);
   if (modifiers & CONSTRUCTOR) {
      fConstructors.push_back(fFunctionMembers[fFunctionMembers.size()-1]);
   }
   // setting the destructor is not needed because it is always provided when building the union
}

//______________________________________________________________________________
void Reflex::Union::RemoveDataMember(const Member& dm) const
{
// Remove data member dm from this union.
   ScopeBase::RemoveDataMember(dm);
}

//______________________________________________________________________________
void Reflex::Union::RemoveFunctionMember(const Member& fm) const
{
// Remove function member from this union.
   ScopeBase::RemoveFunctionMember(fm);
}

//______________________________________________________________________________
inline Reflex::TypeName* Reflex::Union::TypeNameGet() const
{
// Return the TypeName* of this union.
   return fTypeName;
}

