// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
# define REFLEX_BUILD
#endif

#include "Reflex/Scope.h"

#include "Reflex/internal/ScopeBase.h"
#include "Reflex/Member.h"
#include "Reflex/Type.h"
#include "Reflex/TypeTemplate.h"
#include "Reflex/MemberTemplate.h"
#include "Reflex/Base.h"
#include "Reflex/Builder/TypeBuilder.h"

#include "Reflex/Tools.h"
#include "Class.h"

//-------------------------------------------------------------------------------
Reflex::Scope&
Reflex::Scope::__NIRVANA__() {
//-------------------------------------------------------------------------------
// static wraper around NIRVANA, the base of the top scope.
   static Scope s = Scope(new ScopeName(Literal("@N@I@R@V@A@N@A@"), 0));
   return s;
}


//-------------------------------------------------------------------------------
Reflex::Scope::operator
Reflex::Type() const {
//-------------------------------------------------------------------------------
// Conversion operator to Type. If this scope is not a Type, returns the empty type.
   if (*this) {
      return *(fScopeName->fScopeBase);
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
Reflex::Base
Reflex::Scope::BaseAt(size_t nth) const {
//-------------------------------------------------------------------------------
// Return nth base class info.
   if (*this) {
      return fScopeName->fScopeBase->BaseAt(nth);
   }
   return Dummy::Base();
}


//-------------------------------------------------------------------------------
size_t
Reflex::Scope::BaseSize() const {
//-------------------------------------------------------------------------------
// Return number of base classes.
   if (*this) {
      return fScopeName->fScopeBase->BaseSize();
   }
   return 0;
}


//-------------------------------------------------------------------------------
Reflex::Scope
Reflex::Scope::ByName(const std::string& name) {
//-------------------------------------------------------------------------------
// Lookup a Scope by it's fully qualified name.
   return ScopeName::ByName(name);
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::Scope::DataMemberAt(size_t nth,
                            EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return the nth data member of this scope.
   if (*this) {
      return fScopeName->fScopeBase->DataMemberAt(nth, inh);
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::Scope::DataMemberByName(const std::string& name,
                                EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return a data member by it's name.
   if (*this) {
      return fScopeName->fScopeBase->DataMemberByName(name, inh);
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
size_t
Reflex::Scope::DataMemberSize(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return number of data mebers of this scope.
   if (*this) {
      return fScopeName->fScopeBase->DataMemberSize(inh);
   }
   return 0;
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::Scope::FunctionMemberAt(size_t nth,
                                EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return nth function member of this socpe.
   if (*this) {
      return fScopeName->fScopeBase->FunctionMemberAt(nth, inh);
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::Scope::FunctionMemberByName(const std::string& name,
                                    EMEMBERQUERY inh,
                                    EDELAYEDLOADSETTING allowDelayedLoad) const {
//-------------------------------------------------------------------------------
// Return a function member by it's name.
   if (*this) {
      return fScopeName->fScopeBase->FunctionMemberByName(name, Type(), 0, inh, allowDelayedLoad);
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::Scope::FunctionMemberByName(const std::string& name,
                                    const Type& signature,
                                    unsigned int modifiers_mask,
                                    EMEMBERQUERY inh,
                                    EDELAYEDLOADSETTING allowDelayedLoad) const {
//-------------------------------------------------------------------------------
// Return a function member by it's name, qualified by it's signature type.
   if (*this) {
      return fScopeName->fScopeBase->FunctionMemberByName(name, signature, modifiers_mask, inh, allowDelayedLoad);
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::Scope::FunctionMemberByNameAndSignature(const std::string& name,
                                                const Type& signature,
                                                unsigned int modifiers_mask,
                                                EMEMBERQUERY inh,
                                                EDELAYEDLOADSETTING allowDelayedLoad) const {
//-------------------------------------------------------------------------------
// Return a function member by it's name, qualified by it's signature type.
   if (*this) {
      return fScopeName->fScopeBase->FunctionMemberByNameAndSignature(name, signature, modifiers_mask, inh, allowDelayedLoad);
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
size_t
Reflex::Scope::FunctionMemberSize(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return number of function members of this scope.
   if (*this) {
      return fScopeName->fScopeBase->FunctionMemberSize(inh);
   }
   return 0;
}


//-------------------------------------------------------------------------------
bool
Reflex::Scope::HasBase(const Type& cl) const {
//-------------------------------------------------------------------------------
// Return base info if type has base cl.
   if (*this) {
      return fScopeName->fScopeBase->HasBase(cl);
   }
   return false;
}


//-------------------------------------------------------------------------------
bool
Reflex::Scope::IsPrivate() const {
//-------------------------------------------------------------------------------
// True if the the type's access is private.
   return operator Type().IsPrivate();
}


//-------------------------------------------------------------------------------
bool
Reflex::Scope::IsProtected() const {
//-------------------------------------------------------------------------------
// True if the the type's access is protected.
   return operator Type().IsProtected();
}


//-------------------------------------------------------------------------------
bool
Reflex::Scope::IsPublic() const {
//-------------------------------------------------------------------------------
// True if the the type is publicly accessible.
   return operator Type().IsPublic();
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::Scope::LookupMember(const std::string& nam) const {
//-------------------------------------------------------------------------------
// Lookup a member from this scope.
   if (*this) {
      return fScopeName->fScopeBase->LookupMember(nam, *this);
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::Scope::LookupType(const std::string& nam) const {
//-------------------------------------------------------------------------------
// Lookup a type from this scope.
   if (*this) {
      return fScopeName->fScopeBase->LookupType(nam, *this);
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
Reflex::Scope
Reflex::Scope::LookupScope(const std::string& nam) const {
//-------------------------------------------------------------------------------
// Lookup a scope from this scope.
   if (*this) {
      return fScopeName->fScopeBase->LookupScope(nam, *this);
   }
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::Scope::MemberByName(const std::string& name,
                            EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return a member from this scope, by name.
   if (*this) {
      return fScopeName->fScopeBase->MemberByName(name, Type(), inh);
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::Scope::MemberByName(const std::string& name,
                            const Type& signature,
                            EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return a member in this scope, looked up by name and signature (for functions)
   if (*this) {
      return fScopeName->fScopeBase->MemberByName(name, signature, inh);
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
Reflex::Member_Iterator
Reflex::Scope::Member_Begin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return the begin iterator of member container.
   if (*this) {
      return fScopeName->fScopeBase->Member_Begin(inh);
   }
   return Dummy::MemberCont().begin();
}


//-------------------------------------------------------------------------------
Reflex::Member_Iterator
Reflex::Scope::Member_End(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return the end iterator of member container.
   if (*this) {
      return fScopeName->fScopeBase->Member_End(inh);
   }
   return Dummy::MemberCont().end();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_Member_Iterator
Reflex::Scope::Member_RBegin(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return the rbegin iterator of member container.
   if (*this) {
      return fScopeName->fScopeBase->Member_RBegin(inh);
   }
   return Dummy::MemberCont().rbegin();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_Member_Iterator
Reflex::Scope::Member_REnd(EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return the rend iterator of member container.
   if (*this) {
      return fScopeName->fScopeBase->Member_REnd(inh);
   }
   return Dummy::MemberCont().rend();
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::Scope::MemberAt(size_t nth,
                        EMEMBERQUERY inh) const {
//-------------------------------------------------------------------------------
// Return the nth member of this scope.
   if (*this) {
      return fScopeName->fScopeBase->MemberAt(nth, inh);
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
Reflex::MemberTemplate
Reflex::Scope::MemberTemplateAt(size_t nth) const {
//-------------------------------------------------------------------------------
// Return the nth memer template in this scope.
   if (*this) {
      return fScopeName->fScopeBase->MemberTemplateAt(nth);
   }
   return Dummy::MemberTemplate();
}


//-------------------------------------------------------------------------------
size_t
Reflex::Scope::MemberTemplateSize() const {
//-------------------------------------------------------------------------------
// Return the number of member templates in this scope.
   if (*this) {
      return fScopeName->fScopeBase->MemberTemplateSize();
   }
   return 0;
}


//-------------------------------------------------------------------------------
Reflex::MemberTemplate
Reflex::Scope::MemberTemplateByName(const std::string& nam) const {
//-------------------------------------------------------------------------------
// Look up a member template in this scope by name and return it.
   if (*this) {
      return fScopeName->fScopeBase->MemberTemplateByName(nam);
   }
   return Dummy::MemberTemplate();
}


//-------------------------------------------------------------------------------
std::string
Reflex::Scope::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
// Return the name of this scope, scoped if requested.
   if (*this) {
      return fScopeName->fScopeBase->Name(mod);
   } else if (fScopeName) {
      if (0 != (mod & (SCOPED | S))) {
         return fScopeName->Name();
      } else { return Tools::GetBaseName(fScopeName->Name()); }
   } else {
      return "";
   }
}


//-------------------------------------------------------------------------------
Reflex::Scope
Reflex::Scope::ScopeAt(size_t nth) {
//-------------------------------------------------------------------------------
// Return the nth scope in the Reflex database.
   return ScopeName::ScopeAt(nth);
}


//-------------------------------------------------------------------------------
size_t
Reflex::Scope::ScopeSize() {
//-------------------------------------------------------------------------------
// Return the number of scopes defined.
   return ScopeName::ScopeSize();
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::Scope::SubTypeAt(size_t nth) const {
//-------------------------------------------------------------------------------
// Return the nth sub type of this scope.
   if (*this) {
      return fScopeName->fScopeBase->SubTypeAt(nth);
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
size_t
Reflex::Scope::SubTypeSize() const {
//-------------------------------------------------------------------------------
// Return the number of sub types.
   if (*this) {
      return fScopeName->fScopeBase->SubTypeSize();
   }
   return 0;
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::Scope::SubTypeByName(const std::string& nam) const {
//-------------------------------------------------------------------------------
// Look up a sub type by name and return it.
   if (*this) {
      return fScopeName->fScopeBase->SubTypeByName(nam);
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::Scope::TemplateArgumentAt(size_t nth) const {
//-------------------------------------------------------------------------------
// Return the nth template argument of this scope (ie. class).
   return operator Type().TemplateArgumentAt(nth);
}


//-------------------------------------------------------------------------------
Reflex::TypeTemplate
Reflex::Scope::TemplateFamily() const {
//-------------------------------------------------------------------------------
// Return the template family related to this scope.
   return operator Type().TemplateFamily();
}


//-------------------------------------------------------------------------------
Reflex::TypeTemplate
Reflex::Scope::SubTypeTemplateAt(size_t nth) const {
//-------------------------------------------------------------------------------
// Return the nth sub type template.
   if (*this) {
      return fScopeName->fScopeBase->SubTypeTemplateAt(nth);
   }
   return Dummy::TypeTemplate();
}


//-------------------------------------------------------------------------------
size_t
Reflex::Scope::SubTypeTemplateSize() const {
//-------------------------------------------------------------------------------
// Return the number of type templates in this scope.
   if (*this) {
      return fScopeName->fScopeBase->SubTypeTemplateSize();
   }
   return 0;
}


//-------------------------------------------------------------------------------
Reflex::TypeTemplate
Reflex::Scope::SubTypeTemplateByName(const std::string& nam) const {
//-------------------------------------------------------------------------------
// Lookup a sub type template by string and return it.
   if (*this) {
      return fScopeName->fScopeBase->SubTypeTemplateByName(nam);
   }
   return Dummy::TypeTemplate();
}


//-------------------------------------------------------------------------------
void
Reflex::Scope::AddDataMember(const Member& dm) const {
//-------------------------------------------------------------------------------
// Add data member dm to this scope.
   if (*this) {
      fScopeName->fScopeBase->AddDataMember(dm);
   }
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::Scope::AddDataMember(const char* name,
                             const Type& type,
                             size_t offset,
                             unsigned int modifiers /* = 0 */,
                             char* interpreterOffset /* = 0 */) const {
//-------------------------------------------------------------------------------
// Add data member to this scope.
   if (*this) {
      return fScopeName->fScopeBase->AddDataMember(name, type, offset, modifiers, interpreterOffset);
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
void
Reflex::Scope::RemoveDataMember(const Member& dm) const {
//-------------------------------------------------------------------------------
// Remove data member dm from this scope.
   if (*this) {
      fScopeName->fScopeBase->RemoveDataMember(dm);
   }
}


//-------------------------------------------------------------------------------
void
Reflex::Scope::AddFunctionMember(const Member& fm) const {
//-------------------------------------------------------------------------------
// Add function member fm to this scope.
   if (*this) {
      fScopeName->fScopeBase->AddFunctionMember(fm);
   }
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::Scope::AddFunctionMember(const char* nam,
                                 const Type& typ,
                                 StubFunction stubFP,
                                 void* stubCtx,
                                 const char* params,
                                 unsigned int modifiers) const {
//-------------------------------------------------------------------------------
// Add function member to this scope.
   if (*this) {
      return fScopeName->fScopeBase->AddFunctionMember(nam, typ, stubFP, stubCtx, params, modifiers);
   }
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
void
Reflex::Scope::RemoveFunctionMember(const Member& fm) const {
//-------------------------------------------------------------------------------
// Remove function member fm from this scope.
   if (*this) {
      fScopeName->fScopeBase->RemoveFunctionMember(fm);
   }
}


//-------------------------------------------------------------------------------
void
Reflex::Scope::AddSubType(const Type& ty) const {
//-------------------------------------------------------------------------------
// Add sub type ty to this scope.
   if (*this) {
      fScopeName->fScopeBase->AddSubType(ty);
   }
}


//-------------------------------------------------------------------------------
void
Reflex::Scope::AddSubType(const char* type,
                          size_t size,
                          TYPE typeType,
                          const std::type_info& typeInfo,
                          unsigned int modifiers) const {
//-------------------------------------------------------------------------------
// Add sub type to this scope.
   if (*this) {
      fScopeName->fScopeBase->AddSubType(type,
                                         size,
                                         typeType,
                                         typeInfo,
                                         modifiers);
   }
}


//-------------------------------------------------------------------------------
void
Reflex::Scope::RemoveSubType(const Type& ty) const {
//-------------------------------------------------------------------------------
// Remove sub type ty from this scope.
   if (*this) {
      fScopeName->fScopeBase->RemoveSubType(ty);
   }
}


//-------------------------------------------------------------------------------
void
Reflex::Scope::AddMemberTemplate(const MemberTemplate& mt) const {
//-------------------------------------------------------------------------------
// Add member template mt to this scope.
   if (*this) {
      fScopeName->fScopeBase->AddMemberTemplate(mt);
   }
}


//-------------------------------------------------------------------------------
void
Reflex::Scope::RemoveMemberTemplate(const MemberTemplate& mt) const {
//-------------------------------------------------------------------------------
// Remove member template mt from this scope.
   if (*this) {
      fScopeName->fScopeBase->RemoveMemberTemplate(mt);
   }
}


//-------------------------------------------------------------------------------
void
Reflex::Scope::AddSubTypeTemplate(const TypeTemplate& tt) const {
//-------------------------------------------------------------------------------
// Add type template tt to this scope.
   if (*this) {
      fScopeName->fScopeBase->AddSubTypeTemplate(tt);
   }
}


//-------------------------------------------------------------------------------
void
Reflex::Scope::RemoveSubTypeTemplate(const TypeTemplate& tt) const {
//-------------------------------------------------------------------------------
// Remove type template tt from this scope.
   if (*this) {
      fScopeName->fScopeBase->RemoveSubTypeTemplate(tt);
   }
}


//-------------------------------------------------------------------------------
size_t
Reflex::Scope::TemplateArgumentSize() const {
//-------------------------------------------------------------------------------
// Return the number of template arguments.
   return operator Type().TemplateArgumentSize();
}


//-------------------------------------------------------------------------------
Reflex::Type_Iterator
Reflex::Scope::TemplateArgument_Begin() const {
//-------------------------------------------------------------------------------
// Return the 'begin' of the iterator on the template arguments.
   return operator Type().TemplateArgument_Begin();
}


//-------------------------------------------------------------------------------
Reflex::Type_Iterator
Reflex::Scope::TemplateArgument_End() const {
//-------------------------------------------------------------------------------
// Return the 'end' of the iterator on the template arguments.
   return operator Type().TemplateArgument_End();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_Type_Iterator
Reflex::Scope::TemplateArgument_RBegin() const {
//-------------------------------------------------------------------------------
// Return the 'begin' of the reverse iterator on the template arguments.
   return operator Type().TemplateArgument_RBegin();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_Type_Iterator
Reflex::Scope::TemplateArgument_REnd() const {
//-------------------------------------------------------------------------------
// Return the 'end' of the reverse iterator on the template arguments.
   return operator Type().TemplateArgument_REnd();
}


//-------------------------------------------------------------------------------
void
Reflex::Scope::GenerateDict(DictionaryGenerator& generator) const {
//-------------------------------------------------------------------------------
// Generate Dictionary information about itself.
   if (*this) {
      fScopeName->fScopeBase->GenerateDict(generator);
   }
}


//-------------------------------------------------------------------------------
void
Reflex::Scope::Unload() const {
//-------------------------------------------------------------------------------
// Unload a scope, i.e. delete the ScopeName's ScopeBase object.
   if (fScopeName)
      const_cast<Reflex::ScopeName*>(fScopeName)->Unload();
}


//-------------------------------------------------------------------------------
void
Reflex::Scope::UpdateMembers() const {
//-------------------------------------------------------------------------------
// UpdateMembers will update the list of Function/Data/Members with all
// members of base classes currently availabe in the system, switching
// INHERITEDMEMBERS_DEFAULT to INHERITEDMEMBERS_ALSO.

   if (*this) {
      fScopeName->fScopeBase->UpdateMembers();
   }
}


#ifdef REFLEX_CINT_MERGE
bool
Reflex::Scope::operator &&(const Scope& right) const
{ return operator bool() && (bool) right; }

bool
Reflex::Scope::operator &&(const Type& right) const
{ return operator bool() && (bool) right; }

bool
Reflex::Scope::operator &&(const Member& right) const
{ return operator bool() && (bool) right; }

bool
Reflex::Scope::operator ||(const Scope& right) const
{ return operator bool() && (bool) right; }

bool
Reflex::Scope::operator ||(const Type& right) const
{ return operator bool() && (bool) right; }

bool
Reflex::Scope::operator ||(const Member& right) const
{ return operator bool() || (bool) right; }

#endif
