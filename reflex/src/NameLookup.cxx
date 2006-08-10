// @(#)root/reflex:$Name:  $:$Id: NameLookup.cxx,v 1.8 2006/08/07 14:20:07 axel Exp $
// Author: Stefan Roiser 2006

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

#include "NameLookup.h"
#include "Reflex/Base.h"
#include "Reflex/Scope.h"
#include "Reflex/Type.h"
#include "Reflex/Tools.h"
#include "Reflex/internal/OwnedMember.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::NameLookup::NameLookup(const std::string& name, const Scope& current):
   fCurrentScope(current), fLookupName(name), fPartialSuccess(false), 
   fPosNamePart(0), fPosNamePartLen(std::string::npos) {
   // Initialize a NameLookup object used internally to keep track of lookup
   // states.
   }
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
const ROOT::Reflex::Type &
ROOT::Reflex::NameLookup::LookupType( const std::string & nam, 
                                      const Scope & current ) {
//-------------------------------------------------------------------------------
// Lookup up a (possibly scoped) type name appearing in the scope context 
// current. This is the public interface for type lookup.
   NameLookup lookup(nam, current);
   return lookup.LookupType();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Type &
ROOT::Reflex::NameLookup::LookupType() {
//-------------------------------------------------------------------------------
// Lookup a type using fLookupName, fCurrentScope.
   fPartialSuccess = false;
   fPosNamePart = 0;
   fPosNamePartLen = std::string::npos;
   FindNextScopePos();

   if ( fPosNamePart == 2 ) {
      fLookedAtUsingDir.clear();
      // ::A...
      fCurrentScope = Scope::GlobalScope();
      return LookupTypeInScope();
   } else
      // A...
      return LookupTypeInUnknownScope();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Type &
ROOT::Reflex::NameLookup::LookupTypeInScope() {
//-------------------------------------------------------------------------------
// Lookup a type in fCurrentScope.
// Checks sub-types, sub-scopes, using directives, and base classes for
// the name given by fLookupName, fPosNamePart, and fPosNamePartLen.
// If the name part is found, and another name part follows in fPosNamePart,
// LookupTypeInScope requests the scope found to lookup the next name
// part. fPartialMatch reflexts that the left part of the name was matched;
// even if the trailing part of fLookupName cannot be found, the lookup 
// cannot continue on declaring scopes and has to fail.
// A list of lookups performed in namespaces pulled in via using directives is 
// kept in fLookedAtUsingDir, to prevent infinite loops due to 
//   namespace A{using namespace B;} namespace B{using namespace A;}
// loops.
// The lookup does not take the declaration order into account; the result of
// parts of the lookup algorithm which depend on the order will be unpredictable.

   if (!fCurrentScope) return Dummy::Type();
   if (fLookedAtUsingDir.find(fCurrentScope) != fLookedAtUsingDir.end())
      // prevent inf loop from
      // ns A { using ns B; } ns B {using ns A;}
      return Dummy::Type();

   for ( Type_Iterator it = fCurrentScope.SubType_Begin(); it != fCurrentScope.SubType_End(); ++it ) {
      if ( 0 == fLookupName.compare(fPosNamePart, fPosNamePartLen, (*it).Name() ) ) {
         fPartialSuccess = true;
         fLookedAtUsingDir.clear();
         FindNextScopePos();
         if (fPosNamePart == std::string::npos) return *it;
         if (it->IsTypedef()) fCurrentScope = it->FinalType();
         else fCurrentScope = *it;
         return LookupTypeInScope();
      }
   }

   for ( Scope_Iterator in = fCurrentScope.SubScope_Begin(); in != fCurrentScope.SubScope_End(); ++in ) {
      // only take namespaces into account - classes were checked as part of SubType
      if (in->IsNamespace() && 
         0 == fLookupName.compare(fPosNamePart, fPosNamePartLen, (*in).Name() ) ) {
         fPartialSuccess = true;
         fLookedAtUsingDir.clear();
         FindNextScopePos();
         if (fPosNamePart == std::string::npos) return Dummy::Type(); // namespace is no a type
         return LookupTypeInScope();
      }
   }

   if (fCurrentScope.UsingDirectiveSize()) {
      fLookedAtUsingDir.insert(fCurrentScope);
      Scope storeCurrentScope = fCurrentScope;
      for ( Scope_Iterator si = storeCurrentScope.UsingDirective_Begin(); si != storeCurrentScope.UsingDirective_End(); ++si ) {
         fCurrentScope = *si;
         const Type & t = LookupTypeInScope();
         if (fPartialSuccess) return t;
      }
      fCurrentScope = storeCurrentScope;
   }

   if (fPosNamePart == 0) // only for "BaseClass...", not for "A::BaseClass..."
      for ( Base_Iterator bi = fCurrentScope.Base_Begin(); bi != fCurrentScope.Base_End(); ++bi ) {
         if ( 0 == fLookupName.compare(fPosNamePart, fPosNamePartLen, (*bi).Name() ) ) {
            fPartialSuccess = true;
            fLookedAtUsingDir.clear();
            FindNextScopePos();
            if (fPosNamePart == std::string::npos) return bi->ToType();
            fCurrentScope = bi->ToType().FinalType();
            return LookupTypeInScope();
         }
      }

   if (fCurrentScope.BaseSize()) {
      Scope storeCurrentScope = fCurrentScope;
      for ( Base_Iterator bi = storeCurrentScope.Base_Begin(); bi != storeCurrentScope.Base_End(); ++bi ) {
         fCurrentScope = bi->ToScope();
         const Type & t = LookupTypeInScope();
         if ( fPartialSuccess) return t;
      }
      fCurrentScope = storeCurrentScope;
   }

   return Dummy::Type();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Type &
ROOT::Reflex::NameLookup::LookupTypeInUnknownScope() {
//-------------------------------------------------------------------------------
// Lookup a type in fCurrentScope and its declaring scopes.
   for (fPartialSuccess = false; !fPartialSuccess && fCurrentScope; fCurrentScope = fCurrentScope.DeclaringScope()) {
      fLookedAtUsingDir.clear();
      const Type & t = LookupTypeInScope();
      if (fPartialSuccess) return t;
      if (fCurrentScope.IsTopScope()) break;
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Member &
ROOT::Reflex::NameLookup::LookupMember( const std::string & nam, 
                                        const Scope & current ) {
//-------------------------------------------------------------------------------
// Lookup a member.
   if ( Tools::GetBasePosition(nam)) return LookupMemberQualified( nam );
   else                              return LookupMemberUnqualified( nam, current );

}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Member &
ROOT::Reflex::NameLookup::LookupMemberQualified( const std::string & nam ) {
//-------------------------------------------------------------------------------
// Lookup of a qualified member.
   const Scope & bscope = Scope::ByName(Tools::GetScopeName(nam));
   if ( bscope ) {
      return LookupMemberUnqualified( Tools::GetBaseName(nam), bscope);
   }
   else {
      return Dummy::Member();
   }

}



//-------------------------------------------------------------------------------
const ROOT::Reflex::Member &
ROOT::Reflex::NameLookup::LookupMemberUnqualified( const std::string & nam,
                                                   const Scope & current ) {
//-------------------------------------------------------------------------------
// Lookup of an unqualified member.
   const Member & m0 = current.MemberByName(nam);
   if ( m0 ) return m0;
      
   for ( Scope_Iterator si = current.UsingDirective_Begin(); si != current.UsingDirective_End(); ++si ) {
      const Member & m1 = LookupMember( nam, *si );
      if ( m1 ) return m1;
   }

   for ( Base_Iterator bi = current.Base_Begin(); bi != current.Base_End(); ++ bi ) {
      const Member & m2 = LookupMember( nam, bi->ToScope() );
      if ( m2 ) return m2;
   }
         
   if ( ! current.IsTopScope() ) return LookupMember( nam, current.DeclaringScope() );

   return Dummy::Member();

}



//-------------------------------------------------------------------------------
const ROOT::Reflex::Type & ROOT::Reflex::NameLookup::AccessControl( const Type & typ,
                                                                    const Scope & /* current */ ) {
//-------------------------------------------------------------------------------
// Check access .
   
   //if ( typ.IsPublic()) return true;

   //else if ( typ.IsProtected() && current.HasBase(typ.DeclaringScope()) ) return true;

   return typ;

}

//-------------------------------------------------------------------------------
void 
ROOT::Reflex::NameLookup::FindNextScopePos() {
//-------------------------------------------------------------------------------
// Move fPosNamePart to point to the next scope in fLookupName, updating 
// fPosNamePartLen. If fPosNamePartLen == std::string::npos, initialize
// fPosNamePart and fPosNamePartLen. If there is no next scope left, fPosNamePart
// will be set to std::string::npos and fPosNamePartLen will be set to 0.
   if (fPosNamePartLen != std::string::npos) {
      // we know the length, so jump
      fPosNamePart += fPosNamePartLen + 2;
      if (fPosNamePart > fLookupName.length()) {
         // past the string's end?
         fPosNamePart = std::string::npos;
         fPosNamePartLen = 0;
         return;
      }
   }
   else {
      // uninitialized
      // set fPosNamePartLen and check that fLookupName doesn't start with '::'
      if (fLookupName.compare(0, 2, "::") == 0)
         fPosNamePart = 2;
      else fPosNamePart = 0;
   }
   fPosNamePartLen = Tools::GetFirstScopePosition(fLookupName.substr(fPosNamePart));
   if (fPosNamePartLen == 0)
      // no next "::"
      fPosNamePartLen = fLookupName.length();
   else fPosNamePartLen -= 2; // no "::"
   // fPosNamePartLen -= fPosNamePart; No! We already look in substr()
}
