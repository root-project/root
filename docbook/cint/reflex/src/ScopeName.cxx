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

#include "Reflex/internal/ScopeName.h"

#include "Reflex/Scope.h"
#include "Reflex/internal/ScopeBase.h"
#include "Reflex/Type.h"

#include "Reflex/Tools.h"
#include "Reflex/internal/OwnedMember.h"

#include "stl_hash.h"
#include <vector>


//-------------------------------------------------------------------------------
typedef __gnu_cxx::hash_map<const char**, Reflex::Scope> Name2Scope_t;
typedef std::vector<Reflex::Scope> ScopeVec_t;

//-------------------------------------------------------------------------------
static Name2Scope_t&
sScopes() {
//-------------------------------------------------------------------------------
// Static wrapper around scope map.
   static Name2Scope_t* m = 0;

   if (!m) {
      m = new Name2Scope_t;
   }
   return *m;
}


//-------------------------------------------------------------------------------
static ScopeVec_t&
sScopeVec() {
//-------------------------------------------------------------------------------
// Static wrapper around scope vector.
   static ScopeVec_t* m = 0;

   if (!m) {
      m = new ScopeVec_t;
   }
   return *m;
}


//-------------------------------------------------------------------------------
Reflex::ScopeName::ScopeName(const char* name,
                             ScopeBase* scopeBase):
   fName(name),
   fScopeBase(scopeBase) {
//-------------------------------------------------------------------------------
// Create the scope name dictionary info.
   fThisScope = new Scope(this);
   sScopes()[fName.key()] = *fThisScope;
   sScopeVec().push_back(*fThisScope);

   //---Build recursively the declaring scopeNames
   if (fName != "@N@I@R@V@A@N@A@") {
      std::string decl_name = Tools::GetScopeName(std::string(fName.c_str()));

      if (!Scope::ByName(decl_name).Id()) {
         new ScopeName(decl_name.c_str(), 0);
      }
   }
}


//-------------------------------------------------------------------------------
Reflex::ScopeName::~ScopeName() {
//-------------------------------------------------------------------------------
// Destructor.
}


//-------------------------------------------------------------------------------
Reflex::Scope
Reflex::ScopeName::ByName(const std::string& name) {
//-------------------------------------------------------------------------------
// Lookup a scope by fully qualified name.
   Name2Scope_t::iterator it;

   if (name.size() > 2 && name[0] == ':' && name[1] == ':') {
      const std::string& k = name.substr(2);
      const char* kcstr = k.c_str();
      it = sScopes().find(&kcstr);
   } else {
      const char* ncstr = name.c_str();
      it = sScopes().find(&ncstr);
   }

   if (it != sScopes().end()) {
      return it->second;
   } else {
      // HERE STARTS AN UGLY HACK WHICH HAS TO BE UNDONE ASAP
      // (also remove inlcude Reflex/Type.h)
      Type t = Type::ByName(name);

      if (t && t.IsTypedef()) {
         while (t.IsTypedef())
            t = t.ToType();

         if (t.IsClass() || t.IsEnum() || t.IsUnion()) {
            return t.operator Scope();
         }
      }
      return Dummy::Scope();
   }
   // END OF UGLY HACK
} // ByName


//-------------------------------------------------------------------------------
void
Reflex::ScopeName::CleanUp() {
//-------------------------------------------------------------------------------
// Cleanup memory allocations for scopes.
   ScopeVec_t::iterator it;

   for (it = sScopeVec().begin(); it != sScopeVec().end(); ++it) {
      Scope* s = ((ScopeName*) it->Id())->fThisScope;

      if (*s) {
         s->Unload();
      }
      delete s;
   }

   for (it = sScopeVec().begin(); it != sScopeVec().end(); ++it) {
      delete ((ScopeName*) it->Id());
   }
} // CleanUp


//-------------------------------------------------------------------------------
void
Reflex::ScopeName::DeleteScope() const {
//-------------------------------------------------------------------------------
// Delete the scope base information.
   delete fScopeBase;
   fScopeBase = 0;
}


//-------------------------------------------------------------------------------
void
Reflex::ScopeName::HideName() {
//-------------------------------------------------------------------------------
// Append the string " @HIDDEN@" to a scope name.
   if (fName.length() == 0 || fName[fName.length() - 1] != '@') {
      sScopes().erase(fName.key());
      fName += " @HIDDEN@";
      sScopes()[fName.key()] = this;
   }
}


//-------------------------------------------------------------------------------
void
Reflex::ScopeName::UnhideName() {
   //-------------------------------------------------------------------------------
   // Remove the string " @HIDDEN@" to a scope name.
   static const unsigned int len = strlen(" @HIDDEN@");

   if (fName.length() > len && fName[fName.length() - 1] == '@' && 0 == strcmp(" @HIDDEN@", fName.c_str() + fName.length() - len)) {
      sScopes().erase(fName.key());
      fName.erase(fName.length() - len);
      sScopes()[fName.key()] = this;
   }
}


//-------------------------------------------------------------------------------
Reflex::Scope
Reflex::ScopeName::ThisScope() const {
//-------------------------------------------------------------------------------
// Return the scope corresponding to this scope.
   return *fThisScope;
}


//-------------------------------------------------------------------------------
Reflex::Scope
Reflex::ScopeName::ScopeAt(size_t nth) {
//-------------------------------------------------------------------------------
// Return the nth scope defined in Reflex.
   if (nth < sScopeVec().size()) {
      return sScopeVec()[nth];
   }
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
size_t
Reflex::ScopeName::ScopeSize() {
//-------------------------------------------------------------------------------
// Return the number of scopes defined in Reflex.
   return sScopeVec().size();
}


//-------------------------------------------------------------------------------
Reflex::Scope_Iterator
Reflex::ScopeName::Scope_Begin() {
//-------------------------------------------------------------------------------
// Return the begin iterator of the scope collection.
   return sScopeVec().begin();
}


//-------------------------------------------------------------------------------
Reflex::Scope_Iterator
Reflex::ScopeName::Scope_End() {
//-------------------------------------------------------------------------------
// Return the end iterator of the scope collection.
   return sScopeVec().end();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_Scope_Iterator
Reflex::ScopeName::Scope_RBegin() {
//-------------------------------------------------------------------------------
// Return the rbegin iterator of the scope collection.
   return ((const std::vector<Scope> &)sScopeVec()).rbegin();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_Scope_Iterator
Reflex::ScopeName::Scope_REnd() {
//-------------------------------------------------------------------------------
// Return the rend iterator of the scope collection.
   return ((const std::vector<Scope> &)sScopeVec()).rend();
}

//-------------------------------------------------------------------------------
void
Reflex::ScopeName::Unload() {
//-------------------------------------------------------------------------------
// Unload reflection information for this type.
   if (Reflex::Instance::State() != Reflex::Instance::kHasShutDown) {
      delete fScopeBase;
      fScopeBase = 0;
      if (Reflex::Instance::State() != Reflex::Instance::kTearingDown) {
         fName.ToHeap();
      }
   }
}

