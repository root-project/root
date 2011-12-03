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

#include "Reflex/internal/TypeName.h"

#include "Reflex/Type.h"
#include "Reflex/internal/OwnedMember.h"

#include "stl_hash.h"
#include <vector>


//-------------------------------------------------------------------------------
typedef __gnu_cxx::hash_map<const char**, Reflex::TypeName*> Name2Type_t;
typedef __gnu_cxx::hash_map<const char*, Reflex::TypeName*> TypeId2Type_t;
typedef std::vector<Reflex::Type> TypeVec_t;


//-------------------------------------------------------------------------------
static Name2Type_t&
sTypes() {
//-------------------------------------------------------------------------------
// Static wrapper for type map.
   static Name2Type_t* m = 0;

   if (!m) {
      m = new Name2Type_t;
   }
   return *m;
}


//-------------------------------------------------------------------------------
static TypeId2Type_t&
sTypeInfos() {
//-------------------------------------------------------------------------------
// Static wrapper for type map (type_infos).
   static TypeId2Type_t* m;

   if (!m) {
      m = new TypeId2Type_t;
   }
   return *m;
}


//-------------------------------------------------------------------------------
static TypeVec_t&
sTypeVec() {
//-------------------------------------------------------------------------------
// Static wrapper for type vector.
   static TypeVec_t* m = 0;

   if (!m) {
      m = new TypeVec_t;
   }
   return *m;
}


//-------------------------------------------------------------------------------
Reflex::TypeName::TypeName(const char* nam,
                           TypeBase* typeBas,
                           const std::type_info* ti):
   fName(nam),
   fTypeBase(typeBas) {
//-------------------------------------------------------------------------------
// Construct a type name.
   fThisType = new Type(this);
   sTypes()[fName.key()] = this;
   sTypeVec().push_back(*fThisType);

   if (ti) {
      sTypeInfos()[ti->name()] = this;
   }
}


//-------------------------------------------------------------------------------
Reflex::TypeName::~TypeName() {
//-------------------------------------------------------------------------------
// Destructor.
}


//-------------------------------------------------------------------------------
void
Reflex::TypeName::CleanUp() {
//-------------------------------------------------------------------------------
// Cleanup memory allocations for types.
   for (TypeVec_t::iterator it = sTypeVec().begin(); it != sTypeVec().end(); ++it) {
      TypeName* tn = (TypeName*) it->Id();
      Type* t = tn->fThisType;

      if (*t) {
         t->Unload();
      }
      delete t;
      delete tn;
   }
}


//-------------------------------------------------------------------------------
void
Reflex::TypeName::DeleteType() const {
//-------------------------------------------------------------------------------
// Delete the type base information.
   delete fTypeBase;
   fTypeBase = 0;
}


//-------------------------------------------------------------------------------
void
Reflex::TypeName::SetTypeId(const std::type_info& ti) const {
//-------------------------------------------------------------------------------
// Add a type_info to the map.
   sTypeInfos()[ti.name()] = const_cast<TypeName*>(this);
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypeName::ByName(const std::string& key) {
//-------------------------------------------------------------------------------
// Lookup a type by name.
   Name2Type_t::const_iterator it;
   const Name2Type_t& n2t = sTypes();

   if (key.size() > 2 && key[0] == ':' && key[1] == ':') {
      const std::string& k = key.substr(2);
      const char* kcstr = k.c_str();
      it = n2t.find(&kcstr);
   } else {
      const char* kcstr = key.c_str();
      it = n2t.find(&kcstr);
   }

   if (it != n2t.end()) {
      return it->second->ThisType();
   } else { return Dummy::Type(); }
} // ByName


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypeName::ByTypeInfo(const std::type_info& ti) {
//-------------------------------------------------------------------------------
// Lookup a type by type_info.
   const TypeId2Type_t& id2t = sTypeInfos();
   TypeId2Type_t::const_iterator it = id2t.find(ti.name());

   if (it != id2t.end()) {
      return it->second->ThisType();
   } else { return Dummy::Type(); }
}


//-------------------------------------------------------------------------------
void
Reflex::TypeName::HideName() {
//-------------------------------------------------------------------------------
// Append the string " @HIDDEN@" to a type name.
   if (fName.length() == 0 || fName[fName.length() - 1] != '@') {
      sTypes().erase(fName.key());
      fName += " @HIDDEN@";
      sTypes()[fName.key()] = this;
   }
}


//-------------------------------------------------------------------------------
void
Reflex::TypeName::UnhideName() {
   //-------------------------------------------------------------------------------
   // Remove the string " @HIDDEN@" to a type name.
   static const unsigned int len = strlen(" @HIDDEN@");

   if (fName.length() > len && fName[fName.length() - 1] == '@' && 0 == strcmp(" @HIDDEN@", fName.c_str() + fName.length() - len)) {
      sTypes().erase(fName.key());
      fName.erase(fName.length() - len);
      sTypes()[fName.key()] = this;
   }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypeName::ThisType() const {
//-------------------------------------------------------------------------------
// Return Type of this TypeName.
   return *fThisType;
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypeName::TypeAt(size_t nth) {
//-------------------------------------------------------------------------------
// Return nth type in Reflex.
   if (nth < sTypeVec().size()) {
      return sTypeVec()[nth];
   }
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
size_t
Reflex::TypeName::TypeSize() {
//-------------------------------------------------------------------------------
// Return number of types in Reflex.
   return sTypeVec().size();
}


//-------------------------------------------------------------------------------
Reflex::Type_Iterator
Reflex::TypeName::Type_Begin() {
//-------------------------------------------------------------------------------
// Return begin iterator of the type container.
   return sTypeVec().begin();
}


//-------------------------------------------------------------------------------
Reflex::Type_Iterator
Reflex::TypeName::Type_End() {
//-------------------------------------------------------------------------------
// Return end iterator of the type container.
   return sTypeVec().end();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_Type_Iterator
Reflex::TypeName::Type_RBegin() {
//-------------------------------------------------------------------------------
// Return rbegin iterator of the type container.
   return ((const std::vector<Type> &)sTypeVec()).rbegin();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_Type_Iterator
Reflex::TypeName::Type_REnd() {
//-------------------------------------------------------------------------------
// Return rend iterator of the type container.
   return ((const std::vector<Type> &)sTypeVec()).rend();
}


//-------------------------------------------------------------------------------
void
Reflex::TypeName::Unload() {
//-------------------------------------------------------------------------------
// Unload reflection information for this type.
   if (Reflex::Instance::State() != Reflex::Instance::kHasShutDown) {
      delete fTypeBase;
      fTypeBase = 0;
      if (Reflex::Instance::State() != Reflex::Instance::kTearingDown) {
         fName.ToHeap();
      }
   } else {
      // Still invalidate this instance.
      fTypeBase = 0;
   }
}

