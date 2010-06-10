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
# define REFLEX_BUILD
#endif

#include "Reflex/internal/MemberBase.h"

#include "Reflex/internal/OwnedMember.h"
#include "Reflex/Scope.h"
#include "Reflex/Type.h"
#include "Reflex/Base.h"
#include "Reflex/Object.h"
#include "Reflex/internal/OwnedPropertyList.h"
#include "Reflex/DictionaryGenerator.h"

#include "Reflex/Tools.h"
#include "Class.h"

//-------------------------------------------------------------------------------
Reflex::MemberBase::MemberBase(const char* name,
                               const Type& type,
                               TYPE memberType,
                               unsigned int modifiers)
//-------------------------------------------------------------------------------
   : fType(type, modifiers & (CONST | VOLATILE | REFERENCE), Type::APPEND),
   fModifiers(modifiers),
   fName(name),
   fScope(Scope()),
   fMemberType(memberType),
   fPropertyList(OwnedPropertyList(new PropertyListImpl())) {
// Construct the dictionary info for a member
   fThisMember = new Member(this);
}


//-------------------------------------------------------------------------------
Reflex::MemberBase::~MemberBase() {
//-------------------------------------------------------------------------------
// Destructor.
   delete fThisMember;
   fPropertyList.Delete();
}


//-------------------------------------------------------------------------------
Reflex::MemberBase::operator
Reflex::Member() const {
//-------------------------------------------------------------------------------
// Conversion operator to Member.
   return *fThisMember;
}


//-------------------------------------------------------------------------------
void*
Reflex::MemberBase::CalculateBaseObject(const Object& obj) const {
//-------------------------------------------------------------------------------
// Return the object address a member lives in.
   char* mem = (char*) obj.Address();
   // check if its a dummy object
   Type cl = obj.TypeOf();

   while (cl && cl.IsTypedef()) {
      cl = cl.ToType();
   }

   // if the object type is not implemented return the Address of the object
   if (!cl) {
      return mem;
   }

   if (cl.IsClass()) {
      const Class* clTB = 0;
      if (DeclaringScope()) {
         const Class* declCl =  dynamic_cast<const Class*>(DeclaringScope().ToScopeBase());
         if (declCl && cl.Id() != declCl->ThisType().Id()) {
            // now we know that the Member type is an inherited one
            clTB = dynamic_cast<const Class*>(cl.ToTypeBase());
         }
      }
      if (clTB) {
         std::vector<OffsetFunction> basePath = clTB->PathToBase(DeclaringScope());

         if (basePath.size()) {
            // there is a path described from the object to the class containing the Member
            std::vector<OffsetFunction>::iterator pIter;

            for (pIter = basePath.begin(); pIter != basePath.end(); ++pIter) {
               mem += (*pIter)(mem);
            }
         } else {
            throw RuntimeError(std::string(": ERROR: There is no path available from class ")
                               + cl.Name(SCOPED) + " to " + Name(SCOPED));
         }
      }
   } else {
      throw RuntimeError(std::string("Object ") + cl.Name(SCOPED) + " does not represent a class");
   }
   return (void*) mem;
} // CalculateBaseObject


//-------------------------------------------------------------------------------
Reflex::Scope
Reflex::MemberBase::DeclaringScope() const {
//-------------------------------------------------------------------------------
// Return the scope the member lives in.
   return fScope;
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::MemberBase::DeclaringType() const {
//-------------------------------------------------------------------------------
// Return the type the member lives in.
   return DeclaringScope();
}


//-------------------------------------------------------------------------------
std::string
Reflex::MemberBase::MemberTypeAsString() const {
//-------------------------------------------------------------------------------
// Remember type of the member as a string.
   switch (fMemberType) {
   case DATAMEMBER:
      return "DataMember";
      break;
   case FUNCTIONMEMBER:
      return "FunctionMember";
      break;
   default:
      return Reflex::Argv0() + ": ERROR: Member " + Name() +
             " has no Species associated";
   }
}


//-------------------------------------------------------------------------------
Reflex::PropertyList
Reflex::MemberBase::Properties() const {
//-------------------------------------------------------------------------------
// Return the property list attached to this member.
   return fPropertyList;
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::MemberBase::TemplateArgumentAt(size_t /* nth */) const {
//-------------------------------------------------------------------------------
// Return the nth template argument (in FunMemTemplInstance)
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
void
Reflex::MemberBase::GenerateDict(DictionaryGenerator& /* generator */) const {
//-------------------------------------------------------------------------------
// Generate Dictionary information about itself.
}
