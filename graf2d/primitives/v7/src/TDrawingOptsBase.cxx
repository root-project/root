/// \file TDrawingOptsBase.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-02-20
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TDrawingOptsBase.hxx"

#include "ROOT/TDrawingAttr.hxx"
#include "ROOT/TLogger.hxx"

#include "TClass.h"
#include "TDataMember.h"
#include "TMemberInspector.h"


// pin vtable.
ROOT::Experimental::TDrawingOptsBase::~TDrawingOptsBase() = default;

namespace {
class TAttrInspector: public TMemberInspector {
   ROOT::Experimental::TDrawingOptsBase::VisitFunc_t fFunc;
   TClass* fClDrawingAttrBase = nullptr;
   bool fTriedToSetClDrawingAttrBase = false;

   void InitClDrawingAttrBase() {
      if (fTriedToSetClDrawingAttrBase)
         return;
      fTriedToSetClDrawingAttrBase = true;
      fClDrawingAttrBase = TClass::GetClass("ROOT::Experimental::TDrawingAttrBase");
      if (!fClDrawingAttrBase) {
         R__ERROR_HERE("Graf2s") << "Cannot find dictionary for class ROOT::Experimental::TDrawingAttrBase";
      }
   }

   bool InheritsFromDrawingAttrBase(const char *memberFullTypeName) {
      static constexpr const char clNameAttrTag[] = "ROOT::Experimental::TDrawingAttr<";
      static constexpr const int lenNameAttrTag = sizeof(clNameAttrTag) - 1;
      if (!strncmp(memberFullTypeName, clNameAttrTag, lenNameAttrTag)) {
         return true;
      } else if (TClass* clMember = TClass::GetClass(memberFullTypeName)) {
         InitClDrawingAttrBase();
         if (fClDrawingAttrBase &&clMember->InheritsFrom(fClDrawingAttrBase)) {
            return true;
         }
      }
      return false;
   }

public:
   TAttrInspector(const ROOT::Experimental::TDrawingOptsBase::VisitFunc_t func): fFunc(func) {}

   using TMemberInspector::Inspect;
   void Inspect(TClass *cl, const char *parent, const char *name, const void *addr, Bool_t /*isTransient*/) {
      // Skip nested objects:
      if (parent && parent[0])
         return;

      if (TDataMember* dm = cl->GetDataMember(name)) {
         if (const char* memberFullTypeName = dm->GetFullTypeName()) {
            if (InheritsFromDrawingAttrBase(memberFullTypeName)) {
               auto pAttr = reinterpret_cast<const ROOT::Experimental::TDrawingAttrBase*>(addr);
               fFunc(*const_cast<ROOT::Experimental::TDrawingAttrBase*>(pAttr));
            }
         }
      }
   }
};
}

/// Invoke func with each attribute as argument.
void ROOT::Experimental::TDrawingOptsBase::VisitAttributes(const TDrawingOptsBase::VisitFunc_t &func)
{
   TClass* clThis = TClass::GetClass(typeid(*this));
   if (!clThis) {
      R__ERROR_HERE("Graf2s") << "Cannot find dictionary for the derived class with typeid " << typeid(*this).name();
      return;
   }

   TAttrInspector insp(func);
   if (!clThis->CallShowMembers(this, insp)) {
      R__ERROR_HERE("Graf2s") << "Unable to inspect members of class with typeid " << typeid(*this).name();
      return;
   }
}

/// Synchronize all shared attributes into their local copy.
void ROOT::Experimental::TDrawingOptsBase::Snapshot() {
   VisitAttributes([](TDrawingAttrBase& attr) { attr.Snapshot(); });
}
