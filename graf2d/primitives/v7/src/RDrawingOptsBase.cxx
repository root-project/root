/// \file RDrawingOptsBase.cxx
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

#include "ROOT/RDrawingOptsBase.hxx"

#include "ROOT/RDrawingAttr.hxx"
#include "ROOT/TLogger.hxx"

#include "TClass.h"
#include "TDataMember.h"
#include "TMemberInspector.h"


// pin vtable.
ROOT::Experimental::RDrawingOptsBase::~RDrawingOptsBase() = default;

namespace {
class RAttrInspector: public TMemberInspector {
   ROOT::Experimental::RDrawingOptsBase::VisitFunc_t fFunc;
   TClass* fClDrawingAttrBase = nullptr;
   bool fTriedToSetClDrawingAttrBase = false;

   void InitClDrawingAttrBase() {
      if (fTriedToSetClDrawingAttrBase)
         return;
      fTriedToSetClDrawingAttrBase = true;
      fClDrawingAttrBase = TClass::GetClass("ROOT::Experimental::RDrawingAttrBase");
      if (!fClDrawingAttrBase) {
         R__ERROR_HERE("Graf2d") << "Cannot find dictionary for class ROOT::Experimental::RDrawingAttrBase";
      }
   }

   bool InheritsFromDrawingAttrBase(const char *memberFullTypeName) {
      static constexpr const char clNameAttrTag[] = "ROOT::Experimental::RDrawingAttr<";
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
   RAttrInspector(const ROOT::Experimental::RDrawingOptsBase::VisitFunc_t func): fFunc(func) {}

   using TMemberInspector::Inspect;
   void Inspect(TClass *cl, const char *parent, const char *name, const void *addr, Bool_t /*isTransient*/) {
      // Skip nested objects:
      if (parent && parent[0])
         return;

      if (TDataMember* dm = cl->GetDataMember(name)) {
         if (const char* memberFullTypeName = dm->GetFullTypeName()) {
            if (InheritsFromDrawingAttrBase(memberFullTypeName)) {
               auto pAttr = reinterpret_cast<const ROOT::Experimental::RDrawingAttrBase*>(addr);
               fFunc(*const_cast<ROOT::Experimental::RDrawingAttrBase*>(pAttr));
            }
         }
      }
   }
};
}

/// Invoke func with each attribute as argument.
void ROOT::Experimental::RDrawingOptsBase::VisitAttributes(const RDrawingOptsBase::VisitFunc_t &func)
{
   TClass* clThis = ROOT::GetClass(this);
   if (!clThis) {
      R__ERROR_HERE("Graf2d") << "Cannot find dictionary for the derived class with typeid " << typeid(*this).name();
      return;
   }

   RAttrInspector insp(func);
   if (!clThis->CallShowMembers(this, insp)) {
      R__ERROR_HERE("Graf2d") << "Unable to inspect members of class with typeid " << typeid(*this).name();
      return;
   }
}

/// Synchronize all shared attributes into their local copy.
void ROOT::Experimental::RDrawingOptsBase::Snapshot() {
   VisitAttributes([](RDrawingAttrBase& attr) { attr.Snapshot(); });
}
