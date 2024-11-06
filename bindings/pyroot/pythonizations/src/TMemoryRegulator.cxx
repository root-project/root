
// Author: Enric Tejedor CERN  08/2019
// Author: Vincenzo Eduardo Padulano CERN 05/2024

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMemoryRegulator.h"

#include "../../cppyy/CPyCppyy/src/ProxyWrappers.h"
#include "../../cppyy/CPyCppyy/src/CPPInstance.h"

namespace {
// We use a map because we do not control the deregistration anyway, that is
// managed by the Python GC.
// key: object address; value: object class id
using TrackedObjects_t = std::unordered_map<Cppyy::TCppObject_t, Cppyy::TCppType_t>;

TrackedObjects_t &GetTrackedObjects()
{
   thread_local TrackedObjects_t trackedObjects{};
   return trackedObjects;
}

Cppyy::TCppType_t &GetTObjectTypeID()
{
   static Cppyy::TCppType_t tobjectTypeID{Cppyy::GetScope("TObject")};
   return tobjectTypeID;
}

auto GetObjectIt(const TrackedObjects_t &objects, Cppyy::TCppObject_t key)
{
   return objects.find(key);
}

std::pair<bool, bool> RegisterHook(Cppyy::TCppObject_t cppobj, Cppyy::TCppType_t klass)
{
   if (Cppyy::IsSubtype(klass, GetTObjectTypeID())) {
      GetTrackedObjects().insert({cppobj, klass});
   }
   return {true, true};
}

std::pair<bool, bool> UnregisterHook(Cppyy::TCppObject_t cppobj, Cppyy::TCppType_t klass)
{
   if (Cppyy::IsSubtype(klass, GetTObjectTypeID())) {
      if (auto it = GetObjectIt(GetTrackedObjects(), cppobj); it != GetTrackedObjects().end()) {
         GetTrackedObjects().erase(it);
      }
   }
   return {true, true};
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// \brief Injects custom registration/deregistration logic into cppyy's memory regulator
PyROOT::RegulatorCleanup::RegulatorCleanup()
{
   CPyCppyy::MemoryRegulator::SetRegisterHook(RegisterHook);
   CPyCppyy::MemoryRegulator::SetUnregisterHook(UnregisterHook);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Get the class id of the TObject being deleted and run Cppyy's
///        RecursiveRemove.
/// \param[in] object Object being destructed.
void PyROOT::CallCppyyRecursiveRemove(TObject *object)
{
   auto &trackedObjects = GetTrackedObjects();
   if (auto it = GetObjectIt(trackedObjects, reinterpret_cast<Cppyy::TCppObject_t>(object));
       it != trackedObjects.end()) {
      // The iterator may be invalidated in RecursiveRemove, so we erase it from our tracked objects first.
      const auto cppobj = it->first;
      const auto klassid = it->second;
      trackedObjects.erase(it);
      CPyCppyy::MemoryRegulator::RecursiveRemove(cppobj, klassid);
   }
}
