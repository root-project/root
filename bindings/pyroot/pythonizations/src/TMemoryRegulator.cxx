
// Author: Enric Tejedor CERN  08/2019
// Author: Vincenzo Eduardo Padulano CERN 05/2024

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMemoryRegulator.h"

#include "../../cppyy/CPyCppyy/src/ProxyWrappers.h"
#include "../../cppyy/CPyCppyy/src/CPPInstance.h"

////////////////////////////////////////////////////////////////////////////
/// \brief Constructor. Registers the hooks to run on Cppyy's object
///        construction and destruction
PyROOT::TMemoryRegulator::TMemoryRegulator()
{
   CPyCppyy::MemoryRegulator::SetRegisterHook(
      [this](Cppyy::TCppObject_t cppobj, Cppyy::TCppType_t klass) { return this->RegisterHook(cppobj, klass); });
   CPyCppyy::MemoryRegulator::SetUnregisterHook(
      [this](Cppyy::TCppObject_t cppobj, Cppyy::TCppType_t klass) { return this->UnregisterHook(cppobj, klass); });
}

////////////////////////////////////////////////////////////////////////////
/// \brief Register a hook that Cppyy runs when constructing an object.
/// \param[in] cppobj Address of the object.
/// \param[in] klass Class id of the object.
/// \return Pair of two booleans. First indicates success, second tells
///         Cppyy if we want to continue running RegisterPyObject
std::pair<bool, bool> PyROOT::TMemoryRegulator::RegisterHook(Cppyy::TCppObject_t cppobj, Cppyy::TCppType_t klass)
{
   static Cppyy::TCppType_t tobjectTypeID = (Cppyy::TCppType_t)Cppyy::GetScope("TObject");

   if (Cppyy::IsSubtype(klass, tobjectTypeID)) {
      fObjectMap.insert({cppobj, klass});
   }

   return {true, true};
}

////////////////////////////////////////////////////////////////////////////
/// \brief Register a hook that Cppyy runs when deleting an object.
/// \param[in] cppobj Address of the object.
/// \param[in] klass Class id of the object.
/// \return Pair of two booleans. First indicates success, second tells
///         Cppyy if we want to continue running UnRegisterPyObject
std::pair<bool, bool> PyROOT::TMemoryRegulator::UnregisterHook(Cppyy::TCppObject_t cppobj, Cppyy::TCppType_t klass)
{

   static Cppyy::TCppType_t tobjectTypeID = (Cppyy::TCppType_t)Cppyy::GetScope("TObject");

   if (Cppyy::IsSubtype(klass, tobjectTypeID)) {
      if (auto it = fObjectMap.find(cppobj); it != fObjectMap.end())
         fObjectMap.erase(it);
   }

   return {true, true};
}

////////////////////////////////////////////////////////////////////////////
/// \brief Get the class id of the TObject being deleted and run Cppyy's
///        RecursiveRemove.
/// \param[in] object Object being destructed.
void PyROOT::TMemoryRegulator::CallCppyyRecursiveRemove(TObject *object)
{
   auto cppobj = reinterpret_cast<Cppyy::TCppObject_t>(object);

   if (auto it = fObjectMap.find(cppobj); it != fObjectMap.end()) {
      CPyCppyy::MemoryRegulator::RecursiveRemove(cppobj, it->second);
      fObjectMap.erase(it);
   }
}

////////////////////////////////////////////////////////////////////////////
/// \brief Clean up all tracked objects.
void PyROOT::TMemoryRegulator::ClearProxiedObjects()
{
   while (!fObjectMap.empty()) {
      auto elem = fObjectMap.begin();
      auto cppobj = elem->first;
      auto klassid = elem->second;
      auto pyclass = CPyCppyy::CreateScopeProxy(klassid);
      auto pyobj = (CPyCppyy::CPPInstance *)CPyCppyy::MemoryRegulator::RetrievePyObject(cppobj, pyclass);

      if (pyobj && (pyobj->fFlags & CPyCppyy::CPPInstance::kIsOwner)) {
         // Only delete the C++ object if the Python proxy owns it.
         // If it is a value, cppyy deletes it in RecursiveRemove as part of
         // the proxy cleanup.
         auto o = static_cast<TObject *>(cppobj);
         bool isValue = pyobj->fFlags & CPyCppyy::CPPInstance::kIsValue;
         CallCppyyRecursiveRemove(o);
         if (!isValue)
            delete o;
      } else {
         // Non-owning proxy, just unregister to clean tables.
         // The proxy deletion by Python will have no effect on C++, so all good
         bool ret = CPyCppyy::MemoryRegulator::UnregisterPyObject(pyobj, pyclass);
         if (!ret) {
            fObjectMap.erase(elem);
         }
      }
   }
}
