
// Author: Enric Tejedor CERN  08/2019

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMemoryRegulator.h"

PyROOT::ObjectMap_t PyROOT::TMemoryRegulator::fObjectMap = PyROOT::ObjectMap_t();

////////////////////////////////////////////////////////////////////////////
/// \brief Constructor. Registers the hooks to run on Cppyy's object
///        construction and destruction
PyROOT::TMemoryRegulator::TMemoryRegulator()
{
   CPyCppyy::MemoryRegulator::SetRegisterHook(PyROOT::TMemoryRegulator::RegisterHook);
   CPyCppyy::MemoryRegulator::SetUnregisterHook(PyROOT::TMemoryRegulator::UnregisterHook);
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
      ObjectMap_t::iterator ppo = fObjectMap.find(cppobj);
      if (ppo == fObjectMap.end()) {
         // Set cleanup bit so RecursiveRemove is tried on registered object
         ((TObject*)cppobj)->SetBit(TObject::kMustCleanup);
         fObjectMap.insert({cppobj, klass});
      }
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
      ObjectMap_t::iterator ppo = fObjectMap.find(cppobj);
      if (ppo != fObjectMap.end()) {
         fObjectMap.erase(ppo);
      }
   }

   return {true, true};
}

////////////////////////////////////////////////////////////////////////////
/// \brief Get the class id of the TObject being deleted and run Cppyy's
///        RecursiveRemove.
/// \param[in] object Object being destructed.
void PyROOT::TMemoryRegulator::RecursiveRemove(TObject *object)
{
   auto cppobj = (Cppyy::TCppObject_t)object;
   Cppyy::TCppType_t klass = 0;

   ObjectMap_t::iterator ppo = fObjectMap.find(cppobj);
   if (ppo != fObjectMap.end()) {
      klass = ppo->second;
      CPyCppyy::MemoryRegulator::RecursiveRemove(cppobj, klass);
   }
}
