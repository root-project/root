
// Author: Enric Tejedor CERN  08/2019
// Author: Vincenzo Eduardo Padulano CERN 05/2024

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef PYROOT_TMEMORYREGULATOR_H
#define PYROOT_TMEMORYREGULATOR_H

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMemoryRegulator                                                     //
//                                                                      //
// Sets hooks in Cppyy's MemoryRegulator to keep track of the TObjects  //
// that are constructed and destructed. For those objects, a map is     //
// filled, where the key is the address of the object and the value is  //
// the class to which the object belongs.                               //
//                                                                      //
// The TMemoryRegulator object, created in PyROOTWrapper.cxx, is added  //
// to the list of cleanups and its RecursiveRemove method is called by  //
// ROOT to manage the memory of TObjects being deleted.                 //
// In RecursiveRemove, the object being deleted is already a TNamed, so //
// the information about its actual class is not available anymore.     //
// To solve the problem, the map above is used to know the class of the //
// object, so that Cppyy's RecursiveRemove can be called passing the    //
// class as argument.                                                   //
//////////////////////////////////////////////////////////////////////////

// Bindings
// CPyCppyy.h must be go first, since it includes Python.h, which must be
// included before any standard header
#include "../../cppyy/CPyCppyy/src/CPyCppyy.h"
#include "../../cppyy/CPyCppyy/src/MemoryRegulator.h"

// ROOT
#include "TObject.h"

// Stl
#include <unordered_map>

namespace PyROOT {

class RegulatorCleanup;

/// Manages TObject-derived objects created in a PyROOT application
///
/// This class is responsible to keep track of the creation of the objects
/// that need further memory management within ROOT. The `ClearProxiedObjects`
/// function is only called at PyROOT shutdown time. The `CallCppyyRecursiveRemove`
/// is called as part of the global list of cleanups object destruction.
///
/// This class is intentionally not derived from TObject. See the
/// `PyROOT::RegulatorCleanup` class for more info.
///
/// \note This class is not thread-safe on its own. We create one thread-local
///       object in PyROOTWrapper.cxx.
class TMemoryRegulator final {
   using ObjectMap_t = std::unordered_map<Cppyy::TCppObject_t, Cppyy::TCppType_t>;

   ObjectMap_t fObjectMap{}; // key: object address; value: object class id

   std::pair<bool, bool> RegisterHook(Cppyy::TCppObject_t, Cppyy::TCppType_t);

   std::pair<bool, bool> UnregisterHook(Cppyy::TCppObject_t, Cppyy::TCppType_t);

   void CallCppyyRecursiveRemove(TObject *object);

   void ClearProxiedObjects();

   TMemoryRegulator();

   friend class RegulatorCleanup;
};

/// A TObject-derived class to inject the memory regulation logic in the ROOT list of cleanups.
///
/// The purpose of this class is to keep the responsibilities separate between
/// the TMemoryRegulator logic and the rest of ROOT.
class RegulatorCleanup final : public TObject {
   TMemoryRegulator fRegulator{};

public:
   void RecursiveRemove(TObject *object) final { fRegulator.CallCppyyRecursiveRemove(object); }
   void CallClearProxiedObjects() { fRegulator.ClearProxiedObjects(); }
   ClassDefInlineNV(RegulatorCleanup, 0);
};

} // namespace PyROOT

#endif // !PYROOT_TMEMORYREGULATOR_H
