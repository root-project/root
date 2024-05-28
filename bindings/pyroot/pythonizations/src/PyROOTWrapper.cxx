// Author: Enric Tejedor CERN  06/2018
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Bindings
#include "PyROOTWrapper.h"
#include "TMemoryRegulator.h"

// Cppyy
#include "CPyCppyy/API.h"

// ROOT
#include "TROOT.h"
#include "TSystem.h"
#include "TClass.h"
#include "TInterpreter.h"
#include "DllImport.h"

namespace PyROOT {
R__EXTERN PyObject *gRootModule;
}

using namespace PyROOT;

namespace {

static void AddToGlobalScope(const char *label, TObject *obj, const char *classname)
{
   // Bind the given object with the given class in the global scope with the
   // given label for its reference.
   PyModule_AddObject(gRootModule, label, CPyCppyy::Instance_FromVoidPtr(obj, classname));
}

} // unnamed namespace

PyROOT::RegulatorCleanup &GetRegulatorCleanup()
{
   // The object is thread-local because it can happen that we call into
   // C++ code (from the PyROOT CPython extension, from CPyCppyy or from cling)
   // from different Python threads. A notable example is within a distributed
   // RDataFrame application running on Dask.
   thread_local PyROOT::RegulatorCleanup m;
   return m;
}

void PyROOT::Init()
{
   // Initialize and acquire the GIL to allow for threading in ROOT
#if PY_VERSION_HEX < 0x03090000
   PyEval_InitThreads();
#endif

   // Memory management
   gROOT->GetListOfCleanups()->Add(&GetRegulatorCleanup());

   // Bind ROOT globals that will be needed in ROOT.py
   AddToGlobalScope("gROOT", gROOT, gROOT->IsA()->GetName());
   AddToGlobalScope("gSystem", gSystem, gSystem->IsA()->GetName());
   AddToGlobalScope("gInterpreter", gInterpreter, gInterpreter->IsA()->GetName());
}

PyObject *PyROOT::ClearProxiedObjects(PyObject * /* self */, PyObject * /* args */)
{
   // Delete all memory-regulated objects
   GetRegulatorCleanup().CallClearProxiedObjects();
   Py_RETURN_NONE;
}
