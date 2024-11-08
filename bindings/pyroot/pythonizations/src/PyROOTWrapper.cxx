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

// ROOT
#include "TROOT.h"

namespace PyROOT {
R__EXTERN PyObject *gRootModule;
}

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
}
