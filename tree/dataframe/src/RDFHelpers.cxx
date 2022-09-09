// Author: Stefan Wunsch, Enrico Guiraud CERN  09/2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDFHelpers.hxx"
#include "TROOT.h"      // IsImplicitMTEnabled
#include "TError.h"     // Warning
#include "RConfigure.h" // R__USE_IMT
#include "ROOT/RDF/RLoopManager.hxx" // for RLoopManager
#include "ROOT/RResultHandle.hxx"    // for RResultHandle, RunGraphs
#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#endif // R__USE_IMT

#include <algorithm>
#include <set>

using ROOT::RDF::RResultHandle;

void ROOT::RDF::RunGraphs(std::vector<RResultHandle> handles)
{
   if (handles.empty()) {
      Warning("RunGraphs", "Got an empty list of handles, now quitting.");
      return;
   }

   // Check that there are results which have not yet been run
   const unsigned int nToRun =
      std::count_if(handles.begin(), handles.end(), [](const auto &h) { return !h.IsReady(); });
   if (nToRun < handles.size()) {
      Warning("RunGraphs", "Got %lu handles from which %lu link to results which are already ready.", handles.size(),
              handles.size() - nToRun);
   }
   if (nToRun == 0u)
      return;

   // Find the unique event loops
   auto sameGraph = [](const RResultHandle &a, const RResultHandle &b) { return a.fLoopManager < b.fLoopManager; };
   std::set<RResultHandle, decltype(sameGraph)> s(handles.begin(), handles.end(), sameGraph);
   std::vector<RResultHandle> uniqueLoops(s.begin(), s.end());

   // Trigger the unique event loops
   auto run = [](RResultHandle &h) { if (h.fLoopManager) h.fLoopManager->Run(); };
#ifdef R__USE_IMT
   if (ROOT::IsImplicitMTEnabled()) {
      ROOT::TThreadExecutor{}.Foreach(run, uniqueLoops);
      return;
   }
#endif // R__USE_IMT
   std::for_each(uniqueLoops.begin(), uniqueLoops.end(), run);
}
