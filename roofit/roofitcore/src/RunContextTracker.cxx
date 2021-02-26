// Author: Jonas Rembser, CERN  Feb 2021

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2020, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include "RunContextTracker.h"

#include "RooAbsArg.h"
#include "RooAbsReal.h"
#include "RunContext.h"
#include "RooMsgService.h"

#include "ROOT/RMakeUnique.hxx"

// Create a RunContextTracker that installs RooChangeTrackers for all arguments
// where results are cached in the RunContext.
/// \param[in] runContext The runContext used to look up the RooAbsArgs that need to be tracked.
RunContextTracker::RunContextTracker(RooBatchCompute::RunContext const &runContext)
{
   for (auto const &item : runContext.spans) {
      addTracker(item.first);
   }
}

/// Reset all change trackers to a clean state.
/// This should be called after evaluating the computation graph. In this way,
/// one can figure out which nodes need to be recomputed in the next evaluation
/// round depending on which fundamental parameters are changed in the meantime.
void RunContextTracker::resetTrackers()
{
   for (auto const &item : _trackers) {
      auto &tracker = *item.second;
      tracker.hasChanged(true);
   }
}

/// Clean a RunContext from results that need to be recalculated.
/// This should be called just before evaluating the computation graph.
/// \param[in] caller The class that is calling this function (information used for debug messages).
/// \param[in] runContext The runContext that will be cleaned from results.
void RunContextTracker::cleanRunContext(RooAbsArg const &caller, RooBatchCompute::RunContext &runContext)
{
   for (auto const &item : _trackers) {
      auto const *arg = item.first;
      auto &tracker = *item.second;
      if (tracker.hasChanged(false)) {
         auto found = runContext.spans.find(arg);
         if (found != runContext.spans.end()) {
            runContext.spans.erase(found);
         }
      } else {
         if (oodologD(&caller, FastEvaluations)) {
            if (runContext.spans.find(arg) != runContext.spans.end()) {
               oocxcoutD(&caller, FastEvaluations)
                  << "Value of " << arg->GetName() << " kept in RunContext by RunContextTracker::cleanRunContext"
                  << std::endl;
            }
         }
      }
   }
}

/// Add a RooChangeTracker that tracks only a given RooAbsReal.
/// \param[in] absReal The RooAbsReal to create a RooChangeTracker for.
void RunContextTracker::addTracker(const RooAbsReal *absReal)
{
   if (absReal && _trackers.find(absReal) == _trackers.end()) {
      auto trackerName = std::string(absReal->GetName()) + "_tracker";
      _trackers[absReal] =
         std::make_unique<RooChangeTracker>(trackerName.c_str(), trackerName.c_str(), RooArgSet{*absReal});
   }
}
