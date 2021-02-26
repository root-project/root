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

#ifndef roofit_roofitcore_RunContextTracker_h
#define roofit_roofitcore_RunContextTracker_h

#include "RooChangeTracker.h"

#include <memory>
#include <unordered_map>

class RooAbsArg;
class RooAbsReal;
class RooChangeTracker;

namespace RooBatchCompute {
struct RunContext;
}

/// Keeping track of which results can be erased from a RooBatchCompute::RunContext to clean a RunContext if needed.
/// This is achieved by owning a RooChangeTracker for each RooAbsReal whose result is cached in a RunContext.
/// Enable logging by adding this line to your script:
/// ~~~{.cpp}
/// RooMsgService::instance().addStream(DEBUG, Topic(FastEvaluations));
/// ~~~
class RunContextTracker {

public:
   RunContextTracker(RooBatchCompute::RunContext const &runContext);

   void resetTrackers();
   void cleanRunContext(RooAbsArg const &caller, RooBatchCompute::RunContext &runContext);

private:
   void addTracker(const RooAbsReal *absReal);
   void checkIfCovers(RooBatchCompute::RunContext const &runContext) const;

   // member variables
   std::unordered_map<const RooAbsReal *, std::unique_ptr<RooChangeTracker>> _trackers;
};

#endif
