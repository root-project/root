// @(#)root/base:$Id$
// Author: Kristjan Gulbrandsen   11/05/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TVirtualPerfStats
\ingroup Base

Provides the interface for the PROOF internal performance measurement
and event tracing.
*/


#include "TVirtualPerfStats.h"
#include "TThreadSlots.h"


ClassImp(TVirtualPerfStats);


static const char *gEventTypeNames[] = {
   "UnDefined",
   "Packet",
   "Start",
   "Stop",
   "File",
   "FileOpen",
   "FileRead",
   "Rate"
};

////////////////////////////////////////////////////////////////////////////////
/// Return the current ROOT perf stats if any.

TVirtualPerfStats *&TVirtualPerfStats::CurrentPerfStats()
{
   static TVirtualPerfStats *currentPerfStats = nullptr;
   if (!gThreadTsd) {
      return currentPerfStats;
   } else {
      return *(TVirtualPerfStats**)(*gThreadTsd)(&currentPerfStats,ROOT::kPerfStatsThreadSlot);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return the name of the event type.

const char *TVirtualPerfStats::EventType(EEventType type)
{
   if (type < kUnDefined || type >= kNumEventType) {
      return "Illegal EEventType";
   } else {
      return gEventTypeNames[type];
   }
}
