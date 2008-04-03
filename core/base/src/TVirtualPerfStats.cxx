// @(#)root/base:$Id$
// Author: Kristjan Gulbrandsen   11/05/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualPerfStats                                                    //
//                                                                      //
// Provides the interface for the PROOF internal performance measurment //
// and event tracing.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TVirtualPerfStats.h"


ClassImp(TVirtualPerfStats)


TVirtualPerfStats *gPerfStats = 0;

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

//______________________________________________________________________________
const char *TVirtualPerfStats::EventType(EEventType type)
{
   // Return the name of the event type.
   if (type < kUnDefined || type >= kNumEventType) {
      return "Illegal EEventType";
   } else {
      return gEventTypeNames[type];
   }
}
