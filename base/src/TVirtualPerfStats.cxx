// @(#)root/base:$Name:  $:$Id: TVirtualPerfStats.cxx,v 1.2 2004/05/30 23:14:18 rdm Exp $
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

static char *EventTypeNames[] = {
	"UnDefined",
	"Packet",
	"Start",
	"Stop",
	"File",
	"FileOpen",
	"FileRead",
};

const char *TVirtualPerfStats::EventType(EEventType type)
{
   if (type < kUnDefined || type >= kNumEventType) {
      return "Illegal EEventType";
   } else {
      return EventTypeNames[type];
   }
}
