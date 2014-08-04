// @(#)root/geom:$Id$
// Author: Andrei Gheata   01/11/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_____________________________________________________________________________
//   TGeoAtt - visualization and tracking attributes for volumes and nodes
// ==========
// The TGeoAtt class is a utility for volume/node visibility and tracking
// activity. By default the attributes are set to visible/active
//_____________________________________________________________________________

#include "TGeoManager.h"
#include "TGeoAtt.h"

ClassImp(TGeoAtt)

//-----------------------------------------------------------------------------
TGeoAtt::TGeoAtt()
{
// Default constructor
   fGeoAtt = 0;
   if (!gGeoManager) return;
   SetActivity(kTRUE);
   SetActiveDaughters(kTRUE);
   SetVisibility(kTRUE);
   SetVisDaughters(kTRUE);
   SetVisStreamed(kFALSE);
   SetVisTouched(kFALSE);
   SetVisLeaves();
}
//-----------------------------------------------------------------------------
TGeoAtt::TGeoAtt(Option_t * /*vis_opt*/, Option_t * /*activity_opt*/, Option_t * /*optimization_opt*/)
{
// Constructor
   fGeoAtt = 0;
   SetActivity(kTRUE);
   SetActiveDaughters(kTRUE);
   SetVisibility(kTRUE);
   SetVisDaughters(kTRUE);
   SetVisStreamed(kFALSE);
   SetVisTouched(kFALSE);
   SetVisLeaves();
}
//-----------------------------------------------------------------------------
TGeoAtt::~TGeoAtt()
{
// Destructor
}

//-----------------------------------------------------------------------------
void TGeoAtt::SetVisBranch()
{
// Set branch type visibility.
   SetAttBit(kVisBranch, kTRUE);
   SetAttBit(kVisContainers, kFALSE);
   SetAttBit(kVisOnly, kFALSE);
}

//-----------------------------------------------------------------------------
void TGeoAtt::SetVisContainers(Bool_t flag)
{
// Set branch type visibility.
   SetVisLeaves(!flag);
}

//-----------------------------------------------------------------------------
void TGeoAtt::SetVisLeaves(Bool_t flag)
{
// Set branch type visibility.
   SetAttBit(kVisBranch, kFALSE);
   SetAttBit(kVisContainers, !flag);
   SetAttBit(kVisOnly, kFALSE);
}

//-----------------------------------------------------------------------------
void TGeoAtt::SetVisOnly(Bool_t flag)
{
// Set branch type visibility.
   SetAttBit(kVisBranch, kFALSE);
   SetAttBit(kVisContainers, kFALSE);
   SetAttBit(kVisOnly, flag);
}

//-----------------------------------------------------------------------------
void TGeoAtt::SetVisibility(Bool_t vis)
{
// Set visibility for this object
   if (vis)  SetAttBit(kVisThis);
   else      ResetAttBit(kVisThis);
   if (gGeoManager && gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}
//-----------------------------------------------------------------------------
void TGeoAtt::SetVisDaughters(Bool_t vis)
{
// Set visibility for the daughters.
   if (vis)  SetAttBit(kVisDaughters);
   else      ResetAttBit(kVisDaughters);
   if (gGeoManager && gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}
//-----------------------------------------------------------------------------
void TGeoAtt::SetVisStreamed(Bool_t vis)
{
// Mark attributes as "streamed to file".
   if (vis)  SetAttBit(kVisStreamed);
   else      ResetAttBit(kVisStreamed);
}
//-----------------------------------------------------------------------------
void TGeoAtt::SetVisTouched(Bool_t vis)
{
// Mark visualization attributes as "modified".
   if (vis)  SetAttBit(kVisTouched);
   else      ResetAttBit(kVisTouched);
}
//-----------------------------------------------------------------------------
void TGeoAtt::SetOptimization(Option_t * /*option*/)
{
// Set optimization flags.
}

