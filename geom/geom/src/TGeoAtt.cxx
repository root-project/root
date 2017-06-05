// @(#)root/geom:$Id$
// Author: Andrei Gheata   01/11/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGeoAtt
\ingroup Geometry_classes

Visualization and tracking attributes for volumes and nodes.

The TGeoAtt class is an utility for volume/node visibility and tracking
activity. By default the attributes are set to visible/active
*/

#include "TGeoAtt.h"

#include "TGeoManager.h"
#include "Rtypes.h"

ClassImp(TGeoAtt);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TGeoAtt::TGeoAtt()
{
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
////////////////////////////////////////////////////////////////////////////////
/// Constructor

TGeoAtt::TGeoAtt(Option_t * /*vis_opt*/, Option_t * /*activity_opt*/, Option_t * /*optimization_opt*/)
{
   fGeoAtt = 0;
   SetActivity(kTRUE);
   SetActiveDaughters(kTRUE);
   SetVisibility(kTRUE);
   SetVisDaughters(kTRUE);
   SetVisStreamed(kFALSE);
   SetVisTouched(kFALSE);
   SetVisLeaves();
}
////////////////////////////////////////////////////////////////////////////////
/// Destructor

TGeoAtt::~TGeoAtt()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Set branch type visibility.

void TGeoAtt::SetVisBranch()
{
   SetAttBit(kVisBranch, kTRUE);
   SetAttBit(kVisContainers, kFALSE);
   SetAttBit(kVisOnly, kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// Set branch type visibility.

void TGeoAtt::SetVisContainers(Bool_t flag)
{
   SetVisLeaves(!flag);
}

////////////////////////////////////////////////////////////////////////////////
/// Set branch type visibility.

void TGeoAtt::SetVisLeaves(Bool_t flag)
{
   SetAttBit(kVisBranch, kFALSE);
   SetAttBit(kVisContainers, !flag);
   SetAttBit(kVisOnly, kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// Set branch type visibility.

void TGeoAtt::SetVisOnly(Bool_t flag)
{
   SetAttBit(kVisBranch, kFALSE);
   SetAttBit(kVisContainers, kFALSE);
   SetAttBit(kVisOnly, flag);
}

////////////////////////////////////////////////////////////////////////////////
/// Set visibility for this object

void TGeoAtt::SetVisibility(Bool_t vis)
{
   if (vis)  SetAttBit(kVisThis);
   else      ResetAttBit(kVisThis);
   if (gGeoManager && gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}
////////////////////////////////////////////////////////////////////////////////
/// Set visibility for the daughters.

void TGeoAtt::SetVisDaughters(Bool_t vis)
{
   if (vis)  SetAttBit(kVisDaughters);
   else      ResetAttBit(kVisDaughters);
   if (gGeoManager && gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}
////////////////////////////////////////////////////////////////////////////////
/// Mark attributes as "streamed to file".

void TGeoAtt::SetVisStreamed(Bool_t vis)
{
   if (vis)  SetAttBit(kVisStreamed);
   else      ResetAttBit(kVisStreamed);
}
////////////////////////////////////////////////////////////////////////////////
/// Mark visualization attributes as "modified".

void TGeoAtt::SetVisTouched(Bool_t vis)
{
   if (vis)  SetAttBit(kVisTouched);
   else      ResetAttBit(kVisTouched);
}
////////////////////////////////////////////////////////////////////////////////
/// Set optimization flags.

void TGeoAtt::SetOptimization(Option_t * /*option*/)
{
}

