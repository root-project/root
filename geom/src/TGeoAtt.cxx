// @(#)root/geom:$Name:  $:$Id: TGeoAtt.cxx,v 1.7 2005/07/27 10:32:28 brun Exp $
// Author: Andrei Gheata   01/11/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGeoManager.h"
#include "TGeoAtt.h"

/*************************************************************************
 * TGeoAtt - visualization and tracking attributes for volumes and nodes
 *
 *
 *
 *************************************************************************/

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
}
//-----------------------------------------------------------------------------
TGeoAtt::~TGeoAtt()
{
// Destructor
}
//-----------------------------------------------------------------------------
void TGeoAtt::SetVisibility(Bool_t vis)
{
// Set visibility for this object
   if (vis)  SetAttBit(kVisThis);
   else      ResetAttBit(kVisThis);
   if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}
//-----------------------------------------------------------------------------
void TGeoAtt::SetVisDaughters(Bool_t vis)
{
// Set visibility for the daughters.
   if (vis)  SetAttBit(kVisDaughters);
   else      ResetAttBit(kVisDaughters);
   if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
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

