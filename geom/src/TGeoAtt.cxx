/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata - date

////////////////////////////////////////////////////////////////////////////////
// Full description with examples and pictures
//
//
//
//
//Begin_Html
/*
<img src=".gif">
*/
//End_Html

#include "TGeoManager.h"
#include "TGeoAtt.h"

// statics and globals

ClassImp(TGeoAtt)

//-----------------------------------------------------------------------------
TGeoAtt::TGeoAtt()
{
// Default constructor
   fGeoAtt = 0;
   SetVisibility(kTRUE);
   SetVisDaughters(kTRUE);
   SetVisStreamed(kFALSE);
   SetVisTouched(kFALSE);
}
//-----------------------------------------------------------------------------
TGeoAtt::TGeoAtt(Option_t *vis_opt, Option_t *activity_opt, Option_t *optimization_opt)
{
// constructor
   fGeoAtt = 0;
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
// set visibility for this object
   if (vis)  SetBit(kVisThis);
   else      ResetBit(kVisThis);
   if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}
//-----------------------------------------------------------------------------
void TGeoAtt::SetVisDaughters(Bool_t vis)
{
// set visibility for this object
   if (vis)  SetBit(kVisDaughters);
   else      ResetBit(kVisDaughters);
   if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}
//-----------------------------------------------------------------------------
void TGeoAtt::SetVisStreamed(Bool_t vis)
{
// set visibility for this object
   if (vis)  SetBit(kVisStreamed);
   else      ResetBit(kVisStreamed);
}
//-----------------------------------------------------------------------------
void TGeoAtt::SetVisTouched(Bool_t vis)
{
// set visibility for this object
   if (vis)  SetBit(kVisTouched);
   else      ResetBit(kVisTouched);
}
//-----------------------------------------------------------------------------
void TGeoAtt::SetActivity(Option_t *option)
{
// set activity flags 
}
//-----------------------------------------------------------------------------
void TGeoAtt::SetOptimization(Option_t *option)
{
// set optimization flags 
}

