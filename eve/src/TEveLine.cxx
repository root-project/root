// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveLine.h"

//______________________________________________________________________________
// TEveLine
//
// An arbitrary polyline with fixed line and marker attributes.

ClassImp(TEveLine)

//______________________________________________________________________________
TEveLine::TEveLine(Int_t n_points, TreeVarType_e tv_type) :
   TEvePointSet(n_points, tv_type),
   fRnrLine   (kTRUE),
   fRnrPoints (kFALSE)
{
   fMainColorPtr = &fLineColor;
}

//______________________________________________________________________________
TEveLine::TEveLine(const Text_t* name, Int_t n_points, TreeVarType_e tv_type) :
   TEvePointSet(name, n_points, tv_type),
   fRnrLine   (kTRUE),
   fRnrPoints (kFALSE)
{
   fMainColorPtr = &fLineColor;
}

//______________________________________________________________________________
TEveLine::~TEveLine()
{}
