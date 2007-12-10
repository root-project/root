// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEvePad.h"
#include "THashList.h"

//______________________________________________________________________________
// TEvePad
//
// This was intended as a TPad wrapper to allow smart updates of
// groups of pads. Uses THashList instead of TList for faster removal
// of objects from the pad.

ClassImp(TEvePad)

//______________________________________________________________________________
TEvePad::TEvePad()
{
   // Default constructor.

   fPrimitives = new THashList;
}

//______________________________________________________________________________
TEvePad::TEvePad(const char *name, const char *title, Double_t xlow,
                 Double_t ylow, Double_t xup, Double_t yup,
                 Color_t color, Short_t bordersize, Short_t bordermode)
   : TPad(name,title,xlow,ylow,xup,yup,color,bordersize,bordermode)
{
   // Constructor.

   delete fPrimitives;
   fPrimitives = new THashList;
}
