// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEvePad
#define ROOT_TEvePad

#include "TPad.h"

class TEvePad : public TPad
{
public:
   TEvePad();
   TEvePad(const char* name, const char* title,
           Double_t xlow, Double_t ylow, Double_t xup, Double_t yup,
           Color_t color = -1, Short_t bordersize = -1, Short_t bordermode = -2);
   ~TEvePad() override {}

   Bool_t    IsBatch() const override { return kTRUE; }

   void      Update() override { PaintModified(); }

   TVirtualViewer3D *GetViewer3D(Option_t * /*type*/ = "") override
   { return fViewer3D; }

   ClassDefOverride(TEvePad, 1); // Internal TEveUtil pad class (sub-class of TPad) overriding handling of updates and 3D-viewers.
};

#endif
