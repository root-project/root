// @(#)root/gpad:$Id$
// Author: Rene Brun   23/11/96

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TSlider.h"
#include "TSliderBox.h"
#include "TMath.h"

#include <cstring>



/** \class TSliderBox
\ingroup gpad

The moving box in a TSlider
*/

////////////////////////////////////////////////////////////////////////////////
/// SliderBox default constructor.

TSliderBox::TSliderBox(): TWbox()
{
   fSlider = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// SliderBox normal constructor.

TSliderBox::TSliderBox(Double_t x1, Double_t y1,Double_t x2, Double_t  y2, Color_t color, Short_t bordersize, Short_t bordermode)
           :TWbox(x1,y1,x2,y2,color,bordersize,bordermode)
{
   fSlider = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// SliderBox default destructor.

TSliderBox::~TSliderBox()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Interaction with a slider.

void TSliderBox::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   Bool_t vertical = fSlider ? fSlider->GetAbsWNDC() < fSlider->GetAbsHNDC() : kTRUE;

   TWbox::ExecuteEvent((vertical ? 20000 : 10000) + event, px, py);

   if (!fSlider)
      return;

   Int_t bordersize = fSlider->GetBorderSize();
   Double_t dx = fSlider->PixeltoX(bordersize);
   Double_t dy = fSlider->PixeltoY(-bordersize);
   Double_t v1, v2, d;

   // Give control to object using the slider
   if (vertical) {
      v1 = GetY1();
      v2 = GetY2();
      d = TMath::Min(0.3, dy);
   } else {
      v1 = GetX1();
      v2 = GetX2();
      d = TMath::Min(0.3, dx);
   }
   fSlider->SetMinimum(TMath::Max(0., (v1 - d) / (1 - 2*d)));
   fSlider->SetMaximum(TMath::Min(1., (v2 - d) / (1 - 2*d)));

   if (event == kButton1Up) {
      if (vertical) {
         SetX1(dx); SetX2(1 - dx);
      } else {
         SetY1(dy); SetY2(1 - dy);
      }
   }

   //A user method to execute?
   if (event == kButton1Up && (strlen(fSlider->GetMethod()) > 0)) {
      gPad->SetCursor(kWatch);
      gROOT->ProcessLine(fSlider->GetMethod());
      return;
   }

   //An object connected to this slider?
   TObject *obj = fSlider->GetObject();
   if (obj)
      obj->ExecuteEvent(event, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TSliderBox::SavePrimitive(std::ostream &, Option_t * /*= ""*/)
{
}
