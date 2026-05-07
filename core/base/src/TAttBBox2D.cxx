// @(#)root/graf:$Id$
// Author: Anna-Pia Lohfink 27.3.2014

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TAttBBox2D.h"

#include "TPoint.h"
#include "TVirtualPad.h"


/** \class TAttBBox2D
\ingroup Base
\ingroup GraphicsAtt

Abstract base class for elements drawn in the editor.
Classes inheriting from TAttBBox2D implementing the TAttBBox2D
virtual methods, and using TPad::ShowGuideLines in ExecuteEvent
will automatically get the guide lines drawn when moved in the pad.
All methods work with pixel coordinates.
*/

////////////////////////////////////////////////////////////////////////////////
// TAttBBox2D destructor.

TAttBBox2D::~TAttBBox2D()
{
}

////////////////////////////////////////////////////////////////////////////////
// Returns BBox center
// By default returns center of rectangle returned by GetBBox() method

TPoint TAttBBox2D::GetBBoxCenter()
{
   auto box = GetBBox();
   return { (Short_t) (box.fX + box.fWidth/2), (Short_t) (box.fY + box.fHeight/2)};
}

////////////////////////////////////////////////////////////////////////////////
// Set BBox center
// By default calls SetBBoxCenterX and SetBBoxCenterY

void TAttBBox2D::SetBBoxCenter(const TPoint &p)
{
   SetBBoxCenterX(p.GetX());
   SetBBoxCenterY(p.GetY());
}

////////////////////////////////////////////////////////////////////////////////
/// Return user X coordinate for pixel X value
/// Can return ndc or normal values
/// Also one can specify to use absolute coordinates for input parameter x
/// Used in derived classes to implement SetBBox... methods

Double_t TAttBBox2D::GetXCoord(const Int_t x, Bool_t is_ndc, Bool_t is_absolute)
{
   if (!gPad)
      return 0.;

   if (!is_ndc)
      return gPad->PadtoX(is_absolute ? gPad->AbsPixeltoX(x) : gPad->PixeltoX(x));

   if (is_absolute) {
      Double_t ww = gPad->GetWw();
      Double_t wndc = gPad->GetAbsWNDC();
      return ww > 0 && wndc > 0 ? (x / ww - gPad->GetAbsXlowNDC()) / wndc : 0.;
   }
   Int_t pw = gPad->GetPadWidth();
   return pw > 0 ? 1. * x / pw : 0.;
}

////////////////////////////////////////////////////////////////////////////////
// Return user Y coordinate for pixel Y value
/// Can return ndc or normal values
/// Also one can specify to use absolute coordinates for input parameter x
// Used in derived classes to implement SetBBox... methods

Double_t TAttBBox2D::GetYCoord(const Int_t y, Bool_t is_ndc, Bool_t is_absolute)
{
   if (!gPad)
      return 0.;

   if (!is_ndc)
      return gPad->PadtoY(is_absolute ? gPad->AbsPixeltoY(y) : gPad->PixeltoY(y - gPad->VtoPixel(0)));

   if (is_absolute) {
      Double_t wh = gPad->GetWh();
      Double_t hndc = gPad->GetAbsHNDC();
      return wh > 0 && hndc > 0 ? ((1. - y / wh) - gPad->GetAbsYlowNDC()) / hndc : 0.;
   }

   Int_t ph = gPad->GetPadHeight();
   return ph > 0 ? 1. - 1. * y / ph : 0.;
}
