// @(#)root/gl:$Id$
// Author:  Sergey Linev  04/03/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "TPadPainterPS.h"
#include "TVirtualPS.h"
#include "TVirtualX.h"
#include "TCanvas.h"
#include "TPoint.h"
#include "TError.h"
#include "TImage.h"
#include "TROOT.h"
#include "TMath.h"
#include "TPad.h"

/** \class TPadPainter
\ingroup gpad

Implement TVirtualPadPainter which abstracts painting operations.
*/

////////////////////////////////////////////////////////////////////////////////
/// Consructor
/// Assigns TVirtualPS instance which will be used by the painter

TPadPainterPS::TPadPainterPS(TVirtualPS *ps)
{
   fPS = ps;
}

/*
Line/fill/etc. attributes can be set inside TPad, but not only where:
many of them are set by base sub-objects of 2d primitives
(2d primitives usually inherit TAttLine or TAttFill etc.).  And these sub-objects
call gVirtualPS->SetLineWidth ... etc. So, if I save some attributes in my painter,
it will be mess - at any moment I do not know, where to take line attribute - from
gVirtualX or from my own member. So! All attributed, _ALL_ go to/from gVirtualPS.
*/


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

Color_t TPadPainterPS::GetLineColor() const
{
   return gVirtualPS->GetLineColor();
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

Style_t TPadPainterPS::GetLineStyle() const
{
   return gVirtualPS->GetLineStyle();
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

Width_t TPadPainterPS::GetLineWidth() const
{
   return gVirtualPS->GetLineWidth();
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

void TPadPainterPS::SetLineColor(Color_t lcolor)
{
   gVirtualPS->SetLineColor(lcolor);
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

void TPadPainterPS::SetLineStyle(Style_t lstyle)
{
   gVirtualPS->SetLineStyle(lstyle);
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

void TPadPainterPS::SetLineWidth(Width_t lwidth)
{
   gVirtualPS->SetLineWidth(lwidth);
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

Color_t TPadPainterPS::GetFillColor() const
{
   return gVirtualPS->GetFillColor();
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

Style_t TPadPainterPS::GetFillStyle() const
{
   return gVirtualPS->GetFillStyle();
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

Bool_t TPadPainterPS::IsTransparent() const
{
   //IsTransparent is implemented as inline function in TAttFill.
   return gVirtualPS->IsTransparent();
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

void TPadPainterPS::SetFillColor(Color_t fcolor)
{
   gVirtualPS->SetFillColor(fcolor);
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

void TPadPainterPS::SetFillStyle(Style_t fstyle)
{
   gVirtualPS->SetFillStyle(fstyle);
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

void TPadPainterPS::SetOpacity(Int_t percent)
{
   // FIXME!!!
//   gVirtualPS->SetOpacity(percent);
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

Short_t TPadPainterPS::GetTextAlign() const
{
   return gVirtualPS->GetTextAlign();
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

Float_t TPadPainterPS::GetTextAngle() const
{
   return gVirtualPS->GetTextAngle();
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

Color_t TPadPainterPS::GetTextColor() const
{
   return gVirtualPS->GetTextColor();
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

Font_t TPadPainterPS::GetTextFont() const
{
   return gVirtualPS->GetTextFont();
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

Float_t TPadPainterPS::GetTextSize() const
{
   return gVirtualPS->GetTextSize();
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

Float_t TPadPainterPS::GetTextMagnitude() const
{
   // FIXME!!!
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

void TPadPainterPS::SetTextAlign(Short_t align)
{
   gVirtualPS->SetTextAlign(align);
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

void TPadPainterPS::SetTextAngle(Float_t tangle)
{
   gVirtualPS->SetTextAngle(tangle);
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

void TPadPainterPS::SetTextColor(Color_t tcolor)
{
   gVirtualPS->SetTextColor(tcolor);
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

void TPadPainterPS::SetTextFont(Font_t tfont)
{
   gVirtualPS->SetTextFont(tfont);
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

void TPadPainterPS::SetTextSize(Float_t tsize)
{
   gVirtualPS->SetTextSize(tsize);
}


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualPS.

void TPadPainterPS::SetTextSizePixels(Int_t npixels)
{
   gVirtualPS->SetTextSizePixels(npixels);
}


////////////////////////////////////////////////////////////////////////////////
/// Create a gVirtualX Pixmap - not implemented

Int_t TPadPainterPS::CreateDrawable(UInt_t, UInt_t)
{
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Clear the current gVirtualX window - not implemented

void TPadPainterPS::ClearDrawable()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Copy a gVirtualX pixmap - not implemented

void TPadPainterPS::CopyDrawable(Int_t, Int_t, Int_t py)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Close the current gVirtualX pixmap - not implemented

void TPadPainterPS::DestroyDrawable(Int_t)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Select the window in which the graphics will go - not implemented

void TPadPainterPS::SelectDrawable(Int_t device)
{
}

////////////////////////////////////////////////////////////////////////////////
///Noop, for non-gl pad TASImage calls gVirtualX->CopyArea.

void TPadPainterPS::DrawPixels(const unsigned char * /*pixelData*/, UInt_t /*width*/, UInt_t /*height*/,
                             Int_t /*dstX*/, Int_t /*dstY*/, Bool_t /*enableAlphaBlending*/)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Paint a simple line.

void TPadPainterPS::DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   if (GetLineWidth() <= 0)
      return;
   Double_t x[2] = {x1, x2}, y[2] = {y1, y2};
   fPS->DrawPS(2, x, y);
}


////////////////////////////////////////////////////////////////////////////////
/// Paint a simple line in normalized coordinates.

void TPadPainterPS::DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2)
{
   if (GetLineWidth() <= 0)
      return;

   Double_t xw[2], yw[2];

   xw[0] = (1 - u1) * fPad->GetX1() + u1 * fPad->GetX2();
   xw[1] = (1 - u2) * fPad->GetX1() + u2 * fPad->GetX2();
   yw[0] = (1 - v1) * fPad->GetY1() + v1 * fPad->GetY2();
   yw[1] = (1 - v2) * fPad->GetY1() + v2 * fPad->GetY2();
   fPS->DrawPS(2, xw, yw);
}


////////////////////////////////////////////////////////////////////////////////
/// Paint a simple box.

void TPadPainterPS::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, EBoxMode mode)
{
   Int_t style0 = -1;

   if (mode == TVirtualPadPainter::kHollow) {
      if (GetLineWidth() <= 0)
         return;
      style0 = fPS->GetFillStyle();
      if (style0 > 0)
         fPS->SetFillStyle(0);
   }

   fPS->DrawBox(x1, y1, x2, y2);

   if (style0 > 0)
      fPS->SetFillStyle(style0);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint filled area.

void TPadPainterPS::DrawFillArea(Int_t nPoints, const Double_t *xs, const Double_t *ys)
{
   if (nPoints < 3) {
      ::Error("TPadPainterPS::DrawFillArea", "invalid number of points %d", nPoints);
      return;
   }

   fPS->DrawPS(-nPoints, const_cast<Double_t *>(xs), const_cast<Double_t *>(ys));
}


////////////////////////////////////////////////////////////////////////////////
/// Paint filled area.

void TPadPainterPS::DrawFillArea(Int_t nPoints, const Float_t *xs, const Float_t *ys)
{
   if (nPoints < 3) {
      ::Error("TPadPainterPS::DrawFillArea", "invalid number of points %d", nPoints);
      return;
   }

   fPS->DrawPS(-nPoints, const_cast<Float_t *>(xs), const_cast<Float_t *>(ys));
}

////////////////////////////////////////////////////////////////////////////////
/// Paint Polyline.

void TPadPainterPS::DrawPolyLine(Int_t n, const Double_t *xs, const Double_t *ys)
{
   if (GetLineWidth() <= 0)
      return;

   if (n < 2) {
      ::Error("TPadPainterPS::DrawPolyLine", "invalid number of points");
      return;
   }

   fPS->DrawPS(n, const_cast<Double_t *>(xs), const_cast<Double_t *>(ys));
}


////////////////////////////////////////////////////////////////////////////////
/// Paint polyline.

void TPadPainterPS::DrawPolyLine(Int_t n, const Float_t *xs, const Float_t *ys)
{
   if (GetLineWidth() <= 0)
      return;

   if (n < 2) {
      ::Error("TPadPainterPS::DrawPolyLine", "invalid number of points");
      return;
   }

   fPS->DrawPS(n, const_cast<Float_t *>(xs), const_cast<Float_t *>(ys));
}


////////////////////////////////////////////////////////////////////////////////
/// Paint polyline in normalized coordinates.

void TPadPainterPS::DrawPolyLineNDC(Int_t n, const Double_t *u, const Double_t *v)
{
   if (GetLineWidth() <= 0)
      return;

   if (n < 2) {
      ::Error("TPadPainterPS::DrawPolyLineNDC", "invalid number of points %d", n);
      return;
   }

   std::vector<Double_t> xw(n), yw(n);
   for (Int_t i = 0; i < n; i++) {
      xw[i] = (1 - u[i]) * fPad->GetX1() + u[i] * fPad->GetX2();
      yw[i] = (1 - v[i]) * fPad->GetY1() + v[i] * fPad->GetY2();
   }
   fPS->DrawPS(n, xw.data(), yw.data());
}

////////////////////////////////////////////////////////////////////////////////
/// Paint N segments on the pad

void TPadPainterPS::DrawSegments(Int_t n, Double_t *x, Double_t *y)
{
   if (GetLineWidth() <= 0)
      return;

   if (n < 1) {
      ::Error("TPadPainterPS::DrawSegments", "invalid number of segments %d", n);
      return;
   }

   fPS->DrawSegments(n, x, y);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint N segments in normalized coordinates on the pad

void TPadPainterPS::DrawSegmentsNDC(Int_t n, Double_t *u, Double_t *v)
{
   if (GetLineWidth() <= 0)
      return;

   if (n < 1) {
      ::Error("TPadPainterPS::DrawSegmentsNDC", "invalid number of segments %d", n);
      return;
   }
   // recalculate values into normal coordiantes
   for (Int_t i = 0; i < 2*n; i++) {
      u[i] = (1 - u[i]) * fPad->GetX1() + u[i] * fPad->GetX2();
      v[i] = (1 - v[i]) * fPad->GetY1() + v[i] * fPad->GetY2();
   }
   fPS->DrawSegments(n, u, v);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker.

void TPadPainterPS::DrawPolyMarker(Int_t n, const Double_t *x, const Double_t *y)
{
   if (n < 1) {
      ::Error("TPadPainterPS::DrawPolyMarker", "invalid number of points %d", n);
      return;
   }

   fPS->DrawPolyMarker(n, const_cast<Double_t *>(x), const_cast<Double_t *>(y));
}


////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker.

void TPadPainterPS::DrawPolyMarker(Int_t n, const Float_t *x, const Float_t *y)
{
   if (n < 1) {
      ::Error("TPadPainterPS::DrawPolyMarker", "invalid number of points %d", n);
      return;
   }

   fPS->DrawPolyMarker(n, const_cast<Float_t *>(x), const_cast<Float_t *>(y));
}


////////////////////////////////////////////////////////////////////////////////
/// Paint text.

void TPadPainterPS::DrawText(Double_t x, Double_t y, const char *text, ETextMode mode)
{
   const Int_t px = gPad->XtoPixel(x);
   const Int_t py = gPad->YtoPixel(y);
   const Double_t angle = GetTextAngle();
   const Double_t mgn = GetTextMagnitude();
}


////////////////////////////////////////////////////////////////////////////////
/// Special version working with wchar_t and required by TMathText.

void TPadPainterPS::DrawText(Double_t x, Double_t y, const wchar_t *text, ETextMode mode)
{
   const Int_t px = gPad->XtoPixel(x);
   const Int_t py = gPad->YtoPixel(y);
   const Double_t angle = GetTextAngle();
   const Double_t mgn = GetTextMagnitude();
}


////////////////////////////////////////////////////////////////////////////////
/// Paint text in normalized coordinates.

void TPadPainterPS::DrawTextNDC(Double_t u, Double_t v, const char *text, ETextMode mode)
{
   const Int_t px = gPad->UtoPixel(u);
   const Int_t py = gPad->VtoPixel(v);
   const Double_t angle = GetTextAngle();
   const Double_t mgn = GetTextMagnitude();
}


////////////////////////////////////////////////////////////////////////////////
/// Save the image displayed in the canvas pointed by "pad" into a binary file.

void TPadPainterPS::SaveImage(TVirtualPad *, const char *, Int_t) const
{
}


////////////////////////////////////////////////////////////////////////////////
/// Paint text in normalized coordinates.

void TPadPainterPS::DrawTextNDC(Double_t u, Double_t v, const wchar_t *text, ETextMode mode)
{
   const Int_t px = gPad->UtoPixel(u);
   const Int_t py = gPad->VtoPixel(v);
   const Double_t angle = GetTextAngle();
   const Double_t mgn = GetTextMagnitude();
}
