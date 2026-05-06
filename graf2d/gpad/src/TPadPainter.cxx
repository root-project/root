// @(#)root/gl:$Id$
// Author:  Olivier Couet, Timur Pocheptsov (vertex merge)  06/05/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "TPadPainter.h"
#include "TVirtualX.h"
#include "TCanvas.h"
#include "TPoint.h"
#include "TError.h"
#include "TImage.h"
#include "TROOT.h"
#include "TMath.h"
#include "TPad.h"

namespace {

//Typedef is fine, but let's pretend we look cool and modern:
using size_type = std::vector<TPoint>::size_type;

template<typename T>
void ConvertPoints(TVirtualPad *pad, unsigned nPoints, const T *xs, const T *ys,
                   std::vector<TPoint> &dst);
inline
void MergePointsX(std::vector<TPoint> &points, unsigned nMerged, SCoord_t yMin,
                  SCoord_t yMax, SCoord_t yLast);

inline
size_type MergePointsInplaceY(std::vector<TPoint> &dst, size_type nMerged, SCoord_t xMin,
                              SCoord_t xMax, SCoord_t xLast, size_type first);

template<typename T>
void ConvertPointsAndMergePassX(TVirtualPad *pad, unsigned nPoints, const T *x, const T *y,
                                std::vector<TPoint> &dst);

void ConvertPointsAndMergeInplacePassY(std::vector<TPoint> &dst);

template<class T>
void DrawFillAreaAux(TVirtualPad *pad, WinContext_t cont, Int_t nPoints, const T *xs, const T *ys, Bool_t add_first_point);

template<typename T>
void DrawPolyLineAux(TVirtualPad *pad, WinContext_t cont, unsigned nPoints, const T *xs, const T *ys);

template<class T>
void DrawPolyMarkerAux(TVirtualPad *pad, WinContext_t cont, Bool_t double_buffer, unsigned nPoints, const T *xs, const T *ys);


}


/** \class TPadPainter
\ingroup gpad

Implement TVirtualPadPainter which abstracts painting operations.
*/

////////////////////////////////////////////////////////////////////////////////
///Empty ctor. We need it only because of explicit copy ctor.

TPadPainter::TPadPainter()
{
   fDoubleBuffer = 1;
   fWinContext = (WinContext_t) 0;
}

/*
Line/fill/etc. attributes can be set inside TPad, but not only where:
many of them are set by base sub-objects of 2d primitives
(2d primitives usually inherit TAttLine or TAttFill etc.).  And these sub-objects
call gVirtualX->SetLineWidth ... etc. So, if I save some attributes in my painter,
it will be mess - at any moment I do not know, where to take line attribute - from
gVirtualX or from my own member. So! All attributed, _ALL_ go to/from gVirtualX.
*/


////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualX.

void TPadPainter::SetOpacity(Int_t percent)
{
   gVirtualX->SetOpacityW(fWinContext, percent);
}

////////////////////////////////////////////////////////////////////////////////
/// Delegate to gVirtualX.

Float_t TPadPainter::GetTextMagnitude() const
{
   return gVirtualX->GetTextMagnitude();
}

////////////////////////////////////////////////////////////////////////////////
/// Create a gVirtualX Pixmap.

Int_t TPadPainter::CreateDrawable(UInt_t w, UInt_t h)
{
   return gVirtualX->OpenPixmap(Int_t(w), Int_t(h));
}

////////////////////////////////////////////////////////////////////////////////
/// Resize a gVirtualX Pixmap.

Int_t TPadPainter::ResizeDrawable(Int_t device, UInt_t w, UInt_t h)
{
   return gVirtualX->ResizePixmap(device, w, h);
}


////////////////////////////////////////////////////////////////////////////////
/// Returns true when cocoa backend is used

Bool_t TPadPainter::IsCocoa() const
{
   return gVirtualX->InheritsFrom("TGCocoa");
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if trasnparent colors are supported

Bool_t TPadPainter::IsSupportAlpha() const
{
   return gVirtualX->InheritsFrom("TGQuartz");
}

////////////////////////////////////////////////////////////////////////////////
/// Clear the current gVirtualX window - calling gVirtualX->ClearWindowW

void TPadPainter::ClearDrawable()
{
   if (fWinContext)
      gVirtualX->ClearWindowW(fWinContext);
}

////////////////////////////////////////////////////////////////////////////////
/// Clear specified window - calling gVirtualX->ClearWindowW

void TPadPainter::ClearWindow(Int_t device)
{
   auto ctxt = gVirtualX->GetWindowContext(device);
   if (ctxt)
      gVirtualX->ClearWindowW(ctxt);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy a gVirtualX pixmap.

void TPadPainter::CopyDrawable(Int_t device, Int_t px, Int_t py)
{
   gVirtualX->CopyPixmapW(fWinContext, device, px, py);
}

////////////////////////////////////////////////////////////////////////////////
/// Close the current gVirtualX pixmap.

void TPadPainter::DestroyDrawable(Int_t device)
{
   gVirtualX->SelectWindow(device);
   gVirtualX->ClosePixmap();
   fWinContext = (WinContext_t) 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Select the window in which the graphics will go.

void TPadPainter::SelectDrawable(Int_t device)
{
   gVirtualX->SelectWindow(device);
   fWinContext = gVirtualX->GetWindowContext(device);
}

////////////////////////////////////////////////////////////////////////////////
/// Call low-level update of selected drawable, redirect to gVirtualX.

void TPadPainter::UpdateDrawable(Int_t mode)
{
   gVirtualX->UpdateWindowW(fWinContext, mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Set drawing mode for specified device

void TPadPainter::SetDrawMode(Int_t device, Int_t mode)
{
   gVirtualX->SetDrawModeW(gVirtualX->GetWindowContext(device), (TVirtualX::EDrawMode) mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Set double buffer mode for specified device

void TPadPainter::SetDoubleBuffer(Int_t device, Int_t mode)
{
   // important flag - when disabled canvas pixmap used directly
   // so one need to use absolute coordinates
   fDoubleBuffer = mode;

   gVirtualX->SetDoubleBuffer(device, mode);
}

////////////////////////////////////////////////////////////////////////////////
///Noop, for non-gl pad TASImage calls gVirtualX->CopyArea.

void TPadPainter::DrawPixels(const unsigned char * /*pixelData*/, UInt_t /*width*/, UInt_t /*height*/,
                             Int_t /*dstX*/, Int_t /*dstY*/, Bool_t /*enableAlphaBlending*/)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Set fill attributes

void TPadPainter::SetAttFill(const TAttFill &att)
{
   TPadPainterBase::SetAttFill(att);

   TAttFill fill = GetAttFillInternal(IsCocoa());

   gVirtualX->SetAttFill(fWinContext, fill);
}

////////////////////////////////////////////////////////////////////////////////
/// Set line attributes

void TPadPainter::SetAttLine(const TAttLine &att)
{
   TPadPainterBase::SetAttLine(att);

   gVirtualX->SetAttLine(fWinContext, att);
}

////////////////////////////////////////////////////////////////////////////////
/// Set marker attributes

void TPadPainter::SetAttMarker(const TAttMarker &att)
{
   TPadPainterBase::SetAttMarker(att);

   gVirtualX->SetAttMarker(fWinContext, att);
}

////////////////////////////////////////////////////////////////////////////////
/// Set text attributes

void TPadPainter::SetAttText(const TAttText &att)
{
   TPadPainterBase::SetAttText(att);

   // TODO: in ROOT7 move text size handling directly to correspondent PS engine
   //       One not need to recalculate text size many time back and forth

   if (!fPad)
      Fatal("SetAttText", "Pad not specified");

   TAttText attm(att);
   attm.SetTextSize(att.GetTextSizePixels(*fPad));

   gVirtualX->SetAttText(fWinContext, attm);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint a simple line.

void TPadPainter::DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   if (fAttLine.GetLineWidth() <= 0)
      return;

   const Int_t px1 = fDoubleBuffer ? gPad->XtoPixel(x1) : gPad->XtoAbsPixel(x1);
   const Int_t px2 = fDoubleBuffer ? gPad->XtoPixel(x2) : gPad->XtoAbsPixel(x2);
   const Int_t py1 = fDoubleBuffer ? gPad->YtoPixel(y1) : gPad->YtoAbsPixel(y1);
   const Int_t py2 = fDoubleBuffer ? gPad->YtoPixel(y2) : gPad->YtoAbsPixel(y2);
   gVirtualX->DrawLineW(fWinContext, px1, py1, px2, py2);
}


////////////////////////////////////////////////////////////////////////////////
/// Paint a simple line in normalized coordinates.

void TPadPainter::DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2)
{
   if (fAttLine.GetLineWidth() <= 0)
      return;

   const Int_t px1 = fDoubleBuffer ? gPad->UtoPixel(u1) : gPad->UtoAbsPixel(u1);
   const Int_t py1 = fDoubleBuffer ? gPad->VtoPixel(v1) : gPad->VtoAbsPixel(v1);
   const Int_t px2 = fDoubleBuffer ? gPad->UtoPixel(u2) : gPad->UtoAbsPixel(u2);
   const Int_t py2 = fDoubleBuffer ? gPad->VtoPixel(v2) : gPad->VtoAbsPixel(v2);
   gVirtualX->DrawLineW(fWinContext, px1, py1, px2, py2);
}


////////////////////////////////////////////////////////////////////////////////
/// Paint a simple box.

void TPadPainter::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, EBoxMode mode)
{
   if (fAttLine.GetLineWidth() <= 0 && mode == TVirtualPadPainter::kHollow)
      return;

   if (fFullyTransparent && mode == TVirtualPadPainter::kFilled)
      return;

   Int_t px1 = fDoubleBuffer ? gPad->XtoPixel(x1) : gPad->XtoAbsPixel(x1);
   Int_t px2 = fDoubleBuffer ? gPad->XtoPixel(x2) : gPad->XtoAbsPixel(x2);
   Int_t py1 = fDoubleBuffer ? gPad->YtoPixel(y1) : gPad->YtoAbsPixel(y1);
   Int_t py2 = fDoubleBuffer ? gPad->YtoPixel(y2) : gPad->YtoAbsPixel(y2);

   // Box width must be at least one pixel (WTF is this code???)
   if (TMath::Abs(px2 - px1) < 1)
      px2 = px1 + 1;
   if (TMath::Abs(py1 - py2) < 1)
      py1 = py2 + 1;

   gVirtualX->DrawBoxW(fWinContext, px1, py1, px2, py2, (TVirtualX::EBoxMode)mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint filled area.

void TPadPainter::DrawFillArea(Int_t nPoints, const Double_t *xs, const Double_t *ys)
{
   if (nPoints < 3) {
      ::Error("TPadPainter::DrawFillArea", "invalid number of points %d", nPoints);
      return;
   }

   // if fully transparent, add first point to draw line
   DrawFillAreaAux(gPad, fWinContext, nPoints, xs, ys, fFullyTransparent);
}


////////////////////////////////////////////////////////////////////////////////
/// Paint filled area.

void TPadPainter::DrawFillArea(Int_t nPoints, const Float_t *xs, const Float_t *ys)
{
   if (nPoints < 3) {
      ::Error("TPadPainter::DrawFillArea", "invalid number of points %d", nPoints);
      return;
   }

   // if fully transparent, add first point to draw line
   DrawFillAreaAux(gPad, fWinContext, nPoints, xs, ys, fFullyTransparent);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint Polyline.

void TPadPainter::DrawPolyLine(Int_t n, const Double_t *xs, const Double_t *ys)
{
   if (fAttLine.GetLineWidth() <= 0)
      return;

   if (n < 2) {
      ::Error("TPadPainter::DrawPolyLine", "invalid number of points");
      return;
   }

   DrawPolyLineAux(gPad, fWinContext, n, xs, ys);
}


////////////////////////////////////////////////////////////////////////////////
/// Paint polyline.

void TPadPainter::DrawPolyLine(Int_t n, const Float_t *xs, const Float_t *ys)
{
   if (fAttLine.GetLineWidth() <= 0)
      return;

   if (n < 2) {
      ::Error("TPadPainter::DrawPolyLine", "invalid number of points");
      return;
   }

   DrawPolyLineAux(gPad, fWinContext, n, xs, ys);
}


////////////////////////////////////////////////////////////////////////////////
/// Paint polyline in normalized coordinates.

void TPadPainter::DrawPolyLineNDC(Int_t n, const Double_t *u, const Double_t *v)
{
   if (fAttLine.GetLineWidth() <= 0)
      return;

   if (n < 2) {
      ::Error("TPadPainter::DrawPolyLineNDC", "invalid number of points %d", n);
      return;
   }

   std::vector<TPoint> xy(n);

   for (Int_t i = 0; i < n; ++i) {
      xy[i].fX = (SCoord_t)gPad->UtoPixel(u[i]);
      xy[i].fY = (SCoord_t)gPad->VtoPixel(v[i]);
   }

   gVirtualX->DrawPolyLineW(fWinContext, n, &xy[0]);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint N segments on the pad

void TPadPainter::DrawSegments(Int_t n, Double_t *x, Double_t *y)
{
   if (fAttLine.GetLineWidth() <= 0)
      return;

   if (n < 1) {
      ::Error("TPadPainter::DrawSegments", "invalid number of segments %d", n);
      return;
   }

   std::vector<TPoint> xy(n*2);
   Int_t cnt = 0;
   for (Int_t i = 0; i < n*2; ++i) {
      if ((i % 2 == 0) && (x[i] == x[i+1]) && (y[i] == y[i+1])) {
         // exclude empty segment
         i++;
         continue;
      }

      xy[cnt].fX = (SCoord_t)gPad->XtoPixel(x[i]);
      xy[cnt].fY = (SCoord_t)gPad->YtoPixel(y[i]);
      cnt++;
   }

   if (cnt > 1)
      gVirtualX->DrawLinesSegmentsW(fWinContext, cnt/2, &xy[0]);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint N segments in normalized coordinates on the pad

void TPadPainter::DrawSegmentsNDC(Int_t n, Double_t *u, Double_t *v)
{
   if (fAttLine.GetLineWidth() <= 0)
      return;

   if (n < 1) {
      ::Error("TPadPainter::DrawSegmentsNDC", "invalid number of segments %d", n);
      return;
   }

   std::vector<TPoint> xy(n*2);
   Int_t cnt = 0;
   for (Int_t i = 0; i < n*2; ++i) {
      if ((i % 2 == 0) && (u[i] == u[i+1]) && (v[i] == v[i+1])) {
         // exclude empty segment
         i++;
         continue;
      }

      xy[cnt].fX = (SCoord_t)gPad->UtoPixel(u[i]);
      xy[cnt].fY = (SCoord_t)gPad->VtoPixel(v[i]);
      cnt++;
   }

   if (cnt > 1)
      gVirtualX->DrawLinesSegmentsW(fWinContext, cnt/2, &xy[0]);
}



////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker.

void TPadPainter::DrawPolyMarker(Int_t n, const Double_t *x, const Double_t *y)
{
   if (n < 1) {
      ::Error("TPadPainter::DrawPolyMarker", "invalid number of points %d", n);
      return;
   }

   DrawPolyMarkerAux(gPad, fWinContext, fDoubleBuffer, n, x, y);
}


////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker.

void TPadPainter::DrawPolyMarker(Int_t n, const Float_t *x, const Float_t *y)
{
   if (n < 1) {
      ::Error("TPadPainter::DrawPolyMarker", "invalid number of points %d", n);
      return;
   }

   DrawPolyMarkerAux(gPad, fWinContext, fDoubleBuffer, n, x, y);
}


////////////////////////////////////////////////////////////////////////////////
/// Paint text.

void TPadPainter::DrawText(Double_t x, Double_t y, const char *text, ETextMode mode)
{
   const Int_t px = gPad->XtoPixel(x);
   const Int_t py = gPad->YtoPixel(y);
   const Double_t angle = GetTextAngle();
   const Double_t mgn = GetTextMagnitude();
   gVirtualX->DrawTextW(fWinContext, px, py, angle, mgn, text, (TVirtualX::ETextMode)mode);
}


////////////////////////////////////////////////////////////////////////////////
/// Special version working with wchar_t and required by TMathText.

void TPadPainter::DrawText(Double_t x, Double_t y, const wchar_t *text, ETextMode mode)
{
   const Int_t px = gPad->XtoPixel(x);
   const Int_t py = gPad->YtoPixel(y);
   const Double_t angle = GetTextAngle();
   const Double_t mgn = GetTextMagnitude();
   gVirtualX->DrawTextW(fWinContext, px, py, angle, mgn, text, (TVirtualX::ETextMode)mode);
}


////////////////////////////////////////////////////////////////////////////////
/// Paint text in normalized coordinates.

void TPadPainter::DrawTextNDC(Double_t u, Double_t v, const char *text, ETextMode mode)
{
   const Int_t px = gPad->UtoPixel(u);
   const Int_t py = gPad->VtoPixel(v);
   const Double_t angle = GetTextAngle();
   const Double_t mgn = GetTextMagnitude();
   gVirtualX->DrawTextW(fWinContext, px, py, angle, mgn, text, (TVirtualX::ETextMode)mode);
}


////////////////////////////////////////////////////////////////////////////////
/// Save the image displayed in the canvas pointed by "pad" into a binary file.

void TPadPainter::SaveImage(TVirtualPad *pad, const char *fileName, Int_t type) const
{
   if (gVirtualX->InheritsFrom("TGCocoa") && !gROOT->IsBatch() &&
      pad->GetCanvas() && pad->GetCanvas()->GetCanvasID() != -1) {

      TCanvas * const canvas = pad->GetCanvas();
      //Force TCanvas::CopyPixmaps.
      canvas->Flush();

      const UInt_t w = canvas->GetWw();
      const UInt_t h = canvas->GetWh();

      const std::unique_ptr<unsigned char[]>
               pixelData(gVirtualX->GetColorBits(canvas->GetCanvasID(), 0, 0, w, h));

      if (pixelData.get()) {
         const std::unique_ptr<TImage> image(TImage::Create());
         if (image.get()) {
            image->DrawRectangle(0, 0, w, h);
            if (unsigned char *argb = (unsigned char *)image->GetArgbArray()) {
               //Ohhh.
               if (sizeof(UInt_t) == 4) {
                  //For sure the data returned from TGCocoa::GetColorBits,
                  //it's 4 * w * h bytes with what TASImage considers to be argb.
                  std::copy(pixelData.get(), pixelData.get() + 4 * w * h, argb);
               } else {
                  //A bit paranoid, don't you think so?
                  //Will Quartz/TASImage work at all on such a fancy platform? ;)
                  const unsigned shift = std::numeric_limits<unsigned char>::digits;
                  //
                  unsigned *dstPixel = (unsigned *)argb, *end = dstPixel + w * h;
                  const unsigned char *srcPixel = pixelData.get();
                  for (;dstPixel != end; ++dstPixel, srcPixel += 4) {
                     //Looks fishy but should work, trust me :)
                     *dstPixel = srcPixel[0] & (srcPixel[1] << shift) &
                                               (srcPixel[2] << 2 * shift) &
                                               (srcPixel[3] << 3 * shift);
                  }
               }

               image->WriteImage(fileName, (TImage::EImageFileTypes)type);
               //Success.
               return;
            }
         }
      }
   }

   if (type == TImage::kGif) {
      Int_t wid = (pad == pad->GetCanvas()) ? pad->GetCanvasID() : pad->GetPixmapID();
      auto ctxt = gVirtualX->GetWindowContext(wid);
      // TODO: if fail, one can use TImage functionality instead
      gVirtualX->WriteGIFW(ctxt, fileName);
   } else {
      const std::unique_ptr<TImage> img(TImage::Create());
      if (img.get()) {
         img->FromPad(pad);
         img->WriteImage(fileName, (TImage::EImageFileTypes)type);
      }
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Paint text in normalized coordinates.

void TPadPainter::DrawTextNDC(Double_t u, Double_t v, const wchar_t *text, ETextMode mode)
{
   const Int_t px = gPad->UtoPixel(u);
   const Int_t py = gPad->VtoPixel(v);
   const Double_t angle = GetTextAngle();
   const Double_t mgn = GetTextMagnitude();
   gVirtualX->DrawTextW(fWinContext, px, py, angle, mgn, text, (TVirtualX::ETextMode)mode);
}

//Aux. private functions.
namespace {

////////////////////////////////////////////////////////////////////////////////
///I'm using 'pad' pointer to get rid of this damned gPad.
///Unfortunately, TPadPainter itself still has to use it.
///But at least this code does not have to be fixed.

template<typename T>
void ConvertPoints(TVirtualPad *pad, unsigned nPoints, const T *x, const T *y,
                   std::vector<TPoint> &dst)
{
   if (!nPoints)
      return;

   dst.resize(nPoints);

   for (unsigned i = 0; i < nPoints; ++i) {
      dst[i].fX = (SCoord_t)pad->XtoPixel(x[i]);
      dst[i].fY = (SCoord_t)pad->YtoPixel(y[i]);
   }
}

////////////////////////////////////////////////////////////////////////////////

inline void MergePointsX(std::vector<TPoint> &points, unsigned nMerged, SCoord_t yMin,
                         SCoord_t yMax, SCoord_t yLast)
{
   const auto firstPointX = points.back().fX;
   const auto firstPointY = points.back().fY;

   if (nMerged == 2) {
      points.push_back(TPoint(firstPointX, yLast));//We have not merge anything.
   } else if (nMerged == 3) {
      yMin == firstPointY ? points.push_back(TPoint(firstPointX, yMax)) :
                            points.push_back(TPoint(firstPointX, yMin));
      points.push_back(TPoint(firstPointX, yLast));
   } else {
      points.push_back(TPoint(firstPointX, yMin));
      points.push_back(TPoint(firstPointX, yMax));
      points.push_back(TPoint(firstPointX, yLast));
   }
}

////////////////////////////////////////////////////////////////////////////////
///Indices below are _valid_.

inline size_type MergePointsInplaceY(std::vector<TPoint> &dst, size_type nMerged, SCoord_t xMin,
                                     SCoord_t xMax, SCoord_t xLast, size_type first)
{
   const TPoint &firstPoint = dst[first];//This point is never updated.

   if (nMerged == 2) {
      dst[first + 1].fX = xLast;
      dst[first + 1].fY = firstPoint.fY;
   } else if (nMerged == 3) {
      dst[first + 1].fX = xMin == firstPoint.fX ? xMax : xMin;
      dst[first + 1].fY = firstPoint.fY;
      dst[first + 2].fX = xLast;
      dst[first + 2].fY = firstPoint.fY;
   } else {
      dst[first + 1].fX = xMin;
      dst[first + 1].fY = firstPoint.fY;
      dst[first + 2].fX = xMax;
      dst[first + 2].fY = firstPoint.fY;
      dst[first + 3].fX = xLast;
      dst[first + 3].fY = firstPoint.fY;
      nMerged = 4;//Adjust the shift.
   }

   return nMerged;
}

////////////////////////////////////////////////////////////////////////////////
/// I'm using 'pad' pointer to get rid of this damned gPad.
/// Unfortunately, TPadPainter itself still has to use it.
/// But at least this code does not have to be fixed.

template<typename T>
void ConvertPointsAndMergePassX(TVirtualPad *pad, unsigned nPoints, const T *x, const T *y,
                                std::vector<TPoint> &dst)
{
   //The "first" pass along X axis.
   TPoint currentPoint;
   SCoord_t yMin = 0, yMax = 0, yLast = 0;
   unsigned nMerged = 0;

   //The first pass along X.
   for (unsigned i = 0; i < nPoints;) {
      currentPoint.fX = (SCoord_t)pad->XtoPixel(x[i]);
      currentPoint.fY = (SCoord_t)pad->YtoPixel(y[i]);

      yMin = currentPoint.fY;
      yMax = yMin;

      dst.push_back(currentPoint);
      bool merged = false;
      nMerged = 1;

      for (unsigned j = i + 1; j < nPoints; ++j) {
         const SCoord_t newX = pad->XtoPixel(x[j]);

         if (newX == currentPoint.fX) {
            yLast = pad->YtoPixel(y[j]);
            yMin = TMath::Min(yMin, yLast);
            yMax = TMath::Max(yMax, yLast);//We continue.
            ++nMerged;
         } else {
            if (nMerged > 1)
               MergePointsX(dst, nMerged, yMin, yMax, yLast);
            merged = true;
            break;
         }
      }

      if (!merged && nMerged > 1)
         MergePointsX(dst, nMerged, yMin, yMax, yLast);

      i += nMerged;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// This pass is a bit more complicated, since we have to 'compact' in-place.

void ConvertPointsAndMergeInplacePassY(std::vector<TPoint> &dst)
{
   size_type i = 0;
   for (size_type j = 1, nPoints = dst.size(); i < nPoints;) {
      //i is always less than j, so i is always valid here.
      const TPoint &currentPoint = dst[i];

      SCoord_t xMin = currentPoint.fX;
      SCoord_t xMax = xMin;
      SCoord_t xLast = 0;

      bool merged = false;
      size_type nMerged = 1;

      for (; j < nPoints; ++j) {
         const TPoint &nextPoint = dst[j];

         if (nextPoint.fY == currentPoint.fY) {
            xLast = nextPoint.fX;
            xMin = TMath::Min(xMin, xLast);
            xMax = TMath::Max(xMax, xLast);
            ++nMerged;//and we continue ...
         } else {
            if (nMerged > 1)
               nMerged = MergePointsInplaceY(dst, nMerged, xMin, xMax, xLast, i);
            merged = true;
            break;
         }
      }

      if (!merged && nMerged > 1)
         nMerged = MergePointsInplaceY(dst, nMerged, xMin, xMax, xLast, i);

      i += nMerged;

      if (j < nPoints) {
         dst[i] = dst[j];
         ++j;
      } else
        break;
   }

   dst.resize(i);
}

////////////////////////////////////////////////////////////////////////////////
/// This is a quite simple algorithm, using the fact, that after conversion many
/// subsequent vertices can have the same 'x' or 'y' coordinate and this part of
/// a polygon will look like a line on the screen.

template<typename T>
void ConvertPointsAndMerge(TVirtualPad *pad, unsigned threshold, unsigned nPoints, const T *x,
                           const T *y, std::vector<TPoint> &dst)
{
   //I'm using 'pad' pointer to get rid of this damned gPad.
   //Unfortunately, TPadPainter itself still has to use it.
   //But at least this code does not have to be fixed.

   if (!nPoints)
      return;

   dst.clear();
   dst.reserve(threshold);

   ConvertPointsAndMergePassX(pad, nPoints, x, y, dst);

   if (dst.size() < threshold)
      return;

   ConvertPointsAndMergeInplacePassY(dst);
}

////////////////////////////////////////////////////////////////////////////////

template<class T>
void DrawFillAreaAux(TVirtualPad *pad, WinContext_t cont, Int_t nPoints, const T *xs, const T *ys, Bool_t add_first_point)
{
   std::vector<TPoint> xy;

   const Int_t threshold = Int_t(TMath::Min(pad->GetWw() * pad->GetAbsWNDC(),
                                 pad->GetWh() * pad->GetAbsHNDC())) * 2;

   if (threshold <= 0) {
      //Ooops, pad is invisible or something really bad and stupid happened.
      ::Error("DrawFillAreaAux", "invalid pad's geometry");
      return;
   }

   if (nPoints < threshold)
      ConvertPoints(pad, nPoints, xs, ys, xy);
   else
      ConvertPointsAndMerge(pad, threshold, nPoints, xs, ys, xy);

   //We close the 'polygon' so it can be rendered as a polyline by gVirtualX.
   if (add_first_point)
      xy.push_back(xy.front());

   if (xy.size() > 2)
      gVirtualX->DrawFillAreaW(cont, xy.size(), xy.data());
}

////////////////////////////////////////////////////////////////////////////////

template<typename T>
void DrawPolyLineAux(TVirtualPad *pad, WinContext_t cont, unsigned nPoints, const T *xs, const T *ys)
{
   std::vector<TPoint> xy;

   const Int_t threshold = Int_t(TMath::Min(pad->GetWw() * pad->GetAbsWNDC(),
                                            pad->GetWh() * pad->GetAbsHNDC())) * 2;

   if (threshold <= 0) {//Ooops, pad is invisible or something really bad and stupid happened.
      ::Error("DrawPolyLineAux", "invalid pad's geometry");
      return;
   }

   if (nPoints < (unsigned)threshold)
      ConvertPoints(pad, nPoints, xs, ys, xy);
   else
      ConvertPointsAndMerge(pad, threshold, nPoints, xs, ys, xy);

   if (xy.size() > 1)
      gVirtualX->DrawPolyLineW(cont, xy.size(), &xy[0]);

}

////////////////////////////////////////////////////////////////////////////////

template<class T>
void DrawPolyMarkerAux(TVirtualPad *pad, WinContext_t cont, Bool_t double_buffer, unsigned nPoints, const T *xs, const T *ys)
{
   std::vector<TPoint> xy(nPoints);

   for (unsigned i = 0; i < nPoints; ++i) {
      xy[i].fX = (SCoord_t) (double_buffer ? pad->XtoPixel(xs[i]) : pad->XtoAbsPixel(xs[i]));
      xy[i].fY = (SCoord_t) (double_buffer ? pad->YtoPixel(ys[i]) : pad->YtoAbsPixel(ys[i]));
   }

   gVirtualX->DrawPolyMarkerW(cont, nPoints, &xy[0]);
}

}
