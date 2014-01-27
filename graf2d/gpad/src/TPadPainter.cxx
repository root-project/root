// @(#)root/gl:$Id$
// Author:  Olivier Couet, Timur Pocheptsov(vertex merge)  06/05/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <cassert>
#include <vector>

#include "TPad.h"
#include "TPoint.h"
#include "TPadPainter.h"
#include "TVirtualX.h"
#include "TImage.h"
#include "TMath.h"

// Local scratch buffer for screen points, faster than allocating buffer on heap
const Int_t kPXY = 1002;
static TPoint gPXY[kPXY];


namespace {

//All asserts were commented, since ROOT's build system does not use -DNDEBUG for gpad module (with --build=release).

typedef std::vector<TPoint>::size_type size_type;

//______________________________________________________________________________
template<typename T>
void ConvertPoints(TVirtualPad *pad, unsigned nPoints, const T *x, const T *y, std::vector<TPoint> &dst)
{
   //I'm using 'pad' pointer to get rid of this damned gPad.
   //Unfortunately, TPadPainter itself still has to use it.
   //But at least this code does not have to be fixed.

   if (!nPoints)
      return;

   //assert(pad != 0 && "ConvertPoints, parameter 'pad' is null");
   //assert(x != 0 && "ConvertPoints, parameter 'x' is null");
   //assert(y != 0 && "ConvertPoints, parameter 'y' is null");

   dst.resize(nPoints);
   
   for (unsigned i = 0; i < nPoints; ++i) {
      dst[i].fX = pad->XtoPixel(x[i]);
      dst[i].fY = pad->YtoPixel(y[i]);
   }
}

//______________________________________________________________________________
inline void MergePointsX(std::vector<TPoint> &points, unsigned nMerged, SCoord_t yMin, SCoord_t yMax, SCoord_t yLast)
{
   //assert(points.size() != 0 && "MergePointsX, parameter 'points' is an empty vector, should contain at least 1 point already");
   //assert(nMerged > 1 && "MergePointsX, nothing to merge");
   
   const SCoord_t firstPointX = points.back().fX;
   const SCoord_t firstPointY = points.back().fY;
   
   if (nMerged == 2) {
      points.push_back(TPoint(firstPointX, yLast));//We have not merge anything.
   } else if (nMerged == 3) {
      yMin == firstPointY ? points.push_back(TPoint(firstPointX, yMax)) : points.push_back(TPoint(firstPointX, yMin));
      points.push_back(TPoint(firstPointX, yLast));
   } else {
      points.push_back(TPoint(firstPointX, yMin));
      points.push_back(TPoint(firstPointX, yMax));
      points.push_back(TPoint(firstPointX, yLast));
   }
}

//______________________________________________________________________________
inline size_type MergePointsInplaceY(std::vector<TPoint> &dst, size_type nMerged, SCoord_t xMin, SCoord_t xMax, SCoord_t xLast, size_type first)
{
   //assert(nMerged > 1 && "MergePointsInplaceY, nothing to merge");
   //assert(first < dst.size() && "MergePointsInplaceY, parameter 'first' is out of range");
   //assert(dst.size() - first >= nMerged && "MergePointsInplaceY, invalid index 'first + nMerged'");
   
   //Indices below are _valid_ - see asserts above.
   
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

//______________________________________________________________________________
template<typename T>
void ConvertPointsAndMergePassX(TVirtualPad *pad, unsigned nPoints, const T *x, const T *y, std::vector<TPoint> &dst)
{
   //I'm using 'pad' pointer to get rid of this damned gPad.
   //Unfortunately, TPadPainter itself still has to use it.
   //But at least this code does not have to be fixed.

   //assert(pad != 0 && "ConvertPointsAndMergePassX, parameter 'pad' is null");
   //assert(x != 0 && "ConvertPointsAndMergePassX, parameter 'x' is null");
   //assert(y != 0 && "ConvertPointsAndMergePassX, parameter 'y' is null");

   //The "first" pass along X axis.
   TPoint currentPoint;
   SCoord_t yMin = 0, yMax = 0, yLast = 0;
   unsigned nMerged = 0;
   
   //The first pass along X.
   for (unsigned i = 0; i < nPoints;) {
      currentPoint.fX = SCoord_t(pad->XtoPixel(x[i]));
      currentPoint.fY = SCoord_t(pad->YtoPixel(y[i]));
      
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

//______________________________________________________________________________
void ConvertPointsAndMergeInplacePassY(std::vector<TPoint> &dst)
{
   //assert(dst.size() != 0 && "ConvertPointsAndMergeInplacePassY, nothing to merge");

   //This pass is a bit more complicated, since we have
   //to 'compact' in-place.

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

//______________________________________________________________________________
template<typename T>
void ConvertPointsAndMerge(TVirtualPad *pad, unsigned threshold, unsigned nPoints, const T *x, const T *y, std::vector<TPoint> &dst)
{
   //This is a quite simple algorithm, using the fact, that after conversion many subsequent vertices
   //can have the same 'x' or 'y' coordinate and this part of a polygon will look like a line
   //on the screen.
   //Please NOTE: even if there are some problems (like invalid polygons), the algorithm can be
   //fixed (I'm not sure at the moment if it's important) and remembering the order of yMin/yMax or xMin/xMax (see
   //aux. functions above) - this should help if there any problems.

   //I'm using 'pad' pointer to get rid of this damned gPad.
   //Unfortunately, TPadPainter itself still has to use it.
   //But at least this code does not have to be fixed.

   if (!nPoints)
      return;

   //assert(pad != 0 && "ConvertPointsAndMerge, parameter 'pad' is null");
   //assert(threshold != 0 && "ConvertPointsAndMerge, parameter 'threshold' must be > 0");
   //assert(x != 0 && "ConvertPointsAndMerge, parameter 'x' is null");
   //assert(y != 0 && "ConvertPointsAndMerge, parameter 'y' is null");

   dst.clear();
   dst.reserve(threshold);

   ConvertPointsAndMergePassX(pad, nPoints, x, y, dst);
   
   if (dst.size() < threshold)
      return;

   ConvertPointsAndMergeInplacePassY(dst);
}

}

ClassImp(TPadPainter)


//______________________________________________________________________________
TPadPainter::TPadPainter()
{
   //Empty ctor. Here only because of explicit copy ctor.
}

/*
Line/fill/etc. attributes can be set inside TPad, but not only where:
many of them are set by base sub-objects of 2d primitives
(2d primitives usually inherit TAttLine or TAttFill etc.).  And these sub-objects
call gVirtualX->SetLineWidth ... etc. So, if I save some attributes in my painter,
it will be mess - at any moment I do not know, where to take line attribute - from
gVirtualX or from my own member. So! All attributed, _ALL_ go to/from gVirtualX.
*/


//______________________________________________________________________________
Color_t TPadPainter::GetLineColor() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetLineColor();
}


//______________________________________________________________________________
Style_t TPadPainter::GetLineStyle() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetLineStyle();
}


//______________________________________________________________________________
Width_t TPadPainter::GetLineWidth() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetLineWidth();
}


//______________________________________________________________________________
void TPadPainter::SetLineColor(Color_t lcolor)
{
   // Delegate to gVirtualX.

   gVirtualX->SetLineColor(lcolor);
}


//______________________________________________________________________________
void TPadPainter::SetLineStyle(Style_t lstyle)
{
   // Delegate to gVirtualX.

   gVirtualX->SetLineStyle(lstyle);
}


//______________________________________________________________________________
void TPadPainter::SetLineWidth(Width_t lwidth)
{
   // Delegate to gVirtualX.

   gVirtualX->SetLineWidth(lwidth);
}


//______________________________________________________________________________
Color_t TPadPainter::GetFillColor() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetFillColor();
}


//______________________________________________________________________________
Style_t TPadPainter::GetFillStyle() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetFillStyle();
}


//______________________________________________________________________________
Bool_t TPadPainter::IsTransparent() const
{
   // Delegate to gVirtualX.

   //IsTransparent is implemented as inline function in TAttFill.
   return gVirtualX->IsTransparent();
}


//______________________________________________________________________________
void TPadPainter::SetFillColor(Color_t fcolor)
{
   // Delegate to gVirtualX.

   gVirtualX->SetFillColor(fcolor);
}


//______________________________________________________________________________
void TPadPainter::SetFillStyle(Style_t fstyle)
{
   // Delegate to gVirtualX.

   gVirtualX->SetFillStyle(fstyle);
}


//______________________________________________________________________________
void TPadPainter::SetOpacity(Int_t percent)
{
   // Delegate to gVirtualX.

   gVirtualX->SetOpacity(percent);
}


//______________________________________________________________________________
Short_t TPadPainter::GetTextAlign() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetTextAlign();
}


//______________________________________________________________________________
Float_t TPadPainter::GetTextAngle() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetTextAngle();
}


//______________________________________________________________________________
Color_t TPadPainter::GetTextColor() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetTextColor();
}


//______________________________________________________________________________
Font_t TPadPainter::GetTextFont() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetTextFont();
}


//______________________________________________________________________________
Float_t TPadPainter::GetTextSize() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetTextSize();
}


//______________________________________________________________________________
Float_t TPadPainter::GetTextMagnitude() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetTextMagnitude();
}


//______________________________________________________________________________
void TPadPainter::SetTextAlign(Short_t align)
{
   // Delegate to gVirtualX.

   gVirtualX->SetTextAlign(align);
}


//______________________________________________________________________________
void TPadPainter::SetTextAngle(Float_t tangle)
{
   // Delegate to gVirtualX.

   gVirtualX->SetTextAngle(tangle);
}


//______________________________________________________________________________
void TPadPainter::SetTextColor(Color_t tcolor)
{
   // Delegate to gVirtualX.

   gVirtualX->SetTextColor(tcolor);
}


//______________________________________________________________________________
void TPadPainter::SetTextFont(Font_t tfont)
{
   // Delegate to gVirtualX.

   gVirtualX->SetTextFont(tfont);
}


//______________________________________________________________________________
void TPadPainter::SetTextSize(Float_t tsize)
{
   // Delegate to gVirtualX.
   
   gVirtualX->SetTextSize(tsize);
}


//______________________________________________________________________________
void TPadPainter::SetTextSizePixels(Int_t npixels)
{
   // Delegate to gVirtualX.

   gVirtualX->SetTextSizePixels(npixels);
}


//______________________________________________________________________________
Int_t TPadPainter::CreateDrawable(UInt_t w, UInt_t h)
{
   // Create a gVirtualX Pixmap.

   return gVirtualX->OpenPixmap(Int_t(w), Int_t(h));
}


//______________________________________________________________________________
void TPadPainter::ClearDrawable()
{
   // Clear the current gVirtualX window.

   gVirtualX->ClearWindow();
}


//______________________________________________________________________________
void TPadPainter::CopyDrawable(Int_t id, Int_t px, Int_t py)
{
   // Copy a gVirtualX pixmap.

   gVirtualX->CopyPixmap(id, px, py);
}


//______________________________________________________________________________
void TPadPainter::DestroyDrawable()
{
   // Close the current gVirtualX pixmap.

   gVirtualX->ClosePixmap();
}


//______________________________________________________________________________
void TPadPainter::SelectDrawable(Int_t device)
{
   // Select the window in which the graphics will go.

   gVirtualX->SelectWindow(device);
}


//______________________________________________________________________________
void TPadPainter::DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   // Paint a simple line.

   Int_t px1 = gPad->XtoPixel(x1);
   Int_t px2 = gPad->XtoPixel(x2);
   Int_t py1 = gPad->YtoPixel(y1);
   Int_t py2 = gPad->YtoPixel(y2);

   gVirtualX->DrawLine(px1, py1, px2, py2);
}


//______________________________________________________________________________
void TPadPainter::DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2)
{
   // Paint a simple line in normalized coordinates.

   Int_t px1 = gPad->UtoPixel(u1);
   Int_t py1 = gPad->VtoPixel(v1);
   Int_t px2 = gPad->UtoPixel(u2);
   Int_t py2 = gPad->VtoPixel(v2);
   gVirtualX->DrawLine(px1, py1, px2, py2);
}


//______________________________________________________________________________
void TPadPainter::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, EBoxMode mode)
{
   // Paint a simple box.

   Int_t px1 = gPad->XtoPixel(x1);
   Int_t px2 = gPad->XtoPixel(x2);
   Int_t py1 = gPad->YtoPixel(y1);
   Int_t py2 = gPad->YtoPixel(y2);

   // Box width must be at least one pixel
   if (TMath::Abs(px2-px1) < 1) px2 = px1+1;
   if (TMath::Abs(py1-py2) < 1) py1 = py2+1;

   gVirtualX->DrawBox(px1,py1,px2,py2,(TVirtualX::EBoxMode)mode);
}


//______________________________________________________________________________
void TPadPainter::DrawFillArea(Int_t n, const Double_t *x, const Double_t *y)
{
   // Paint filled area.
/*
   TPoint *pxy = &gPXY[0];
   if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   Int_t fillstyle = gVirtualX->GetFillStyle();
   for (Int_t i=0; i<n; i++) {
      pxy[i].fX = gPad->XtoPixel(x[i]);
      pxy[i].fY = gPad->YtoPixel(y[i]);
   }
   if (fillstyle == 0) {
      pxy[n].fX = pxy[0].fX;
      pxy[n].fY = pxy[0].fY;
      gVirtualX->DrawFillArea(n+1,pxy);
   } else {
      gVirtualX->DrawFillArea(n,pxy);
   }
   if (n >= kPXY) delete [] pxy;
*/
   if (n < 3)
      return;
   
   std::vector<TPoint> xy;
   
   const Int_t threshold = unsigned(TMath::Min(gPad->GetWw() * gPad->GetAbsWNDC(),
                                    gPad->GetWh() * gPad->GetAbsHNDC())) * 2;

   if (threshold <= 0)//Ooops, pad is invisible or something really bad and stupid happened.
      return;

   if (n < threshold)
      ConvertPoints(gPad, n, x, y, xy);
   else
      ConvertPointsAndMerge(gPad, threshold, n, x, y, xy);

   if (!gVirtualX->GetFillStyle())//We close the 'polygon' and it'll be rendered as a polyline by gVirtualX.
      xy.push_back(xy.front());
   
   if (xy.size() > 2)
      gVirtualX->DrawFillArea(xy.size(), &xy[0]);

}


//______________________________________________________________________________
void TPadPainter::DrawFillArea(Int_t n, const Float_t *x, const Float_t *y)
{
   // Paint filled area.
/*
   TPoint *pxy = &gPXY[0];
   if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   Int_t fillstyle = gVirtualX->GetFillStyle();
   for (Int_t i=0; i<n; i++) {
      pxy[i].fX = gPad->XtoPixel(x[i]);
      pxy[i].fY = gPad->YtoPixel(y[i]);
   }
   if (fillstyle == 0) {
      pxy[n].fX = pxy[0].fX;
      pxy[n].fY = pxy[0].fY;
      gVirtualX->DrawFillArea(n+1,pxy);
   } else {
      gVirtualX->DrawFillArea(n,pxy);
   }
   if (n >= kPXY) delete [] pxy;
*/
   // Paint filled area.
   if (n < 3)
      return;
   
   std::vector<TPoint> xy;

   const Int_t threshold = Int_t(TMath::Min(gPad->GetWw() * gPad->GetAbsWNDC(),
                                            gPad->GetWh() * gPad->GetAbsHNDC())) * 2;

   if (threshold <= 0)//Ooops, pad is invisible or something really bad and stupid happened.
      return;

   if (n < threshold)
      ConvertPoints(gPad, n, x, y, xy);
   else
      ConvertPointsAndMerge(gPad, threshold, n, x, y, xy);

   if (!gVirtualX->GetFillStyle())//We close the 'polygon' and it'll be rendered as a polyline by gVirtualX.
      xy.push_back(xy.front());
   
   if (xy.size() > 2)
      gVirtualX->DrawFillArea(xy.size(), &xy[0]);
}


//______________________________________________________________________________
void TPadPainter::DrawPolyLine(Int_t n, const Double_t *x, const Double_t *y)
{
   // Paint polyline.
/*
   TPoint *pxy = &gPXY[0];
   if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   for (Int_t i=0; i<n; i++) {
      pxy[i].fX = gPad->XtoPixel(x[i]);
      pxy[i].fY = gPad->YtoPixel(y[i]);
   }
   gVirtualX->DrawPolyLine(n,pxy);
   if (n >= kPXY) delete [] pxy;
*/
   std::vector<TPoint> xy;

   const Int_t threshold = Int_t(TMath::Min(gPad->GetWw() * gPad->GetAbsWNDC(),
                                            gPad->GetWh() * gPad->GetAbsHNDC())) * 2;

   if (threshold <= 0)//Ooops, pad is invisible or something really bad and stupid happened.
      return;

   if (n < threshold)
      ConvertPoints(gPad, n, x, y, xy);
   else
      ConvertPointsAndMerge(gPad, threshold, n, x, y, xy);

   if (xy.size() > 1)
      gVirtualX->DrawPolyLine(xy.size(), &xy[0]);
}


//______________________________________________________________________________
void TPadPainter::DrawPolyLine(Int_t n, const Float_t *x, const Float_t *y)
{
   // Paint polyline.
/*
   TPoint *pxy = &gPXY[0];
   if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   for (Int_t i=0; i<n; i++) {
      pxy[i].fX = gPad->XtoPixel(x[i]);
      pxy[i].fY = gPad->YtoPixel(y[i]);
   }
   gVirtualX->DrawPolyLine(n,pxy);
   if (n >= kPXY) delete [] pxy;*/
   std::vector<TPoint> xy;

   const Int_t threshold = Int_t(TMath::Min(gPad->GetWw() * gPad->GetAbsWNDC(),
                                            gPad->GetWh() * gPad->GetAbsHNDC())) * 2;

   if (threshold <= 0)//Ooops, pad is invisible or something really bad and stupid happened.
      return;

   if (n < threshold)
      ConvertPoints(gPad, n, x, y, xy);
   else
      ConvertPointsAndMerge(gPad, threshold, n, x, y, xy);

   if (xy.size() > 1)
      gVirtualX->DrawPolyLine(xy.size(), &xy[0]);
}


//______________________________________________________________________________
void TPadPainter::DrawPolyLineNDC(Int_t n, const Double_t *u, const Double_t *v)
{
   // Paint polyline in normalized coordinates.

   TPoint *pxy = &gPXY[0];
   if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   for (Int_t i=0; i<n; i++) {
      pxy[i].fX = gPad->UtoPixel(u[i]);
      pxy[i].fY = gPad->VtoPixel(v[i]);
   }
   gVirtualX->DrawPolyLine(n,pxy);
   if (n >= kPXY) delete [] pxy;
}


//______________________________________________________________________________
void TPadPainter::DrawPolyMarker(Int_t n, const Double_t *x, const Double_t *y)
{
   // Paint polymarker.

   TPoint *pxy = &gPXY[0];
   if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   for (Int_t i=0; i<n; i++) {
      pxy[i].fX = gPad->XtoPixel(x[i]);
      pxy[i].fY = gPad->YtoPixel(y[i]);
   }
   gVirtualX->DrawPolyMarker(n,pxy);
   if (n >= kPXY)   delete [] pxy;
}


//______________________________________________________________________________
void TPadPainter::DrawPolyMarker(Int_t n, const Float_t *x, const Float_t *y)
{
   // Paint polymarker.

   TPoint *pxy = &gPXY[0];
   if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   for (Int_t i=0; i<n; i++) {
      pxy[i].fX = gPad->XtoPixel(x[i]);
      pxy[i].fY = gPad->YtoPixel(y[i]);
   }
   gVirtualX->DrawPolyMarker(n,pxy);
   if (n >= kPXY)   delete [] pxy;
}


//______________________________________________________________________________
void TPadPainter::DrawText(Double_t x, Double_t y, const char *text, ETextMode mode)
{
   // Paint text.

   Int_t px = gPad->XtoPixel(x);
   Int_t py = gPad->YtoPixel(y);
   Double_t angle = GetTextAngle();
   Double_t mgn = GetTextMagnitude();
   gVirtualX->DrawText(px, py, angle, mgn, text, (TVirtualX::ETextMode)mode);
}


//______________________________________________________________________________
void TPadPainter::DrawText(Double_t x, Double_t y, const wchar_t *text, ETextMode mode)
{
   // Paint text.

   Int_t px = gPad->XtoPixel(x);
   Int_t py = gPad->YtoPixel(y);
   Double_t angle = GetTextAngle();
   Double_t mgn = GetTextMagnitude();
   gVirtualX->DrawText(px, py, angle, mgn, text, (TVirtualX::ETextMode)mode);
}


//______________________________________________________________________________
void TPadPainter::DrawTextNDC(Double_t u, Double_t v, const char *text, ETextMode mode)
{
   // Paint text in normalized coordinates.

   Int_t px = gPad->UtoPixel(u);
   Int_t py = gPad->VtoPixel(v);
   Double_t angle = GetTextAngle();
   Double_t mgn = GetTextMagnitude();
   gVirtualX->DrawText(px, py, angle, mgn, text, (TVirtualX::ETextMode)mode);
}


//______________________________________________________________________________
void TPadPainter::SaveImage(TVirtualPad *pad, const char *fileName, Int_t type) const
{
   // Save the image displayed in the canvas pointed by "pad" into a 
   // binary file.

   if (type == TImage::kGif) {
      gVirtualX->WriteGIF((char*)fileName);
   } else {
      TImage *img = TImage::Create();
      if (img) {
         img->FromPad(pad);
         img->WriteImage(fileName, (TImage::EImageFileTypes)type);
         delete img;
      }
   }
}


//______________________________________________________________________________
void TPadPainter::DrawTextNDC(Double_t u, Double_t v, const wchar_t *text, ETextMode mode)
{
   // Paint text in normalized coordinates.

   Int_t px = gPad->UtoPixel(u);
   Int_t py = gPad->VtoPixel(v);
   Double_t angle = GetTextAngle();
   Double_t mgn = GetTextMagnitude();
   gVirtualX->DrawText(px, py, angle, mgn, text, (TVirtualX::ETextMode)mode);
}
