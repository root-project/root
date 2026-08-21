// @(#)root/postscript:$Id$
// Author: Valeriy Onuchin

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TImageDump
\ingroup PS

\brief Save canvas as an image (GIF, JPEG, PNG, XPM, TIFF etc.).

Example:
~~~ {.cpp}
         $ root -b
         root [0] .x hsimple.C
         root [1] c1->Print("c1.gif");
~~~
TImageDump can be used in any mode (batch, interactive) as follows
~~~ {.cpp}
         TCanvas *c1;
         TImageDump *imgdump = new TImageDump("test.png");
         c1->Paint();
         imgdump->Close();
~~~
*/

#include "TImageDump.h"
#include "TImage.h"
#include "TMath.h"
#include "TPoint.h"
#include "TColor.h"
#include "TVirtualPad.h"
#include "TVirtualX.h"
#include "TROOT.h"
#include "TText.h"
#include "RStipples.h"
#include "TList.h"
#include "TStyle.h"
#include "TObjString.h"
#include "TObjArray.h"



////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TImageDump::TImageDump() : TVirtualPS()
{
   gVirtualPS = this;
   SetTitle("IMG");
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize batch image interface
///
/// fname : image file name
///
/// The possible workstation types are:
///  - 111 - Portrait
///  - 112 - Landscape
///  - 114 - preview, keep in memory (do not write on delete)

TImageDump::TImageDump(const char *fname, Int_t wtype) : TVirtualPS(fname, wtype)
{
   Open(fname, wtype);
   gVirtualPS = this;
   SetTitle("IMG");
}

////////////////////////////////////////////////////////////////////////////////
/// Open a image file

void TImageDump::Open(const char *fname, Int_t type)
{
   fImage  = TImage::Create();
   fType   = type;
   SetName(fname);
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TImageDump::~TImageDump()
{
   Close();

   delete fImage;
   fImage = nullptr;

   gVirtualPS = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Close a image file

void TImageDump::Close(Option_t *)
{
   // if fType == 114 - do not write image
   if (!fImage || (fType == 114)) {
      return;
   }

   //if (fType == 112) fImage->Flip(90);
   fImage->WriteImage(GetName());
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a Box

void TImageDump::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t  y2)
{
   if (!gPad || !fImage)
      return;

   fImage->BeginPaint();

   Double_t x[4], y[4];
   Int_t ix1 = x1 < x2 ? XtoPixel(x1) : XtoPixel(x2);
   Int_t ix2 = x1 < x2 ? XtoPixel(x2) : XtoPixel(x1);
   Int_t iy1 = y1 < y2 ? YtoPixel(y1) : YtoPixel(y2);
   Int_t iy2 = y1 < y2 ? YtoPixel(y2) : YtoPixel(y1);

   if (ix1<0 || ix2 <0 || iy1 < 0 || iy2 <0) return; // box is not visible

   if (TMath::Abs(ix2-ix1) < 1) ix2 = ix1+1;
   if (TMath::Abs(iy1-iy2) < 1) iy1 = iy2+1;

   Int_t fillis = fFillStyle/1000;
   Int_t fillsi = fFillStyle%1000;

   TColor *col = gROOT->GetColor(fFillColor);
   if (!col) { // no color, set it white
      fFillColor = 10;
      col = gROOT->GetColor(fFillColor);
      if (!col) return;
   }

   TColor *linecol = gROOT->GetColor(fLineColor);
   if (!linecol) { // no color, set it to black
      fLineColor = 1;
      linecol = gROOT->GetColor(fLineColor);
   }

   if ((fillis == 3) || (fillis == 2)) {
      if (fillsi > 99) {
         x[0] = x1;   y[0] = y1;
         x[1] = x2;   y[1] = y1;
         x[2] = x2;   y[2] = y2;
         x[3] = x1;   y[3] = y2;
         return;
      }
      if ((fillsi > 0) && (fillsi < 26)) {
         x[0] = x1;   y[0] = y1;
         x[1] = x2;   y[1] = y1;
         x[2] = x2;   y[2] = y2;
         x[3] = x1;   y[3] = y2;
         DrawPS(-4, &x[0], &y[0]);
      }
      if (fillsi == -3) {
         // fill style = -3 ... which is NEVER used now
      }
   }

   if (fillis == 1) {
      fImage->DrawBox(ix1, iy1, ix2, iy2, col->AsHexString(), 1, TVirtualX::kFilled);
   }

   if (fillis == 0) {
      if (fLineWidth<=0) return;
      fImage->DrawBox(ix1, iy1, ix2, iy2, linecol->AsHexString(), fLineWidth, TVirtualX::kHollow);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a Frame around a box
///
///  - mode = -1  the box looks as it is behind the screen
///  - mode =  1  the box looks as it is in front of the screen
/// border is the border size in already pre-computed dark is the
/// color for the dark part of the frame light is the color for the light
/// part of the frame

void TImageDump::DrawFrame(Double_t x1, Double_t y1, Double_t x2, Double_t  y2,
                            Int_t mode, Int_t bordersize, Int_t dark, Int_t light)
{
   if (!gPad || !fImage) {
      return;
   }

   fImage->BeginPaint();

   bordersize = bordersize < 1 ? 1 : bordersize;

   TColor *col;
   TColor *lo = gROOT->GetColor(dark);
   if (!lo) {
      lo = gROOT->GetColor(10);
   }
   TColor *hi = gROOT->GetColor(light);
   if (!hi) {
      hi = gROOT->GetColor(10);
   }

   Short_t pxl,pyl,pxt,pyt,px1,py1,px2,py2;

   px1 = XtoPixel(x1);   py1 = YtoPixel(y1);
   px2 = XtoPixel(x2);   py2 = YtoPixel(y2);
   if (px1 < px2) {pxl = px1; pxt = px2;}
   else           {pxl = px2; pxt = px1;}
   if (py1 > py2) {pyl = py1; pyt = py2;}
   else           {pyl = py2; pyt = py1;}

   if (bordersize == 1) {
      col = gROOT->GetColor(fLineColor);
      if (!col) {
         fLineColor = 1;
         col = gROOT->GetColor(fLineColor);
         if (!col) return;
      }
      fImage->DrawBox(pxl, pyl, pxt, pyt-1, col->AsHexString(),  TVirtualX::kFilled);
      return;
   }

   if (!fImage->IsValid()) {
      col = gROOT->GetColor(light);
      if (!col) {
         col = gROOT->GetColor(10);
         if (!col) return;
      }
      fImage->DrawBox(pxl, pyl, pxt, pyt, // force image creation and resizing
                      "#ffffffff", 1,  TVirtualX::kFilled);
   }

   TPoint frame[6];

   frame[0].fX = pxl;                 frame[0].fY = pyl;
   frame[1].fX = pxl + bordersize;    frame[1].fY = pyl - bordersize;
   frame[2].fX = pxl + bordersize;    frame[2].fY = pyt + bordersize;
   frame[3].fX = pxt - bordersize;    frame[3].fY = pyt + bordersize;
   frame[4].fX = pxt;                 frame[4].fY = pyt;
   frame[5].fX = pxl;                 frame[5].fY = pyt;

   if (mode == -1) col = lo;
   else            col = hi;

   fImage->DrawFillArea(6, frame, col->AsHexString());

   frame[0].fX = pxl;                 frame[0].fY = pyl;
   frame[1].fX = pxl + bordersize;    frame[1].fY = pyl - bordersize;
   frame[2].fX = pxt - bordersize;    frame[2].fY = frame[1].fY;
   frame[3].fX = frame[2].fX;         frame[3].fY = pyt + bordersize;
   frame[4].fX = pxt;                 frame[4].fY = pyt;
   frame[5].fX = pxt;                 frame[5].fY = pyl;

   if (mode == -1) col = hi;
   else            col = lo;

   fImage->DrawFillArea(6, frame, col->AsHexString());
}

////////////////////////////////////////////////////////////////////////////////
/// draw polymarker

template<typename T>
void TImageDump::DrawPolyMarkerShape(Int_t n, T *xw, T *yw)
{
   if (!gPad || !fImage)
      return;

   TColor *col = gROOT->GetColor(GetMarkerColor());
   if (!col)
      col = gROOT->GetColor(1);

   if (!col)
      return;
   TString colHex = col->AsHexString();

   fImage->BeginPaint();

   auto markerLineWidth = TAttMarker::GetMarkerLineWidth(GetMarkerStyle());
   Int_t markerSize = 0;          ///< size of simple markers
   std::vector<TPoint> markerShape;   ///< marker shape points
   // prefer to use triangles while image not always correctly fill complex polygon
   auto markerType = GetMarkerShape(markerSize, markerShape, gStyle->GetImageScaling(), TAttMarker::kPreferTriangles | TAttMarker::kDotAsLines);

   for (Int_t i = 0; i < n; i++) {
      auto ix = XtoPixel(xw[i]);
      auto iy = YtoPixel(yw[i]);
      for (auto &pnt : markerShape) {
         pnt.fX += ix;
         pnt.fY += iy;
      }
      switch(markerType) {
         case TAttMarker::kShapeDot:
            fImage->PutPixel(ix, iy, colHex.Data());
            break;
         case TAttMarker::kShapeCircle:
            // hollow circle
            fImage->DrawCircle(ix, iy, markerSize/2, colHex.Data(), markerLineWidth);
            break;
         case TAttMarker::kShapeFilledCircle:
            // filled circle
            fImage->DrawCircle(ix, iy, markerSize/2, colHex.Data(), -1);
            break;
         case TAttMarker::kShapePolyLine:
            // hollow polygon
            fImage->DrawPolyLine(markerShape.size(), markerShape.data(), colHex.Data(), markerLineWidth);
            break;
         case TAttMarker::kShapeFilledArea:
            // filled polygon
            fImage->FillPolygon(markerShape.size(), markerShape.data(), colHex.Data());
            break;
         case TAttMarker::kShapeSegments:
            // segmented line
            for (std::size_t s = 0; s < markerShape.size(); s += 2)
               fImage->DrawLine(markerShape[s].fX, markerShape[s].fY, markerShape[s + 1].fX, markerShape[s + 1].fY, colHex.Data(), markerLineWidth);
            break;
         case TAttMarker::kShapeTriangles:
            // filled triangles
            for (std::size_t t = 0; t < markerShape.size(); t += 3)
               fImage->FillPolygon(3, markerShape.data() + t, colHex.Data());
            break;
      }
      for (auto &pnt : markerShape) {
         pnt.fX -= ix;
         pnt.fY -= iy;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// draw polymarker

void TImageDump::DrawPolyMarker(Int_t n, Float_t *xw, Float_t *yw)
{
   DrawPolyMarkerShape<Float_t>(n, xw, yw);
}

////////////////////////////////////////////////////////////////////////////////
/// draw polymarker

void TImageDump::DrawPolyMarker(Int_t n, Double_t *xw, Double_t *yw)
{
   DrawPolyMarkerShape<Double_t>(n, xw, yw);
}

////////////////////////////////////////////////////////////////////////////////
/// This function defines a path with xw and yw and draw it according the
/// value of nn:
///
///  - If nn > 0 a line is drawn.
///  - If nn < 0 a closed polygon is drawn.

void TImageDump::DrawPS(Int_t nn, Double_t *x, Double_t *y)
{
   if (!gPad || !fImage || !nn)
      return;

   fImage->BeginPaint();

   Bool_t line = nn > 1;
   UInt_t n = TMath::Abs(nn);
   Int_t fais = fFillStyle / 1000;
   Int_t fasi = fFillStyle % 1000;

   // SetLineStyle
   std::vector<char> dashList;

   TColor *fcol = gROOT->GetColor(fFillColor);
   if (!fcol) { // no color, set it white
      fFillColor = 10;
      fcol = gROOT->GetColor(fFillColor);
   }

   TColor *lcol = gROOT->GetColor(fLineColor);
   if (!lcol) { // no color, make it black
      fLineColor = 1;
      lcol = gROOT->GetColor(fLineColor);
   }

   if (line) {
      if (fLineWidth <= 0)
         return;
      // dash lines
      if (fLineStyle > 1) {
         TString st = gStyle->GetLineStyleString(fLineStyle);
         std::unique_ptr<TObjArray> tokens(st.Tokenize(" "));

         if (tokens)
            for (int j = 0; j < tokens->GetEntries(); j++) {
               Int_t it;
               sscanf(tokens->At(j)->GetName(), "%d", &it);
               dashList.emplace_back((char)(it/4));
            }
      }
   }

   if (n == 1) {  // point
      auto px1 = XtoPixel(x[0]);
      auto py1 = YtoPixel(y[0]);
      if (fcol)
         fImage->PutPixel(px1, py1, fcol->AsHexString());
      return;
   }

   if (n == 2) {  // line
      auto px1 = XtoPixel(x[0]);
      auto py1 = YtoPixel(y[0]);
      auto px2 = XtoPixel(x[1]);
      auto py2 = YtoPixel(y[1]);

      // SetLineColor
      if (lcol) {
         if (fLineStyle < 2) {
            fImage->DrawLine(px1, py1, px2, py2, lcol->AsHexString(), fLineWidth);
         } else {
            fImage->DrawDashLine(px1, py1, px2, py2, dashList.size(), dashList.data(),
                                 lcol->AsHexString(), fLineWidth);
         }
      }
      return;
   }

   if (!line && ((fais == 3) || (fais == 2)) && (fasi > 100))
      return;

   std::vector<TPoint> pt(n+1);

   for (UInt_t i = 0; i < n; i++) {
      pt[i].fX = XtoPixel(x[i]);
      pt[i].fY = YtoPixel(y[i]);
   }
   pt[n].fX = pt[0].fX;
   pt[n].fY = pt[0].fY;

   const char *stipple = (fais == 3) && (fasi > 0) && (fasi < 26) ? (const char*)gStipples[fasi] : nullptr;

   // filled polygon
   if (!line && fFillStyle && (fFillStyle != 4000) && fcol) {
      if (n < 5)    // convex
         fImage->FillPolygon(n, pt.data(), fcol->AsHexString(), stipple);
      else        // non-convex fill area
         fImage->DrawFillArea(n, pt.data(), fcol->AsHexString(), stipple);
   }

   // hollow polygon or polyline is drawn
   if (line || !fFillStyle || (fFillStyle == 4000)) {
      if (!line) {
         if (fcol)
            fImage->DrawPolyLine(n+1, pt.data(), fcol->AsHexString(), 1);
      } else if (lcol) {
         if (fLineStyle < 2) { // solid
            fImage->DrawPolyLine(n, pt.data(), lcol->AsHexString(), fLineWidth);
         } else { // dashed
            DrawDashPolyLine(n, pt.data(), dashList.size(), dashList.data(),
                            lcol->AsHexString(), fLineWidth);
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// not used

void TImageDump::DrawPS(Int_t, Float_t *, Float_t *)
{
   if (!gPad || !fImage) {
      return;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// draw dashed polyline

void TImageDump::DrawDashPolyLine(Int_t nn, TPoint *xy, UInt_t nDash,
                                    const char* pDash, const char* col, UInt_t thick)
{
   Int_t x0 = xy[0].GetX();
   Int_t y0 = xy[0].GetY();

   for (Int_t i = 1; i < nn; i++) {
      Int_t x = xy[i].GetX();
      Int_t y = xy[i].GetY();

      fImage->DrawDashLine(x0, y0, x, y, nDash, pDash, col, thick);

      x0 = x;
      y0 = y;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// new page

void TImageDump::NewPage()
{
   fX0 = fY0 = 0;
   if (gPad && fImage) {
      UInt_t w = gPad->GetWw() * gStyle->GetImageScaling();
      UInt_t h = gPad->GetWh() * gStyle->GetImageScaling();
      if (gPad != gPad->GetMother()) {
         fX0 = gPad->XtoAbsPixel(gPad->GetX1())* gStyle->GetImageScaling();
         fY0 = gPad->YtoAbsPixel(gPad->GetY2())* gStyle->GetImageScaling();
         w = w * gPad->GetAbsWNDC();
         h = h * gPad->GetAbsHNDC();
      }
      fImage->DrawRectangle(0, 0, w, h, "#ffffffff");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Draw text
///
///  - x: x position of the text
///  - y: y position of the text

void TImageDump::Text(Double_t x, Double_t y, const char *chars)
{
   if (!gPad || !fImage)
      return;

   fImage->BeginPaint();

   TText t(x, y, chars);
   t.SetTextSize(fTextSize*gStyle->GetImageScaling());
   t.SetTextFont(fTextFont);
   t.SetTextAlign(fTextAlign);
   t.SetTextAngle(fTextAngle);
   t.SetTextColor(fTextColor);
   fImage->DrawTextOnPad(&t, XtoPixel(x), YtoPixel(y), gPad, fX0, fY0);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw text
///
///  - x: x position of the text
///  - y: y position of the text

void TImageDump::Text(Double_t x, Double_t y, const wchar_t *chars)
{
   if (!gPad || !fImage)
      return;

   fImage->BeginPaint();

   TText t(x, y, chars);
   t.SetTextSize(fTextSize*gStyle->GetImageScaling());
   t.SetTextFont(fTextFont);
   t.SetTextAlign(fTextAlign);
   t.SetTextAngle(fTextAngle);
   t.SetTextColor(fTextColor);
   fImage->DrawTextOnPad(&t, XtoPixel(x), YtoPixel(y), gPad, fX0, fY0);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw text with URL. Same as Text.
///

void TImageDump::TextUrl(Double_t x, Double_t y, const char *chars, const char *)
{
   Text(x, y, chars);
}

////////////////////////////////////////////////////////////////////////////////
///cell array begin

void TImageDump::CellArrayBegin(Int_t w, Int_t h, Double_t x1, Double_t x2,
                                Double_t y1, Double_t y2)
{
   if (!gPad || !fImage || (w <= 0) || (h <= 0))
      return;

   fImage->BeginPaint();

   fCellArrayW = w;
   fCellArrayH = h;
   fCellArrayColors.resize(w * h);
   fCellArrayIdx = 0;

   fCellArrayX1 = x1 < x2 ? XtoPixel(x1) : XtoPixel(x2);
   fCellArrayX2 = x1 > x2 ? XtoPixel(x2) : XtoPixel(x1);
   fCellArrayY1 = y1 < y2 ? YtoPixel(y1) : YtoPixel(y2);
   fCellArrayY2 = y1 < y2 ? YtoPixel(y2) : YtoPixel(y1);
}

////////////////////////////////////////////////////////////////////////////////
/// Cell array fill

void TImageDump::CellArrayFill(Int_t r, Int_t g, Int_t b)
{
   if (fCellArrayIdx >= fCellArrayColors.size())
      return;

   fImage->BeginPaint();

   fCellArrayColors[fCellArrayIdx++] = ((r & 0xFF) << 16) + ((g & 0xFF) << 8) + (b & 0xFF);
}

////////////////////////////////////////////////////////////////////////////////
/// Cell array end

void TImageDump::CellArrayEnd()
{
   if (!fImage || fCellArrayColors.empty() || !fCellArrayW || !fCellArrayH)
      return;

   fImage->BeginPaint();

   fImage->DrawCellArray(fCellArrayX1, fCellArrayX2, fCellArrayY1, fCellArrayY2,
                         fCellArrayW, fCellArrayH, fCellArrayColors.data());

   fCellArrayColors.clear();
   fCellArrayIdx = 0;
   fCellArrayW = 0;
   fCellArrayH = 0;
   fCellArrayX1 = 0;
   fCellArrayX2 = 0;
   fCellArrayY1 = 0;
   fCellArrayY2 = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Set color with its R G B components
///
///  - r: % of red in [0,1]
///  - g: % of green in [0,1]
///  - b: % of blue in [0,1]

void TImageDump::SetColor(Float_t /*r*/, Float_t /*g*/, Float_t /*b*/)
{
}

////////////////////////////////////////////////////////////////////////////////
/// x to pixel

Int_t TImageDump::XtoPixel(Double_t x)
{
   return gPad->XtoAbsPixel(x)*gStyle->GetImageScaling() - fX0;
}

////////////////////////////////////////////////////////////////////////////////
/// y to pixel

Int_t TImageDump::YtoPixel(Double_t y)
{
   return  gPad->YtoAbsPixel(y)*gStyle->GetImageScaling() - fY0;
}
