// @(#)root/postscript:$Id$
// Author: Valeriy Onuchin

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TImageDump                                                           //
//                                                                      //
// save canvas as an image (GIF, JPEG, PNG, XPM, TIFF etc.)             //
// in batch mode.  Example:                                             //
//                                                                      //
//         $ root -b                                                    //
//         root [0] .x hsimple.C                                        //
//         root [1] c1->Print("c1.gif");                                //
//                                                                      //
// TImageDump can be used in any mode (batch, interactive) as follows   //
//                                                                      //
//    TCanvas *c1;                                                      //
//    TImageDump *imgdump = new TImageDump("test.png");                 //
//    c1->Paint();                                                      //
//    imgdump->Close();                                                 //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TImageDump.h"
#include "TImage.h"
#include "TMath.h"
#include "TPoint.h"
#include "TColor.h"
#include "TVirtualPad.h"
#include "TEnv.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TText.h"
#include "RStipples.h"
#include "TList.h"
#include "TStyle.h"
#include "TObjString.h"
#include "TObjArray.h"


ClassImp(TImageDump)


//______________________________________________________________________________
TImageDump::TImageDump() : TVirtualPS()
{
   // Default SVG constructor

   fStream    = 0;
   fImage     = 0;
   gVirtualPS = this;
   fType      = 0;
}

//______________________________________________________________________________
TImageDump::TImageDump(const char *fname, Int_t wtype) : TVirtualPS(fname, wtype)
{
   // Initialize batch image interface
   //
   //  fname : image file name
   //
   //    The possible workstation types are:
   //  111 - Portrait
   //  112 - Landscape
   //  114 - preview, keep in memory (do not write on delete)

   Open(fname, wtype);
   gVirtualPS = this;
}

//______________________________________________________________________________
void TImageDump::Open(const char *fname, Int_t type)
{
   // Open a image file

   fStream = 0;
   fImage  = TImage::Create();
   fType   = type;
   SetName(fname);
}

//______________________________________________________________________________
TImageDump::~TImageDump()
{
   // destructor

   Close();

   delete fImage;
   fImage = 0;

   gVirtualPS = 0;
}

//______________________________________________________________________________
void TImageDump::Close(Option_t *)
{
   // Close a image file

   // if fType == 114 - do not write image
   if (!fImage || (fType == 114)) {
      return;
   }

   //if (fType == 112) fImage->Flip(90);
   fImage->WriteImage(GetName());
}

//______________________________________________________________________________
void TImageDump::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t  y2)
{
   // Draw a Box

   if (!gPad || !fImage) {
      return;
   }

   fImage->BeginPaint();

   static Double_t x[4], y[4];
   Int_t ix1 = x1 < x2 ? XtoPixel(x1) : XtoPixel(x2);
   Int_t ix2 = x1 < x2 ? XtoPixel(x2) : XtoPixel(x1);
   Int_t iy1 = y1 < y2 ? YtoPixel(y1) : YtoPixel(y2);
   Int_t iy2 = y1 < y2 ? YtoPixel(y2) : YtoPixel(y1);
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
      fImage->DrawBox(ix1, iy1, ix2, iy2, linecol->AsHexString(), fLineWidth, TVirtualX::kHollow);
   }
}

//______________________________________________________________________________
void TImageDump::DrawFrame(Double_t x1, Double_t y1, Double_t x2, Double_t  y2,
                            Int_t mode, Int_t bordersize, Int_t dark, Int_t light)
{
   // Draw a Frame around a box
   //
   // mode = -1  the box looks as it is behind the screen
   // mode =  1  the box looks as it is in front of the screen
   // border is the border size in already pre-computed dark is the
   // color for the dark part of the frame light is the color for the light
   // part of the frame

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
   frame[3].fX = pxt - bordersize;    frame[3].fY = pyt + bordersize;;
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

//______________________________________________________________________________
void TImageDump::DrawPolyMarker(Int_t, Float_t *, Float_t *)
{
   // not used

   if (!gPad || !fImage) {
      return;
   }
}

//______________________________________________________________________________
void TImageDump::DrawPolyMarker(Int_t n, Double_t *xw, Double_t *yw)
{
   // draw polymarker

   if (!gPad || !fImage) {
      return;
   }

   fImage->BeginPaint();

   Int_t ms = TMath::Abs(fMarkerStyle);
   static TPoint pt[20];

   if (ms > 7 && ms <= 19) ms = 20;
   if (ms == 4) ms = 24;

   // Define the marker size
   const Int_t kBASEMARKER = 8;
   Double_t msize = fMarkerSize * kBASEMARKER;
   if (ms == 6) msize *= 0.2;
   if (ms == 7) msize *= 0.3;
   Double_t m  = msize;
   Double_t m2 = m/2;
   Double_t m3 = m/3;
   Double_t m6 = m/6;

   TColor *col = gROOT->GetColor(fMarkerColor);
   if (!col) { // no color
      fMarkerColor = 1;
      col = gROOT->GetColor(fMarkerColor);
      if (!col) return;
   }

   // Draw the marker according to the type
   Short_t ix,iy;
   for (Int_t i=0;i<n;i++) {
      ix = XtoPixel(xw[i]);
      iy = YtoPixel(yw[i]);

      switch (ms) {
      // Dots (.) big, medium and small
      case 7:
         fImage->PutPixel((UInt_t)ix-1, (UInt_t)iy-1, col->AsHexString());
         fImage->PutPixel((UInt_t)ix-1, (UInt_t)iy+1, col->AsHexString());
         fImage->PutPixel((UInt_t)ix+1, (UInt_t)iy+1, col->AsHexString());
         fImage->PutPixel((UInt_t)ix+1, (UInt_t)iy-1, col->AsHexString());
      case 6:
         fImage->PutPixel((UInt_t)ix,   (UInt_t)iy-1, col->AsHexString());
         fImage->PutPixel((UInt_t)ix,   (UInt_t)iy+1, col->AsHexString());
         fImage->PutPixel((UInt_t)ix-1, (UInt_t)iy,   col->AsHexString());
         fImage->PutPixel((UInt_t)ix+1, (UInt_t)iy,   col->AsHexString());
      case 1:
         fImage->PutPixel((UInt_t)ix,   (UInt_t)iy,   col->AsHexString());
         break;
      // Plus (+)
      case 2:
         fImage->DrawLine(UInt_t(ix-m2), UInt_t(iy), UInt_t(ix+m2), UInt_t(iy), col->AsHexString());
         fImage->DrawLine(UInt_t(ix), UInt_t(iy-m2), UInt_t(ix), UInt_t(iy+m2), col->AsHexString());
         break;
      // X shape (X)
      case 5:
         fImage->DrawLine(UInt_t(ix-m2), UInt_t(iy-m2), UInt_t(ix+m2), UInt_t(iy+m2), col->AsHexString());
         fImage->DrawLine(UInt_t(ix-m2), UInt_t(iy+m2), UInt_t(ix+m2), UInt_t(iy-m2), col->AsHexString());
         break;
      // Asterisk shape (*)
      case 3:
      case 31:
         fImage->DrawLine(UInt_t(ix-m2), UInt_t(iy), UInt_t(ix+m2), UInt_t(iy), col->AsHexString());
         fImage->DrawLine(UInt_t(ix), UInt_t(iy-m2), UInt_t(ix), UInt_t(iy+m2), col->AsHexString());
         fImage->DrawLine(UInt_t(ix-m2), UInt_t(iy-m2), UInt_t(ix+m2), UInt_t(iy+m2), col->AsHexString());
         fImage->DrawLine(UInt_t(ix-m2), UInt_t(iy+m2), UInt_t(ix+m2), UInt_t(iy-m2), col->AsHexString());
         break;
      // Circle
      case 4:
      case 24:
         fImage->DrawCircle(ix, iy, Int_t(msize/2), col->AsHexString(), 1);
         break;
      // Circle
      case 8:
      case 20:
         fImage->DrawCircle(ix, iy, Int_t(msize/2), col->AsHexString(), -1);
         break;
      // Square
      case 21:
         fImage->FillRectangle(col->AsHexString(), UInt_t(ix-m2), UInt_t(iy-m2), UInt_t(m), UInt_t(m));
         break;
      case 25:
         fImage->DrawRectangle(UInt_t(ix-m2), UInt_t(iy-m2), UInt_t(m), UInt_t(m), col->AsHexString());
         break;
      // Down triangle
      case 23:
      case 32:
         pt[0].fX = Short_t(ix-m2); pt[0].fY = Short_t(iy-m2);
         pt[1].fX = Short_t(ix+m2); pt[1].fY = Short_t(iy-m2);
         pt[2].fX = Short_t(ix);    pt[2].fY = Short_t(iy+m2);
         pt[3].fX = Short_t(ix-m2); pt[3].fY = Short_t(iy-m2);
	 ms == 32 ? fImage->DrawPolyLine(4, pt, col->AsHexString()) :
                    fImage->FillPolygon(3, pt, col->AsHexString());
         break;
      // Up triangle
      case 22:
      case 26:
         pt[0].fX = Short_t(ix);    pt[0].fY = Short_t(iy-m2);
         pt[1].fX = Short_t(ix+m2); pt[1].fY = Short_t(iy+m2);
         pt[2].fX = Short_t(ix-m2); pt[2].fY = Short_t(iy+m2);
         pt[3].fX = Short_t(ix);    pt[3].fY = Short_t(iy-m2);
         ms == 26 ? fImage->DrawPolyLine(4, pt, col->AsHexString()) :
                    fImage->FillPolygon(3, pt, col->AsHexString());
         break;
      case 27:
      case 33:
         pt[0].fX = Short_t(ix);    pt[0].fY = Short_t(iy-m2);
         pt[1].fX = Short_t(ix+m3); pt[1].fY = Short_t(iy);
         pt[2].fX = Short_t(ix);    pt[2].fY = Short_t(iy+m2);
         pt[3].fX = Short_t(ix-m3); pt[3].fY = Short_t(iy);
         pt[4].fX = Short_t(ix);    pt[4].fY = Short_t(iy-m2);
         ms == 27 ? fImage->DrawPolyLine(5, pt, col->AsHexString()) :
                    fImage->FillPolygon(4, pt, col->AsHexString());
         break;
      case 28:
      case 34:
         pt[0].fX = Short_t(ix-m6);  pt[0].fY = Short_t(iy-m6);
         pt[1].fX = Short_t(ix-m6);  pt[1].fY = Short_t(iy-m2);
         pt[2].fX = Short_t(ix+m6);  pt[2].fY = Short_t(iy-m2);
         pt[3].fX = Short_t(ix+m6);  pt[3].fY = Short_t(iy-m6);
         pt[4].fX = Short_t(ix+m2);  pt[4].fY = Short_t(iy-m6);
         pt[5].fX = Short_t(ix+m2);  pt[5].fY = Short_t(iy+m6);
         pt[6].fX = Short_t(ix+m6);  pt[6].fY = Short_t(iy+m6);
         pt[7].fX = Short_t(ix+m6);  pt[7].fY = Short_t(iy+m2);
         pt[8].fX = Short_t(ix-m6);  pt[8].fY = Short_t(iy+m2);
         pt[9].fX = Short_t(ix-m6);  pt[9].fY = Short_t(iy+m6);
         pt[10].fX = Short_t(ix-m2); pt[10].fY = Short_t(iy+m6);
         pt[11].fX = Short_t(ix-m2); pt[11].fY = Short_t(iy-m6);
         pt[12].fX = Short_t(ix-m6); pt[12].fY = Short_t(iy-m6);
         ms == 28 ? fImage->DrawPolyLine(13, pt, col->AsHexString()) :
                    fImage->FillPolygon(12, pt, col->AsHexString());
         break;
      case 29:
      case 30:
         pt[0].fX = Short_t(ix);             pt[0].fY = Short_t(iy+m2);
         pt[1].fX = Short_t(ix+0.112255*m);  pt[1].fY = Short_t(iy+0.15451*m);
         pt[2].fX = Short_t(ix+0.47552*m);   pt[2].fY = Short_t(iy+0.15451*m);
         pt[3].fX = Short_t(ix+0.181635*m);  pt[3].fY = Short_t(iy-0.05902*m);
         pt[4].fX = Short_t(ix+0.29389*m);   pt[4].fY = Short_t(iy-0.40451*m);
         pt[5].fX = Short_t(ix);             pt[5].fY = Short_t(iy-0.19098*m);
         pt[6].fX = Short_t(ix-0.29389*m);   pt[6].fY = Short_t(iy-0.40451*m);
         pt[7].fX = Short_t(ix-0.181635*m);  pt[7].fY = Short_t(iy-0.05902*m);
         pt[8].fX = Short_t(ix-0.47552*m);   pt[8].fY = Short_t(iy+0.15451*m);
         pt[9].fX = Short_t(ix-0.112255*m);  pt[9].fY = Short_t(iy+0.15451*m);
         pt[10].fX = Short_t(ix);             pt[10].fY = Short_t(iy+m2);
         ms == 30 ? fImage->DrawPolyLine(11, pt, col->AsHexString()) :
                    fImage->DrawFillArea(10, pt, col->AsHexString());
         break;
      default:
         fImage->PutPixel(UInt_t(ix), UInt_t(iy), col->AsHexString());
         break;
      }
   }
}

//______________________________________________________________________________
void TImageDump::DrawPS(Int_t nn, Double_t *x, Double_t *y)
{
   // This function defines a path with xw and yw and draw it according the
   // value of nn:
   //
   //  If nn > 0 a line is drawn.
   //  If nn < 0 a closed polygon is drawn.

   if (!gPad || !fImage || !nn) {
      return;
   }

   fImage->BeginPaint();

   TColor *col = 0;
   Int_t  fais = 0 , fasi = 0;
   Bool_t line = nn > 1;
   UInt_t n = TMath::Abs(nn);

   fais = fFillStyle/1000;
   fasi = fFillStyle%1000;

   Short_t px1, py1, px2, py2;
   static const UInt_t gCachePtSize = 200;
   static TPoint gPointCache[gCachePtSize];
   Bool_t del = kTRUE;


   // SetLineStyle
   Int_t ndashes = 0;
   char *dash = 0;
   static char dashList[10];
   Int_t dashLength = 0;
   Int_t dashSize = 0;

   if (line) {
      // dash lines
      if (fLineStyle > 1) {
         TString st = gStyle->GetLineStyleString(fLineStyle);
         TObjArray *tokens = st.Tokenize(" ");
         ndashes = tokens->GetEntries();
         dash = new char[ndashes];

         for (int j = 0; j < ndashes; j++) {
            Int_t it;
            sscanf(((TObjString*)tokens->At(j))->GetName(), "%d", &it);
            dash[j] = (char)(it/4);
         }

         dashSize = TMath::Min((int)sizeof(dashList), ndashes);
         dashLength = 0;
         for (int i = 0; i < dashSize; i++ ) {
            dashList[i] = dash[i];
            dashLength += dashList[i];
         }
         delete tokens;
         delete [] dash;
      }

      // SetLineColor
      col = gROOT->GetColor(fLineColor);
      if (!col) { // no color, make it black
         fLineColor = 1;
         col = gROOT->GetColor(fLineColor);
         if (!col) return;
      }
   }

   if (n == 1) {  // point
      col = gROOT->GetColor(fFillColor);
      if (!col) { // no color, make it black
         fFillColor = 1;
         col = gROOT->GetColor(fFillColor);
         if (!col) return;
      }
      px1 = XtoPixel(x[0]);   py1 = YtoPixel(y[0]);
      fImage->PutPixel(px1, py1, col->AsHexString());
      return;
   }

   if (n == 2) {  // line
      px1 = XtoPixel(x[0]);   py1 = YtoPixel(y[0]);
      px2 = XtoPixel(x[1]);   py2 = YtoPixel(y[1]);

      // SetLineColor
      col = gROOT->GetColor(fLineColor);
      if (!col) { // no color, make it black
         fLineColor = 1;
         col = gROOT->GetColor(fLineColor);
         if (!col) return;
      }
      if (fLineStyle < 2) {
         fImage->DrawLine(px1, py1, px2, py2, col->AsHexString(), fLineWidth);
      } else {
         fImage->DrawDashLine(px1, py1, px2, py2, dashSize, (const char*)dashList,
                                 col->AsHexString(), fLineWidth);
      }
      return;
   }

   if (!line && ((fais == 3) || (fais == 2)) && (fasi > 100) ) {
      return;
   }

   TPoint *pt = 0;
   if (n+1 < gCachePtSize) {
      pt = (TPoint*)&gPointCache;
      del = kFALSE;
   } else {
      pt = new TPoint[n+1];
      del = kTRUE;
   }

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

   for (UInt_t i = 0; i < n; i++) {
      pt[i].fX = XtoPixel(x[i]);
      pt[i].fY = YtoPixel(y[i]);
   }
   pt[n].fX = pt[0].fX;
   pt[n].fY = pt[0].fY;

   const char *stipple = (fais == 3) && (fasi > 0) && (fasi < 26) ? (const char*)gStipples[fasi] : 0;

   // filled polygon
   if (!line && fFillStyle && (fFillStyle != 4000)) {
      if (!fcol) return;

      if (n < 5) {   // convex
         fImage->FillPolygon(n, pt, fcol->AsHexString(), stipple);
      } else {       // non-convex fill area
         fImage->DrawFillArea(n, pt, fcol->AsHexString(), stipple);
      }
   }

   // hollow polygon or polyline is drawn
   if (line || !fFillStyle || (fFillStyle == 4000)) {
      if (!lcol) return;
      if (!line) {
         fImage->DrawPolyLine(n+1, pt, fcol->AsHexString(), 1);
      } else {
         if (fLineStyle < 2) { // solid
            fImage->DrawPolyLine(n, pt, lcol->AsHexString(), fLineWidth);
         } else { // dashed
            DrawDashPolyLine(n, pt,  dashSize, (const char*)dashList,
                            lcol->AsHexString(), fLineWidth);
         }
      }
   }
   if (del) delete [] pt;
}

//______________________________________________________________________________
void TImageDump::DrawPS(Int_t, Float_t *, Float_t *)
{
   // not used

   if (!gPad || !fImage) {
      return;
   }
}
//______________________________________________________________________________
void TImageDump::DrawDashPolyLine(Int_t nn, TPoint *xy, UInt_t nDash,
                                    const char* pDash, const char* col, UInt_t thick)
{
   // draw dashed polyline

   Int_t x0 = xy[0].GetX();
   Int_t y0 = xy[0].GetY();
   Int_t x = 0;
   Int_t y = 0;

   for (Int_t i = 1; i < nn; i++) {
      x = xy[i].GetX();
      y = xy[i].GetY();

      fImage->DrawDashLine(x0, y0, x, y, nDash, pDash, col, thick);

      x0 = x;
      y0 = y;
   }
}

//______________________________________________________________________________
void TImageDump::NewPage()
{
   // new page

   if (gPad && fImage) {
      UInt_t w = UInt_t(gPad->GetWw()*gPad->GetWNDC());
      UInt_t h = UInt_t(gPad->GetWh()*gPad->GetHNDC());
      fImage->DrawRectangle(0, 0, w, h, "#ffffffff");
   }
   return;
}

//______________________________________________________________________________
void TImageDump::Text(Double_t x, Double_t y, const char *chars)
{
   // Draw text
   //
   // x: x position of the text
   // y: y position of the text

   if (!gPad || !fImage) {
      return;
   }

   fImage->BeginPaint();

   TText t(x, y, chars);
   t.SetTextSize(fTextSize);
   t.SetTextFont(fTextFont);
   t.SetTextAlign(fTextAlign);
   t.SetTextAngle(fTextAngle);
   t.SetTextColor(fTextColor);
   fImage->DrawText(&t, XtoPixel(x), YtoPixel(y));
}


////////////////////////// CellArray code ////////////////////////////////////
static UInt_t *gCellArrayColors = 0;
static Int_t   gCellArrayN = 0;
static Int_t   gCellArrayW = 0;
static Int_t   gCellArrayH = 0;
static Int_t   gCellArrayX1 = 0;
static Int_t   gCellArrayX2 = 0;
static Int_t   gCellArrayY1 = 0;
static Int_t   gCellArrayY2 = 0;
static Int_t   gCellArrayIdx = 0;

//______________________________________________________________________________
void TImageDump::CellArrayBegin(Int_t w, Int_t h, Double_t x1, Double_t x2,
                                Double_t y1, Double_t y2)
{
   //cell array begin

   if (!gPad || !fImage || (w <= 0) || (h <= 0)) {
      return;
   }

   if (gCellArrayColors) {
      delete [] gCellArrayColors;
   }

   fImage->BeginPaint();

   gCellArrayN = w * h;
   gCellArrayW = w;
   gCellArrayH = h;
   gCellArrayColors = new UInt_t[gCellArrayN];

   gCellArrayX1 = x1 < x2 ? XtoPixel(x1) : XtoPixel(x2);
   gCellArrayX2 = x1 > x2 ? XtoPixel(x2) : XtoPixel(x1);
   gCellArrayY1 = y1 < y2 ? YtoPixel(y1) : YtoPixel(y2);
   gCellArrayY2 = y1 < y2 ? YtoPixel(y2) : YtoPixel(y1);

   gCellArrayIdx = 0;
}

//______________________________________________________________________________
void TImageDump::CellArrayFill(Int_t r, Int_t g, Int_t b)
{
   //Cell array fill

   if (gCellArrayIdx >= gCellArrayN) return;

   fImage->BeginPaint();

   gCellArrayColors[gCellArrayIdx] = ((r & 0xFF) << 16) + ((g & 0xFF) << 8) + (b & 0xFF);
   gCellArrayIdx++;
}

//______________________________________________________________________________
void TImageDump::CellArrayEnd()
{
   //Cell array end

   if (!fImage || !gCellArrayColors || !gCellArrayW || !gCellArrayH) {
      return;
   }

   fImage->BeginPaint();

   fImage->DrawCellArray(gCellArrayX1, gCellArrayX2, gCellArrayY1, gCellArrayY2,
                         gCellArrayW, gCellArrayH, gCellArrayColors);

   delete [] gCellArrayColors;
   gCellArrayColors = 0;
   gCellArrayN = 0;
   gCellArrayW = 0;
   gCellArrayH = 0;
   gCellArrayX1 = 0;
   gCellArrayX2 = 0;
   gCellArrayY1 = 0;
   gCellArrayY2 = 0;
   gCellArrayIdx = 0;
}

//______________________________________________________________________________
void TImageDump::SetColor(Float_t /*r*/, Float_t /*g*/, Float_t /*b*/)
{
   // Set color with its R G B components
   //
   //  r: % of red in [0,1]
   //  g: % of green in [0,1]
   //  b: % of blue in [0,1]

}

//______________________________________________________________________________
Int_t TImageDump::XtoPixel(Double_t x)
{
   // x to pixel

   return  gPad->XtoAbsPixel(x);
}

//______________________________________________________________________________
Int_t TImageDump::YtoPixel(Double_t y)
{
   // y to pixel

   return  gPad->YtoAbsPixel(y);
}
