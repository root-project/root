// @(#)root/postscript:$Name:  $:$Id: TImageDump.cxx,v 1.22 2006/10/13 08:18:39 rdm Exp $
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
// TImageDump can be used in any mode (batch, interactive) as follows   //                                                                      //
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


ClassImp(TImageDump)


//______________________________________________________________________________
TImageDump::TImageDump() : TVirtualPS()
{
   // Default SVG constructor

   fStream = 0;
   fImage  = 0;
   gVirtualPS = this;
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

   static Double_t x[4], y[4];
   Int_t ix1 = x1 < x2 ? gPad->XtoAbsPixel(x1) : gPad->XtoAbsPixel(x2);
   Int_t ix2 = x1 < x2 ? gPad->XtoAbsPixel(x2) : gPad->XtoAbsPixel(x1);
   Int_t iy1 = y1 < y2 ? gPad->YtoAbsPixel(y1) : gPad->YtoAbsPixel(y2);
   Int_t iy2 = y1 < y2 ? gPad->YtoAbsPixel(y2) : gPad->YtoAbsPixel(y1);

   Int_t fillis = fFillStyle/1000;
   Int_t fillsi = fFillStyle%1000;

   TColor *col = gROOT->GetColor(fFillColor);
   if (!col) { // no color, set it a la hsimple
      fFillColor = 48;
      col = gROOT->GetColor(48);
   }

   TColor *linecol = gROOT->GetColor(fLineColor);
   if (!linecol) { // no color, set it to black
      fLineColor = 1;
      linecol = gROOT->GetColor(1);
   }

   if (fillis == 3 || fillis == 2) {
      if (fillsi > 99) {
         x[0] = x1;   y[0] = y1;
         x[1] = x2;   y[1] = y1;
         x[2] = x2;   y[2] = y2;
         x[3] = x1;   y[3] = y2;
         return;
      }
      if (fillsi > 0 && fillsi < 26) {
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
      fImage->DrawBox(ix1, iy1, ix2, iy2, linecol->AsHexString(), 1, TVirtualX::kHollow);
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
   // border is the border size in already pre-computed SVG units dark is the
   // color for the dark part of the frame light is the color for the light
   // part of the frame

   if (!gPad || !fImage || (bordersize < 1)) {
      return;
   }

   TColor *col;
   TColor *lo = gROOT->GetColor(dark);
   if (!lo) {
      lo = gROOT->GetColor(121);
   }
   TColor *hi = gROOT->GetColor(light);
   if (!hi) {
      hi = gROOT->GetColor(171);
   }

   Short_t pxl,pyl,pxt,pyt,px1,py1,px2,py2;
   Double_t xl, xt, yl, yt;

   px1 = gPad->XtoAbsPixel(x1);   py1 = gPad->YtoAbsPixel(y1);
   px2 = gPad->XtoAbsPixel(x2);   py2 = gPad->YtoAbsPixel(y2);
   if (px1 < px2) {pxl = px1; pxt = px2; xl = x1; xt = x2; }
   else           {pxl = px2; pxt = px1; xl = x2; xt = x1;}
   if (py1 > py2) {pyl = py1; pyt = py2; yl = y1; yt = y2;}
   else           {pyl = py2; pyt = py1; yl = y2; yt = y1;}

   if (bordersize == 1) {
      col = gROOT->GetColor(fLineColor);
      if (!col) {
         fLineColor = 1;
         col = gROOT->GetColor(fLineColor);
      }
      fImage->DrawBox(pxl, pyl, pxt, pyt-1, col->AsHexString());
      return;
   }

   if (!fImage->IsValid()) {
      col = gROOT->GetColor(light);
      if (!col) {
         col = gROOT->GetColor(10);
      }
      fImage->DrawBox(pxl, pyl, pxt, pyt,    // force image creation and resizing
                      col->AsHexString(), 1, TVirtualX::kFilled);
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
   Int_t ms = TMath::Abs(fMarkerStyle);
   static TPoint pt[20];

   if (ms >= 6 && ms <= 19) ms = 20;
   if (ms == 4) ms = 24;

   // Define the marker size
   Double_t msize = 0.20*fMarkerSize*TMath::Max(fImage->GetWidth(), fImage->GetHeight())/20;
   if (ms == 6) msize *= 0.2;
   if (ms == 7) msize *= 0.3;
   Double_t m  = msize;
   Double_t m2 = m/2;
   Double_t m3 = m/3;
//   Double_t m4 = m2*1.333333333333;
   Double_t m6 = m/6;

   TColor *col = gROOT->GetColor(fMarkerColor);

   // Draw the marker according to the type
   Short_t ix,iy;
   for (Int_t i=0;i<n;i++) {
      ix = gPad->XtoAbsPixel(xw[i]);
      iy = gPad->YtoAbsPixel(yw[i]);

      switch (ms) {
      // Dot (.)
      case 1:
         fImage->PutPixel((UInt_t)ix, (UInt_t)iy, col->AsHexString());
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
         if (m < 4) fImage->FillRectangle(col->AsHexString(), UInt_t(ix-m2), UInt_t(iy-m2), UInt_t(m), UInt_t(m));
         else fImage->DrawCircle(ix, iy, Int_t(msize/4), col->AsHexString(), Int_t(msize/8));
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
      case 26:
         pt[0].fX = Short_t(ix-m2); pt[0].fY = Short_t(iy-m2);
         pt[1].fX = Short_t(ix+m2); pt[1].fY = Short_t(iy-m2);
         pt[2].fX = Short_t(ix);    pt[2].fY = Short_t(iy+m2);
         pt[3].fX = Short_t(ix-m2); pt[3].fY = Short_t(iy-m2);
         ms == 26 ? fImage->DrawPolyLine(4, pt, col->AsHexString()) :
                    fImage->FillPolygon(3, pt, col->AsHexString());
         break;
      // Up triangle
      case 22:
         pt[0].fX = Short_t(ix);    pt[0].fY = Short_t(iy-m2);
         pt[1].fX = Short_t(ix+m2); pt[1].fY = Short_t(iy+m2);
         pt[2].fX = Short_t(ix-m2); pt[2].fY = Short_t(iy+m2);
         fImage->FillPolygon(3, pt, col->AsHexString());
         break;
      case 27:
         fImage->DrawLine(UInt_t(ix), UInt_t(iy-m2), UInt_t(ix+m3), UInt_t(iy), col->AsHexString());
         fImage->DrawLine(UInt_t(ix+m3), UInt_t(iy), UInt_t(ix), UInt_t(iy+m2), col->AsHexString());
         fImage->DrawLine(UInt_t(ix), UInt_t(iy+m2), UInt_t(ix-m3), UInt_t(iy), col->AsHexString());
         fImage->DrawLine(UInt_t(ix-m3), UInt_t(iy), UInt_t(ix), UInt_t(iy-m2), col->AsHexString());
         break;
      case 28:
         fImage->DrawLine(UInt_t(ix-m6), UInt_t(iy-m6), UInt_t(ix-m6), UInt_t(iy-m2), col->AsHexString());
         fImage->DrawLine(UInt_t(ix-m6), UInt_t(iy-m2), UInt_t(ix+m6), UInt_t(iy-m2), col->AsHexString());
         fImage->DrawLine(UInt_t(ix+m6), UInt_t(iy-m2), UInt_t(ix+m6), UInt_t(iy-m6), col->AsHexString());
         fImage->DrawLine(UInt_t(ix+m6), UInt_t(iy-m6), UInt_t(ix+m2), UInt_t(iy-m6), col->AsHexString());
         fImage->DrawLine(UInt_t(ix+m2), UInt_t(iy-m6), UInt_t(ix+m2), UInt_t(iy+m6), col->AsHexString());
         fImage->DrawLine(UInt_t(ix+m2), UInt_t(iy+m6), UInt_t(ix+m6), UInt_t(iy+m6), col->AsHexString());
         fImage->DrawLine(UInt_t(ix+m6), UInt_t(iy+m6), UInt_t(ix+m6), UInt_t(iy+m2), col->AsHexString());
         fImage->DrawLine(UInt_t(ix+m6), UInt_t(iy+m2), UInt_t(ix-m6), UInt_t(iy+m2), col->AsHexString());
         fImage->DrawLine(UInt_t(ix-m6), UInt_t(iy+m2), UInt_t(ix-m6), UInt_t(iy+m6), col->AsHexString());
         fImage->DrawLine(UInt_t(ix-m6), UInt_t(iy+m6), UInt_t(ix-m2), UInt_t(iy+m6), col->AsHexString());
         fImage->DrawLine(UInt_t(ix-m2), UInt_t(iy+m6), UInt_t(ix-m2), UInt_t(iy-m6), col->AsHexString());
         fImage->DrawLine(UInt_t(ix-m2), UInt_t(iy-m6), UInt_t(ix-m6), UInt_t(iy-m6), col->AsHexString());
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
                    fImage->FillPolygon(10, pt, col->AsHexString());
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
   TColor *col = 0;
   Int_t  fais = 0 , fasi = 0;
   Bool_t line = nn > 0;
   UInt_t n = TMath::Abs(nn);

   fais = fFillStyle/1000;
   fasi = fFillStyle%1000;

   Short_t px1, py1, px2, py2;
   static const UInt_t gCachePtSize = 200;
   static TPoint gPointCache[gCachePtSize];
   Bool_t del = kTRUE;

   switch (n) {
      case 1:
         col = gROOT->GetColor(fFillColor);
         if (!col) { // no color, make it black 
            fFillColor = 1;
            col = gROOT->GetColor(1);
         }
         px1 = gPad->XtoAbsPixel(x[0]);   py1 = gPad->YtoAbsPixel(y[0]);
         fImage->PutPixel(px1, py1, col->AsHexString());
         break;

      case 2:
      {
         col = gROOT->GetColor(fLineColor);
         if (!col) { // no color, make it black
            fLineColor = 1;
            col = gROOT->GetColor(1);
         }
         px1 = gPad->XtoAbsPixel(x[0]);   py1 = gPad->YtoAbsPixel(y[0]);
         px2 = gPad->XtoAbsPixel(x[1]);   py2 = gPad->YtoAbsPixel(y[1]);

         switch (fLineStyle) {
            case 1:
            {
               fImage->DrawLine(px1, py1, px2, py2, col->AsHexString(), fLineWidth);
               break;
            }
            case 2:
               fImage->DrawDashLine(px1, py1, px2, py2, 2, "\x5\x5", col->AsHexString(), fLineWidth);
               break;
            case 3:
               fImage->DrawDashLine(px1, py1, px2, py2, 2, "\x1\x3", col->AsHexString(), fLineWidth);
               break;
            case 4:
               fImage->DrawDashLine(px1, py1, px2, py2, 4,"\x5\x3\x1\x3", col->AsHexString(), fLineWidth);
               break;
            default:
               fImage->DrawLine(px1, py1, px2, py2, col->AsHexString());
               break;
         }
         break;
      }
      default:
      {
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
         if (!fcol) { // no color, set it a la hsimple
            fFillColor = 48;
            fcol = gROOT->GetColor(48);
         }
         TColor *lcol = gROOT->GetColor(fLineColor);
         if (!lcol) { // no color, make it black
            fLineColor = 1;
            lcol = gROOT->GetColor(1);
         }

         for (UInt_t i = 0; i < n; i++) {
            pt[i].fX = gPad->XtoAbsPixel(x[i]);
            pt[i].fY = gPad->YtoAbsPixel(y[i]);
         }
         pt[n].fX = pt[0].fX;
         pt[n].fY = pt[0].fY;

         const char *stipple = (fais == 3) && (fasi > 0) && (fasi < 26) ? gStipples[fasi] : 0;

         if (!line && fFillStyle) {
            if (!fcol) return;
            if (n < 5) {
               fImage->FillPolygon(n, pt, fcol->AsHexString(), stipple);
            } else {
               fImage->DrawFillArea(n, pt, fcol->AsHexString(), stipple);
            }
         }
         if (line || !fFillStyle || stipple) {
            if (!lcol) return;
            fImage->DrawPolyLine(line ? n : n+1, pt, lcol->AsHexString(), fLineWidth);
         }
         if (del) delete [] pt;
      }
      break;
   }
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
void TImageDump::NewPage()
{
   // nothing 

   return;
}

//______________________________________________________________________________
void TImageDump::Text(Double_t xx, Double_t yy, const char *chars)
{
   // Draw text
   //
   // xx: x position of the text
   // yy: y position of the text

   // To scale fonts to the same size as the old TT version
   const Float_t kScale = 1;
   const Double_t kDEGRAD = TMath::Pi()/180.;

   if (!gPad || !fImage || (fTextSize < 0)) {
      return;
   }

   Double_t x = gPad->XtoAbsPixel(xx);
   Double_t y = gPad->YtoAbsPixel(yy);

   // Added by cholm for use of DFSG - fonts - based on Kevins fix.
   // Table of Microsoft and (for non-MSFT operating systems) backup
   // FreeFont TTF fonts.
   static const char *fonttable[][2] = {
      // fontnumber/10  MSFT font   Free font
      /* 0 */ { "arialbd.ttf",   "FreeSansBold.ttf"        },
      /* 1 */ { "timesi.ttf",    "FreeSerifItalic.ttf"     },
      /* 2 */ { "timesbd.ttf",   "FreeSerifBold.ttf"       },
      /* 3 */ { "timesbi.ttf",   "FreeSerifBoldItalic.ttf" },
      /* 4 */ { "arial.ttf",     "FreeSans.ttf"            },
      /* 5 */ { "ariali.ttf",    "FreeSansOblique.ttf"     },
      /* 6 */ { "arialbd.ttf",   "FreeSansBold.ttf"        },
      /* 7 */ { "arialbi.ttf",   "FreeSansBoldOblique.ttf" },
      /* 8 */ { "cour.ttf",      "FreeMono.ttf"            },
      /* 9 */ { "couri.ttf",     "FreeMonoOblique.ttf"     },
      /*10 */ { "courbd.ttf",    "FreeMonoBold.ttf"        },
      /*11 */ { "courbi.ttf",    "FreeMonoBoldOblique.ttf" },
      /*12 */ { "symbol.ttf",    "symbol.ttf"              },
      /*13 */ { "times.ttf",     "FreeSerif.ttf"           },
      /*14 */ { "wingding.ttf",  "opens___.ttf"            }
   };

   int fontid = fTextFont / 10;
   if (fontid < 0 || fontid > 14) fontid = 0;

    // try to load font (font must be in Root.TTFontPath resource)
   const char *ttpath = gEnv->GetValue("Root.TTFontPath",
# ifdef TTFFONTDIR
                                       TTFFONTDIR);
# else
                                       "$(ROOTSYS)/fonts");
# endif

   char *ttfont = gSystem->Which(ttpath, fonttable[fontid][0], kReadPermission);

   if (!ttfont)
      // try backup free font
      ttfont = gSystem->Which(ttpath, fonttable[fontid][1], kReadPermission);

   if (!ttfont) {
      Error("Text", "font file %s not found in path", fonttable[fontid][0]);
      return;
   }

   Double_t wh = (Double_t)gPad->XtoPixel(gPad->GetX2());
   Double_t hh = (Double_t)gPad->YtoPixel(gPad->GetY1());

   Int_t ttfsize;

   if (wh < hh) {
      ttfsize = (Int_t)(fTextSize*wh*kScale);
   } else {
      ttfsize = (Int_t)(fTextSize*hh*kScale);
   }

   if (ttfsize <= 0) return;

   // Text alignment
   Int_t txalh = fTextAlign/10;
   if (txalh < 1) txalh = 1;
   if (txalh > 3) txalh = 3;

   Int_t txalv = fTextAlign%10;
   if (txalv < 1) txalv = 1;
   if (txalv > 3) txalv = 3;

   UInt_t w, h;

   TText t;
   t.SetTextSize(fTextSize);
   t.SetTextFont(fTextFont);
   t.GetTextExtent(w, h, chars);

   if (txalh == 2) x -= (w>>1);
   if (txalh == 3) x -= w;

   Float_t angle = kDEGRAD*fTextAngle;

   if (txalv == 3) {
      y += (fTextAngle != 0. ? h * TMath::Cos(angle) : h);
   }
   if (txalv == 2) {
      y += (fTextAngle != 0. ? (h>>1) * TMath::Cos(angle) : h>>1);
   }

   TColor *col = gROOT->GetColor(fTextColor);
   fImage->DrawText((int)x, angle ? (int)y - w: (int)y - h, chars, ttfsize, col->AsHexString(),
                     ttfont, TImage::kPlain, 0, fTextAngle);

   delete [] ttfont;
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

   gCellArrayN = w * h;
   gCellArrayW = w;
   gCellArrayH = h;
   gCellArrayColors = new UInt_t[gCellArrayN];

   gCellArrayX1 = x1 < x2 ? gPad->XtoPixel(x1) : gPad->XtoPixel(x2);
   gCellArrayX2 = x1 > x2 ? gPad->XtoPixel(x2) : gPad->XtoPixel(x1);
   gCellArrayY1 = y1 < y2 ? gPad->YtoPixel(y1) : gPad->YtoPixel(y2);
   gCellArrayY2 = y1 < y2 ? gPad->YtoPixel(y2) : gPad->YtoPixel(y1);

   gCellArrayIdx = 0;
}

//______________________________________________________________________________
void TImageDump::CellArrayFill(Int_t r, Int_t g, Int_t b)
{
   //Cell array fill

   if (gCellArrayIdx >= gCellArrayN) return;

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

