// @(#)root/postscript:$Name:  $:$Id: TPDF.cxx,v 1.0
// Author: Olivier Couet

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPDF                                                                 //
//                                                                      //
// Graphics interface to PDF.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef WIN32
#pragma optimize("",off)
#endif

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "Riostream.h"
#include "TROOT.h"
#include "TColor.h"
#include "TVirtualPad.h"
#include "TPoints.h"
#include "TPDF.h"
#include "TStyle.h"
#include "TMath.h"
#include "TStorage.h"
#include "TText.h"
#include "zlib.h"
#include "TObjString.h"

// To scale fonts to the same size as the old TT version
const Float_t kScale = 0.93376068;

// Objects numbers
const Int_t kObjRoot      = 1; // Root object
const Int_t kObjInfo      = 2; // Info object
const Int_t kObjOutlines  = 3; // Outlines object
const Int_t kObjPages     = 4; // Pages object (pages index)
const Int_t kObjResources = 5; // Pages Resources object
const Int_t kObjFont      = 6; // Font object
const Int_t kObjFirstPage = 7; // First page object

ClassImp(TPDF)


//______________________________________________________________________________
//
// PDF driver
//


//______________________________________________________________________________
TPDF::TPDF() : TVirtualPS()
{
   // Default PDF constructor

   fStream    = 0;
   fType      = 0;
   fCompress  = kFALSE;
   gVirtualPS = this;
}


//______________________________________________________________________________
TPDF::TPDF(const char *fname, Int_t wtype) : TVirtualPS(fname, wtype)
{
   // Initialize the PDF interface
   //
   //  fname : PDF file name
   //  wtype : PDF workstation type. Not used in the PDF driver. But as TPDF
   //          inherits from TVirtualPS it should be kept. Anyway it is not
   //          necessary to specify this parameter at creation time because it
   //          has a default value (which is ignore in the PDF case).

   fStream   = 0;
   fCompress = kFALSE;
   Open(fname, wtype);
}


//______________________________________________________________________________
TPDF::~TPDF()
{
   // Default PDF destructor

   Close();

   if (fObjPos) delete [] fObjPos;
}


//______________________________________________________________________________
void TPDF::CellArrayBegin(Int_t, Int_t, Double_t, Double_t, Double_t,
                          Double_t)
{
   Warning("TPDF::CellArrayBegin", "not yet implemented");
}


//______________________________________________________________________________
void TPDF::CellArrayFill(Int_t, Int_t, Int_t)
{
   Warning("TPDF::CellArrayFill", "not yet implemented");
}


//______________________________________________________________________________
void TPDF::CellArrayEnd()
{
   Warning("TPDF::CellArrayEnd", "not yet implemented");
}


//______________________________________________________________________________
void TPDF::Close(Option_t *)
{
   // Close a PDF file

   Int_t i;

   if (!gVirtualPS) return;
   if (!fStream) return;
   if (gPad) gPad->Update();

   // Close the currently opened page
   WriteCompressedBuffer();
   PrintStr("endstream@");
   Int_t StreamLength = fNByte-fStartStream-10;
   PrintStr("endobj@");
   NewObject(3*(fNbPage-1)+kObjFirstPage+2);
   WriteInteger(StreamLength, 0);
   PrintStr("endobj@");

   // List of all the pages
   NewObject(kObjPages);
   PrintStr("<<@");
   PrintStr("/Type /Pages@");
   PrintStr("/Count");
   WriteInteger(fNbPage);
   PrintStr("@");
   PrintStr("/Kids [");
   for (i=1; i<=fNbPage; i++) {
      WriteInteger(3*(i-1)+kObjFirstPage);
      PrintStr(" 0 R");
   }
   PrintStr(" ]");
   PrintStr("@");
   PrintStr(">>@");
   PrintStr("endobj@");

   // Cross-Reference Table
   Int_t RefInd = fNByte;
   PrintStr("xref@");
   PrintStr("0");
   WriteInteger(fNbObj+1);
   PrintStr("@");
   PrintStr("0000000000 65535 f @");
   char str[21];
   for (i=0; i<fNbObj; i++) {
      sprintf(str,"%10.10d 00000 n @",fObjPos[i]);
      PrintStr(str);
   }

   // Trailer
   PrintStr("trailer@");
   PrintStr("<<@");
   PrintStr("/Size");
   WriteInteger(fNbObj+1);
   PrintStr("@");
   PrintStr("/Root");
   WriteInteger(kObjRoot);
   PrintStr(" 0 R");
   PrintStr("@");
   PrintStr("/Info");
   WriteInteger(kObjInfo);
   PrintStr(" 0 R");
   PrintStr(">>@");
   PrintStr("startxref@");
   WriteInteger(RefInd, 0);
   PrintStr("%%EOF@");

   // Close file stream
   if (fStream) { fStream->close(); delete fStream; fStream = 0;}

   gVirtualPS = 0;
}


//______________________________________________________________________________
void TPDF::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t  y2)
{
   // Draw a Box

   static Double_t x[4], y[4];
   Double_t ix1 = XtoPDF(x1);
   Double_t ix2 = XtoPDF(x2);
   Double_t iy1 = YtoPDF(y1);
   Double_t iy2 = YtoPDF(y2);
   Int_t fillis = fFillStyle/1000;
   Int_t fillsi = fFillStyle%1000;

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
         SetColor(5);
         WriteReal(ix1);
         WriteReal(iy1);
         WriteReal(ix2 - ix1);
         WriteReal(iy2 - iy1);
         PrintFast(6," re f*");
      }
   }
   if (fillis == 1) {
      SetColor(fFillColor);
      WriteReal(ix1);
      WriteReal(iy1);
      WriteReal(ix2 - ix1);
      WriteReal(iy2 - iy1);
      PrintFast(6," re f*");
   }
   if (fillis == 0) {
      SetColor(fLineColor);
      WriteReal(ix1);
      WriteReal(iy1);
      WriteReal(ix2 - ix1);
      WriteReal(iy2 - iy1);
      PrintFast(5," re S");
   }
}


//______________________________________________________________________________
void TPDF::DrawFrame(Double_t xl, Double_t yl, Double_t xt, Double_t  yt,
                            Int_t mode, Int_t border, Int_t dark, Int_t light)
{
   // Draw a Frame around a box
   //
   // mode = -1  box looks as it is behind the screen
   // mode =  1  box looks as it is in front of the screen
   // border is the border size in already precomputed PostScript units
   // dark  is the color for the dark part of the frame
   // light is the color for the light part of the frame

   static Double_t xps[7], yps[7];
   Int_t i;

   // Draw top&left part of the box
   if (mode == -1) SetColor(dark);
   else            SetColor(light);
   xps[0] = XtoPDF(xl);          yps[0] = YtoPDF(yl);
   xps[1] = xps[0] + border;     yps[1] = yps[0] + border;
   xps[2] = xps[1];              yps[2] = YtoPDF(yt) - border;
   xps[3] = XtoPDF(xt) - border; yps[3] = yps[2];
   xps[4] = XtoPDF(xt);          yps[4] = YtoPDF(yt);
   xps[5] = xps[0];              yps[5] = yps[4];
   xps[6] = xps[0];              yps[6] = yps[0];

   MoveTo(xps[0], yps[0]);
   for (i=1;i<7;i++) LineTo(xps[i], yps[i]);
   PrintFast(3," f*");

   // Draw bottom&right part of the box
   if (mode == -1) SetColor(light);
   else            SetColor(dark);
   xps[0] = XtoPDF(xl);          yps[0] = YtoPDF(yl);
   xps[1] = xps[0] + border;     yps[1] = yps[0] + border;
   xps[2] = XtoPDF(xt) - border; yps[2] = yps[1];
   xps[3] = xps[2];              yps[3] = YtoPDF(yt) - border;
   xps[4] = XtoPDF(xt);          yps[4] = YtoPDF(yt);
   xps[5] = xps[4];              yps[5] = yps[0];
   xps[6] = xps[0];              yps[6] = yps[0];

   MoveTo(xps[0], yps[0]);
   for (i=1;i<7;i++) LineTo(xps[i], yps[i]);
   PrintFast(3," f*");
}


//______________________________________________________________________________
void TPDF::DrawHatch(Float_t, Float_t, Int_t, Float_t *, Float_t *)
{
   // Draw Fill area with hatch styles

   Warning("DrawHatch", "hatch fill style not yet implemented");
}


//______________________________________________________________________________
void TPDF::DrawHatch(Float_t, Float_t, Int_t, Double_t *, Double_t *)
{
   // Draw Fill area with hatch styles

   Warning("DrawHatch", "hatch fill style not yet implemented");
}


//______________________________________________________________________________
void TPDF::DrawPolyLine(Int_t nn, TPoints *xy)
{
   // Draw a PolyLine
   //
   //  Draw a polyline through  the points xy.
   //  If NN=1 moves only to point x,y.
   //  If NN=0 the x,y are  written  in the PDF file
   //     according to the current transformation.
   //  If NN>0 the line is clipped as a line.
   //  If NN<0 the line is clipped as a fill area.

   Int_t  n;

   Style_t linestylesav = fLineStyle;
   Width_t linewidthsav = fLineWidth;

   if (nn > 0) {
      n = nn;
      SetLineStyle(fLineStyle);
      SetLineWidth(fLineWidth);
      SetColor(Int_t(fLineColor));
   } else {
      n = -nn;
      SetLineStyle(1);
      SetLineWidth(1);
      SetColor(Int_t(fLineColor));
   }

   WriteReal(XtoPDF(xy[0].GetX()));
   WriteReal(YtoPDF(xy[0].GetY()));
   if( n <= 1) {
      if( n == 0) return;
      PrintFast(2," m");
      return;
   }

   PrintFast(2," m");

   for (Int_t i=1;i<n;i++) LineTo(XtoPDF(xy[i].GetX()), YtoPDF(xy[i].GetY()));

   if (nn > 0 ) {
      if (xy[0].GetX() == xy[n-1].GetX() && xy[0].GetY() == xy[n-1].GetY()) PrintFast(3," cl");
      PrintFast(2," S");
   } else {
      PrintFast(3," f*");
   }

   SetLineStyle(linestylesav);
   SetLineWidth(linewidthsav);
}


//______________________________________________________________________________
void TPDF::DrawPolyLineNDC(Int_t nn, TPoints *xy)
{
   // Draw a PolyLine in NDC space
   //
   //  Draw a polyline through  the points  xy.
   //  If NN=1 moves only to point x,y.
   //  If NN=0 the x,y are  written  in the PDF        file
   //     according to the current transformation.
   //  If NN>0 the line is clipped as a line.
   //  If NN<0 the line is clipped as a fill area.

   Int_t  n;

   Style_t linestylesav = fLineStyle;
   Width_t linewidthsav = fLineWidth;

   if (nn > 0) {
      n = nn;
      SetLineStyle(fLineStyle);
      SetLineWidth(fLineWidth);
      SetColor(Int_t(fLineColor));
   } else {
      n = -nn;
      SetLineStyle(1);
      SetLineWidth(1);
      SetColor(Int_t(fLineColor));
   }

   WriteReal(UtoPDF(xy[0].GetX()));
   WriteReal(VtoPDF(xy[0].GetY()));
   if( n <= 1) {
      if( n == 0) return;
      PrintFast(2," m");
      return;
   }

   PrintFast(2," m");

   for (Int_t i=1;i<n;i++) LineTo(UtoPDF(xy[i].GetX()), VtoPDF(xy[i].GetY()));

   if (nn > 0 ) {
      if (xy[0].GetX() == xy[n-1].GetX() && xy[0].GetY() == xy[n-1].GetY()) PrintFast(3," cl");
      PrintFast(2," S");
   } else {
      PrintFast(3," f*");
   }

   SetLineStyle(linestylesav);
   SetLineWidth(linewidthsav);
}


//______________________________________________________________________________
void TPDF::DrawPolyMarker(Int_t n, Float_t *xw, Float_t *yw)
{
   // Draw markers at the n WC points xw, yw

   Style_t linestylesav = fLineStyle;
   Width_t linewidthsav = fLineWidth;
   SetLineStyle(1);
   SetLineWidth(1);
   SetColor(Int_t(fMarkerColor));
   Int_t ms = abs(fMarkerStyle);

   if (ms >= 6 && ms <= 19) ms = 20;
   if (ms == 4) ms = 24;

   // Define the marker size
   Double_t msize = 0.23*fMarkerSize*TMath::Max(fXsize,fYsize)/20;
   if (ms == 6) msize *= 0.2;
   if (ms == 7) msize *= 0.3;
   Double_t m  = CMtoPDF(msize);
   Double_t m2 = m/2;
   Double_t m3 = m/3;
   Double_t m4 = m2*1.333333333333;
   Double_t m6 = m/6;

   // Draw the marker according to the type
   Double_t ix,iy;
   for (Int_t i=0;i<n;i++) {
      ix = XtoPDF(xw[i]);
      iy = YtoPDF(yw[i]);
      // Dot (.)
      if (ms == 1) {
         MoveTo(ix-1, iy);
         LineTo(ix  , iy);
      // Plus (+)
      } else if (ms == 2) {
         MoveTo(ix-m2, iy);
         LineTo(ix+m2, iy);
         MoveTo(ix   , iy-m2);
         LineTo(ix   , iy+m2);
      // X shape (X)
      } else if (ms == 5) {
         MoveTo(ix-m2, iy-m2);
         LineTo(ix+m2, iy+m2);
         MoveTo(ix-m2, iy+m2);
         LineTo(ix+m2, iy-m2);
      // Asterisk shape (*)
      } else if (ms == 3 || ms == 31) {
         MoveTo(ix-m2, iy);
         LineTo(ix+m2, iy);
         MoveTo(ix   , iy-m2);
         LineTo(ix   , iy+m2);
         MoveTo(ix-m2, iy-m2);
         LineTo(ix+m2, iy+m2);
         MoveTo(ix-m2, iy+m2);
         LineTo(ix+m2, iy-m2);
      // Circle
      } else if (ms == 24 || ms == 20) {
         MoveTo(ix-m2, iy);
         WriteReal(ix-m2); WriteReal(iy+m4);
         WriteReal(ix+m2); WriteReal(iy+m4);
         WriteReal(ix+m2); WriteReal(iy)   ; PrintFast(2," c");
         WriteReal(ix+m2); WriteReal(iy-m4);
         WriteReal(ix-m2); WriteReal(iy-m4);
         WriteReal(ix-m2); WriteReal(iy)   ; PrintFast(4," c h");
      // Square
      } else if (ms == 25 || ms == 21) {
         WriteReal(ix-m2); WriteReal(iy-m2);
         WriteReal(m)    ; WriteReal(m)    ; PrintFast(3," re");
      // Down triangle
      } else if (ms == 23) {
         MoveTo(ix   , iy-m2);
         LineTo(ix+m2, iy+m2);
         LineTo(ix-m2, iy+m2);
         PrintFast(2," h");
      // Up triangle
      } else if (ms == 26 || ms == 22) {
         MoveTo(ix-m2, iy-m2);
         LineTo(ix+m2, iy-m2);
         LineTo(ix   , iy+m2);
         PrintFast(2," h");
      } else if (ms == 27) {
         MoveTo(ix   , iy-m2);
         LineTo(ix+m3, iy);
         LineTo(ix   , iy+m2);
         LineTo(ix-m3, iy)   ;
         PrintFast(2," h");
      } else if (ms == 28) {
         MoveTo(ix-m6, iy-m6);
         LineTo(ix-m6, iy-m2);
         LineTo(ix+m6, iy-m2);
         LineTo(ix+m6, iy-m6);
         LineTo(ix+m2, iy-m6);
         LineTo(ix+m2, iy+m6);
         LineTo(ix+m6, iy+m6);
         LineTo(ix+m6, iy+m2);
         LineTo(ix-m6, iy+m2);
         LineTo(ix-m6, iy+m6);
         LineTo(ix-m2, iy+m6);
         LineTo(ix-m2, iy-m6);
         PrintFast(2," h");
      } else if (ms == 29 || ms == 30) {
         MoveTo(ix           , iy+m2);
         LineTo(ix+0.112255*m, iy+0.15451*m);
         LineTo(ix+0.47552*m , iy+0.15451*m);
         LineTo(ix+0.181635*m, iy-0.05902*m);
         LineTo(ix+0.29389*m , iy-0.40451*m);
         LineTo(ix           , iy-0.19098*m);
         LineTo(ix-0.29389*m , iy-0.40451*m);
         LineTo(ix-0.181635*m, iy-0.05902*m);
         LineTo(ix-0.47552*m , iy+0.15451*m);
         LineTo(ix-0.112255*m, iy+0.15451*m);
         PrintFast(2," h");
      } else {
         MoveTo(ix-1, iy);
         LineTo(ix  , iy);
      }
   }

   if ((ms > 19 && ms < 24) || ms == 29) {
      PrintFast(2," f");
   } else {
      PrintFast(2," S");
   }

   SetLineStyle(linestylesav);
   SetLineWidth(linewidthsav);
}


//______________________________________________________________________________
void TPDF::DrawPolyMarker(Int_t n, Double_t *xw, Double_t *yw)
{
   // Draw markers at the n WC points xw, yw

   Style_t linestylesav = fLineStyle;
   Width_t linewidthsav = fLineWidth;
   SetLineStyle(1);
   SetLineWidth(1);
   SetColor(Int_t(fMarkerColor));
   Int_t ms = abs(fMarkerStyle);

   if (ms >= 6 && ms <= 19) ms = 20;
   if (ms == 4) ms = 24;

   // Define the marker size
   Double_t msize = 0.23*fMarkerSize*TMath::Max(fXsize,fYsize)/20;
   if (ms == 6) msize *= 0.2;
   if (ms == 7) msize *= 0.3;
   Double_t m  = CMtoPDF(msize);
   Double_t m2 = m/2;
   Double_t m3 = m/3;
   Double_t m4 = m2*1.333333333333;
   Double_t m6 = m/6;

   // Draw the marker according to the type
   Double_t ix,iy;
   for (Int_t i=0;i<n;i++) {
      ix = XtoPDF(xw[i]);
      iy = YtoPDF(yw[i]);
      // Dot (.)
      if (ms == 1) {
         MoveTo(ix-1, iy);
         LineTo(ix  , iy);
      // Plus (+)
      } else if (ms == 2) {
         MoveTo(ix-m2, iy);
         LineTo(ix+m2, iy);
         MoveTo(ix   , iy-m2);
         LineTo(ix   , iy+m2);
      // X shape (X)
      } else if (ms == 5) {
         MoveTo(ix-m2, iy-m2);
         LineTo(ix+m2, iy+m2);
         MoveTo(ix-m2, iy+m2);
         LineTo(ix+m2, iy-m2);
      // Asterisk shape (*)
      } else if (ms == 3 || ms == 31) {
         MoveTo(ix-m2, iy);
         LineTo(ix+m2, iy);
         MoveTo(ix   , iy-m2);
         LineTo(ix   , iy+m2);
         MoveTo(ix-m2, iy-m2);
         LineTo(ix+m2, iy+m2);
         MoveTo(ix-m2, iy+m2);
         LineTo(ix+m2, iy-m2);
      // Circle
      } else if (ms == 24 || ms == 20) {
         MoveTo(ix-m2, iy);
         WriteReal(ix-m2); WriteReal(iy+m4);
         WriteReal(ix+m2); WriteReal(iy+m4);
         WriteReal(ix+m2); WriteReal(iy)   ; PrintFast(2," c");
         WriteReal(ix+m2); WriteReal(iy-m4);
         WriteReal(ix-m2); WriteReal(iy-m4);
         WriteReal(ix-m2); WriteReal(iy)   ; PrintFast(4," c h");
      // Square
      } else if (ms == 25 || ms == 21) {
         WriteReal(ix-m2); WriteReal(iy-m2);
         WriteReal(m)    ; WriteReal(m)    ; PrintFast(3," re");
      // Down triangle
      } else if (ms == 23) {
         MoveTo(ix   , iy-m2);
         LineTo(ix+m2, iy+m2);
         LineTo(ix-m2, iy+m2);
         PrintFast(2," h");
      // Up triangle
      } else if (ms == 26 || ms == 22) {
         MoveTo(ix-m2, iy-m2);
         LineTo(ix+m2, iy-m2);
         LineTo(ix   , iy+m2);
         PrintFast(2," h");
      } else if (ms == 27) {
         MoveTo(ix   , iy-m2);
         LineTo(ix+m3, iy);
         LineTo(ix   , iy+m2);
         LineTo(ix-m3, iy)   ;
         PrintFast(2," h");
      } else if (ms == 28) {
         MoveTo(ix-m6, iy-m6);
         LineTo(ix-m6, iy-m2);
         LineTo(ix+m6, iy-m2);
         LineTo(ix+m6, iy-m6);
         LineTo(ix+m2, iy-m6);
         LineTo(ix+m2, iy+m6);
         LineTo(ix+m6, iy+m6);
         LineTo(ix+m6, iy+m2);
         LineTo(ix-m6, iy+m2);
         LineTo(ix-m6, iy+m6);
         LineTo(ix-m2, iy+m6);
         LineTo(ix-m2, iy-m6);
         PrintFast(2," h");
      } else if (ms == 29 || ms == 30) {
         MoveTo(ix           , iy+m2);
         LineTo(ix+0.112255*m, iy+0.15451*m);
         LineTo(ix+0.47552*m , iy+0.15451*m);
         LineTo(ix+0.181635*m, iy-0.05902*m);
         LineTo(ix+0.29389*m , iy-0.40451*m);
         LineTo(ix           , iy-0.19098*m);
         LineTo(ix-0.29389*m , iy-0.40451*m);
         LineTo(ix-0.181635*m, iy-0.05902*m);
         LineTo(ix-0.47552*m , iy+0.15451*m);
         LineTo(ix-0.112255*m, iy+0.15451*m);
         PrintFast(2," h");
      } else {
         MoveTo(ix-1, iy);
         LineTo(ix  , iy);
      }
   }

   if ((ms > 19 && ms < 24) || ms == 29) {
      PrintFast(2," f");
   } else {
      PrintFast(2," S");
   }

   SetLineStyle(linestylesav);
   SetLineWidth(linewidthsav);
}


//______________________________________________________________________________
void TPDF::DrawPS(Int_t nn, Float_t *xw, Float_t *yw)
{
   // Draw a PolyLine
   //
   //  Draw a polyline through the points xw,yw.
   //  If nn=1 moves only to point xw,yw.
   //  If nn=0 the XW(1) and YW(1) are  written  in the PostScript file
   //          according to the current NT.
   //  If nn>0 the line is clipped as a line.
   //  If nn<0 the line is clipped as a fill area.

   static Float_t dyhatch[24] = {.0075,.0075,.0075,.0075,.0075,.0075,.0075,.0075,
                                 .01  ,.01  ,.01  ,.01  ,.01  ,.01  ,.01  ,.01  ,
                                 .015 ,.015 ,.015 ,.015 ,.015 ,.015 ,.015 ,.015};
   static Float_t anglehatch[24] = {180, 90,135, 45,150, 30,120, 60,
                                    180, 90,135, 45,150, 30,120, 60,
                                    180, 90,135, 45,150, 30,120, 60};
   Int_t  n, fais = 0 , fasi = 0;

   Style_t linestylesav = fLineStyle;
   Width_t linewidthsav = fLineWidth;

   if (nn > 0) {
      n = nn;
      SetLineStyle(fLineStyle);
      SetLineWidth(fLineWidth);
      SetColor(Int_t(fLineColor));
   } else {
      n = -nn;
      SetLineStyle(1);
      SetLineWidth(1);
      SetColor(Int_t(fFillColor));
      fais = fFillStyle/1000;
      fasi = fFillStyle%1000;
      if (fais == 3 || fais == 2) {
         if (fasi > 100 && fasi <125) {
            DrawHatch(dyhatch[fasi-101],anglehatch[fasi-101], n, xw, yw);
            SetLineStyle(linestylesav);
            SetLineWidth(linewidthsav);
            return;
         }
         if (fasi > 0 && fasi < 26) {
            SetFillPatterns(fasi, Int_t(fFillColor));
         }
      }
   }

   WriteReal(XtoPDF(xw[0]));
   WriteReal(YtoPDF(yw[0]));
   if( n <= 1) {
      if( n == 0) return;
      PrintFast(2," m");
      return;
   }

   PrintFast(2," m");

   for (Int_t i=1;i<n;i++) LineTo(XtoPDF(xw[i]), YtoPDF(yw[i]));

   if (nn > 0 ) {
      if (xw[0] == xw[n-1] && yw[0] == yw[n-1]) PrintFast(2," h");
      PrintFast(2," S");
   } else {
      if (fais == 0) {PrintFast(2," s"); return;}
      if (fais == 3 || fais == 2) {
         if (fasi > 0 && fasi < 26) {
            PrintFast(3," f*");
            fRed   = -1;
            fGreen = -1;
            fBlue  = -1;
         }
         SetLineStyle(linestylesav);
         SetLineWidth(linewidthsav);
         return;
      }
      PrintFast(3," f*");
   }

   SetLineStyle(linestylesav);
   SetLineWidth(linewidthsav);
}


//______________________________________________________________________________
void TPDF::DrawPS(Int_t nn, Double_t *xw, Double_t *yw)
{
   // Draw a PolyLine
   //
   // Draw a polyline through  the points xw,yw.
   // If nn=1 moves only to point xw,yw.
   // If nn=0 the xw(1) and YW(1) are  written  in the PostScript file
   //         according to the current NT.
   // If nn>0 the line is clipped as a line.
   // If nn<0 the line is clipped as a fill area.

   static Float_t dyhatch[24] = {.0075,.0075,.0075,.0075,.0075,.0075,.0075,.0075,
                                 .01  ,.01  ,.01  ,.01  ,.01  ,.01  ,.01  ,.01  ,
                                 .015 ,.015 ,.015 ,.015 ,.015 ,.015 ,.015 ,.015};
   static Float_t anglehatch[24] = {180, 90,135, 45,150, 30,120, 60,
                                    180, 90,135, 45,150, 30,120, 60,
                                    180, 90,135, 45,150, 30,120, 60};
   Int_t  n, fais = 0, fasi = 0;

   Style_t linestylesav = fLineStyle;
   Width_t linewidthsav = fLineWidth;

   if (nn > 0) {
      n = nn;
      SetLineStyle(fLineStyle);
      SetLineWidth(fLineWidth);
      SetColor(Int_t(fLineColor));
   } else {
      n = -nn;
      SetLineStyle(1);
      SetLineWidth(1);
      SetColor(Int_t(fFillColor));
      fais = fFillStyle/1000;
      fasi = fFillStyle%1000;
      if (fais == 3 || fais == 2) {
         if (fasi > 100 && fasi <125) {
            DrawHatch(dyhatch[fasi-101],anglehatch[fasi-101], n, xw, yw);
            SetLineStyle(linestylesav);
            SetLineWidth(linewidthsav);
            return;
         }
         if (fasi > 0 && fasi < 26) {
            SetFillPatterns(fasi, Int_t(fFillColor));
         }
      }
   }

   WriteReal(XtoPDF(xw[0]));
   WriteReal(YtoPDF(yw[0]));
   if( n <= 1) {
      if( n == 0) return;
      PrintFast(2," m");
      return;
   }

   PrintFast(2," m");

   for (Int_t i=1;i<n;i++) LineTo(XtoPDF(xw[i]), YtoPDF(yw[i]));

   if (nn > 0 ) {
      if (xw[0] == xw[n-1] && yw[0] == yw[n-1]) PrintFast(2," h");
      PrintFast(2," S");
   } else {
      if (fais == 0) {PrintFast(2," s"); return;}
      if (fais == 3 || fais == 2) {
         if (fasi > 0 && fasi < 26) {
            PrintFast(3," f*");
            fRed   = -1;
            fGreen = -1;
            fBlue  = -1;
         }
         SetLineStyle(linestylesav);
         SetLineWidth(linewidthsav);
         return;
      }
      PrintFast(3," f*");
   }

   SetLineStyle(linestylesav);
   SetLineWidth(linewidthsav);
}


//______________________________________________________________________________
void TPDF::FontEncode()
{
   // Font encoding

   static const char *sdtfonts[] = {
   "/Times-Italic"         , "/Times-Bold"         , "/Times-BoldItalic",
   "/Helvetica"            , "/Helvetica-Oblique"  , "/Helvetica-Bold"  ,
   "/Helvetica-BoldOblique", "/Courier"            , "/Courier-Oblique" ,
   "/Courier-Bold"         , "/Courier-BoldOblique", "/Symbol"          ,
   "/Times-Roman"          , "/ZapfDingbats"};

   NewObject(kObjFont);
   PrintStr("<<@");
   for (Int_t i=0; i<14; i++) {
      PrintStr("/F");
      WriteInteger(i+1,0);
      PrintStr(" <<");
      PrintStr("/Type/Font");
      PrintStr("/Subtype/Type1");
      PrintStr("/BaseFont");
      PrintStr(sdtfonts[i]);
      PrintStr(">>");
      PrintStr("@");
   }
   PrintStr(">>");
   PrintStr("endobj@");
}


//______________________________________________________________________________
void TPDF::LineTo(Double_t x, Double_t y)
{
   // Draw a line to a new position

   WriteReal(x);
   WriteReal(y);
   PrintFast(2," l");
}


//______________________________________________________________________________
void TPDF::MoveTo(Double_t x, Double_t y)
{
   // Move to a new position

   WriteReal(x);
   WriteReal(y);
   PrintFast(2," m");
}


//______________________________________________________________________________
void TPDF::NewObject(Int_t n)
{
   // Create a new object in the PDF file

   if (!fObjPos || n >= fObjPosSize) {
      Int_t newN = TMath::Max(2*fObjPosSize,n+1);
      Int_t *saveo = new Int_t [newN];
      if (fObjPos && fObjPosSize) {
         memcpy(saveo,fObjPos,fObjPosSize*sizeof(Int_t));
         memset(&saveo[fObjPosSize],0,(newN-fObjPosSize)*sizeof(Int_t));
         delete [] fObjPos;
      }
      fObjPos     = saveo;
      fObjPosSize = newN;
   }
   fObjPos[n-1] = fNByte;
   fNbObj       = TMath::Max(fNbObj,n);
   WriteInteger(n, 0);
   PrintStr(" 0 obj");
   PrintStr("@");
}


//______________________________________________________________________________
void TPDF::NewPage()
{
   // Start a new PDF page.

   // Compute pad conversion coefficients
   if (gPad) {
      Double_t ww   = gPad->GetWw();
      Double_t wh   = gPad->GetWh();
      fYsize        = fXsize*wh/ww;
   } else {
      fYsize = 27;
   }

   fNbPage++;

   if (fNbPage>1) {
      // Close the currently opened page
      WriteCompressedBuffer();
      PrintStr("endstream@");
      Int_t StreamLength = fNByte-fStartStream-10;
      PrintStr("endobj@");
      NewObject(3*(fNbPage-2)+kObjFirstPage+2);
      WriteInteger(StreamLength, 0);
      PrintStr("endobj@");
   }

   // Start a new page
   NewObject(3*(fNbPage-1)+kObjFirstPage);
   PrintStr("<<@");
   PrintStr("/Type /Page@");
   PrintStr("@");
   PrintStr("/Parent");
   WriteInteger(kObjPages);
   PrintStr(" 0 R");
   PrintStr("@");

   Double_t xlow=0, ylow=0, xup=1, yup=1;
   if (gPad) {
      xlow = gPad->GetAbsXlowNDC();
      xup  = xlow + gPad->GetAbsWNDC();
      ylow = gPad->GetAbsYlowNDC();
      yup  = ylow + gPad->GetAbsHNDC();
   }

   PrintStr("/MediaBox [");
   Double_t width, height;
   switch (fPageFormat) {
      case 100 :
         width  = 8.5*2.54;
         height = 11.*2.54;
         break;
      case 200 :
         width  = 8.5*2.54;
         height = 14.*2.54;
         break;
      case 300 :
         width  = 11.*2.54;
         height = 17.*2.54;
         break;
      default  :
         width  = 21.0*TMath::Power(TMath::Sqrt(2.), 4-fPageFormat);
         height = 29.7*TMath::Power(TMath::Sqrt(2.), 4-fPageFormat);
   };
   WriteReal(CMtoPDF(fXsize*xlow));
   WriteReal(CMtoPDF(fYsize*ylow));
   WriteReal(CMtoPDF(width));
   WriteReal(CMtoPDF(height));
   PrintStr("]");
   PrintStr("@");

   Double_t xmargin = CMtoPDF(0.7);
   Double_t ymargin = 0;
   if (fPageOrientation == 1) ymargin = CMtoPDF(TMath::Sqrt(2.)*0.7);
   if (fPageOrientation == 2) ymargin = CMtoPDF(height)-CMtoPDF(0.7);

   PrintStr("/CropBox [");
   if (fPageOrientation == 1) {
      WriteReal(xmargin);
      WriteReal(ymargin);
      WriteReal(xmargin+CMtoPDF(fXsize*xup));
      WriteReal(ymargin+CMtoPDF(fYsize*yup));
   }
   if (fPageOrientation == 2) {
      WriteReal(xmargin);
      WriteReal(CMtoPDF(height)-CMtoPDF(fXsize*xup)-xmargin);
      WriteReal(xmargin+CMtoPDF(fYsize*yup));
      WriteReal(CMtoPDF(height)-xmargin);
   }
   PrintStr("]");
   PrintStr("@");

   PrintStr("/Rotate 0@");

   PrintStr("/Resources");
   WriteInteger(kObjResources);
   PrintStr(" 0 R");
   PrintStr("@");

   PrintStr("/Contents");
   WriteInteger(3*(fNbPage-1)+kObjFirstPage+1);
   PrintStr(" 0 R");
   PrintStr(">>@");
   PrintStr("endobj@");

   NewObject(3*(fNbPage-1)+kObjFirstPage+1);
   PrintStr("<<@");
   PrintStr("/Length");
   WriteInteger(3*(fNbPage-1)+kObjFirstPage+2);
   PrintStr(" 0 R");
   PrintStr("/Filter [/FlateDecode]@");
   PrintStr(">>@");
   PrintStr("stream@");
   fStartStream = fNByte;
   fCompress = kTRUE;

   PrintStr("1 0 0 1");
   WriteReal(xmargin);
   WriteReal(ymargin);
   PrintStr(" cm");
   if (fPageOrientation == 2) PrintStr(" 0 -1 1 0 0 0 cm");
}


//______________________________________________________________________________
void TPDF::Off()
{
   // Deactivate an already open PDF file

   gVirtualPS = 0;
}


//______________________________________________________________________________
void TPDF::On()
{
   // Activate an already open PDF file

   // fType is used to know if the PDF file is open. Unlike TPostScript, TPDF
   // has no "workstation type".

   if (!fType) {
      Error("On", "no PDF file open");
      Off();
      return;
   }
   gVirtualPS = this;
}


//______________________________________________________________________________
void TPDF::Open(const char *fname, Int_t wtype)
{
   // Open a PDF file

   if (fStream) {
      Warning("Open", "PDF file already open");
      return;
   }

   fLenBuffer = 0;
   fRed       = -1;
   fGreen     = -1;
   fBlue      = -1;
   fType      = abs(wtype);
   gStyle->GetPaperSize(fXsize, fYsize);
   Float_t xrange, yrange;
   if (gPad) {
      Double_t ww = gPad->GetWw();
      Double_t wh = gPad->GetWh();
      if (fType == 113) {
         ww *= gPad->GetWNDC();
         wh *= gPad->GetHNDC();
      }
      Double_t ratio = wh/ww;
      xrange = fXsize;
      yrange = fXsize*ratio;
      if (yrange > fYsize) { yrange = fYsize; xrange = yrange/ratio;}
      fXsize = xrange; fYsize = yrange;
   }

   // Open OS file
   fStream = new ofstream();
#ifdef R__WIN32
      fStream->open(fname, ofstream::out | ofstream::binary);
#else
      fStream->open(fname, ofstream::out);
#endif
   if (fStream == 0) {
      printf("ERROR in TPDF::Open: Cannot open file:%s\n",fname);
      return;
   }

   gVirtualPS = this;

   for (Int_t i=0;i<fSizBuffer;i++) fBuffer[i] = ' ';

   // The page orientation is last digit of PDF workstation type
   //  orientation = 1 for portrait
   //  orientation = 2 for landscape
   fPageOrientation = fType%10;
   if( fPageOrientation < 1 || fPageOrientation > 2) {
      Error("Open", "Invalid page orientation %d", fPageOrientation);
      return;
   }

   // format = 0-99 is the European page format (A4,A3 ...)
   // format = 100 is the US format  8.5x11.0 inch
   // format = 200 is the US format  8.5x14.0 inch
   // format = 300 is the US format 11.0x17.0 inch
   fPageFormat = fType/1000;
   if( fPageFormat == 0 )  fPageFormat = 4;
   if( fPageFormat == 99 ) fPageFormat = 0;

   fRange = kFALSE;

   // Set a default range
   Range(fXsize, fYsize);

   fObjPos = 0;
   fObjPosSize = 0;
   fNbObj = 0;
   fNbPage = 0;

   PrintStr("%PDF-1.4@");

   NewObject(kObjRoot);
   PrintStr("<<@");
   PrintStr("/Type /Catalog@");
   PrintStr("/Outlines");
   WriteInteger(kObjOutlines);
   PrintStr(" 0 R");
   PrintStr("@");
   PrintStr("/Pages");
   WriteInteger(kObjPages);
   PrintStr(" 0 R");
   PrintStr(">>@");
   PrintStr("endobj@");

   NewObject(kObjInfo);
   PrintStr("<<@");
   PrintStr("/Creator (ROOT Version ");
   PrintStr(gROOT->GetVersion());
   PrintStr(")");
   PrintStr("@");
   PrintStr("/CreationDate (");
   TDatime t;
   char str[17];
   sprintf(str,"D:%4.4d%2.2d%2.2d%2.2d%2.2d%2.2d",
                t.GetYear()  , t.GetMonth(),
                t.GetDay()   , t.GetHour(),
                t.GetMinute(), t.GetSecond());
   PrintStr(str);
   PrintStr(")");
   PrintStr("@");
   PrintStr("/Title (");
   PrintStr(GetName());
   PrintStr(")");
   PrintStr("@");
   PrintStr("/Keywords (ROOT)");
   PrintStr(">>@");
   PrintStr("endobj@");

   NewObject(kObjOutlines);
   PrintStr("<<@");
   PrintStr("/Type /Outlines@");
   PrintStr("/Count 0@");
   PrintStr(">>@");
   PrintStr("endobj@");

   NewObject(kObjResources);
   PrintStr("<<@");
   PrintStr("/ProcSet [/PDF /Text]@");
   PrintStr("/Font");
   WriteInteger(kObjFont);
   PrintStr(" 0 R");
   PrintStr("@");
   PrintStr(">>@");
   PrintStr("endobj@");

   FontEncode();

   NewPage();
}


//______________________________________________________________________________
void TPDF::PrintStr(const char *str)
{
   // Output the string str in the output buffer

   Int_t len = strlen(str);
   if (len == 0) return;

   if (fCompress) {
      if (fLenBuffer+len >= fSizBuffer) {
         fBuffer  = TStorage::ReAllocChar(fBuffer, 2*fSizBuffer, fSizBuffer);
         fSizBuffer = 2*fSizBuffer;
      }
      strcpy(fBuffer + fLenBuffer, str);
      fLenBuffer += len;
      return;
   }

   TVirtualPS::PrintStr(str);
}


//______________________________________________________________________________
void TPDF::PrintFast(Int_t len, const char *str)
{
   // Fast version of Print

   if (fCompress) {
      if (fLenBuffer+len >= fSizBuffer) {
         fBuffer  = TStorage::ReAllocChar(fBuffer, 2*fSizBuffer, fSizBuffer);
         fSizBuffer = 2*fSizBuffer;
      }
      strcpy(fBuffer + fLenBuffer, str);
      fLenBuffer += len;
      return;
   }

   TVirtualPS::PrintFast(len, str);
}


//______________________________________________________________________________
void TPDF::Range(Float_t xsize, Float_t ysize)
{
   // Set the range for the paper in centimetres

   Float_t xps, yps, xncm, yncm, dxwn, dywn, xwkwn, ywkwn, xymax;

   fXsize = xsize;
   fYsize = ysize;

   xps = xsize;
   yps = ysize;

   if( xsize <= xps && ysize < yps) {
      if ( xps > yps ) xymax = xps;
      else             xymax = yps;
      xncm  = xsize/xymax;
      yncm  = ysize/xymax;
      dxwn  = ((xps/xymax)-xncm)/2;
      dywn  = ((yps/xymax)-yncm)/2;
   } else {
      if (xps/yps < 1) xwkwn = xps/yps;
      else             xwkwn = 1;
      if (yps/xps < 1) ywkwn = yps/xps;
      else             ywkwn = 1;

      if (xsize < ysize)  {
         xncm = ywkwn*xsize/ysize;
         yncm = ywkwn;
         dxwn = (xwkwn-xncm)/2;
         dywn = 0;
         if( dxwn < 0) {
            xncm = xwkwn;
            dxwn = 0;
            yncm = xwkwn*ysize/xsize;
            dywn = (ywkwn-yncm)/2;
         }
      } else {
         xncm = xwkwn;
         yncm = xwkwn*ysize/xsize;
         dxwn = 0;
         dywn = (ywkwn-yncm)/2;
         if( dywn < 0) {
            yncm = ywkwn;
            dywn = 0;
            xncm = ywkwn*xsize/ysize;
            dxwn = (xwkwn-xncm)/2;
         }
      }
   }
   fRange = kTRUE;
}


//______________________________________________________________________________
void TPDF::SetColor(Int_t color)
{
   // Set color with its color index

   if (color < 0) color = 0;
   TColor *col = gROOT->GetColor(color);
   if (col) {
      SetColor(col->GetRed(), col->GetGreen(), col->GetBlue());
   } else {
      SetColor(1., 1., 1.);
   }
}


//______________________________________________________________________________
void TPDF::SetColor(Float_t r, Float_t g, Float_t b)
{
   // Set color with its R G B components
   //
   //  r: % of red in [0,1]
   //  g: % of green in [0,1]
   //  b: % of blue in [0,1]

   if (r == fRed && g == fGreen && b == fBlue) return;

   fRed   = r;
   fGreen = g;
   fBlue  = b;
   if (fRed   <= 0.000001) fRed=0;
   if (fGreen <= 0.000001) fGreen=0;
   if (fBlue  <= 0.000001) fBlue=0;

   WriteReal(fRed);
   WriteReal(fGreen);
   WriteReal(fBlue);
   PrintFast(3," RG");

   WriteReal(fRed);
   WriteReal(fGreen);
   WriteReal(fBlue);
   PrintFast(3," rg");
}


//______________________________________________________________________________
void TPDF::SetFillColor( Color_t cindex )
{
   // Set color index for fill areas

   fFillColor = cindex;
   if (gStyle->GetFillColor() <= 0) cindex = 0;
}


//______________________________________________________________________________
void TPDF::SetFillPatterns(Int_t /*ipat*/, Int_t /*color*/)
{
}


//______________________________________________________________________________
void TPDF::SetLineColor( Color_t cindex )
{
   // Set color index for lines

   fLineColor = cindex;
}


//______________________________________________________________________________
void TPDF::SetLineStyle(Style_t linestyle)
{
   // Change the line style
   //
   // linestyle = 2 dashed
   //           = 3 dotted
   //           = 4 dash-dotted
   //           = else solid (1 in is used most of the time)

   if ( linestyle == fLineStyle) return;
   fLineStyle = linestyle;
   TString st = (TString)gStyle->GetLineStyleString(linestyle);
   PrintFast(2," [");
   TObjArray *tokens = st.Tokenize(" ");
   for (Int_t j = 0; j<tokens->GetEntries(); j++) {
      Int_t it;
      sscanf(((TObjString*)tokens->At(j))->GetName(), "%d", &it);
      WriteInteger((Int_t)(it/4));
   }
   delete tokens;
   PrintFast(5,"] 0 d");
}


//______________________________________________________________________________
void TPDF::SetLineWidth(Width_t linewidth)
{
   // Change the line width

   if (linewidth == fLineWidth) return;
   fLineWidth = linewidth;
   WriteReal(fLineWidth);
   PrintFast(2," w");
}


//______________________________________________________________________________
void TPDF::SetMarkerColor( Color_t cindex )
{
   // Set color index for markers.

   fMarkerColor = cindex;
}


//______________________________________________________________________________
void TPDF::SetTextColor( Color_t cindex )
{
   // Set color index for text

   fTextColor = cindex;
}


//______________________________________________________________________________
void TPDF::Text(Double_t xx, Double_t yy, const char *chars)
{
   // Draw text
   //
   // xx: x position of the text
   // yy: y position of the text
   // chars: text to be drawn

   const Double_t kDEGRAD = TMath::Pi()/180.;
   char str[8];
   Double_t x = xx;
   Double_t y = yy;

   // Text color
   SetColor(Int_t(fTextColor));

   // Start the text
   if (!fCompress) PrintStr("@");
   PrintStr(" BT");

   // Font and text size
   sprintf(str," /F%d",abs(fTextFont)/10);
   PrintStr(str);
   Double_t wh = (Double_t)gPad->XtoPixel(gPad->GetX2());
   Double_t hh = (Double_t)gPad->YtoPixel(gPad->GetY1());
   Float_t tsize, ftsize;
   if (wh < hh) {
      tsize = fTextSize*wh;
      Int_t TTFsize = (Int_t)(tsize*kScale+0.5); // TTF size
      ftsize = (TTFsize*fXsize*gPad->GetAbsWNDC())/wh;
   } else {
      tsize = fTextSize*hh;
      Int_t TTFsize = (Int_t)(tsize*kScale+0.5); // TTF size
      ftsize = (TTFsize*fYsize*gPad->GetAbsHNDC())/hh;
   }
   Double_t fontsize = 72*(ftsize)/2.54;
   if( fontsize <= 0) return;
   WriteReal(fontsize);
   PrintStr(" Tf");

   // Text alignment
   Float_t tsizex = gPad->AbsPixeltoX(Int_t(tsize))-gPad->AbsPixeltoX(0);
   Float_t tsizey = gPad->AbsPixeltoY(0)-gPad->AbsPixeltoY(Int_t(tsize));
   Int_t txalh = fTextAlign/10;
   if (txalh < 1) txalh = 1; if (txalh > 3) txalh = 3;
   Int_t txalv = fTextAlign%10;
   if (txalv < 1) txalv = 1; if (txalv > 3) txalv = 3;
   if( txalv == 3) {
     y -= 0.8*tsizey*TMath::Cos(kDEGRAD*fTextAngle);
     x += 0.8*tsizex*TMath::Sin(kDEGRAD*fTextAngle);
   } else if( txalv == 2) {
     y -= 0.4*tsizey*TMath::Cos(kDEGRAD*fTextAngle);
     x += 0.4*tsizex*TMath::Sin(kDEGRAD*fTextAngle);
   }
   if (txalh > 1) {
      TText t;
      UInt_t w, h;
      t.SetTextSize(fTextSize);
      t.SetTextFont(fTextFont);
      t.GetTextExtent(w, h, chars);
      if(txalh == 2) x = x - (gPad->AbsPixeltoX(w)-gPad->AbsPixeltoX(0))/2;
      if(txalh == 3) x = x - (gPad->AbsPixeltoX(w)-gPad->AbsPixeltoX(0));
   }

   // Text angle
   if(fTextAngle == 0) {
      WriteReal(XtoPDF(x));
      WriteReal(YtoPDF(y));
      PrintStr(" Td");
   } else if (fTextAngle == 90) {
      PrintStr(" 0 1 -1 0");
      WriteReal(XtoPDF(x));
      WriteReal(YtoPDF(y));
      PrintStr(" Tm");
   } else {
      WriteReal(TMath::Cos(kDEGRAD*fTextAngle));
      WriteReal(TMath::Sin(kDEGRAD*fTextAngle));
      WriteReal(-TMath::Sin(kDEGRAD*fTextAngle));
      WriteReal(TMath::Cos(kDEGRAD*fTextAngle));
      WriteReal(XtoPDF(x));
      WriteReal(YtoPDF(y));
      PrintStr(" Tm");
   }

   // Ouput the text. Escape some characters if needed
   PrintStr(" (");
   Int_t len=strlen(chars);
   for (Int_t i=0; i<len;i++) {
      if (chars[i]!='\n') {
         if (chars[i]=='(' || chars[i]==')') {
            sprintf(str,"\\%c",chars[i]);
         } else {
            sprintf(str,"%c",chars[i]);
         }
         PrintStr(str);
      }
   }
   PrintStr(") Tj ET");
   if (!fCompress) PrintStr("@");
}


//______________________________________________________________________________
void TPDF::TextNDC(Double_t u, Double_t v, const char *chars)
{
   // Write a string of characters in NDC

   Double_t x = gPad->GetX1() + u*(gPad->GetX2() - gPad->GetX1());
   Double_t y = gPad->GetY1() + v*(gPad->GetY2() - gPad->GetY1());
   Text(x, y, chars);
}


//______________________________________________________________________________
Double_t TPDF::UtoPDF(Double_t u)
{
   // Convert U from NDC coordinate to PDF

   Double_t cm = fXsize*(gPad->GetAbsXlowNDC() + u*gPad->GetAbsWNDC());
   return 72*cm/2.54;
}


//______________________________________________________________________________
Double_t TPDF::VtoPDF(Double_t v)
{
   // Convert V from NDC coordinate to PDF

   Double_t cm = fYsize*(gPad->GetAbsYlowNDC() + v*gPad->GetAbsHNDC());
   return 72*cm/2.54;
}


//______________________________________________________________________________
Double_t TPDF::XtoPDF(Double_t x)
{
   // Convert X from world coordinate to PDF

   Double_t u = (x - gPad->GetX1())/(gPad->GetX2() - gPad->GetX1());
   return  UtoPDF(u);
}


//______________________________________________________________________________
Double_t TPDF::YtoPDF(Double_t y)
{
   // Convert Y from world coordinate to PDF

   Double_t v = (y - gPad->GetY1())/(gPad->GetY2() - gPad->GetY1());
   return  VtoPDF(v);
}


//______________________________________________________________________________
void TPDF::WriteCompressedBuffer()
{
   // Write the buffer in a compressed way

   z_stream stream;
   int err;
   char *out = new char[2*fLenBuffer];

   stream.next_in   = (Bytef*)fBuffer;
   stream.avail_in  = (uInt)fLenBuffer;
   stream.next_out  = (Bytef*)out;
   stream.avail_out = (uInt)2*fLenBuffer;
   stream.zalloc    = (alloc_func)0;
   stream.zfree     = (free_func)0;
   stream.opaque    = (voidpf)0;

   err = deflateInit(&stream, Z_DEFAULT_COMPRESSION);
   if (err != Z_OK) {
      Error("WriteCompressedBuffer", "error in deflateInit (zlib)");
      return;
   }

   err = deflate(&stream, Z_FINISH);
   if (err != Z_STREAM_END) {
      deflateEnd(&stream);
      Error("WriteCompressedBuffer", "error in deflate (zlib)");
      return;
   }

   err = deflateEnd(&stream);

   fStream->write(out, stream.total_out);

   fNByte += stream.total_out;
   fStream->write("\n",1); fNByte++;
   fLenBuffer = 0;
   delete [] out;
   fCompress = kFALSE;
}
