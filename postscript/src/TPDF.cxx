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

// to scale fonts to the same size as the old TT version
const Float_t kScale = 0.93376068;

ClassImp(TPDF)

//______________________________________________________________________________
//
// PDF driver
//

//______________________________________________________________________________
TPDF::TPDF() : TVirtualPS()
{
   // Default PDF constructor

   fStream = 0;
   fType   = 0;
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

   fStream = 0;
   Open(fname, wtype);
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
      ww *= gPad->GetWNDC();
      wh *= gPad->GetHNDC();
      Double_t ratio = wh/ww;
      xrange = fXsize;
      yrange = fXsize*ratio;
      if (yrange > fYsize) { yrange = fYsize; xrange = yrange/ratio;}
      fXsize = xrange; fYsize = yrange;
   }

   // Open OS file
   fStream   = new ofstream(fname,ios::out);
   if (fStream == 0) {
      printf("ERROR in TPDF::Open: Cannot open file:%s\n",fname);
      return;
   }

   gVirtualPS = this;

   for (Int_t i=0;i<512;i++) fBuffer[i] = ' ';

   fRange = kFALSE;

   // Set a default range
   Range(fXsize, fYsize);

   fObjPos = 0;
   fObjPosSize = 0;
   fNbObj = 0;
   fNbPage = 0;

   PrintStr("%PDF-1.4@");
   
   NewObject(1);
   PrintStr("<<@");
   PrintStr("/Type /Catalog@");
   PrintStr("/Outlines 2 0 R@");
   PrintStr("/Pages 3 0 R@");
   PrintStr(">>@");
   PrintStr("endobj@");
   
   NewObject(2);
   PrintStr("<<@");
   PrintStr("/Type /Outlines@");
   PrintStr("/Count 0@");
   PrintStr(">>@");
   PrintStr("endobj@");
   
   NewObject(4);
   PrintStr("[/PDF /Text]@");
   PrintStr("endobj@");
  
   FontEncode();

   NewPage();
}

//______________________________________________________________________________
TPDF::~TPDF()
{
   // Default PDF destructor

   Close();

   if (fObjPos) delete [] fObjPos;
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
   PrintStr("endstream@");
   Int_t StreamLength = fNByte-fStartStream-10;
   PrintStr("endobj@");
   NewObject(3*fNbPage+18);
   WriteInteger(StreamLength, 0);
   PrintStr("endobj@");

   // List of all the pages
   NewObject(3);
   PrintStr("<<@");
   PrintStr("/Type /Pages@");
   PrintStr("/Count");
   WriteInteger(fNbPage);
   PrintStr("@");
   PrintStr("/Kids [");
   for (i=1; i<=fNbPage; i++) {
      WriteInteger(3*fNbPage+16);
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
   char str[20];
   for (i=0; i<fNbObj; i++) {
      sprintf(str,"%10.10d 00000 n @",fObjPos[i]);
      PrintStr(str);
   }

   // Trailer
   PrintStr("trailer@");
   PrintStr("<<@"); 
   PrintStr("/Size 8@"); 
   PrintStr("/Root 1 0 R@");
   PrintStr(">>@");
   PrintStr("startxref@");
   WriteInteger(RefInd, 0);
   PrintStr("%%EOF@");

   // Close file stream
   if (fStream) { fStream->close(); delete fStream; fStream = 0;}

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
void TPDF::Off()
{
   // Deactivate an already open PDF file

   gVirtualPS = 0;
}

//______________________________________________________________________________
void TPDF::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t  y2)
{
   // Draw a Box

   static Double_t x[4], y[4];
   Int_t ix1 = XtoPDF(x1);
   Int_t ix2 = XtoPDF(x2);
   Int_t iy1 = YtoPDF(y1);
   Int_t iy2 = YtoPDF(y2);
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
         WriteInteger(ix1);
         WriteInteger(iy1);
         WriteInteger(ix2 - ix1);
         WriteInteger(iy2 - iy1);
         PrintFast(6," re f*");
      }
   }
   if (fillis == 1) {
      SetColor(fFillColor);
      WriteInteger(ix1);
      WriteInteger(iy1);
      WriteInteger(ix2 - ix1);
      WriteInteger(iy2 - iy1);
      PrintFast(6," re f*");
   }
   if (fillis == 0) {
      SetColor(fLineColor);
      WriteInteger(ix1);
      WriteInteger(iy1);
      WriteInteger(ix2 - ix1);
      WriteInteger(iy2 - iy1);
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

   static Int_t xps[7], yps[7];
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

   WriteInteger(xps[0]);
   WriteInteger(yps[0]);
   PrintFast(2," m");

   for (i=1;i<7;i++) MovePS(xps[i], yps[i]);
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

   WriteInteger(xps[0]);
   WriteInteger(yps[0]);
   PrintFast(2," m");

   for (i=1;i<7;i++) MovePS(xps[i], yps[i]);
   PrintFast(3," f*");
}

//______________________________________________________________________________
void TPDF::DrawPolyLine(Int_t nn, TPoints *xy)
{
   // Draw a PolyLine
   //
   //  Draw a polyline through  the points  xy.
   //  If NN=1 moves only to point x,y.
   //  If NN=0 the x,y are  written  in the PDF file
   //     according to the current transformation.
   //  If NN>0 the line is clipped as a line.
   //  If NN<0 the line is clipped as a fill area.

   Int_t  n;

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

   WriteInteger(XtoPDF(xy[0].GetX()));
   WriteInteger(YtoPDF(xy[0].GetY()));
   if( n <= 1) {
      if( n == 0) return;
      PrintFast(2," m");
      return;
   }

   PrintFast(2," m");

   for (Int_t i=1;i<n;i++) MovePS(XtoPDF(xy[i].GetX()), YtoPDF(xy[i].GetY()));

   if (nn > 0 ) {
      if (xy[0].GetX() == xy[n-1].GetX() && xy[0].GetY() == xy[n-1].GetY()) PrintFast(3," cl");
      PrintFast(2," S");
   } else {
      PrintFast(3," f*");
   }
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

   WriteInteger(UtoPDF(xy[0].GetX()));
   WriteInteger(VtoPDF(xy[0].GetY()));
   if( n <= 1) {
      if( n == 0) return;
      PrintFast(2," m");
      return;
   }

   PrintFast(2," m");

   for (Int_t i=1;i<n;i++) MovePS(UtoPDF(xy[i].GetX()), VtoPDF(xy[i].GetY()));

   if (nn > 0 ) {
      if (xy[0].GetX() == xy[n-1].GetX() && xy[0].GetY() == xy[n-1].GetY()) PrintFast(3," cl");
      PrintFast(2," S");
   } else {
      PrintFast(3," f*");
   }
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
            return;
         }
         if (fasi > 0 && fasi < 26) {
            SetFillPatterns(fasi, Int_t(fFillColor));
         }
      }
   }

   WriteInteger(XtoPDF(xw[0]));
   WriteInteger(YtoPDF(yw[0]));
   if( n <= 1) {
      if( n == 0) return;
      PrintFast(2," m");
      return;
   }

   PrintFast(2," m");

   for (Int_t i=1;i<n;i++) MovePS(XtoPDF(xw[i]), YtoPDF(yw[i]));

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
         return;
      }
      PrintFast(3," f*");
   }
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
            return;
         }
         if (fasi > 0 && fasi < 26) {
            SetFillPatterns(fasi, Int_t(fFillColor));
         }
      }
   }

   WriteInteger(XtoPDF(xw[0]));
   WriteInteger(YtoPDF(yw[0]));
   if( n <= 1) {
      if( n == 0) return;
      PrintFast(2," m");
      return;
   }

   PrintFast(2," m");

   for (Int_t i=1;i<n;i++) MovePS(XtoPDF(xw[i]), YtoPDF(yw[i]));

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
         return;
      }
      PrintFast(3," f*");
   }
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
void TPDF::FontEncode()
{
   // Font encoding

   static const char *sdtfonts[] = {
   "/Times-Italic"         , "/Times-Bold"         , "/Times-BoldItalic",
   "/Helvetica"            , "/Helvetica-Oblique"  , "/Helvetica-Bold"  ,
   "/Helvetica-BoldOblique", "/Courier"            , "/Courier-Oblique" ,
   "/Courier-Bold"         , "/Courier-BoldOblique", "/Symbol"          ,
   "/Times-Roman"          , "/ZapfDingbats"};

   for (Int_t i=0; i<14; i++) {
      NewObject(5+i);
      PrintStr("<<@");
      PrintStr("/Type /Font@");
      PrintStr("/Subtype /Type1@");
      PrintStr("/BaseFont ");
      PrintStr(sdtfonts[i]);
      PrintStr("@");
      PrintStr(">>@");
      PrintStr("endobj@");
   }
}

//______________________________________________________________________________
void TPDF::MovePS(Int_t ix, Int_t iy)
{
   // Move to a new position

   WriteInteger(ix);
   WriteInteger(iy);
   PrintFast(2," l");
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
   // Start a new  PDF page.

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
      PrintStr("endstream@");
      Int_t StreamLength = fNByte-fStartStream-10;
      PrintStr("endobj@");
      NewObject(3*(fNbPage-1)+18);
      WriteInteger(StreamLength, 0);
      PrintStr("endobj@");
   }

   // Start a new page
   NewObject(3*fNbPage+16);
   PrintStr("<<@");
   PrintStr("/Type /Page@");
   PrintStr("/Parent 3 0 R@");
   PrintStr("/Resources <<@");
   PrintStr("/Font <<@");
   PrintStr("/F01 5 0 R ");
   PrintStr("/F02 6 0 R ");
   PrintStr("/F03 7 0 R ");
   PrintStr("/F04 8 0 R ");
   PrintStr("/F05 9 0 R ");
   PrintStr("/F06 10 0 R ");
   PrintStr("/F07 11 0 R ");
   PrintStr("/F08 12 0 R ");
   PrintStr("/F09 13 0 R ");
   PrintStr("/F10 14 0 R ");
   PrintStr("/F11 15 0 R ");
   PrintStr("/F12 16 0 R ");
   PrintStr("/F13 17 0 R ");
   PrintStr("/F14 18 0 R ");
   PrintStr(">>@");
   PrintStr("/ProcSet 4 0 R >>@");

   Double_t xlow=0, ylow=0, xup=1, yup=1;
   if (gPad) {
      xlow = gPad->GetAbsXlowNDC();
      xup  = xlow + gPad->GetAbsWNDC();
      ylow = gPad->GetAbsYlowNDC();
      yup  = ylow + gPad->GetAbsHNDC();
   }
   PrintStr("/MediaBox [");
   WriteInteger(CMtoPDF(fXsize*xlow),0);
   WriteInteger(CMtoPDF(fYsize*ylow));
   WriteInteger(CMtoPDF(fXsize*xup));
   WriteInteger(CMtoPDF(fYsize*yup));
   PrintStr("]");
   PrintStr("@");
   PrintStr("/CropBox [");
   WriteInteger(CMtoPDF(fXsize*xlow),0);
   WriteInteger(CMtoPDF(fYsize*ylow));
   WriteInteger(CMtoPDF(fXsize*xup));
   WriteInteger(CMtoPDF(fYsize*yup));
   PrintStr("]");
   PrintStr("@");
   
   PrintStr("/Rotate 0@");
   PrintStr("/Contents");
   WriteInteger(3*fNbPage+17);
   PrintStr(" 0 R");
   PrintStr("@");
   PrintStr(">>@");
   PrintStr("endobj@");
   
   NewObject(3*fNbPage+17);
   PrintStr("<</Length");
   WriteInteger(3*fNbPage+18);
   PrintStr(" 0 R >>");
   PrintStr("@");
   PrintStr("stream@");
   fStartStream = fNByte;
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
   if      (linestyle == 2) {PrintStr(" [3 3] 0 d");}
   else if (linestyle == 3) {PrintStr(" [1 2] 0 d");}
   else if (linestyle == 4) {PrintStr(" [3 4 1 4] 0 d");}
   else                     {PrintStr(" [] 0 d");}
}

//______________________________________________________________________________
void TPDF::SetLineWidth(Width_t linewidth)
{
   // Change the line width

   if (linewidth == fLineWidth) return;
   fLineWidth = linewidth;
   WriteInteger(Int_t(fLineWidth));
   PrintFast(2," w");
}

//______________________________________________________________________________
void TPDF::SetMarkerColor( Color_t cindex )
{
   // Set color index for markers.

   fMarkerColor = cindex;
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

   char str[8];

   // Start the text
   PrintStr("@");
   PrintStr("BT");

   // Font and text size
   sprintf(str," /F%2.2d",abs(fTextFont)/10);
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

   // Text angle
   if(fTextAngle == 0) {
      WriteReal(XtoPDF(xx));
      WriteReal(YtoPDF(yy));
      PrintStr(" Td");
   } else if (fTextAngle == 90) {
      PrintStr(" 0 1 -1 0");
      WriteReal(XtoPDF(xx));
      WriteReal(YtoPDF(yy));
      PrintStr(" Tm");
   } else {
      Double_t DegRad = TMath::Pi()/180.;
      WriteReal(TMath::Cos(DegRad*fTextAngle));
      WriteReal(TMath::Sin(DegRad*fTextAngle));
      WriteReal(-TMath::Sin(DegRad*fTextAngle));
      WriteReal(TMath::Cos(DegRad*fTextAngle));
      WriteReal(XtoPDF(xx));
      WriteReal(YtoPDF(yy));
      PrintStr(" Tm");
   }

   // Ouput the text. Escape some characters if needed
   PrintStr(" (");
   Int_t len=strlen(chars);
   for (Int_t i=0; i<len;i++) {
      if (chars[i]=='(' || chars[i]==')') {
         sprintf(str,"\\%c",chars[i]);
      } else {
         sprintf(str,"%c",chars[i]);
      }
      PrintStr(str);
   }
   PrintStr(") Tj");
   PrintStr(" ET");
   PrintStr("@");
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
Int_t TPDF::UtoPDF(Double_t u)
{
   // Convert U from NDC coordinate to PDF

   Double_t cm = fXsize*(gPad->GetAbsXlowNDC() + u*gPad->GetAbsWNDC());
   return Int_t(0.5 + 72*cm/2.54);
}

//______________________________________________________________________________
Int_t TPDF::VtoPDF(Double_t v)
{
   // Convert V from NDC coordinate to PDF

   Double_t cm = fYsize*(gPad->GetAbsYlowNDC() + v*gPad->GetAbsHNDC());
   return Int_t(0.5 + 72*cm/2.54);
}

//______________________________________________________________________________
Int_t TPDF::XtoPDF(Double_t x)
{
   // Convert X from world coordinate to PDF

   Double_t u = (x - gPad->GetX1())/(gPad->GetX2() - gPad->GetX1());
   return  UtoPDF(u);
}

//______________________________________________________________________________
Int_t TPDF::YtoPDF(Double_t y)
{
   // Convert Y from world coordinate to PDF

   Double_t v = (y - gPad->GetY1())/(gPad->GetY2() - gPad->GetY1());
   return  VtoPDF(v);
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
void TPDF::DrawPolyMarker(Int_t, Float_t *, Float_t *)
{
   Warning("TPDF::DrawPolyMarker", "not yet implemented");
}

//______________________________________________________________________________
void TPDF::DrawPolyMarker(Int_t, Double_t *, Double_t *)
{
   Warning("TPDF::DrawPolyMarker", "not yet implemented");
}
