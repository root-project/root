// @(#)root/postscript:$Name:  $:$Id: TSVG.cxx,v 1.0 2002/02/05 14:00:00 couet Exp $
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
// TSVG                                                                 //
//                                                                      //
// Graphics interface to SVG.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef WIN32
#pragma optimize("",off)
#endif

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <fstream.h>

#include "TROOT.h"
#include "TColor.h"
#include "TVirtualPad.h"
#include "TPoints.h"
#include "TSVG.h"
#include "TStyle.h"
#include "TMath.h"

ClassImp(TSVG)

//______________________________________________________________________________
//Begin_Html
/*
<a href="http://www.w3.org/Graphics/SVG/Overview.htm8"><b>SVG</b></a> is a 
language for describing two-dimensional graphics in XML. <b>SVG</b> allows for
three types of graphic objects: vector graphic shapes, images and text. 
Graphical objects can be grouped, styled, transformed and composed into 
previously rendered objects. The feature set includes nested transformations,
clipping paths, alpha masks, filter effects and template objects. <b>SVG</b> 
drawings can be interactive and dynamic. Animations can be defined and 
triggered either declaratively or via scripting.
<p>
The way to access <b>SVG</b> in <b>ROOT</b> (in my private version only) is the 
following:
<PRE>
   <A HREF="html/TSVG.html">TSVG</A> mysvg(&quot;myfile.svg&quot;)
   object-&gt;Draw();
   mysvg.Close();
</PRE>
The result is the ASCII file <tt>myfile.svg</tt>, it is best viewed with 
Internet Explorer and you need the
<a href="http://www.adobe.com/svg/viewer/install/main.html">Adobe <b>SVG</b>
Viewer</a>. To zoom using the Adobe <b>SVG</b> Viewer, position the mouse over
the area you want to zoom and click the right button. To define the zoom area,
use Control+drag to mark the boundaries of the zoom area. To pan, use Alt+drag.
By clicking with the right mouse button on the <b>SVG</b> graphics you will get
a pop-up menu giving other ways to interact with the image.
<p>
<b>SVG</b> files can be used directly in compressed mode to minimize the time
transfer over the network. Compressed <b>SVG</b> files should be created using
<tt>gzip</tt> on a normal ASCII <b>SVG</b> file and should then be renamed
using the file extension <tt>.svgz</tt>.
*/
//End_Html

//______________________________________________________________________________
TSVG::TSVG() : TVirtualPS()
{
   // Default SVG constructor

   fStream = 0;
   fType   = 0;
   gVirtualPS = this;
}

//______________________________________________________________________________
TSVG::TSVG(const char *fname, Int_t wtype) : TVirtualPS(fname, wtype)
{
   // Initialize the SVG interface
   //
   //  fname : SVG file name
   //  wtype : SVG workstation type. Not used in the SVG driver. But as TSVG
   //          inherits from TVirtualPS it should be kept. Anyway it is not 
   //          necessary to specify this parameter at creation time because it
   //          a default value (which is ignore in the SVG case).

   fStream = 0;
   Open(fname, wtype);
}

//______________________________________________________________________________
void TSVG::Open(const char *fname, Int_t wtype)
{
   // Open a SVG file

   if (fStream) {
      Warning("Open", "SVG file already open");
      return;
   }
 
   fType = abs(wtype);
   SetLineScale(gStyle->GetLineScalePS());
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
      printf("ERROR in TSVG::Open: Cannot open file:%s\n",fname);
      return;
   }

   gVirtualPS = this;

   for (Int_t i=0;i<512;i++) fBuffer[i] = ' ';

   fBoundingBox = kFALSE;

   fRange       = kFALSE;

   // Set a default range
   Range(fXsize, fYsize);

   NewPage();
}

//______________________________________________________________________________
TSVG::~TSVG()
{
   // Default SVG destructor

   Close();
}

//______________________________________________________________________________
void TSVG::Close(Option_t *)
{
   // Close a SVG file
   if (!gVirtualPS) return;
   if (gPad) gPad->Update();
   PrintStr("</svg>@");

   // Close file stream
   if (fStream) { fStream->close(); delete fStream; fStream = 0;}

   gVirtualPS = 0;
}

//______________________________________________________________________________
void TSVG::On()
{
   // Activate an already open SVG file

   // fType is used to know if the SVG file is open. Unlike TPostScript, TSVG
   // has no "workstation type". In fact there is only one SVG type.

   if (!fType) {
      Error("On", "no SVG file open");
      Off();
      return;
   }
   gVirtualPS = this;
}

//______________________________________________________________________________
void TSVG::Off()
{
   // Deactivate an already open SVG file

   gVirtualPS = 0;
}

//______________________________________________________________________________
void TSVG::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t  y2)
{
   // Draw a Box

   static Double_t x[4], y[4];
   Int_t ix1 = XtoSVG(x1);
   Int_t ix2 = XtoSVG(x2);
   Int_t iy1 = YtoSVG(y1);
   Int_t iy2 = YtoSVG(y2);
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
         PrintFast(9,"<rect x=\"");
         WriteInteger(ix1, 0);
         PrintFast(5,"\" y=\"");
         WriteInteger(iy2, 0);
         PrintFast(9,"\" width=\"");
         WriteInteger(ix2-ix1, 0);
         PrintFast(10,"\" height=\"");
         WriteInteger(iy1-iy2, 0);
         PrintFast(7,"\" fill=");
         SetColor(5);
         PrintFast(2,"/>");
      }
   }
   if (fillis == 1) {
      PrintFast(9,"<rect x=\"");
      WriteInteger(ix1, 0);
      PrintFast(5,"\" y=\"");
      WriteInteger(iy2, 0);
      PrintFast(9,"\" width=\"");
      WriteInteger(ix2-ix1, 0);
      PrintFast(10,"\" height=\"");
      WriteInteger(iy1-iy2, 0);
      PrintFast(7,"\" fill=");
      SetColor(fFillColor);
      PrintFast(2,"/>");
   }
   if (fillis == 0) {
      PrintFast(9,"<rect x=\"");
      WriteInteger(ix1, 0);
      PrintFast(5,"\" y=\"");
      WriteInteger(iy2, 0);
      PrintFast(9,"\" width=\"");
      WriteInteger(ix2-ix1, 0);
      PrintFast(10,"\" height=\"");
      WriteInteger(iy1-iy2, 0);
      PrintFast(21,"\" fill=\"none\" stroke=");
      SetColor(fLineColor);
      PrintFast(2,"/>");
   }
}

//______________________________________________________________________________
void TSVG::DrawFrame(Double_t xl, Double_t yl, Double_t xt, Double_t  yt,
                            Int_t mode, Int_t border, Int_t dark, Int_t light)
{
   // Draw a Frame around a box
   //
   // mode = -1  the box looks as it is behind the screen
   // mode =  1  the box looks as it is in front of the screen
   // border is the border size in already precomputed SVG units dark is the 
   // color for the dark part of the frame light is the color for the light 
   // part of the frame

   static Int_t xps[7], yps[7];
   Int_t i, ixd0, iyd0, idx, idy, ixdi, iydi, ix, iy;

   //- Draw top&left part of the box

   xps[0] = XtoSVG(xl);          yps[0] = YtoSVG(yl);
   xps[1] = xps[0] + border;     yps[1] = yps[0] - border;
   xps[2] = xps[1];              yps[2] = YtoSVG(yt) + border;
   xps[3] = XtoSVG(xt) - border; yps[3] = yps[2];
   xps[4] = XtoSVG(xt);          yps[4] = YtoSVG(yt);
   xps[5] = xps[0];              yps[5] = yps[4];
   xps[6] = xps[0];              yps[6] = yps[0];

   ixd0 = xps[0];
   iyd0 = yps[0];
   PrintFast(10,"<path d=\"M");
   WriteInteger(ixd0, 0);
   PrintFast(1,",");
   WriteInteger(iyd0, 0);

   idx = 0;
   idy = 0;
   for (i=1; i<7; i++) {
      ixdi = xps[i];
      iydi = yps[i];
      ix   = ixdi - ixd0;
      iy   = iydi - iyd0;
      ixd0 = ixdi;
      iyd0 = iydi;
      if( ix && iy) {
         if( idx ) { MovePS(idx,0); idx = 0; }
         if( idy ) { MovePS(0,idy); idy = 0; }
         MovePS(ix,iy);
         continue;
      }
      if ( ix ) {
         if( idy )  { MovePS(0,idy); idy = 0; }
         if( !idx ) { idx = ix; continue;}
         if( ix*idx > 0 ) {
            idx += ix;
         } else { 
            MovePS(idx,0);
            idx  = ix; 
         }
         continue;
      }
      if( iy ) {
         if( idx ) { MovePS(idx,0); idx = 0; }
         if( !idy) { idy = iy; continue;}
         if( iy*idy > 0 ) {
            idy += iy;
         } else { 
            MovePS(0,idy);
            idy  = iy;
         }
      }
   }
   if( idx ) MovePS(idx,0);
   if( idy ) MovePS(0,idy);
   PrintFast(8,"z\" fill=");
   if (mode == -1) {
      SetColor(dark);
   } else {
      SetColor(light);
   }
   PrintFast(2,"/>");

   //- Draw bottom&right part of the box
   xps[0] = XtoSVG(xl);          yps[0] = YtoSVG(yl);
   xps[1] = xps[0] + border;     yps[1] = yps[0] - border;
   xps[2] = XtoSVG(xt) - border; yps[2] = yps[1];
   xps[3] = xps[2];              yps[3] = YtoSVG(yt) + border;
   xps[4] = XtoSVG(xt);          yps[4] = YtoSVG(yt);
   xps[5] = xps[4];              yps[5] = yps[0];
   xps[6] = xps[0];              yps[6] = yps[0];

   ixd0 = xps[0];
   iyd0 = yps[0];
   PrintFast(10,"<path d=\"M");
   WriteInteger(ixd0, 0);
   PrintFast(1,",");
   WriteInteger(iyd0, 0);

   idx = 0;
   idy = 0;
   for (i=1;i<7;i++) {
      ixdi = xps[i];
      iydi = yps[i];
      ix   = ixdi - ixd0;
      iy   = iydi - iyd0;
      ixd0 = ixdi;
      iyd0 = iydi;
      if( ix && iy) {
         if( idx ) { MovePS(idx,0); idx = 0; }
         if( idy ) { MovePS(0,idy); idy = 0; }
         MovePS(ix,iy);
         continue;
      }
      if ( ix ) {
         if( idy )  { MovePS(0,idy); idy = 0; }
         if( !idx ) { idx = ix; continue;}
         if( ix*idx > 0 ) {
            idx += ix;
         } else {
            MovePS(idx,0);
            idx  = ix;
         }
         continue;
      }
      if( iy ) {
         if( idx ) { MovePS(idx,0); idx = 0; }
         if( !idy) { idy = iy; continue;}
         if( iy*idy > 0 ) {
            idy += iy;
         } else {
            MovePS(0,idy);
            idy  = iy;
         }
      }
   }
   if( idx ) MovePS(idx,0);
   if( idy ) MovePS(0,idy);
   PrintFast(8,"z\" fill=");
   if (mode == -1) {
      SetColor(light);
   } else {
      SetColor(dark);
   }
   PrintFast(2,"/>");
}

//______________________________________________________________________________
void TSVG::DrawPolyLine(Int_t nn, TPoints *xy)
{
   // Draw a PolyLine
   //
   //  Draw a polyline through  the points  xy.
   //  If NN=1 moves only to point x,y.
   //  If NN=0 the x,y are  written  in the SVG        file
   //     according to the current transformation.
   //  If NN>0 the line is clipped as a line.
   //  If NN<0 the line is clipped as a fill area.

   Int_t  n, ixd0, iyd0, idx, idy, ixdi, iydi, ix, iy;

   if (nn > 0) {
      n = nn;
///     SetLineStyle(fLineStyle);
///     SetLineWidth(fLineWidth);
///     SetColor(Int_t(fLineColor));
   } else {
      n = -nn;
///     SetLineStyle(1);
///     SetLineWidth(1);
///     SetColor(Int_t(fLineColor));
   }

   ixd0 = XtoSVG(xy[0].GetX());
   iyd0 = YtoSVG(xy[0].GetY());
///  WriteInteger(ixd0);
///  WriteInteger(iyd0);
   if( n <= 1) {
      if( n == 0) return;
///     PrintFast(2," m");
      return;
   }

   PrintFast(2," m");
   idx = 0;
   idy = 0;
   for (Int_t i=1;i<n;i++) {
      ixdi = XtoSVG(xy[i].GetX());
      iydi = YtoSVG(xy[i].GetY());
      ix   = ixdi - ixd0;
      iy   = iydi - iyd0;
      ixd0 = ixdi;
      iyd0 = iydi;
      if( ix && iy) {
         if( idx ) { MovePS(idx,0); idx = 0; }
         if( idy ) { MovePS(0,idy); idy = 0; }
         MovePS(ix,iy);
         continue;
      }
      if ( ix ) {
         if( idy )  { MovePS(0,idy); idy = 0; }
         if( !idx ) { idx = ix; continue;}
         if( ix*idx > 0 ) {
            idx += ix;
         } else { 
            MovePS(idx,0);
            idx  = ix;
         }
         continue;
      }
      if( iy ) {
         if( idx ) { MovePS(idx,0); idx = 0; }
         if( !idy) { idy = iy; continue;}
         if( iy*idy > 0 ) {
            idy += iy;
         } else { 
            MovePS(0,idy);
            idy  = iy;
         }
      }
   }
   if( idx ) MovePS(idx,0);
   if( idy ) MovePS(0,idy);

   if (nn > 0 ) {
///     if (xy[0].GetX() == xy[n-1].GetX() && xy[0].GetY() == xy[n-1].GetY()) PrintFast(3," cl");
///     PrintFast(2," s");
   } else {
///     PrintFast(2," f");
   }
}

//______________________________________________________________________________
void TSVG::DrawPolyLineNDC(Int_t nn, TPoints *xy)
{
   // Draw a PolyLine in NDC space
   //
   //  Draw a polyline through  the points  xy.
   //  If NN=1 moves only to point x,y.
   //  If NN=0 the x,y are  written  in the SVG        file
   //     according to the current transformation.
   //  If NN>0 the line is clipped as a line.
   //  If NN<0 the line is clipped as a fill area.

   Int_t  n, ixd0, iyd0, idx, idy, ixdi, iydi, ix, iy;

   if (nn > 0) {
      n = nn;
///     SetLineStyle(fLineStyle);
///     SetLineWidth(fLineWidth);
///     SetColor(Int_t(fLineColor));
   } else {
      n = -nn;
///     SetLineStyle(1);
///     SetLineWidth(1);
///     SetColor(Int_t(fLineColor));
   }

   ixd0 = UtoSVG(xy[0].GetX());
   iyd0 = VtoSVG(xy[0].GetY());
///  WriteInteger(ixd0);
///  WriteInteger(iyd0);
   if( n <= 1) {
      if( n == 0) return;
///     PrintFast(2," m");
      return;
   }

///  PrintFast(2," m");
   idx = 0;
   idy = 0;
   for (Int_t i=1;i<n;i++) {
      ixdi = UtoSVG(xy[i].GetX());
      iydi = VtoSVG(xy[i].GetY());
      ix   = ixdi - ixd0;
      iy   = iydi - iyd0;
      ixd0 = ixdi;
      iyd0 = iydi;
      if( ix && iy) {
         if( idx ) { MovePS(idx,0); idx = 0; }
         if( idy ) { MovePS(0,idy); idy = 0; }
         MovePS(ix,iy);
         continue;
      }
      if ( ix ) {
         if( idy )  { MovePS(0,idy); idy = 0; }
         if( !idx ) { idx = ix; continue;}
         if( ix*idx > 0 ) {
            idx += ix;
         } else { 
            MovePS(idx,0);
            idx  = ix;
         }
         continue;
      }
      if( iy ) {
         if( idx ) { MovePS(idx,0); idx = 0; }
         if( !idy) { idy = iy; continue;}
         if( iy*idy > 0 ) {
            idy += iy;
         } else { 
            MovePS(0,idy);
            idy  = iy;
         }
      }
   }
   if( idx ) MovePS(idx,0);
   if( idy ) MovePS(0,idy);

   if (nn > 0 ) {
      if (xy[0].GetX() == xy[n-1].GetX() && xy[0].GetY() == xy[n-1].GetY()) PrintFast(3," cl");
///     PrintFast(2," s");
   } else {
///     PrintFast(2," f");
   }
}

//______________________________________________________________________________
void TSVG::DrawPS(Int_t nn, Double_t *xw, Double_t *yw)
{
   // This function defines a path with xw and yw and draw it according the 
   // value of nn: 
   //
   //  If nn>0 a line is drawn.
   //  If nn<0 a closed polygon is drawn.

///   static Float_t dyhatch[24] = {.0075,.0075,.0075,.0075,.0075,.0075,.0075,.0075,
///                                 .01  ,.01  ,.01  ,.01  ,.01  ,.01  ,.01  ,.01  ,
///                                 .015 ,.015 ,.015 ,.015 ,.015 ,.015 ,.015 ,.015};
///   static Float_t anglehatch[24] = {180, 90,135, 45,150, 30,120, 60,
///                                    180, 90,135, 45,150, 30,120, 60,
///                                    180, 90,135, 45,150, 30,120, 60};
   Int_t  n, ixd0, iyd0, idx, idy, ixdi, iydi, ix, iy, fais, fasi;
   fais = fasi = 0;

   if (nn > 0) {
      n = nn;
   } else {
      n = -nn;
      fais = fFillStyle/1000;
      fasi = fFillStyle%1000;
      if (fais == 3 || fais == 2) {
         if (fasi > 100 && fasi <125) {
///        DrawHatch(dyhatch[fasi-101],anglehatch[fasi-101], n, xw, yw);
            return;
         }
         if (fasi > 0 && fasi < 26) {
///        SetFillPatterns(fasi, Int_t(fFillColor));
         }
      }
   }

    if( n <= 1) {
       Error("DrawPS", "Two points are needed");
       return;
    }

    ixd0 = XtoSVG(xw[0]);
    iyd0 = YtoSVG(yw[0]);

    PrintFast(10,"<path d=\"M");
    WriteInteger(ixd0, 0);
    PrintFast(1,",");
    WriteInteger(iyd0, 0);

    idx = idy = 0;
    for (Int_t i=1;i<n;i++) {
       ixdi = XtoSVG(xw[i]);
       iydi = YtoSVG(yw[i]);
       ix   = ixdi - ixd0;
       iy   = iydi - iyd0;
       ixd0 = ixdi;
       iyd0 = iydi;
       if( ix && iy) {
          if( idx ) { MovePS(idx,0); idx = 0; }
          if( idy ) { MovePS(0,idy); idy = 0; }
          MovePS(ix,iy);
       } else if ( ix ) {
          if( idy )  { MovePS(0,idy); idy = 0;}
          if( !idx ) { idx = ix;}
          else if( TMath::Sign(ix,idx) == ix )       idx += ix;
          else { MovePS(idx,0);  idx  = ix;}
       } else if( iy ) {
          if( idx ) { MovePS(idx,0); idx = 0;}
          if( !idy) { idy = iy;}
          else if( TMath::Sign(iy,idy) == iy)         idy += iy;
          else { MovePS(0,idy);    idy  = iy;}
       }
    }
    if (idx) MovePS(idx,0);
    if (idy) MovePS(0,idy);
 
    if (nn > 0 ) {
       if (xw[0] == xw[n-1] && yw[0] == yw[n-1]) PrintFast(1,"z");
       PrintFast(21,"\" fill=\"none\" stroke=");
       SetColor(fLineColor);
       if(fLineWidth > 1.) {
          PrintFast(15," stroke-width=\"");
          WriteInteger(Int_t(fLineWidth), 0);
          PrintFast(1,"\"");
       }
       if (fLineStyle == 2) {
          PrintFast(23," stroke-dasharray=\"3,3\"");
       } else if (fLineStyle == 3) {
          PrintFast(23," stroke-dasharray=\"1,4\"");
       } else if (fLineStyle == 4) {
          PrintFast(27," stroke-dasharray=\"3,4,1,4\"");
       }
       PrintFast(2,"/>");
    } else {
       PrintFast(8,"z\" fill=");
       if (fais == 0) {
          PrintFast(14,"\"none\" stroke=");
          SetColor(fFillColor);
///      } else if (fais == 3 || fais == 2) {
///        if (fasi > 0 && fasi < 26) {
///           Put SVG patterns here
       } else {
          SetColor(fFillColor);
       }
       PrintFast(2,"/>");
    }
}

//______________________________________________________________________________
void TSVG::Initialize()
{
   // Initialize the SVG file. The main task of the function is to ouput the 
   // SVG header file which constist in <title>, <desc> and <defs>. The
   // HeaderPS porvided by the user program is written in the <defs> part.

   // Title
   PrintStr("<title>@");
   const char *pstitle = gStyle->GetTitlePS();
   if (!strlen(pstitle)) pstitle = gPad->GetMother()->GetTitle();
   PrintStr(GetName());
   PrintStr(pstitle);
   PrintStr("@");
   PrintStr("</title>@");

   // Description
   PrintStr("<desc>@");
   PrintFast(22,"Creator: ROOT Version ");
   PrintStr(gROOT->GetVersion());
   PrintStr("@");
   PrintFast(14,"CreationDate: ");
   TDatime t;
   PrintStr(t.AsString());
   //Check a special header is defined in the current style
   Int_t nh = strlen(gStyle->GetHeaderPS());
   if (nh) {
      PrintFast(nh,gStyle->GetHeaderPS());
   }
   PrintStr("</desc>@");

   // Definitions
   PrintStr("<defs>@");
   PrintStr("</defs>@");

}

//______________________________________________________________________________
void TSVG::MovePS(Int_t ix, Int_t iy)
{
   // Move to a new position (ix, iy). The move is done in relative coordinates
   // which allows to have short numbers which decraese the sive of the file.
   // This function use the full power of the SVG's paths by using the
   // horizontal and vertical move whenever it is possible.  

   if (ix != 0 && iy != 0)  {
      PrintFast(1,"l");
      WriteInteger(ix, 0);
      PrintFast(1,",");
      WriteInteger(iy, 0);
   } else if (ix != 0)  {
      PrintFast(1,"h");
      WriteInteger(ix, 0);
   } else if (iy != 0)  {
      PrintFast(1,"v");
      WriteInteger(iy, 0);
   }
}

//______________________________________________________________________________
void TSVG::NewPage()
{
   // Start the SVG page. This function initialize the pad conversion 
   // coefficients and ouput the <svg> directive which is close later in the
   // the function Close.

   // Compute pad conversion coefficients
   if (gPad) {
      Double_t ww   = gPad->GetWw();
      Double_t wh   = gPad->GetWh();
      fYsize        = fXsize*wh/ww;
   } else {
      fYsize = 27;
   }
 
   // <svg> directive. It defines the viewBox.
   if(!fBoundingBox) {
      PrintStr("@<svg viewBox=\"0 0");
      WriteInteger(CMtoSVG(fXsize));
      fYsizeSVG = CMtoSVG(fYsize);
      WriteInteger(fYsizeSVG);
      PrintStr("\" xmlns=\"http://www.w3.org/2000/svg\">");
      PrintStr("@");
      Initialize();
      fBoundingBox  = kTRUE;
   }
}

//______________________________________________________________________________
void TSVG::Range(Float_t xsize, Float_t ysize)
{
   // Set the range for the paper in centimeters

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
void TSVG::SetFillColor( Color_t cindex )
{
   // Set color index for fill areas

   fFillColor = cindex;
   if (gStyle->GetFillColor() <= 0) cindex = 0;
}

//______________________________________________________________________________
void TSVG::SetLineColor( Color_t cindex )
{
   // Set color index for lines

   fLineColor = cindex;
}

//______________________________________________________________________________
void TSVG::SetLineStyle(Style_t linestyle)
{
   // Change the line style
   //
   // linestyle = 2 dashed
   //           = 3 dotted
   //           = 4 dash-dotted
   //           = else solid (1 in is used most of the time)

   fLineStyle = linestyle;
}

//______________________________________________________________________________
void TSVG::SetLineWidth(Width_t linewidth)
{
   // Set the lines width.

   fLineWidth = linewidth;
}

//______________________________________________________________________________
void TSVG::SetMarkerColor( Color_t cindex )
{
   // Set color index for markers.

   fMarkerColor = cindex;
}

//______________________________________________________________________________
void TSVG::SetColor(Int_t color)
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
void TSVG::SetColor(Float_t r, Float_t g, Float_t b)
{
   // Set color with its R G B components
   // 
   //  r: % of red in [0,1]
   //  g: % of green in [0,1]
   //  b: % of blue in [0,1]

   if (r <= 0. && g <= 0. && b <= 0. ) {
      PrintFast(7,"\"black\"");
   } else if (r >= 1. && g >= 1. && b >= 1. ) {
      PrintFast(7,"\"white\"");
   } else {
      char str[12];
      sprintf(str,"\"#%2.2x%2.2x%2.2x\"",Int_t(255.*r)
                                        ,Int_t(255.*g)
                                        ,Int_t(255.*b));
      PrintStr(str);
   }
}

//______________________________________________________________________________
void TSVG::SetTextColor( Color_t cindex )
{
   // Set color index for text

   fTextColor = cindex;
}

//______________________________________________________________________________
void TSVG::Text(Double_t xx, Double_t yy, const char *chars)
{
   // Draw text
   //
   // xx: x postion of the text
   // yy: y postion of the text
   // chars: text to be drawn

   Int_t ix    = XtoSVG(xx); 
   Int_t iy    = YtoSVG(yy);
   Int_t txalh = fTextAlign/10;
   if (txalh <1) txalh = 1; if (txalh > 3) txalh = 3;
   Int_t txalv = fTextAlign%10;
   if (txalv <1) txalv = 1; if (txalv > 3) txalv = 3;

   Double_t     wh = (Double_t)gPad->XtoPixel(gPad->GetX2());
   Double_t     hh = (Double_t)gPad->YtoPixel(gPad->GetY1()); 
   Float_t tsize;
   Float_t fontrap = 1.09; //scale down compared to X11
   if (wh < hh)  {
      tsize = fTextSize*wh/fontrap;
   } else {
      tsize = fTextSize*hh/fontrap;
   }
   Float_t ftsize;
   
   Int_t font     = abs(fTextFont)/10;
   if( font > 42 || font < 1) font = 1;
   if (wh < hh) {
      ftsize = fTextSize*fXsize*gPad->GetAbsWNDC();
   } else {
      ftsize = fTextSize*fYsize*gPad->GetAbsHNDC();
   }
				             
   Int_t fontsize = CMtoSVG(ftsize/fontrap);
   if( fontsize <= 0) return;


   if (txalv == 3) iy = iy+fontsize;
   if (txalv == 2) iy = iy+(fontsize/2);

   if (fTextAngle != 0.) {
      PrintFast(21,"<g transform=\"rotate(");
      WriteInteger(-Int_t(fTextAngle), 0);
      PrintFast(1,",");
      WriteInteger(ix, 0);
      PrintFast(1,",");
      WriteInteger(iy, 0);
      PrintFast(3,")\">");
   }

   PrintFast(9,"<text x=\"");
   WriteInteger(ix, 0);
   PrintFast(5,"\" y=\"");
   WriteInteger(iy, 0);
   PrintFast(1,"\"");
   if (txalh == 2) {
      PrintFast(21," text-anchor=\"middle\"");
   } else if (txalh == 3) {
      PrintFast(18," text-anchor=\"end\"");
   }
   PrintFast(6," fill=");
   SetColor(Int_t(fTextColor));
   PrintFast(12," font-size=\"");
   WriteInteger(fontsize, 0);
   PrintFast(2,"\">");
   PrintStr(chars);
   PrintFast(7,"</text>");

   if (fTextAngle != 0.) PrintFast(4,"</g>");
}

//______________________________________________________________________________
void TSVG::TextNDC(Double_t u, Double_t v, const char *chars)
{
   // Write a string of characters in NDC

   Double_t x = gPad->GetX1() + u*(gPad->GetX2() - gPad->GetX1());
   Double_t y = gPad->GetY1() + v*(gPad->GetY2() - gPad->GetY1());
   Text(x, y, chars);
}

//______________________________________________________________________________
Int_t TSVG::UtoSVG(Double_t u)
{
   // Convert U from NDC coordinate to SVG

   Double_t cm = fXsize*(gPad->GetAbsXlowNDC() + u*gPad->GetAbsWNDC());
   return Int_t(0.5 + 72*cm/2.54);
}

//______________________________________________________________________________
Int_t TSVG::VtoSVG(Double_t v)
{
   // Convert V from NDC coordinate to SVG

   Double_t cm = fYsize*(gPad->GetAbsYlowNDC() + v*gPad->GetAbsHNDC());
   return Int_t(0.5 + 72*cm/2.54);
}

//______________________________________________________________________________
Int_t TSVG::XtoSVG(Double_t x)
{
   // Convert X from world coordinate to SVG

   Double_t u = (x - gPad->GetX1())/(gPad->GetX2() - gPad->GetX1());
   return  UtoSVG(u);
}

//______________________________________________________________________________
Int_t TSVG::YtoSVG(Double_t y)
{
   // Convert Y from world coordinate to SVG

   Double_t v = (y - gPad->GetY1())/(gPad->GetY2() - gPad->GetY1());
   return  fYsizeSVG-VtoSVG(v);
}

//______________________________________________________________________________
void TSVG::CellArrayBegin(Int_t W, Int_t H, Double_t x1, Double_t x2,
                                 Double_t y1, Double_t y2)
{
   Warning("TSVG::CellArrayBegin", "not yet implemented");
}

//______________________________________________________________________________
void TSVG::CellArrayFill(Int_t r, Int_t g, Int_t b)
{
   Warning("TSVG::CellArrayFill", "not yet implemented");
}

//______________________________________________________________________________
void TSVG::CellArrayEnd()
{
   Warning("TSVG::CellArrayEnd", "not yet implemented");
}

//______________________________________________________________________________
void TSVG::DrawPolyMarker(Int_t n, Float_t *x, Float_t *y)
{
   Warning("TSVG::DrawPolyMarker", "not yet implemented");
}

//______________________________________________________________________________
void TSVG::DrawPolyMarker(Int_t n, Double_t *x, Double_t *y)
{
   Warning("TSVG::DrawPolyMarker", "not yet implemented");
}

//______________________________________________________________________________
void TSVG::DrawPS(Int_t nn, Float_t *xw, Float_t *yw)
{
   Warning("TSVG::DrawPS", "not yet implemented");
}
