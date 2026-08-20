// @(#)root/postscript:$Id$
// Author: Olivier Couet, Sergey Linev

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef WIN32
#pragma optimize("",off)
#endif

#include <cstdlib>
#include <cstring>
#include <cctype>
#include <fstream>

#include "TROOT.h"
#include "TColor.h"
#include "TVirtualPad.h"
#include "TPoint.h"
#include "TPoints.h"
#include "TTeXDump.h"
#include "TStyle.h"
#include "TMath.h"


/** \class TTeXDump
\ingroup PS

\brief Interface to TeX.

This class allow to generate <b>PGF/TikZ</b> vector graphics output
which can be included in TeX and LaTeX documents.

PGF is a TeX macro package for generating graphics. It is platform
and format-independent and works together with the most important TeX
backend drivers, including pdftex and dvips. It comes with a
user-friendly syntax layer called TikZ.

To generate a such file it is enough to do:
~~~ {.cpp}
   gStyle->SetPaperSize(10.,10.);
   hpx->Draw();
   gPad->Print("hpx.tex");
~~~

Then, the generated file (`hpx.tex`) can be included in a
LaTeX document (`simple.tex`) in the following way:
~~~ {.cpp}
\documentclass{article}
\usepackage{tikz}
\usepackage{changepage}
\usetikzlibrary{patterns}
\usetikzlibrary{plotmarks}
\title{A simple LaTeX example}
\date{August 2026}
\begin{document}
\maketitle
The following image as been generated using the TTeXDump class:
\par
\begin{adjustwidth}{-4cm}{-4cm}
\input{hpx.tex}
\end{adjustwidth}
\end{document}
~~~

Note the four directives needed at the top of the LaTeX file:
~~~ {.cpp}
\usepackage{tikz}
\usepackage{changepage}
\usetikzlibrary{patterns}
\usetikzlibrary{plotmarks}
~~~

Then including the picture in the document is done with the
`\input` directive.

The command `pdflatex simple.tex` will generate the
corresponding pdf file `simple.pdf`.
*/

////////////////////////////////////////////////////////////////////////////////
/// Default TeX constructor

TTeXDump::TTeXDump() : TVirtualPS()
{
   gVirtualPS    = this;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the TeX interface
///
///  --fname : TeX file name
///  - wtype : TeX workstation type. Not used in the TeX driver. But as TTeXDump
///            inherits from TVirtualPS it should be kept. Anyway it is not
///            necessary to specify this parameter at creation time because it
///            has a default value (which is ignore in the TeX case).

TTeXDump::TTeXDump(const char *fname, Int_t wtype) : TVirtualPS(fname, wtype)
{
   gVirtualPS    = this;

   Open(fname, wtype);
}

////////////////////////////////////////////////////////////////////////////////
/// Open a TeX file

void TTeXDump::Open(const char *fname, Int_t wtype)
{
   if (fStream) {
      Warning("Open", "TeX file already open");
      return;
   }

   SetLineScale(gStyle->GetLineScalePS());
   fLenBuffer = 0;
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
   if (!OpenStream(fname)) {
      Error("Open", "Cannot open file:%s", fname);
      return;
   }

   gVirtualPS = this;

   ClearBuffer();

   fBoundingBox = kFALSE;
   fRange       = kFALSE;
   fStandalone  = kFALSE;

   // Set a default range
   Range(fXsize, fYsize);

   if (strstr(GetTitle(),"Standalone"))
      fStandalone = kTRUE;
   if (fStandalone) {
      PrintStr("\\documentclass{standalone}@");
      PrintStr("\\usepackage{tikz}@");
      PrintStr("\\usetikzlibrary{patterns,plotmarks}@");
      PrintStr("\\begin{document}@");
   } else {
      PrintStr("%\\documentclass{standalone}@");
      PrintStr("%\\usepackage{tikz}@");
      PrintStr("%\\usetikzlibrary{patterns,plotmarks}@");
      PrintStr("%\\begin{document}@");
   }

   NewPage();
}

////////////////////////////////////////////////////////////////////////////////
/// Default TeX destructor

TTeXDump::~TTeXDump()
{
   Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Close a TeX file

void TTeXDump::Close(Option_t *)
{
   if (!gVirtualPS || !fStream)
      return;
   if (gPad)
      gPad->Update();
   PrintStr("@");
   PrintStr("\\end{tikzpicture}@");
   if (fStandalone) {
      PrintStr("\\end{document}@");
   } else {
      PrintStr("%\\end{document}@");
   }

   // Close file stream
   CloseStream();

   gVirtualPS = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Activate an already open TeX file

void TTeXDump::On()
{
   // fType is used to know if the TeX file is open. Unlike TPostScript, TTeXDump
   // has no "workstation type". In fact there is only one TeX type.

   if (!fType) {
      Error("On", "no TeX file open");
      Off();
      return;
   }
   gVirtualPS = this;
}

////////////////////////////////////////////////////////////////////////////////
/// Deactivate an already open TeX file

void TTeXDump::Off()
{
   gVirtualPS = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a Box

void TTeXDump::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   Float_t x1c = XtoTeX(x1);
   Float_t y1c = YtoTeX(y1);
   Float_t x2c = XtoTeX(x2);
   Float_t y2c = YtoTeX(y2);

   Int_t fillis = fFillStyle/1000;
   Int_t fillsi = fFillStyle%1000;

   if (fillis==1) {
      SetColor(fFillColor);
      PrintStr("@");
      if (fCurrentAlpha != 1.) {
         PrintStr("\\fill [c");
         PrintStr(", fill opacity=");
         WriteReal(fCurrentAlpha, kFALSE);
      }
      else {
         PrintStr("\\draw [color=c, fill=c");
      }
      PrintStr("] (");
      WriteReal(x1c, kFALSE);
      PrintFast(1,",");
      WriteReal(y1c, kFALSE);
      PrintStr(") rectangle (");
      WriteReal(x2c, kFALSE);
      PrintFast(1,",");
      WriteReal(y2c, kFALSE);
      PrintStr(");");
   }
   if (fillis>1 && fillis<4) {
      SetColor(fFillColor);
      PrintStr("@");
      PrintStr("\\draw [pattern=");
      switch (fillsi) {
      case 1 :
         PrintStr("crosshatch dots");
         break;
      case 2 :
      case 3 :
         PrintStr("dots");
         break;
      case 4 :
         PrintStr("north east lines");
         break;
      case 5 :
         PrintStr("north west lines");
         break;
      case 6 :
         PrintStr("vertical lines");
         break;
      case 7 :
         PrintStr("horizontal lines");
         break;
      case 10 :
          PrintStr("bricks");
         break;
      case 13 :
         PrintStr("crosshatch");
         break;
      }
      PrintStr(", draw=none, pattern color=c");
      if (fCurrentAlpha != 1.) {
         PrintStr(", fill opacity=");
         WriteReal(fCurrentAlpha, kFALSE);
      }
      PrintStr("] (");
      WriteReal(x1c, kFALSE);
      PrintFast(1,",");
      WriteReal(y1c, kFALSE);
      PrintStr(") rectangle (");
      WriteReal(x2c, kFALSE);
      PrintFast(1,",");
      WriteReal(y2c, kFALSE);
      PrintStr(");");
   }
   if (fillis == 0) {
      if (fLineWidth<=0) return;
      SetColor(fLineColor);
      PrintStr("@");
      PrintStr("\\draw [c");
      PrintStr(",line width=");
      WriteReal(0.3*fLineScale*fLineWidth, kFALSE);
      if (fCurrentAlpha != 1.) {
         PrintStr(", opacity=");
         WriteReal(fCurrentAlpha, kFALSE);
      }
      PrintStr("] (");
      WriteReal(x1c, kFALSE);
      PrintFast(1,",");
      WriteReal(y1c, kFALSE);
      PrintStr(") -- (");
      WriteReal(x1c, kFALSE);
      PrintFast(1,",");
      WriteReal(y2c, kFALSE);
      PrintStr(") -- (");
      WriteReal(x2c, kFALSE);
      PrintFast(1,",");
      WriteReal(y2c, kFALSE);
      PrintStr(") -- (");
      WriteReal(x2c, kFALSE);
      PrintFast(1,",");
      WriteReal(y1c, kFALSE);
      PrintStr(") -- (");
      WriteReal(x1c, kFALSE);
      PrintFast(1,",");
      WriteReal(y1c, kFALSE);
      PrintStr(");");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a Frame around a box
///
/// mode = -1  the box looks as it is behind the screen
/// mode =  1  the box looks as it is in front of the screen
/// border is the border size in already pre-computed TeX units dark is the
/// color for the dark part of the frame light is the color for the light
/// part of the frame

void TTeXDump::DrawFrame(Double_t, Double_t, Double_t, Double_t,
                         Int_t, Int_t, Int_t, Int_t)
{
   Warning("DrawFrame", "not yet implemented");
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a PolyLine
///
///  Draw a polyline through  the points  xy.
///  - If NN=1 moves only to point x,y.
///  - If NN=0 the x,y are  written in the TeX file
///       according to the current transformation.
///  - If NN>0 the line is clipped as a line.
///  - If NN<0 the line is clipped as a fill area.

void TTeXDump::DrawPolyLine(Int_t, TPoints *)
{
   Warning("DrawPolyLine", "not yet implemented");
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a PolyLine in NDC space
///
///  Draw a polyline through  the points  xy.
///  - If NN=1 moves only to point x,y.
///  - If NN=0 the x,y are  written in the TeX file
///       according to the current transformation.
///  - If NN>0 the line is clipped as a line.
///  - If NN<0 the line is clipped as a fill area.

void TTeXDump::DrawPolyLineNDC(Int_t, TPoints *)
{
   Warning("DrawPolyLineNDC", "not yet implemented");
}

////////////////////////////////////////////////////////////////////////////////
/// Paint PolyMarker

template<typename T>
void TTeXDump::DrawPolyMarkerShape(Int_t n, T *xw, T *yw)
{
   Int_t markerSize = 0;
   std::vector<TPoint> points;
   auto shape = GetMarkerShape(markerSize, points, 1., kDotAsCircle | kUsePSWidthScale);
   if ((shape == kShapeDot) && (markerSize > 1))
      shape = kShapeFilledCircle;
   auto markerLineWidth = TAttMarker::GetMarkerLineWidth(GetMarkerStyle());
   Bool_t do_fill = (shape == kShapeFilledCircle) || (shape == kShapeFilledArea) || (shape == kShapeTriangles);

   TString name = TString::Format("root_marker%d", (Int_t) GetMarkerStyle());
   if ((shape == kShapeDot) || (shape == kShapeFilledCircle)) {
      name = "*";
      fMarkers[GetMarkerStyle()] = true;
   } else if (shape == kShapeCircle) {
      name = "o";
      fMarkers[GetMarkerStyle()] = true;
   }
   if (!fMarkers[GetMarkerStyle()]) {
      // define marker once
      fMarkers[GetMarkerStyle()] = true;
      Int_t sz0 = 0;
      // get shape for normal marker size to avoid rounding problems
      TAttMarker(1, GetMarkerStyle(), GetMarkerSize() > 2 ? GetMarkerSize() : 2.).GetMarkerShape(sz0, points, 1., kDotAsCircle | kUsePSWidthScale);
      // select coefficient so that relative movements are -1 .. 1
      Float_t k = sz0 > 0 ? 2. / sz0 : 0.02;

      PrintStr(TString::Format("@\\pgfdeclareplotmark{%s} {@", name.Data()));
      switch(shape) {
         case kShapePolyLine:
         case kShapeFilledArea:
            for (std::size_t i = 0; i < points.size(); i++)
               PrintStr(TString::Format("\\pgfpath%s{\\pgfpoint{%4.2f\\pgfplotmarksize}{%4.2f\\pgfplotmarksize}}@",
                        i == 0 ? "moveto" : "lineto", k * points[i].fX, -k * points[i].fY ));
            PrintStr("\\pgfpathclose@");

            if (shape == kShapePolyLine)
               PrintStr("\\pgfusepathqstroke@");
            else
               PrintStr("\\pgfusepathqfillstroke@");
            break;
         case kShapeSegments:
            for (std::size_t i = 0; i < points.size(); i++)
               PrintStr(TString::Format("\\pgfpath%s{\\pgfpoint{%4.2f\\pgfplotmarksize}{%4.2f\\pgfplotmarksize}}@",
                        i % 2 == 0 ? "moveto" : "lineto", k * points[i].fX, -k * points[i].fY ));
            PrintStr("\\pgfpathclose@");
            PrintStr("\\pgfusepathqstroke@");
            break;
         case kShapeTriangles:
            for (std::size_t i = 0; i < points.size(); i++)
               PrintStr(TString::Format("\\pgfpath%s{\\pgfpoint{%4.2f\\pgfplotmarksize}{%4.2f\\pgfplotmarksize}}@",
                        i % 3 == 0 ? "moveto" : "lineto", k * points[i].fX, -k * points[i].fY ));
            PrintStr("\\pgfpathclose@");
            PrintStr("\\pgfusepathqfillstroke@");
            break;
         default:
            // all other shapes handled already
            break;
      }
      PrintStr("}@");
   }

   SetColor(GetMarkerColor());

   PrintStr("@");
   PrintStr("\\foreach \\P in {");

   for (Int_t i = 0; i < n; i++) {
      auto x = XtoTeX(xw[i]);
      auto y = YtoTeX(yw[i]);
      if (i == 0)
         PrintFast(1, "(");
      else
         PrintFast(3, ", (");
      WriteReal(x, kFALSE);
      PrintFast(1, ",");
      WriteReal(y, kFALSE);
      PrintFast(1, ")");
   }

   PrintStr("}{\\draw[mark options={color=c");

   if (do_fill)
      PrintStr(",fill=c");

   if (fCurrentAlpha != 1.) {
      PrintStr(",opacity=");
      WriteReal(fCurrentAlpha, kFALSE);
   }

   PrintStr(TString::Format("}, mark size=%4.2fpt", 0.3 * markerSize));
   // intentionally default line width is 0, only large widths scale differently
   if (!do_fill && (markerLineWidth > 0))
      PrintStr(TString::Format(", line width=%4.2fpt", markerLineWidth > 1 ? 0.2 * gStyle->GetLineScalePS() * markerLineWidth : 0.));
   PrintStr(", mark=");
   PrintStr(name);
   PrintStr("] plot coordinates {\\P};}");
}


////////////////////////////////////////////////////////////////////////////////
/// Paint PolyMarker

void TTeXDump::DrawPolyMarker(Int_t n, Float_t *xw, Float_t *yw)
{
   DrawPolyMarkerShape<Float_t>(n, xw, yw);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint PolyMarker

void TTeXDump::DrawPolyMarker(Int_t n, Double_t *xw, Double_t *yw)
{
   DrawPolyMarkerShape<Double_t>(n, xw, yw);
}

////////////////////////////////////////////////////////////////////////////////
/// This function defines a path with xw and yw and draw it according the
/// value of nn:
///
///  - If nn>0 a line is drawn.
///  - If nn<0 a closed polygon is drawn.

void TTeXDump::DrawPS(Int_t nn, Double_t *xw, Double_t *yw)
{
   Int_t  n = TMath::Abs(nn);
   Float_t x, y;

   if( n <= 1) {
      Error("DrawPS", "Two points are needed");
      return;
   }

   x = XtoTeX(xw[0]);
   y = YtoTeX(yw[0]);

   Int_t fillis = fFillStyle/1000;
   Int_t fillsi = fFillStyle%1000;

   if (nn>0) {
      if (fLineWidth<=0) return;
      SetColor(fLineColor);
      PrintStr("@");
      PrintStr("\\draw [c");
      TString spec = gStyle->GetLineStyleString(fLineStyle);
      TString tikzSpec;
      TString stripped = TString{spec.Strip(TString::kBoth)};
      if (stripped.Length()) {
         tikzSpec.Append(",dash pattern=");
         Ssiz_t i{0}, j{0};
         bool on{true}, iterate{true};

         while (iterate){
            j = stripped.Index(" ", 1, i, TString::kExact);
            if (j == kNPOS){
               iterate = false;
               j = stripped.Length();
            }

            if (on) {
               tikzSpec.Append("on ");
               on = false;
            } else {
               tikzSpec.Append("off ");
               on = true;
            }
            int num = TString{stripped(i, j - i)}.Atoi();
            float pt = 0.2*num;
            tikzSpec.Append(TString::Format("%.2fpt ", pt));
            i = j + 1;
         }
         PrintStr(tikzSpec.Data());
      }
      PrintStr(",line width=");
      WriteReal(0.3*fLineScale*fLineWidth, kFALSE);
      if (fCurrentAlpha != 1.) {
         PrintStr(",opacity=");
         WriteReal(fCurrentAlpha, kFALSE);
      }
   } else {
      SetColor(fFillColor);
      if (fillis==1) {
         PrintStr("@");
         PrintStr("\\draw [c, fill=c");
      } else if (fillis==0) {
         PrintStr("@");
         PrintStr("\\draw [c");
      } else {
         PrintStr("\\draw [pattern=");
         switch (fillsi) {
         case 1 :
            PrintStr("crosshatch dots");
            break;
         case 2 :
         case 3 :
            PrintStr("dots");
            break;
         case 4 :
            PrintStr("north east lines");
            break;
         case 5 :
            PrintStr("north west lines");
            break;
         case 6 :
            PrintStr("vertical lines");
            break;
         case 7 :
            PrintStr("horizontal lines");
            break;
         case 10 :
             PrintStr("bricks");
            break;
         case 13 :
            PrintStr("crosshatch");
            break;
         }
         PrintStr(", draw=none, pattern color=c");
      }
      if (fCurrentAlpha != 1.) {
         PrintStr(", fill opacity=");
         WriteReal(fCurrentAlpha, kFALSE);
      }
   }
   PrintStr("] (");
   WriteReal(x, kFALSE);
   PrintFast(1,",");
   WriteReal(y, kFALSE);
   PrintStr(") -- ");

   for (Int_t i=1;i<n;i++) {
      x = XtoTeX(xw[i]);
      y = YtoTeX(yw[i]);
      PrintFast(1,"(");
      WriteReal(x, kFALSE);
      PrintFast(1,",");
      WriteReal(y, kFALSE);
      PrintFast(1,")");
      if (i<n-1) PrintStr(" -- ");
      else PrintStr(";@");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Start the TeX page. This function starts the tikzpicture environment

void TTeXDump::NewPage()
{
   // Compute pad conversion coefficients
   if (gPad) {
      Double_t ww   = gPad->GetWw();
      Double_t wh   = gPad->GetWh();
      fYsize        = fXsize*wh/ww;
   } else {
      fYsize = 27;
   }

   if(!fBoundingBox) {
      PrintStr("\\begin{tikzpicture}@");
      PrintStr("\\def\\CheckTikzLibraryLoaded#1{ \\ifcsname tikz@library@#1@loaded\\endcsname \\else \\PackageWarning{tikz}{usetikzlibrary{#1} is missing in the preamble.} \\fi }@");
      PrintStr("\\CheckTikzLibraryLoaded{patterns}@");
      PrintStr("\\CheckTikzLibraryLoaded{plotmarks}@");
      fBoundingBox = kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set the range for the paper in centimetres

void TTeXDump::Range(Float_t xsize, Float_t ysize)
{
   fXsize = xsize;
   fYsize = ysize;

   fRange = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set color index for fill areas

void TTeXDump::SetFillColor( Color_t cindex )
{
   fFillColor = cindex;
}

////////////////////////////////////////////////////////////////////////////////
/// Set color index for lines

void TTeXDump::SetLineColor( Color_t cindex )
{
   fLineColor = cindex;
}

////////////////////////////////////////////////////////////////////////////////
/// Change the line style
///
///  - linestyle = 2 dashed
///  - linestyle = 3 dotted
///  - linestyle = 4 dash-dotted
///  - linestyle = else solid (1 in is used most of the time)

void TTeXDump::SetLineStyle(Style_t linestyle)
{
   fLineStyle = linestyle;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the lines width.

void TTeXDump::SetLineWidth(Width_t linewidth)
{
   fLineWidth = linewidth;
}

////////////////////////////////////////////////////////////////////////////////
/// Set size for markers.

void TTeXDump::SetMarkerSize( Size_t msize)
{
   fMarkerSize = msize;
}

////////////////////////////////////////////////////////////////////////////////
/// Set color index for markers.

void TTeXDump::SetMarkerColor( Color_t cindex)
{
   fMarkerColor = cindex;
}

////////////////////////////////////////////////////////////////////////////////
/// Set color with its color index

void TTeXDump::SetColor(Int_t color)
{
   if (color < 0) color = 0;
   TColor *col = gROOT->GetColor(color);

   if (col) {
      SetColor(col->GetRed(), col->GetGreen(), col->GetBlue());
      fCurrentAlpha = col->GetAlpha();
   } else {
      SetColor(1., 1., 1.);
      fCurrentAlpha = 1.;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set color with its R G B components
///
///  - r: % of red in [0,1]
///  - g: % of green in [0,1]
///  - b: % of blue in [0,1]

void TTeXDump::SetColor(Float_t r, Float_t g, Float_t b)
{
   if (fCurrentRed == r && fCurrentGreen == g && fCurrentBlue == b) return;

   fCurrentRed   = r;
   fCurrentGreen = g;
   fCurrentBlue  = b;
   PrintStr("@");
   PrintStr("\\definecolor{c}{rgb}{");
   WriteReal(r, kFALSE);
   PrintFast(1,",");
   WriteReal(g, kFALSE);
   PrintFast(1,",");
   WriteReal(b, kFALSE);
   PrintFast(2,"};");
}

////////////////////////////////////////////////////////////////////////////////
/// Set color index for text

void TTeXDump::SetTextColor( Color_t cindex )
{
   fTextColor = cindex;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw text
///
///  - xx: x position of the text
///  - yy: y position of the text
///  - chars: text to be drawn

void TTeXDump::Text(Double_t x, Double_t y, const char *chars)
{
   Double_t wh = (Double_t)gPad->GetPadWidth();
   Double_t hh = (Double_t)gPad->GetPadHeight();
   Float_t tsize, ftsize;
   if (wh < hh) {
      tsize = fTextSize*wh;
      Int_t sizeTTF = (Int_t)(tsize+0.5);
      ftsize = (sizeTTF*fXsize*gPad->GetAbsWNDC())/wh;
   } else {
      tsize = fTextSize*hh;
      Int_t sizeTTF = (Int_t)(tsize+0.5);
      ftsize = (sizeTTF*fYsize*gPad->GetAbsHNDC())/hh;
   }
   ftsize *= 2.22097;
   if (ftsize <= 0) return;

   TString t(chars);
   if (t.Index("\\")>=0 || t.Index("^{")>=0 || t.Index("_{")>=0) {
      t.Prepend("$");
      t.Append("$");
   } else {
      t.ReplaceAll("<","$<$");
      t.ReplaceAll(">","$>$");
      t.ReplaceAll("_","\\_");
   }
   t.ReplaceAll("&","\\&");
   t.ReplaceAll("#","\\#");
   t.ReplaceAll("%","\\%");

   Int_t txalh = fTextAlign/10;
   if (txalh <1) txalh = 1; else if (txalh > 3) txalh = 3;
   Int_t txalv = fTextAlign%10;
   if (txalv <1) txalv = 1; else if (txalv > 3) txalv = 3;
   SetColor(fTextColor);
   PrintStr("@");
   PrintStr("\\draw");
   if (txalh!=2 || txalv!=2) {
      PrintStr(" [anchor=");
      if (txalv==1) PrintStr("base");
      if (txalv==3) PrintStr("north");
      if (txalh==1) PrintStr(" west");
      if (txalh==3) PrintStr(" east");
      PrintFast(1,"]");
   }
   PrintFast(2," (");
   WriteReal(XtoTeX(x), kFALSE);
   PrintFast(1,",");
   WriteReal(YtoTeX(y), kFALSE);
   PrintStr(") node[scale=");
   WriteReal(ftsize, kFALSE);
   PrintStr(", color=c");
   if (fCurrentAlpha != 1.) {
      PrintStr(",opacity=");
      WriteReal(fCurrentAlpha, kFALSE);
   }
   PrintStr(", rotate=");
   WriteReal(fTextAngle, kFALSE);
   PrintFast(2,"]{");
   PrintStr(t.Data());
   PrintFast(2,"};");
}

////////////////////////////////////////////////////////////////////////////////
/// Draw text with URL. Same as Text.
///

void TTeXDump::TextUrl(Double_t x, Double_t y, const char *chars, const char *)
{
   Text(x, y, chars);
}

////////////////////////////////////////////////////////////////////////////////
/// Write a string of characters in NDC

void TTeXDump::TextNDC(Double_t u, Double_t v, const char *chars)
{
   Double_t x = gPad->GetX1() + u*(gPad->GetX2() - gPad->GetX1());
   Double_t y = gPad->GetY1() + v*(gPad->GetY2() - gPad->GetY1());
   Text(x, y, chars);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert U from NDC coordinate to TeX

Float_t TTeXDump::UtoTeX(Double_t u)
{
   Double_t cm = fXsize*(gPad->GetAbsXlowNDC() + u*gPad->GetAbsWNDC());
   return cm;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert V from NDC coordinate to TeX

Float_t TTeXDump::VtoTeX(Double_t v)
{
   Double_t cm = fYsize*(gPad->GetAbsYlowNDC() + v*gPad->GetAbsHNDC());
   return cm;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert X from world coordinate to TeX

Float_t TTeXDump::XtoTeX(Double_t x)
{
   Double_t u = (x - gPad->GetX1())/(gPad->GetX2() - gPad->GetX1());
   return  UtoTeX(u);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert Y from world coordinate to TeX

Float_t TTeXDump::YtoTeX(Double_t y)
{
   Double_t v = (y - gPad->GetY1())/(gPad->GetY2() - gPad->GetY1());
   return  VtoTeX(v);
}

////////////////////////////////////////////////////////////////////////////////
/// Begin the Cell Array painting

void TTeXDump::CellArrayBegin(Int_t, Int_t, Double_t, Double_t, Double_t,
                          Double_t)
{
   Warning("CellArrayBegin", "not yet implemented");
}

////////////////////////////////////////////////////////////////////////////////
/// Paint the Cell Array

void TTeXDump::CellArrayFill(Int_t, Int_t, Int_t)
{
   Warning("CellArrayFill", "not yet implemented");
}

////////////////////////////////////////////////////////////////////////////////
/// End the Cell Array painting

void TTeXDump::CellArrayEnd()
{
   Warning("CellArrayEnd", "not yet implemented");
}

////////////////////////////////////////////////////////////////////////////////
/// Not needed in TeX case

void TTeXDump::DrawPS(Int_t, Float_t *, Float_t *)
{
   Warning("DrawPS", "not yet implemented");
}
