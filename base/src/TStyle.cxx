// @(#)root/base:$Name:  $:$Id: TStyle.cxx,v 1.13 2001/12/17 17:06:07 brun Exp $
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

#include "TROOT.h"
#include "TStyle.h"
#include "TColor.h"

TStyle  *gStyle;

ClassImp(TStyle)


//______________________________________________________________________________
//
// TStyle objects may be created to define special styles.
// By default ROOT creates a default style that can be accessed via
// the gStyle pointer.
//
// This class includes functions to set the following object attributes.
//   - Canvas
//   - Pad
//   - Histogram axis
//   - Lines
//   - Fill areas
//   - Text
//   - Markers
//   - Functions
//   - Histogram Statistics and Titles
//

//______________________________________________________________________________
TStyle::TStyle() :TNamed()
{

   Reset();
}

//______________________________________________________________________________
TStyle::TStyle(const char *name, const char *title) : TNamed(name,title)
{
// Create a new TStyle.
// The following names are reserved to create special styles
//   -Default: the default style set in TStyle::Reset
//   -Plain: a black&white oriented style
//   -Bold:
//   -Video;
//   -Pub:
//     (see the definition of these styles below).
//
// Note a side-effect of calling gStyle->SetFillColor(0). This is nearly
// equivalent of selecting the "Plain" style.
//
// Many graphics attributes may be set via the TStyle, see in particular
//  - TStyle::SetNdivisions
//  - TStyle::SetAxisColor
//  - TStyle::SetHeaderPS
//  - TStyle::SetTitlePS
//  - TStyle::SetLabelColor
//  - TStyle::SetLabelFont
//  - TStyle::SetLabelOffset
//  - TStyle::SetLabelSize
//  - TStyle::SetOptDate
//  - TStyle::SetLineStyleString
//  - TStyle::SetOptFit
//  - TStyle::SetOptStat
//  - TStyle::SetPaperSize
//  - TStyle::SetTickLength
//  - TStyle::SetTitleOffset
//  - TStyle::SetTitleSize
//  - TStyle::SetPalette
//  - TStyle::SetTimeOffset
//  - TStyle::SetStripDecimals
//
//  The current style is pointed by gStyle.
//  When calling myStyle->cd(), gStyle is set to myStyle.
//  One can also use gROOT to change the current style, eg
//    gROOT->SetStyle("Plain") will change the current style gStyle to the "Plain" style
//  See also TROOT::ForceStyle and TROOT::UseCurrentStyle

   Reset();

   gROOT->GetListOfStyles()->Add(this);

   //may be a standard style to be initialyzed
   if (strcmp(name,"Plain") == 0) {
      SetFrameBorderMode(0);
      SetCanvasBorderMode(0);
      SetPadBorderMode(0);
      SetPadColor(0);
      SetCanvasColor(0);
      SetTitleColor(0);
      SetStatColor(0);
      return;
   }
   if (strcmp(name,"Bold") == 0) {
   // Authors: Art Poskanzer and Jim Thomas, LBNL, Oct. 2000
      SetPalette(1,0);
      SetCanvasColor(10);
      SetCanvasBorderMode(0);
      SetFrameLineWidth(3);
      SetFrameFillColor(10);
      SetPadColor(10);
      SetPadTickX(1);
      SetPadTickY(1);
      SetPadBottomMargin(0.15);
      SetPadLeftMargin(0.15);
      SetHistLineWidth(3);
      SetHistLineColor(kRed);
      SetFuncWidth(3);
      SetFuncColor(kGreen);
      SetLineWidth(3);
      SetLabelSize(0.05,"X");
      SetLabelSize(0.05,"Y");
      SetLabelSize(0.05,"Z");
      SetLabelOffset(0.01,"Y");
      SetLabelColor(kBlue,"X");
      SetLabelColor(kBlue,"Y");
      SetTitleSize(0.06,"X");
      SetTitleSize(0.06,"Y");
      SetTitleSize(0.06,"Z");
      SetTitleOffset(1.3,"Y");
      SetTitleColor(10);
      SetTitleTextColor(kBlue);
      SetStatColor(10);
      return;
   }
   if (strcmp(name,"Video") == 0) {
     // Author: Art Poskanzer, LBNL, Oct. 1999
      SetPalette(1,0);
      SetCanvasColor(10);
      SetCanvasBorderMode(0);
      SetFrameLineWidth(3);
      SetFrameFillColor(10);
      SetPadColor(10);
      SetPadTickX(1);
      SetPadTickY(1);
      SetPadBottomMargin(0.2);
      SetPadLeftMargin(0.2);
      SetHistLineWidth(8);
      SetHistLineColor(kRed);
      SetLabelSize(0.06,"X");
      SetLabelSize(0.06,"Y");
      SetLabelSize(0.06,"Z");
      SetLabelColor(kBlue,"X");
      SetLabelColor(kBlue,"Y");
      SetLabelColor(kBlue,"Z");
      SetTitleSize(0.08,"X");
      SetTitleSize(0.08,"Y");
      SetTitleSize(0.08,"Z");
      SetTitleColor(10);
      SetTitleTextColor(kBlue);
      SetStatColor(10);
      SetFuncWidth(8);
      SetFuncColor(kGreen);
      SetLineWidth(3);
      return;
   }
   if (strcmp(name,"Pub") == 0) {
   // Authors: Art Poskanzer and Jim Thomas, LBNL, Oct. 2000
      SetOptTitle(0);
      SetOptStat(0);
      SetPalette(8,0);
      SetCanvasColor(10);
      SetCanvasBorderMode(0);
      SetFrameLineWidth(3);
      SetFrameFillColor(10);
      SetPadColor(10);
      SetPadTickX(1);
      SetPadTickY(1);
      SetPadBottomMargin(0.15);
      SetPadLeftMargin(0.15);
      SetHistLineWidth(3);
      SetHistLineColor(kRed);
      SetFuncWidth(3);
      SetFuncColor(kGreen);
      SetLineWidth(3);
      SetLabelSize(0.05,"X");
      SetLabelSize(0.05,"Y");
      SetLabelSize(0.05,"Z");
      SetLabelOffset(0.01,"Y");
      SetLabelColor(kBlue,"X");
      SetLabelColor(kBlue,"Y");
      SetLabelColor(kBlue,"Z");
      SetTitleSize(0.06,"X");
      SetTitleSize(0.06,"Y");
      SetTitleSize(0.06,"Z");
      SetTitleOffset(1.3,"Y");
      SetTitleColor(10);
      SetTitleTextColor(kBlue);
      return;
   }

}

//______________________________________________________________________________
TStyle::~TStyle()
{
   gROOT->GetListOfStyles()->Remove(this);
   if (gStyle == this) gStyle = (TStyle*)gROOT->GetListOfStyles()->Last();
}

//______________________________________________________________________________
TStyle::TStyle(const TStyle &style)
{
   ((TStyle&)style).Copy(*this);
}

//______________________________________________________________________________
void TStyle::Browse(TBrowser *)
{
    cd();
}

//______________________________________________________________________________
void TStyle::BuildStyles()
{
    // create some standard styles
   new TStyle("Plain",  "Plain Style (no colors/fill areas)");
   new TStyle("Bold",   "Bold Style");;
   new TStyle("Video",  "Style for video presentation histograms");
   new TStyle("Pub",    "Style for Publications");
   new TStyle("Default","Default Style");
}

//______________________________________________________________________________
void TStyle::cd()
{
//   Change current style
   gStyle = this;
}

//______________________________________________________________________________
void TStyle::Copy(TObject &obj)
{
   TAttLine::Copy(((TStyle&)obj));
   TAttFill::Copy(((TStyle&)obj));
   TAttMarker::Copy(((TStyle&)obj));
   TAttText::Copy(((TStyle&)obj));
   fXaxis.Copy(((TStyle&)obj).fXaxis);
   fYaxis.Copy(((TStyle&)obj).fYaxis);
   fZaxis.Copy(((TStyle&)obj).fZaxis);
   ((TStyle&)obj).fScreenFactor   = fScreenFactor;
   ((TStyle&)obj).fCanvasColor    = fCanvasColor;
   ((TStyle&)obj).fCanvasBorderSize= fCanvasBorderSize;
   ((TStyle&)obj).fCanvasBorderMode= fCanvasBorderMode;
   ((TStyle&)obj).fCanvasDefH     = fCanvasDefH;
   ((TStyle&)obj).fCanvasDefW     = fCanvasDefW;
   ((TStyle&)obj).fCanvasDefX     = fCanvasDefX;
   ((TStyle&)obj).fCanvasDefY     = fCanvasDefY;
   ((TStyle&)obj).fPadColor       = fPadColor;
   ((TStyle&)obj).fPadBorderSize  = fPadBorderSize;
   ((TStyle&)obj).fPadBorderMode  = fPadBorderMode;
   ((TStyle&)obj).fPadBottomMargin= fPadBottomMargin;
   ((TStyle&)obj).fPadTopMargin   = fPadTopMargin;
   ((TStyle&)obj).fPadLeftMargin  = fPadLeftMargin;
   ((TStyle&)obj).fPadRightMargin = fPadRightMargin;
   ((TStyle&)obj).fPadGridX       = fPadGridX;
   ((TStyle&)obj).fPadGridY       = fPadGridY;
   ((TStyle&)obj).fPadTickX       = fPadTickX;
   ((TStyle&)obj).fPadTickY       = fPadTickY;
   ((TStyle&)obj).fFuncColor      = fFuncColor;
   ((TStyle&)obj).fFuncStyle      = fFuncStyle;
   ((TStyle&)obj).fFuncWidth      = fFuncWidth;
   ((TStyle&)obj).fGridColor      = fGridColor;
   ((TStyle&)obj).fGridStyle      = fGridStyle;
   ((TStyle&)obj).fGridWidth      = fGridWidth;
   ((TStyle&)obj).fFrameFillColor = fFrameFillColor;
   ((TStyle&)obj).fFrameFillStyle = fFrameFillStyle;
   ((TStyle&)obj).fFrameLineColor = fFrameLineColor;
   ((TStyle&)obj).fFrameLineStyle = fFrameLineStyle;
   ((TStyle&)obj).fFrameLineWidth = fFrameLineWidth;
   ((TStyle&)obj).fFrameBorderSize= fFrameBorderSize;
   ((TStyle&)obj).fFrameBorderMode= fFrameBorderMode;
   ((TStyle&)obj).fHistFillColor  = fHistFillColor;
   ((TStyle&)obj).fHistFillStyle  = fHistFillStyle;
   ((TStyle&)obj).fHistLineColor  = fHistLineColor;
   ((TStyle&)obj).fHistLineStyle  = fHistLineStyle;
   ((TStyle&)obj).fHistLineWidth  = fHistLineWidth;
   ((TStyle&)obj).fBarWidth       = fBarWidth;
   ((TStyle&)obj).fBarOffset      = fBarOffset;
   ((TStyle&)obj).fDrawBorder     = fDrawBorder;
   ((TStyle&)obj).fOptLogx        = fOptLogx;
   ((TStyle&)obj).fOptLogy        = fOptLogy;
   ((TStyle&)obj).fOptLogz        = fOptLogz;
   ((TStyle&)obj).fOptDate        = fOptDate;
   ((TStyle&)obj).fOptFile        = fOptFile;
   ((TStyle&)obj).fOptFit         = fOptFit;
   ((TStyle&)obj).fOptStat        = fOptStat;
   ((TStyle&)obj).fOptTitle       = fOptTitle;
   ((TStyle&)obj).fErrorMarker    = fErrorMarker;
   ((TStyle&)obj).fErrorMsize     = fErrorMsize;
   ((TStyle&)obj).fErrorX         = fErrorX;
   ((TStyle&)obj).fStatColor      = fStatColor;
   ((TStyle&)obj).fStatTextColor  = fStatTextColor;
   ((TStyle&)obj).fStatBorderSize = fStatBorderSize;
   ((TStyle&)obj).fStatFont       = fStatFont;
   ((TStyle&)obj).fStatFontSize   = fStatFontSize;
   ((TStyle&)obj).fStatStyle      = fStatStyle;
   ((TStyle&)obj).fStatFormat     = fStatFormat;
   ((TStyle&)obj).fStatW          = fStatW;
   ((TStyle&)obj).fStatH          = fStatH ;
   ((TStyle&)obj).fStatX          = fStatX;
   ((TStyle&)obj).fStatY          = fStatY;
   ((TStyle&)obj).fTitleColor     = fTitleColor;
   ((TStyle&)obj).fTitleTextColor = fTitleTextColor;
   ((TStyle&)obj).fTitleFont      = fTitleFont;
   ((TStyle&)obj).fTitleFontSize  = fTitleFontSize;
   ((TStyle&)obj).fTitleStyle     = fTitleStyle;
   ((TStyle&)obj).fTitleBorderSize= fTitleBorderSize;
   ((TStyle&)obj).fTitleW         = fTitleW;
   ((TStyle&)obj).fTitleH         = fTitleH;
   ((TStyle&)obj).fTitleX         = fTitleX;
   ((TStyle&)obj).fTitleY         = fTitleY;
   ((TStyle&)obj).fDateX          = fDateX;
   ((TStyle&)obj).fDateY          = fDateY;
   ((TStyle&)obj).fFitFormat      = fFitFormat;
   ((TStyle&)obj).fShowEventStatus= fShowEventStatus;
   ((TStyle&)obj).fLegoInnerR     = fLegoInnerR;
   ((TStyle&)obj).fStripDecimals  = fStripDecimals;
   Int_t i;
   for (i=0;i<30;i++) {
      ((TStyle&)obj).fLineStyle[i]     = fLineStyle[i];
   }
   Int_t ncolors = fPalette.fN;
   ((TStyle&)obj).fPalette.Set(ncolors);
   for (i=0;i<ncolors;i++) {
      ((TStyle&)obj).fPalette.fArray[i] = fPalette.fArray[i];
   }
   ((TStyle&)obj).fHeaderPS       = fHeaderPS;
   ((TStyle&)obj).fTitlePS        = fTitlePS;
   ((TStyle&)obj).fLineScalePS    = fLineScalePS;
   ((TStyle&)obj).fTimeOffset     = fTimeOffset;
}

//______________________________________________________________________________
void TStyle::Reset(Option_t *)
{
   TAttLine::ResetAttLine();
   TAttFill::ResetAttFill();
   TAttText::ResetAttText();
   TAttMarker::ResetAttMarker();
   SetFillStyle(1001);
   SetFillColor(19);
   fXaxis.ResetAttAxis();
   fYaxis.ResetAttAxis();
   fZaxis.ResetAttAxis();
   fCanvasColor    = 19;
   fCanvasBorderSize= 2;
   fCanvasBorderMode= 1;
   fCanvasDefH     = 500;
   fCanvasDefW     = 700;
   fCanvasDefX     = 10;
   fCanvasDefY     = 10;
   fPadColor       = fCanvasColor;
   fPadBorderSize  = fCanvasBorderSize;
   fPadBorderMode  = fCanvasBorderMode;
   fPadBottomMargin= 0.1;
   fPadTopMargin   = 0.1;
   fPadLeftMargin  = 0.1;
   fPadRightMargin = 0.1;
   fPadGridX       = kFALSE;
   fPadGridY       = kFALSE;
   fPadTickX       = 0;
   fPadTickY       = 0;
   fFuncColor      = 1;
   fFuncStyle      = 1;
   fFuncWidth      = 3;
   fGridColor      = 0;
   fGridStyle      = 3;
   fGridWidth      = 1;
   fHistLineColor  = 1;
   fHistFillColor  = 0;
   fHistFillStyle  = 1001;
   fHistLineStyle  = 1;
   fHistLineWidth  = 1;
   fFrameLineColor = 1;
   fFrameFillColor = 0;
   fFrameFillStyle = 1001;
   fFrameLineStyle = 1;
   fFrameLineWidth = 1;
   fFrameBorderSize= 1;
   fFrameBorderMode= 1;
   fBarWidth       = 1;
   fBarOffset      = 0;
   fDrawBorder     = 0;
   fOptLogx        = 0;
   fOptLogy        = 0;
   fOptLogz        = 0;
   fOptDate        = 0;
   fOptFile        = 0;
   fOptFit         = 0;
   fOptStat        = 1;
   fOptTitle       = 1;
   fErrorMarker    = 21;
   fErrorMsize     = 0.05;
   fErrorX         = 0.5;
   fScreenFactor   = 1;
   fStatColor      = fCanvasColor;
   fStatTextColor  = 1;
   fStatBorderSize = 2;
   fStatFont       = 62;
   fStatFontSize   = 0;
   fStatStyle      = 1001;
   fStatW          = 0.20;
   fStatH          = 0.16;
   fStatX          = 0.98;
   fStatY          = 0.98;
   SetStatFormat();
   SetFitFormat();
   fTitleColor     = fCanvasColor;
   fTitleTextColor = 1;
   fTitleFont      = 62;
   fTitleFontSize  = 0;
   fTitleStyle     = 1001;
   fTitleBorderSize= 2;
   fTitleW         = 0;
   fTitleH         = 0;
   fTitleX         = 0.01;
   fTitleY         = 0.995;
   fShowEventStatus= 0;
   fLegoInnerR     = 0.5;
   fHeaderPS       = "";
   fTitlePS        = "";
   fStripDecimals  = kTRUE;

   SetDateX();
   SetDateY();
   fAttDate.SetTextSize(0.025);
   fAttDate.SetTextAlign(11);
   SetLineScalePS();
   SetLineStyleString(1,"[]");
   SetLineStyleString(2,"[12 12]");
   SetLineStyleString(3,"[4 8]");
   SetLineStyleString(4,"[12 15 4 15]");
   for (Int_t i=5;i<30;i++) SetLineStyleString(i,"[]");

   SetPaperSize();

   SetPalette();

   fTimeOffset = 788918400; // UTC time at 01/01/95
}

//______________________________________________________________________________
Int_t TStyle::AxisChoice( Option_t *axis) const
{
   char achoice = toupper(axis[0]);
   if (achoice == 'Y') return 2;
   if (achoice == 'Z') return 3;
   return 1;
}

//______________________________________________________________________________
Int_t TStyle::GetNdivisions( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetNdivisions();
   if (ax == 2) return fYaxis.GetNdivisions();
   if (ax == 3) return fZaxis.GetNdivisions();
   return 0;
}

//______________________________________________________________________________
Color_t TStyle::GetAxisColor( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetAxisColor();
   if (ax == 2) return fYaxis.GetAxisColor();
   if (ax == 3) return fZaxis.GetAxisColor();
   return 0;
}

//______________________________________________________________________________
Int_t TStyle::GetColorPalette(Int_t i) const
//   return color number i in current palette
{
   Int_t ncolors = GetNumberOfColors();
   if (ncolors == 0) return 0;
   Int_t icol    = i%ncolors;
   if (icol < 0) icol = 0;
   return fPalette.fArray[icol];
}

//______________________________________________________________________________
Color_t TStyle::GetLabelColor( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetLabelColor();
   if (ax == 2) return fYaxis.GetLabelColor();
   if (ax == 3) return fZaxis.GetLabelColor();
   return 0;
}

//______________________________________________________________________________
Style_t TStyle::GetLabelFont( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetLabelFont();
   if (ax == 2) return fYaxis.GetLabelFont();
   if (ax == 3) return fZaxis.GetLabelFont();
   return 0;
}

//______________________________________________________________________________
Float_t TStyle::GetLabelOffset( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetLabelOffset();
   if (ax == 2) return fYaxis.GetLabelOffset();
   if (ax == 3) return fZaxis.GetLabelOffset();
   return 0;
}

//______________________________________________________________________________
Float_t TStyle::GetLabelSize( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetLabelSize();
   if (ax == 2) return fYaxis.GetLabelSize();
   if (ax == 3) return fZaxis.GetLabelSize();
   return 0;
}


//______________________________________________________________________________
const char *TStyle::GetLineStyleString(Int_t i) const
{
//   return line style string (used by Popscript).
//   See SetLineStyleString for more explanations

   if (i < 1 || i > 29) return fLineStyle[0].Data();
   return fLineStyle[i].Data();
}

//______________________________________________________________________________
void TStyle::GetPaperSize(Float_t &xsize, Float_t &ysize)
{
//    Set paper size for PostScript output
//
   xsize = fPaperSizeX;
   ysize = fPaperSizeY;
}

//______________________________________________________________________________
Float_t TStyle::GetTickLength( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetTickLength();
   if (ax == 2) return fYaxis.GetTickLength();
   if (ax == 3) return fZaxis.GetTickLength();
   return 0;
}

//______________________________________________________________________________
Float_t TStyle::GetTitleOffset( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetTitleOffset();
   if (ax == 2) return fYaxis.GetTitleOffset();
   if (ax == 3) return fZaxis.GetTitleOffset();
   return 0;
}

//______________________________________________________________________________
Float_t TStyle::GetTitleSize( Option_t *axis) const
{
   Int_t ax = AxisChoice(axis);
   if (ax == 1) return fXaxis.GetTitleSize();
   if (ax == 2) return fYaxis.GetTitleSize();
   if (ax == 3) return fZaxis.GetTitleSize();
   return 0;
}

//______________________________________________________________________________
void TStyle::SetNdivisions(Int_t n, Option_t *axis)
{
// Set the number of divisions to draw an axis.
//  ndiv      : Number of divisions.
//
//       n = N1 + 100*N2 + 10000*N3
//       N1=number of primary divisions.
//       N2=number of secondary divisions.
//       N3=number of 3rd divisions.
//           e.g.:
//           nndi=0 --> no tick marks.
//           nndi=2 --> 2 divisions, one tick mark in the middle
//                      of the axis.
// axis specifies which axis ("x","y","z"), default = "x"
// if axis="xyz" set all 3 axes

   TString opt = axis;
   opt.ToLower();
   if (opt.Contains("x")) fXaxis.SetNdivisions(n);
   if (opt.Contains("y")) fYaxis.SetNdivisions(n);
   if (opt.Contains("z")) fZaxis.SetNdivisions(n);
}

//______________________________________________________________________________
void TStyle::SetAxisColor(Color_t color, Option_t *axis)
{
// Set color to draw the axis line and tick marks.
// axis specifies which axis ("x","y","z"), default = "x"
// if axis="xyz" set all 3 axes

   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetAxisColor(color);
   if (opt.Contains("y")) fYaxis.SetAxisColor(color);
   if (opt.Contains("z")) fZaxis.SetAxisColor(color);
}

//______________________________________________________________________________
void TStyle::SetHeaderPS(const char *header)
{
// Define a string to be inserted in the Postscript header
// The string in header will be added to the Postscript file
// immediatly following the %%Page line
// For example, this string may contain special Postscript instructions like
//      200 200 translate
// This information is used in TPostScript::Initialize

   fHeaderPS = header;
}

//______________________________________________________________________________
void TStyle::SetTitlePS(const char *pstitle)
{
// Define a string to be used in the %%Title of the Postscript files.
// If this string is not defined, ROOT will use the canvas title.

   fTitlePS = pstitle;
}

//______________________________________________________________________________
void TStyle::SetLabelColor(Color_t color, Option_t *axis)
{
// Set axis labels color.
// axis specifies which axis ("x","y","z"), default = "x"
// if axis="xyz" set all 3 axes

   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetLabelColor(color);
   if (opt.Contains("y")) fYaxis.SetLabelColor(color);
   if (opt.Contains("z")) fZaxis.SetLabelColor(color);
}

//______________________________________________________________________________
void TStyle::SetLabelFont(Style_t font, Option_t *axis)
{
// Set font number used to draw axis labels.
//    font  : Text font code = 10*fontnumber + precision
//             Font numbers must be between 1 and 14
//             precision = 1 fast hardware fonts (steps in the size)
//             precision = 2 scalable and rotatable hardware fonts
// The default font number is 62.
// axis specifies which axis ("x","y","z"), default = "x"
// if axis="xyz" set all 3 axes

   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetLabelFont(font);
   if (opt.Contains("y")) fYaxis.SetLabelFont(font);
   if (opt.Contains("z")) fZaxis.SetLabelFont(font);
}

//______________________________________________________________________________
void TStyle::SetLabelOffset(Float_t offset, Option_t *axis)
{
// Set offset between axis and axis labels.
// The offset is expressed as a percent of the pad height.
// axis specifies which axis ("x","y","z"), default = "x"
// if axis="xyz" set all 3 axes

   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetLabelOffset(offset);
   if (opt.Contains("y")) fYaxis.SetLabelOffset(offset);
   if (opt.Contains("z")) fZaxis.SetLabelOffset(offset);
}

//______________________________________________________________________________
void TStyle::SetLabelSize(Float_t size, Option_t *axis)
{
// Set size of axis labels. The size is expressed as a percent of the pad height.
// axis specifies which axis ("x","y","z"), default = "x"
// if axis="xyz" set all 3 axes

   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetLabelSize(size);
   if (opt.Contains("y")) fYaxis.SetLabelSize(size);
   if (opt.Contains("z")) fZaxis.SetLabelSize(size);
}


//______________________________________________________________________________
void TStyle::SetLineStyleString(Int_t i, const char *text)
{
// Set line style string (used by Postscript)
// PostScript uses the following convention
//  a line is a suite of segments, each segment is described by the
//  number of pixels. For example default line styles are defined as:
//   linestyle 1  "[]"             solid
//   linestyle 2  "[12 12]"        dashed
//   linestyle 3  "[4 8]"          dotted
//   linestyle 4  "[12 15 4 15]"   dash-dotted
//
//  Up to 30 different styles may be defined.
//   The opening and closing brackets may be omitted
//   They will be automaticalled added by this function.

   Int_t nch = strlen(text);
   char *st = new char[nch+10];
   sprintf(st," ");
   if (strstr(text,"[") == 0) strcat(st,"[");
   strcat(st,text);
   if (strstr(text,"]") == 0) strcat(st,"]");
   strcat(st," 0 sd");
   if (i >= 1 && i <= 29) fLineStyle[i] = st;
   delete [] st;
}


//______________________________________________________________________________
void TStyle::SetOptDate(Int_t mode)
{
// if mode is non null, the current date/time will be printed in the canvas.
// The position of the date string can be controlled by:
//    mode = 1 (default) date is printed in the bottom/left corner.
//    mode = 2 date is printed in the bottom/right corner.
//    mode = 3 date is printed in the top/right corner.
//  The date position can also be controlled by:
//    gStyle->SetDateX(x);  x in NDC
//    gStyle->SetDateY(y);  y in NDC
//
//  The date text attributes can be changed with:
//    gStyle->GetAttDate()->SetTextFont(font=62);
//    gStyle->GetAttDate()->SetTextSize(size=0.025);
//    gStyle->GetAttDate()->SetTextAngle(angle=0);
//    gStyle->GetAttDate()->SetTextAlign(align=11);
//    gStyle->GetAttDate()->SetTextColor(color=1);
//
//  The current date attributes can be obtained via:
//    gStyle->GetAttDate()->GetTextxxxx();
//
//  When the date option is active, a text object is created when the pad
//  paint its list of primitives. The text object is named "DATE".
//  The DATE attributes can also be edited interactively (position
//  and attributes) via the normal context menu.

   fOptDate = mode;
   if (mode == 1) {
      SetDateX(0.01);
      SetDateY(0.01);
      fAttDate.SetTextAlign(11);
   }
   if (mode == 2) {
      SetDateX(0.99);
      SetDateY(0.01);
      fAttDate.SetTextAlign(31);
   }
   if (mode == 3) {
      SetDateX(0.99);
      SetDateY(0.99);
      fAttDate.SetTextAlign(33);
   }
}


//______________________________________________________________________________
void TStyle::SetOptFit(Int_t mode)
{
// The type of information about fit parameters printed in the histogram
// statistics box can be selected via the parameter mode.
//  The parameter mode can be = pcev  (default = 0111)
//    v = 1;  print name/values of parameters
//    e = 1;  print errors (if e=1, v must be 1)
//    c = 1;  print Chisquare/Number of degress of freedom
//    p = 1;  print Probability
//  Example: gStyle->SetOptFit(1011);
//           print fit probability, parameter names/values and errors.

   fOptFit = mode;
}

//______________________________________________________________________________
void TStyle::SetOptStat(Int_t mode)
{
// The type of information printed in the histogram statistics box
//  can be selected via the parameter mode.
//  The parameter mode can be = iourmen  (default = 0001111)
//    n = 1;  name of histogram is printed
//    e = 1;  number of entries printed
//    m = 1;  mean value printed
//    r = 1;  rms printed
//    u = 1;  number of underflows printed
//    o = 1;  number of overflows printed
//    i = 1;  integral of bins printed
//  Example: gStyle->SetOptStat(11);
//           print only name of histogram and number of entries.
//
   fOptStat = mode;
}

//______________________________________________________________________________
void TStyle::SetPaperSize(EPaperSize size)
{
   // Set paper size for PostScript output.

   switch (size) {
      case kA4:
         SetPaperSize(20, 26);
         break;
      case kUSLetter:
         SetPaperSize(20, 24);
         break;
      default:
         Error("SetPaperSize", "illegal paper size %d\n", (int)size);
         break;
   }
}

//______________________________________________________________________________
void TStyle::SetPaperSize(Float_t xsize, Float_t ysize)
{
   // Set paper size for PostScript output.
   // The paper size is specified in centimeters. Default is 20x26.
   // See also TPad::Print

   fPaperSizeX = xsize;
   fPaperSizeY = ysize;
}

//______________________________________________________________________________
void TStyle::SetTickLength(Float_t length, Option_t *axis)
{
// set the tick marks length for an axis.
// axis specifies which axis ("x","y","z"), default = "x"
// if axis="xyz" set all 3 axes

   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetTickLength(length);
   if (opt.Contains("y")) fYaxis.SetTickLength(length);
   if (opt.Contains("z")) fZaxis.SetTickLength(length);
}

//______________________________________________________________________________
void TStyle::SetTitleOffset(Float_t offset, Option_t *axis)
{
// Specify a parameter offset to control the distance between the axis
// and the axis title.
// offset = 1 means : use the default distance
// offset = 1.2 means: the distance will be 1.2*(default distance)
// offset = 0.8 means: the distance will be 0.8*(default distance)
//
// axis specifies which axis ("x","y","z"), default = "x"
// if axis="xyz" set all 3 axes

   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetTitleOffset(offset);
   if (opt.Contains("y")) fYaxis.SetTitleOffset(offset);
   if (opt.Contains("z")) fZaxis.SetTitleOffset(offset);
}

//______________________________________________________________________________
void TStyle::SetTitleSize(Float_t size, Option_t *axis)
{
// Set the axis title size.
//
// axis specifies which axis ("x","y","z"), default = "x"
// if axis="xyz" set all 3 axes

   TString opt = axis;
   opt.ToLower();

   if (opt.Contains("x")) fXaxis.SetTitleSize(size);
   if (opt.Contains("y")) fYaxis.SetTitleSize(size);
   if (opt.Contains("z")) fZaxis.SetTitleSize(size);
}

//______________________________________________________________________________
Int_t TStyle::CreateGradientColorTable(UInt_t Number, Double_t* Length,
                              Double_t* Red, Double_t* Green,
                              Double_t* Blue, UInt_t NColors)
{
  // STATIC function.
  // Linear gradient color table:
  // Red, Green and Blue are several RGB colors with values from 0.0 .. 1.0.
  // Their number is "Intervals".
  // Length is the length of the color interval between the RGB-colors:
  // Imaging the whole gradient goes from 0.0 for the first RGB color to 1.0
  // for the last RGB color, then each "Length"-entry in between stands for
  // the length of the intervall between the according RGB colors.
  //
  // This definition is similar to the povray-definition of gradient
  // color tables.
  //
  // In order to create a color table do the following:
  // Define the RGB Colors:
  // > UInt_t Number = 5;
  // > Double_t Red[5]   = { 0.00, 0.09, 0.18, 0.09, 0.00 };
  // > Double_t Green[5] = { 0.01, 0.02, 0.39, 0.68, 0.97 };
  // > Double_t Blue[5]  = { 0.17, 0.39, 0.62, 0.79, 0.97 };
  // Define the length of the (color)-interval between this points
  // > Double_t Stops[5] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
  // i.e. the color interval between Color 2 and Color 3 is
  // 0.79 - 0.62 => 17 % of the total palette area between these colors
  //
  //  Original code by Andreas Zoglauer <zog@mpe.mpg.de>

  UInt_t g, c;
  UInt_t NPalette = 0;
  Int_t *Palette = new Int_t[NColors+1];
  UInt_t NColorsGradient;
  TColor *Color;
  Int_t HighestIndex = 0;

  // Check if all RGB values are between 0.0 and 1.0 and
  // Length goes from 0.0 to 1.0 in increasing order.
  for (c = 0; c < Number; c++) {
    if (Red[c] < 0 || Red[c] > 1.0 ||
        Green[c] < 0 || Green[c] > 1.0 ||
        Blue[c] < 0 || Blue[c] > 1.0 ||
        Length[c] < 0 || Length[c] > 1.0) {
      //Error("CreateGradientColorTable",
      //      "All RGB colors and interval lengths have to be between 0.0 and 1.0");
      delete [] Palette;
      return -1;
    }
    if (c >= 1) {
      if (Length[c-1] > Length[c]) {
        //Error("CreateGradientColorTable",
        //      "The interval lengths have to be in increasing order");
        delete [] Palette;
        return -1;
      }
    }
  }

  // Search for the highest color index not used in ROOT:
  // We do not want to overwrite some colors...
  TSeqCollection *ColorTable = gROOT->GetListOfColors();
  if ((Color = (TColor *) ColorTable->Last()) != 0) {
    if (Color->GetNumber() > HighestIndex) {
      HighestIndex = Color->GetNumber();
    }
    while ((Color = (TColor *) (ColorTable->Before(Color))) != 0) {
      if (Color->GetNumber() > HighestIndex) {
      HighestIndex = Color->GetNumber();
      }
    }
  }
  HighestIndex++;

  // Now create the colors and add them to the default palette:

  // For each defined gradient...
  for (g = 1; g < Number; g++) {
    // create the colors...
    NColorsGradient = (Int_t) (floor(NColors*Length[g]) - floor(NColors*Length[g-1]));
    for (c = 0; c < NColorsGradient; c++) {
      Color = new TColor(HighestIndex,
                         Red[g-1] + c * (Red[g] - Red[g-1])/ NColorsGradient,
                         Green[g-1] + c * (Green[g] - Green[g-1])/ NColorsGradient,
                         Blue[g-1] + c * (Blue[g] - Blue[g-1])/ NColorsGradient,
                         "  ");
      Palette[NPalette] = HighestIndex;
      NPalette++;
      HighestIndex++;
    }
  }

  gStyle->SetPalette(NPalette, Palette);
  delete [] Palette;

  return HighestIndex - NColors;
}

//______________________________________________________________________________
void TStyle::SetPalette(Int_t ncolors, Int_t *colors)
{
// The color palette is used by the histogram classes
//  (see TH1::Draw options).
// For example TH1::Draw("col") draws a 2-D histogram with cells
// represented by a box filled with a color CI function of the cell content.
// if the cell content is N, the color CI used will be the color number
// in colors[N],etc. If the maximum cell content is > ncolors, all
// cell contents are scaled to ncolors.
//
// if ncolors <= 0 a default palette (see below) of 50 colors is defined.
//     the colors defined in this palette are OK for coloring pads, labels
//
// if ncolors == 1 && colors == 0, then
//     a Pretty Palette with a Spectrum Violet->Red is created.
//   It is recommended to use this Pretty palette when drawing legos,
//   surfaces or contours.
//
// if ncolors > 50 and colors=0, the DeepSea palette is used.
//     (see TStyle::CreateGradientColorTable for more details)
//
// if ncolors > 0 and colors = 0, the default palette is used
// with a maximum of ncolors.
//
// The default palette defines:
//   index 0->9   : grey colors from light to dark grey
//   index 10->19 : "brown" colors
//   index 20->29 : "blueish" colors
//   index 30->39 : "redish" colors
//   index 40->49 : basic colors
//
//  The color numbers specified in the palette can be viewed by selecting
//  the item "colors" in the "VIEW" menu of the canvas toolbar.
//  The color parameters can be changed via TColor::SetRGB.

   Int_t i;
   Int_t palette[50] = {19,18,17,16,15,14,13,12,11,20,
                        21,22,23,24,25,26,27,28,29,30, 8,
                        31,32,33,34,35,36,37,38,39,40, 9,
                        41,42,43,44,45,47,48,49,46,50, 2,
                         7, 6, 5, 4, 3, 112,1};
   // set default palette (pad type)
   if (ncolors <= 0) {
      ncolors = 50;
      fPalette.Set(ncolors);
      for (i=0;i<ncolors;i++) fPalette.fArray[i] = palette[i];
      return;
   }

   // set Pretty Palette Spectrum Violet->Red
   if (ncolors == 1 && colors == 0) {
      ncolors = 50;
      fPalette.Set(ncolors);
      for (i=0;i<ncolors;i++) fPalette.fArray[i] = 51+i;
      return;
   }

   // set DeepSea palette
   if (colors == 0 && ncolors > 50) {
      const Int_t NRGBs = 5;
      Double_t Stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
      Double_t Red[NRGBs] = { 0.00, 0.09, 0.18, 0.09, 0.00 };
      Double_t Green[NRGBs] = { 0.01, 0.02, 0.39, 0.68, 0.97 };
      Double_t Blue[NRGBs] = { 0.17, 0.39, 0.62, 0.79, 0.97 };
      CreateGradientColorTable(NRGBs, Stops, Red, Green, Blue, ncolors);
      return;
   }

   // set user defined palette
   fPalette.Set(ncolors);
   if (colors)  for (i=0;i<ncolors;i++) fPalette.fArray[i] = colors[i];
   else         for (i=0;i<ncolors;i++) fPalette.fArray[i] = palette[i];
}

//______________________________________________________________________________
void TStyle::SetTimeOffset(Double_t toffset)
{
// Change the time offset for time plotting.
// Times are expressed in seconds. The corresponding numbers usually have 9
// digits (or more if one takes into account fractions of seconds).
// Thus, since it is very inconvenient to plot very large numbers on a scale,
// one has to set an offset time that will be added to the axis begining,
// in order to plot times correctly and conveniently. A convenient way to
// set the time offset is to use TDatime::Convert().

   fTimeOffset = toffset;
}

//______________________________________________________________________________
void TStyle::SetStripDecimals(Bool_t strip)
{
//  Set option to strip decimals when drawing axis labels.
//  By default, TGaxis::PaintAxis removes trailing 0s after a dot
//  in the axis labels. Ex: {0,0.5,1,1.5,2,2.5, etc}
//  If this function is called with strip=kFALSE, TGAxis::PaintAxis will
//  draw labels with the same number of digits after the dot
//  Ex: (0.0,0.5,1.0,1.5,2.0,2.5,etc}

   fStripDecimals = strip;
}
