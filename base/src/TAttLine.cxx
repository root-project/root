// @(#)root/base:$Name:  $:$Id: TAttLine.cxx,v 1.5 2002/01/24 11:39:27 rdm Exp $
// Author: Rene Brun   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "Strlen.h"
#include "TAttLine.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TVirtualX.h"
#include "TMath.h"
#include "Riostream.h"

ClassImp(TAttLine)

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-*-*Line Attributes class*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =====================
//*-*  Line attributes are:
//*-*    Line Color (see TAttFill for the list of colors)
//*-*    Line Style 1=solid, 2=dash, 3=dot-dot, 4=dash-dot
//*-*    Line Width expressed in pixel units
//*-*
//*-*  This class is used (in general by secondary inheritance)
//*-*  by many other classes (graphics, histograms).
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//______________________________________________________________________________
TAttLine::TAttLine()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*AttLine default constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===========================
//*-*  Default line attributes are taking from the current style
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   if (!gStyle) return;
   fLineColor = gStyle->GetLineColor();
   fLineWidth = gStyle->GetLineWidth();
   fLineStyle = gStyle->GetLineStyle();
}

//______________________________________________________________________________
TAttLine::TAttLine(Color_t color, Style_t style, Width_t width)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*AttLine normal constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===========================
//*-*  Line attributes are taking from the argument list
//*-*    color : must be one of the valid color index
//*-*    style : 1=solid, 2=dash, 3=dash-dot, 4=dot-dot
//*-*    width : expressed in pixel units
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   fLineColor = color;
   fLineWidth = width;
   fLineStyle = style;
}

//______________________________________________________________________________
TAttLine::~TAttLine()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*AttLine destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =================
}

//______________________________________________________________________________
void TAttLine::Copy(TAttLine &attline) const
{
//*-*-*-*-*-*-*-*-*Copy this line attributes to a new attline*-*-*-*-*-*-*-*-*
//*-*              ==========================================
   attline.fLineColor  = fLineColor;
   attline.fLineStyle  = fLineStyle;
   attline.fLineWidth  = fLineWidth;
}

//______________________________________________________________________________
Int_t TAttLine::DistancetoLine(Int_t px, Int_t py, Double_t xp1, Double_t yp1, Double_t xp2, Double_t yp2 )
{
//*-*-*-*-*-*-*-*-*-*-*Compute distance from point px,py to a line*-*-*-*-*-*
//*-*                  ===========================================
//*-*  Compute the closest distance of approach from point px,py to this line.
//*-*  The distance is computed in pixels units.
//*-*
//*-*  Algorithm:
//*-*
//*-*    A(x1,y1)         P                             B(x2,y2)
//*-*    ------------------------------------------------
//*-*                     I
//*-*                     I
//*-*                     I
//*-*                     I
//*-*                    M(x,y)
//*-*
//*-*  Let us call  a = distance AM     A=a**2
//*-*               b = distance BM     B=b**2
//*-*               c = distance AB     C=c**2
//*-*               d = distance PM     D=d**2
//*-*               u = distance AP     U=u**2
//*-*               v = distance BP     V=v**2     c = u + v
//*-*
//*-*  D = A - U
//*-*  D = B - V  = B -(c-u)**2
//*-*     ==> u = (A -B +C)/2c
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Double_t xl, xt, yl, yt;
   Double_t x     = px;
   Double_t y     = py;
   Double_t x1    = gPad->XtoAbsPixel(xp1);
   Double_t y1    = gPad->YtoAbsPixel(yp1);
   Double_t x2    = gPad->XtoAbsPixel(xp2);
   Double_t y2    = gPad->YtoAbsPixel(yp2);
   if (x1 < x2) {xl = x1; xt = x2;}
   else         {xl = x2; xt = x1;}
   if (y1 < y2) {yl = y1; yt = y2;}
   else         {yl = y2; yt = y1;}
   if (x < xl-2 || x> xt+2) return 9999;  //following algorithm only valid in the box
   if (y < yl-2 || y> yt+2) return 9999;  //surrounding the line
   Double_t xx1   = x  - x1;
   Double_t xx2   = x  - x2;
   Double_t x1x2  = x1 - x2;
   Double_t yy1   = y  - y1;
   Double_t yy2   = y  - y2;
   Double_t y1y2  = y1 - y2;
   Double_t A     = xx1*xx1   + yy1*yy1;
   Double_t B     = xx2*xx2   + yy2*yy2;
   Double_t C     = x1x2*x1x2 + y1y2*y1y2;
   if (C <= 0)  return 9999;
   Double_t c     = TMath::Sqrt(C);
   Double_t u     = (A - B + C)/(2*c);
   Double_t D     = TMath::Abs(A - u*u);
   if (D < 0)   return 9999;

   return Int_t(TMath::Sqrt(D) - 0.5*Double_t(fLineWidth));
}

//______________________________________________________________________________
void TAttLine::Modify()
{
//*-*-*-*-*-*-*-*-*-*Change current line attributes if necessary*-*-*-*-*-*-*
//*-*                ===========================================

   if (!gPad) return;
   if (!gPad->IsBatch()) {
      gVirtualX->SetLineColor(fLineColor);
      if (fLineStyle > 0 && fLineStyle < 5) gVirtualX->SetLineStyle(fLineStyle);
      else                                  gVirtualX->SetLineStyle(1);
      gVirtualX->SetLineWidth(fLineWidth);
   }

   if (fLineStyle > 0 && fLineStyle < 30) gPad->SetAttLinePS(fLineColor,fLineStyle,fLineWidth);
   else                                   gPad->SetAttLinePS(fLineColor,1,fLineWidth);
}

//______________________________________________________________________________
void TAttLine::ResetAttLine(Option_t *)
{
//*-*-*-*-*-*-*-*-*Reset this line attributes to default values*-*-*-*-*-*-*
//*-*              ============================================

   fLineColor  = 1;
   fLineStyle  = 1;
   fLineWidth  = 1;
}

//______________________________________________________________________________
void TAttLine::SaveLineAttributes(ofstream &out, const char *name, Int_t coldef, Int_t stydef, Int_t widdef)
{
    // Save line attributes as C++ statement(s) on output stream out

   if (fLineColor != coldef) {
      out<<"   "<<name<<"->SetLineColor("<<fLineColor<<");"<<endl;
   }
   if (fLineStyle != stydef) {
      out<<"   "<<name<<"->SetLineStyle("<<fLineStyle<<");"<<endl;
   }
   if (fLineWidth != widdef) {
      out<<"   "<<name<<"->SetLineWidth("<<fLineWidth<<");"<<endl;
   }
}

//______________________________________________________________________________
void TAttLine::SetLineAttributes()
{
//*-*-*-*-*-*-*-*-*Invoke the DialogCanvas Line attributes*-*-*-*-*-*-*
//*-*              =======================================

   if (gPad) gROOT->SetSelectedPad(gPad->GetSelectedPad());

   TList *lc = (TList*)gROOT->GetListOfCanvases();
   if (!lc->FindObject("R__attline")) {
      gROOT->ProcessLine("TAttLineCanvas *R__attline = "
                         "new TAttLineCanvas(\"R__attline\",\"Line Attributes\","
                         "250,400);");
   }
   gROOT->ProcessLine(Form("R__attline->UpdateLineAttributes(%d,%d,%d);"
                           "R__attline->Show();",fLineColor,fLineStyle,fLineWidth));
}
