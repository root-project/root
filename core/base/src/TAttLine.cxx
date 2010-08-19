// @(#)root/base:$Id$
// Author: Rene Brun   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TAttLine.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TVirtualX.h"
#include "TVirtualPadEditor.h"
#include "TColor.h"
#include <cmath>


ClassImp(TAttLine)


//______________________________________________________________________________
/* Begin_Html
<center><h2>Line Attributes class</h2></center>

This class is used (in general by secondary inheritance)
by many other classes (graphics, histograms). It holds all the line attributes.

<h3>Line attributes</h3>
Line attributes are:
<ul>
<li><a href="#L1">Line Color.</a></li>
<li><a href="#L2">Line Width.</a></li>
<li><a href="#L3">Line Style.</a></li>
</ul>

<a name="L1"></a><h3>Line Color</h3>
The line color is a color index (integer) pointing in the ROOT
color table.
The line color of any class inheriting from <tt>TAttLine</tt> can
be changed using the method <tt>SetLineColor</tt> and retrieved using the
method <tt>GetLineColor</tt>.
The following table shows the first 50 default colors.
End_Html
Begin_Macro(source)
{
   TCanvas *c = new TCanvas("c","Line colors",0,0,500,200);
   c.DrawColorTable();
   return c;
}
End_Macro

Begin_Html
<a name="L2"></a><h3>Line Width</h3>
The line width is expressed in pixel units.
The line width of any class inheriting from <tt>TAttLine</tt> can
be changed using the method <tt>SetLineWidth</tt> and retrieved using the
method <tt>GetLineWidth</tt>.
The following picture shows the line widths from 1 to 10 pixels.
End_Html
Begin_Macro(source)
{
   TCanvas *Lw = new TCanvas("Lw","test",500,200);
   TText  t;
   t.SetTextAlign(32);
   t.SetTextSize(0.08);
   Int_t i=1;
   for (float s=0.1; s<1.0 ; s+=0.092) {
      TLine *lh = new TLine(0.15,s,.85,s);
      lh->SetLineWidth(i);
      t.DrawText(0.1,s,Form("%d",i++));
      lh->Draw();
   }
   return Lw;
}
End_Macro

Begin_Html
<a name="L3"></a><h3>Line Style</h3>
Line styles are identified via integer numbers. The line style of any class
inheriting from <tt>TAttLine</tt> can be changed using the method
<tt>SetLineStyle</tt> and retrieved using the method <tt>GetLineStyle</tt>.
<br>
The first 10 line styles are predefined as shown on the following picture:
End_Html
Begin_Macro(source)
{
   TCanvas *Ls = new TCanvas("Ls","test",500,200);
   TText  t;
   t.SetTextAlign(32);
   t.SetTextSize(0.08);
   Int_t i=1;
   for (float s=0.1; s<1.0 ; s+=0.092) {
      TLine *lh = new TLine(0.15,s,.85,s);
      lh->SetLineStyle(i);
      t.DrawText(0.1,s,Form("%d",i++));
      lh->Draw();
   }
   return Ls;
}
End_Macro 

Begin_Html
Additional line styles can be defined using <tt>TStyle::SetLineStyleString</tt>.
<br>For example the line style number 11 can be defined as follow:
<pre>
   gStyle->SetLineStyleString(11,"400 200");
</pre>
Existing line styles (1 to 10) can be redefined using the same method.
End_Html */


//______________________________________________________________________________
TAttLine::TAttLine()
{
   // AttLine default constructor.

   if (!gStyle) {fLineColor=1; fLineWidth=1; fLineStyle=1; return;}
   fLineColor = gStyle->GetLineColor();
   fLineWidth = gStyle->GetLineWidth();
   fLineStyle = gStyle->GetLineStyle();
}


//______________________________________________________________________________
TAttLine::TAttLine(Color_t color, Style_t style, Width_t width)
{
   // AttLine normal constructor.
   // Line attributes are taking from the argument list
   //   color : must be one of the valid color index
   //   style : 1=solid, 2=dash, 3=dash-dot, 4=dot-dot. New styles can be
   //           defined using TStyle::SetLineStyleString.
   //   width : expressed in pixel units

   fLineColor = color;
   fLineWidth = width;
   fLineStyle = style;
}


//______________________________________________________________________________
TAttLine::~TAttLine()
{
   // AttLine destructor.
}


//______________________________________________________________________________
void TAttLine::Copy(TAttLine &attline) const
{
   // Copy this line attributes to a new TAttLine.

   attline.fLineColor  = fLineColor;
   attline.fLineStyle  = fLineStyle;
   attline.fLineWidth  = fLineWidth;
}


//______________________________________________________________________________
Int_t TAttLine::DistancetoLine(Int_t px, Int_t py, Double_t xp1, Double_t yp1, Double_t xp2, Double_t yp2 )
{
   // Compute distance from point px,py to a line.
   // Compute the closest distance of approach from point px,py to this line.
   // The distance is computed in pixels units.
   //
   // Algorithm:
   //
   //   A(x1,y1)         P                             B(x2,y2)
   //   -----------------+------------------------------
   //                    |
   //                    |
   //                    |
   //                    |
   //                   M(x,y)
   //
   // Let us call  a = distance AM     A=a**2
   //              b = distance BM     B=b**2
   //              c = distance AB     C=c**2
   //              d = distance PM     D=d**2
   //              u = distance AP     U=u**2
   //              v = distance BP     V=v**2     c = u + v
   //
   // D = A - U
   // D = B - V  = B -(c-u)**2
   //    ==> u = (A -B +C)/2c

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
   Double_t a     = xx1*xx1   + yy1*yy1;
   Double_t b     = xx2*xx2   + yy2*yy2;
   Double_t c     = x1x2*x1x2 + y1y2*y1y2;
   if (c <= 0)  return 9999;
   Double_t v     = sqrt(c);
   Double_t u     = (a - b + c)/(2*v);
   Double_t d     = TMath::Abs(a - u*u);
   if (d < 0)   return 9999;

   return Int_t(sqrt(d) - 0.5*Double_t(fLineWidth));
}


//______________________________________________________________________________
void TAttLine::Modify()
{
   // Change current line attributes if necessary.

   if (!gPad) return;
   Int_t lineWidth = TMath::Abs(fLineWidth%100);
   if (!gPad->IsBatch()) {
      gVirtualX->SetLineColor(fLineColor);
      if (fLineStyle > 0 && fLineStyle < 30) gVirtualX->SetLineStyle(fLineStyle);
      else                                   gVirtualX->SetLineStyle(1);
      gVirtualX->SetLineWidth(lineWidth);
   }

   if (fLineStyle > 0 && fLineStyle < 30) gPad->SetAttLinePS(fLineColor,fLineStyle,lineWidth);
   else                                   gPad->SetAttLinePS(fLineColor,1,lineWidth);
}


//______________________________________________________________________________
void TAttLine::ResetAttLine(Option_t *)
{
   // Reset this line attributes to default values.

   fLineColor  = 1;
   fLineStyle  = 1;
   fLineWidth  = 1;
}


//______________________________________________________________________________
void TAttLine::SaveLineAttributes(ostream &out, const char *name, Int_t coldef, Int_t stydef, Int_t widdef)
{
   // Save line attributes as C++ statement(s) on output stream out.

   if (fLineColor != coldef) {
      if (fLineColor > 228) {
         TColor::SaveColor(out, fLineColor);
         out<<"   "<<name<<"->SetLineColor(ci);" << endl;
      } else
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
   // Invoke the DialogCanvas Line attributes.

   TVirtualPadEditor::UpdateLineAttributes(fLineColor,fLineStyle,fLineWidth);
}
