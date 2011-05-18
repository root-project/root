// @(#)root/base:$Id$
// Author: Rene Brun   12/05/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "Strlen.h"
#include "TAttMarker.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TVirtualX.h"
#include "TVirtualPadEditor.h"
#include "TColor.h"

ClassImp(TAttMarker)


//______________________________________________________________________________
/* Begin_Html
<center><h2>Marker Attributes class</h2></center>

This class is used (in general by secondary inheritance)
by many other classes (graphics, histograms). It holds all the markers
attributes.

<h3>Marker attributes</h3>
The marker attributes are:
<ul>
<li><a href="#M1">Marker color.</a></li>
<li><a href="#M2">Marker style.</a></li>
<li><a href="#M3">Marker size.</a></li>
</ul>

<a name="M1"></a><h3>Marker color</h3>
The marker color is a color index (integer) pointing in the ROOT color
table.
The marker color of any class inheriting from <tt>TAttMarker</tt> can
be changed using the method <tt>SetMarkerColor</tt> and retrieved using the
method <tt>GetMarkerColor</tt>.
The following table shows the first 50 default colors.
End_Html
Begin_Macro(source)
{
   TCanvas *c = new TCanvas("c","Marker colors",0,0,500,200);
   c.DrawColorTable();
   return c;
}
End_Macro

Begin_Html
<a name="M2"></a><h3>Marker style</h3>
The Marker style defines the markers' shape.
The marker style of any class inheriting from <tt>TAttMarker</tt> can
be changed using the method <tt>SetMarkerStyle</tt> and retrieved using the
method <tt>GetMarkerStyle</tt>.
The following list gives the currently supported markers (screen
and PostScript) style. Each marker style is identified by an integer number
(first column) corresponding to a marker shape (second column) and can be also
accessed via a global name (third column).
<p>
<pre>
   Marker number         Marker shape          Marker name
        1                    dot                  kDot
        2                    +                    kPlus
        3                    *                    kStar
        4                    o                    kCircle
        5                    x                    kMultiply
        6                    small dot            kFullDotSmall
        7                    medium dot           kFullDotMedium
        8                    large scalable dot   kFullDotLarge
        9 -->19              large scalable dot
       20                    full circle          kFullCircle
       21                    full square          kFullSquare
       22                    full triangle up     kFullTriangleUp
       23                    full triangle down   kFullTriangleDown
       24                    open circle          kOpenCircle
       25                    open square          kOpenSquare
       26                    open triangle up     kOpenTriangleUp
       27                    open diamond         kOpenDiamond
       28                    open cross           kOpenCross
       29                    full star            kOpenStar
       30                    open star            kFullStar
       31                    *
       32                    open triangle down
       33                    full diamond
       34                    full cross
</pre>
End_Html
Begin_Macro(source)
{
   TCanvas *c = new TCanvas("c","Marker types",0,0,500,200);
   TMarker marker;
   marker.DisplayMarkerTypes();
   return c;
}
End_Macro

Begin_Html
<a name="M3"></a><h3>Marker size</h3>
Various marker sizes are shown in the figure below. The default marker size=1
is shown in the top left corner. Marker sizes smaller than 1 can be
specified. The marker size does not refer to any coordinate systems, it is an
absolute value. Therefore the marker size is not affected by any change
in TPad's scale. A marker size equl to 1 correspond to 8 pixels.
That is, a square marker with size 1 will be drawn with a side equal to 8
pixels on the screen.
The marker size of any class inheriting from <tt>TAttMarker</tt> can
be changed using the method <tt>SetMarkerSize</tt> and retrieved using the
method <tt>GetMarkerSize</tt>.
End_Html
Begin_Macro(source)
{
   c = new TCanvas("c","Marker sizes",0,0,500,200);
   TMarker marker;
   marker.SetMarkerStyle(3);
   Double_t x = 0;
   Double_t dx = 1/6.0;
   for (Int_t i=1; i<6; i++) {
      x += dx;
      marker.SetMarkerSize(i*0.2); marker.DrawMarker(x,.165);
      marker.SetMarkerSize(i*0.8); marker.DrawMarker(x,.495);
      marker.SetMarkerSize(i*1.0); marker.DrawMarker(x,.835);
   }
   return c;
}
End_Macro

Begin_Html
Note that the marker styles number 1 6 and 7 (the dots), cannot be scaled. They
are meant to be very fast to draw and are always drawn with the same number of
pixels; therefore <tt>SetMarkerSize</tt> does not apply on them. To have a
"scalable dot" a filled circle should be used instead, i.e. the marker style
number 20. By default (if <tt>SetMarkerStyle</tt> is not specified), the marker
style used is 1. That's the most common one to draw scatter plots.
End_Html */


//______________________________________________________________________________
TAttMarker::TAttMarker()
{
   // TAttMarker default constructor.
   //
   // Default text attributes are taking from the current style.

   if (!gStyle) {fMarkerColor=1; fMarkerStyle=1; fMarkerSize=1; return;}
   fMarkerColor = gStyle->GetMarkerColor();
   fMarkerStyle = gStyle->GetMarkerStyle();
   fMarkerSize  = gStyle->GetMarkerSize();
}


//______________________________________________________________________________
TAttMarker::TAttMarker(Color_t color, Style_t style, Size_t msize)
{
   // TAttMarker normal constructor.
   //
   // Text attributes are taking from the argument list
   //    color : Marker Color Index
   //    style : Marker style (from 1 to 30)
   //    size  : marker size (float)

   fMarkerColor = color;
   fMarkerSize  = msize;
   fMarkerStyle = style;
}


//______________________________________________________________________________
TAttMarker::~TAttMarker()
{
   // TAttMarker destructor.
}


//______________________________________________________________________________
void TAttMarker::Copy(TAttMarker &attmarker) const
{
   // Copy this marker attributes to a new TAttMarker.

   attmarker.fMarkerColor  = fMarkerColor;
   attmarker.fMarkerStyle  = fMarkerStyle;
   attmarker.fMarkerSize   = fMarkerSize;
}


//______________________________________________________________________________
void TAttMarker::Modify()
{
   // Change current marker attributes if necessary.

   if (!gPad) return;
   if (!gPad->IsBatch()) {
      gVirtualX->SetMarkerColor(fMarkerColor);
      gVirtualX->SetMarkerSize (fMarkerSize);
      gVirtualX->SetMarkerStyle(fMarkerStyle);
   }

   gPad->SetAttMarkerPS(fMarkerColor,fMarkerStyle,fMarkerSize);
}


//______________________________________________________________________________
void TAttMarker::ResetAttMarker(Option_t *)
{
   // Reset this marker attributes to the default values.

   fMarkerColor  = 1;
   fMarkerStyle  = 1;
   fMarkerSize   = 1;
}


//______________________________________________________________________________
void TAttMarker::SaveMarkerAttributes(ostream &out, const char *name, Int_t coldef, Int_t stydef, Int_t sizdef)
{
   // Save line attributes as C++ statement(s) on output stream out.

   if (fMarkerColor != coldef) {
      if (fMarkerColor > 228) {
         TColor::SaveColor(out, fMarkerColor);
         out<<"   "<<name<<"->SetMarkerColor(ci);" << endl;
      } else
         out<<"   "<<name<<"->SetMarkerColor("<<fMarkerColor<<");"<<endl;
   }
   if (fMarkerStyle != stydef) {
      out<<"   "<<name<<"->SetMarkerStyle("<<fMarkerStyle<<");"<<endl;
   }
   if (fMarkerSize != sizdef) {
      out<<"   "<<name<<"->SetMarkerSize("<<fMarkerSize<<");"<<endl;
   }
}


//______________________________________________________________________________
void TAttMarker::SetMarkerAttributes()
{
   // Invoke the DialogCanvas Marker attributes.

   TVirtualPadEditor::UpdateMarkerAttributes(fMarkerColor,fMarkerStyle,fMarkerSize);
}
