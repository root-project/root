// @(#)root/base:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TAttFill.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TVirtualX.h"
#include "TVirtualPadEditor.h"
#include "TColor.h"

ClassImp(TAttFill)


//______________________________________________________________________________
/* Begin_Html
<center><h2>Fill Area Attributes class</h2></center>

This class is used (in general by secondary inheritance)
by many other classes (graphics, histograms). It holds all the fill area
attributes.

<h3>Fill Area attributes</h3>
Fill Area attributes are:
<ul>
<li><a href="#F1">Fill Area color.</a></li>
<li><a href="#F2">Fill Area style.</a></li>
</ul>

<a name="F1"></a><h3>Fill Area color</h3>
The fill area color is a color index (integer) pointing in the ROOT
color table.
The fill area color of any class inheriting from <tt>TAttFill</tt> can
be changed using the method <tt>SetFillColor</tt> and retrieved using the
method <tt>GetFillColor</tt>.
The following table shows the first 50 default colors.
End_Html
Begin_Macro(source)
{
   TCanvas *c = new TCanvas("c","Fill Area colors",0,0,500,200);
   c.DrawColorTable();
   return c;
}
End_Macro

Begin_Html
<h4>The ROOT Color Wheel.</h4>
The wheel contains the recommended 216 colors to be used in web applications.
The colors in the Color Wheel are created by TColor::CreateColorWheel.
<p>Using this color set for your text, background or graphics will give your
application a consistent appearance across different platforms and browsers.
<p>Colors are grouped by hue, the aspect most important in human perception 
Touching color chips have the same hue, but with different brightness and vividness.
<p>Colors of slightly different hues <b>clash</b>. If you intend to display
colors of the same hue together, you should pick them from the same group.
<p>Each color chip is identified by a mnemonic (eg kYellow) and a number.
The keywords, kRed, kBlue, kYellow, kPink, etc are defined in the header file <b>Rtypes.h</b>
that is included in all ROOT other header files. We strongly recommend to use these keywords
in your code instead of hardcoded color numbers, eg:
<pre>
   myObject.SetFillColor(kRed);
   myObject.SetFillColor(kYellow-10);
   myLine.SetLineColor(kMagenta+2);
</pre>

End_Html
Begin_Macro(source)
{
   TColorWheel *w = new TColorWheel();
   w->Draw();
   return w->GetCanvas();
}
End_Macro
      
Begin_Html
<h4>Special case forcing black&white output.</h4>
If the current style fill area color is set to 0, then ROOT will force
a black&white output for all objects with a fill area defined and independently
of the object fill style.
   
<a name="F2"></a><h3>Fill Area style</h3>
The fill area style defines the pattern used to fill a polygon.
The fill area style of any class inheriting from <tt>TAttFill</tt> can
be changed using the method <tt>SetFillStyle</tt> and retrieved using the
method <tt>GetFillStyle</tt>.
<h4>Conventions for fill styles:</h4>
<ul>
<li>  0    : hollow                   </li>
<li>  1001 : Solid                    </li>
<li>  2001 : hatch style              </li>
<li>  3000+pattern_number (see below) </li>
<li>  For TPad only:                  </li>
<ul>
   <li>  4000 :the window is transparent.                            </li>
   <li>  4000 to 4100 the window is 100% transparent to 100% opaque. </li>
</ul>
      The pad transparency is visible in binary outputs files like gif, jpg, png etc ..
      but not in vector graphics output files like PS, PDF and SVG.
</ul>

pattern_number can have any value from 1 to 25 (see table), or any
value from 100 to 999. For the latest the numbering convention is the following:
<pre>
      pattern_number = ijk      (FillStyle = 3ijk)
 
      i (1-9) : specify the space between each hatch
                1 = 1/2mm  9 = 6mm
 
      j (0-9) : specify angle between 0 and 90 degrees
                0 = 0
                1 = 10
                2 = 20
                3 = 30
                4 = 45
                5 = Not drawn
                6 = 60
                7 = 70
                8 = 80
                9 = 90
  
      k (0-9) : specify angle between 90 and 180 degrees
                0 = 180
                1 = 170
                2 = 160
                3 = 150
                4 = 135
                5 = Not drawn
                6 = 120
                7 = 110
                8 = 100
                9 = 90
</pre>
The following table shows the list of pattern styles.
The first table displays the 25 fixed patterns. They cannot be
customized unlike the hatches displayed in the second table which be
customized using:
<ul>
<li> <tt>gStyle->SetHatchesSpacing()</tt> to define the spacing between hatches.
<li> <tt>gStyle->SetHatchesLineWidth()</tt> to define the hatches line width.
</ul>
End_Html
Begin_Macro(source)
fillpatterns.C
End_Macro */


//______________________________________________________________________________
TAttFill::TAttFill()
{
   // AttFill default constructor.
   // Default fill attributes are taking from the current style

   if (!gStyle) {fFillColor=1; fFillStyle=0; return;}
   fFillColor = gStyle->GetFillColor();
   fFillStyle = gStyle->GetFillStyle();
}


//______________________________________________________________________________
TAttFill::TAttFill(Color_t color, Style_t style)
{
   // AttFill normal constructor.
   // color Fill Color
   // style Fill Style

   fFillColor = color;
   fFillStyle = style;
}


//______________________________________________________________________________
TAttFill::~TAttFill()
{
   // AttFill destructor.
}


//______________________________________________________________________________
void TAttFill::Copy(TAttFill &attfill) const
{
   // Copy this fill attributes to a new TAttFill.

   attfill.fFillColor  = fFillColor;
   attfill.fFillStyle  = fFillStyle;
}


//______________________________________________________________________________
void TAttFill::Modify()
{
   // Change current fill area attributes if necessary.

   if (!gPad) return;
   if (!gPad->IsBatch()) {
      gVirtualX->SetFillColor(fFillColor);
      gVirtualX->SetFillStyle(fFillStyle);
   }

   gPad->SetAttFillPS(fFillColor,fFillStyle);
}


//______________________________________________________________________________
void TAttFill::ResetAttFill(Option_t *)
{
   // Reset this fill attributes to default values.

   fFillColor = 1;
   fFillStyle = 0;
}


//______________________________________________________________________________
void TAttFill::SaveFillAttributes(ostream &out, const char *name, Int_t coldef, Int_t stydef)
{
    // Save fill attributes as C++ statement(s) on output stream out

   if (fFillColor != coldef) {
      if (fFillColor > 228) {
         TColor::SaveColor(out, fFillColor);
         out<<"   "<<name<<"->SetFillColor(ci);" << endl;
      } else
         out<<"   "<<name<<"->SetFillColor("<<fFillColor<<");"<<endl;
   }
   if (fFillStyle != stydef) {
      out<<"   "<<name<<"->SetFillStyle("<<fFillStyle<<");"<<endl;
   }
}


//______________________________________________________________________________
void TAttFill::SetFillAttributes()
{
   // Invoke the DialogCanvas Fill attributes.

   TVirtualPadEditor::UpdateFillAttributes(fFillColor,fFillStyle);
}
