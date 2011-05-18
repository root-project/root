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
#include "Strlen.h"
#include "TROOT.h"
#include "TAttText.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TVirtualX.h"
#include "TError.h"
#include "TVirtualPadEditor.h"
#include "TColor.h"

ClassImp(TAttText)


//______________________________________________________________________________
/* Begin_Html
<center><h2>Text Attributes class</h2></center>

This class is used (in general by secondary inheritance)
by many other classes (graphics, histograms). It holds all the text attributes.

<h3>Text attributes</h3>
Text attributes are:
<ul>
<li><a href="#T1">Text Alignment.</a></li>
<li><a href="#T2">Text Angle.</a></li>
<li><a href="#T3">Text Color.</a></li>
<li><a href="#T4">Text Size.</a></li>
<li><a href="#T5">Text Font and Precision.</a></li>
<ul>
   <li><a href="#T51">Font quality and speed.</a></li>
   <li><a href="#T52">How to use True Type Fonts.</a></li>
   <li><a href="#T53">List of the currently supported fonts.</a></li>
</ul>
</ul>

<a name="T1"></a><h3>Text Alignment</h3>
The text alignment is an integer number (<tt>align</tt>) allowing to control
the horizontal and vertical position of the text string with respect
to the text position.
The text alignment of any class inheriting from <tt>TAttText</tt> can
be changed using the method <tt>SetTextAlign</tt> and retrieved using the
method <tt>GetTextAlign</tt>.
<pre>
   align = 10*HorizontalAlign + VerticalAlign
</pre>
For horizontal alignment the following convention applies:
<pre>
   1=left adjusted, 2=centered, 3=right adjusted
</pre>
For vertical alignment the following convention applies:
<pre>
   1=bottom adjusted, 2=centered, 3=top adjusted
</pre>
For example:
<pre>
   align = 11 = left adjusted and bottom adjusted
   align = 32 = right adjusted and vertically centered
</pre>
End_Html
Begin_Macro(source)
textalign.C
End_Macro

Begin_Html
<a name="T2"></a><h3>Text Angle</h3>
Text angle in degrees.
The text angle of any class inheriting from <tt>TAttText</tt> can
be changed using the method <tt>SetTextAngle</tt> and retrieved using the
method <tt>GetTextAngle</tt>.
The following picture shows the text angle:
End_Html
Begin_Macro(source)
textangle.C
End_Macro

Begin_Html
<a name="T3"></a><h3>Text Color</h3>
The text color is a color index (integer) pointing in the ROOT
color table.
The text color of any class inheriting from <tt>TAttText</tt> can
be changed using the method <tt>SetTextColor</tt> and retrieved using the
method <tt>GetTextColor</tt>.
The following table shows the first 50 default colors.
End_Html
Begin_Macro(source)
{
   TCanvas *c = new TCanvas("c","Text colors",0,0,500,200);
   c.DrawColorTable();
   return c;
}
End_Macro

Begin_Html
<a name="T4"></a><h3>Text Size</h3>
If the text precision (see next paragraph) is smaller than 3, the text
size (<tt>textsize</tt>) is a fraction of the current pad size. Therefore the
same <tt>textsize</tt> value can generate text outputs with different absolute
sizes in two different pads.
The text size in pixels (<tt>charheight</tt>) is computed the following way:
<p>
<pre>
   pad_width  = gPad->XtoPixel(gPad->GetX2());
   pad_height = gPad->YtoPixel(gPad->GetY1());
   if (pad_width < pad_height)  charheight = textsize*pad_width;
   else                         charheight = textsize*pad_height;
</pre>
<p>
If the text precision is equal to 3, the text size doesn't depend on the pad's
dimensions. A given <tt>textsize</tt> value always generates the same absolute
size. The text size (<tt>charheight</tt>) is given in pixels:
<pre>
   charheight = textsize;
</pre>
<p>
Note that to scale fonts to the same size as the old True Type package a
scale factor of <tt>0.93376068</tt> is apply to the text size before drawing.
<p>
The text size of any class inheriting from <tt>TAttText</tt> can
be changed using the method <tt>SetTextSize</tt> and retrieved using the
method <tt>GetTextSize</tt>.


<a name="T5"></a><h3>Text Font and Precision</h3>
The text font code is combination of the font number and the precision.
<pre>
   Text font code = 10*fontnumber + precision
</pre>
Font numbers must be between 1 and 14.
<p>
The precision can be:

<br><tt>precision = 0</tt> fast hardware fonts (steps in the size)
<br><tt>precision = 1</tt> scalable and rotatable hardware fonts (see below)
<br><tt>precision = 2</tt> scalable and rotatable hardware fonts
<br><tt>precision = 3</tt> scalable and rotatable hardware fonts. Text size
                           is given in pixels.
<p>
The text font and precision of any class inheriting from <tt>TAttText</tt> can
be changed using the method <tt>SetTextFont</tt> and retrieved using the
method <tt>GetTextFont</tt>.

<a name="T51"></a><h4>Font quality and speed</h4>
When precision 0 is used, only the original non-scaled system fonts are
used. The fonts have a minimum (4) and maximum (37) size in pixels. These
fonts are fast and are of good quality. Their size varies with large steps
and they cannot be rotated.
Precision 1 and 2 fonts have a different behaviour depending if the
True Type Fonts (TTF) are used or not. If TTF are used, you always get very good
quality scalable and rotatable fonts. However TTF are slow.

<a name="T52"></a><h4>How to use True Type Fonts</h4>
One can activate the TTF by adding (or activating) the following line
in the <tt>.rootrc</tt> file:
<pre>
   Unix.*.Root.UseTTFonts:     true
</pre>
It is possible to check the TTF are in use in a Root session
with the command:
<pre>
   gEnv->Print();
</pre>
If the TTF are in use the following line will appear at the beginning of the
printout given by this command:
<pre>
   Unix.*.Root.UseTTFonts:   true                           [Global]
</pre>


<a name="T53"></a><h4>List of the currently supported fonts</h4>
<pre>
   Font number         X11 Names             Win32/TTF Names
       1 :       times-medium-i-normal      "Times New Roman"
       2 :       times-bold-r-normal        "Times New Roman"
       3 :       times-bold-i-normal        "Times New Roman"
       4 :       helvetica-medium-r-normal  "Arial"
       5 :       helvetica-medium-o-normal  "Arial"
       6 :       helvetica-bold-r-normal    "Arial"
       7 :       helvetica-bold-o-normal    "Arial"
       8 :       courier-medium-r-normal    "Courier New"
       9 :       courier-medium-o-normal    "Courier New"
      10 :       courier-bold-r-normal      "Courier New"
      11 :       courier-bold-o-normal      "Courier New"
      12 :       symbol-medium-r-normal     "Symbol"
      13 :       times-medium-r-normal      "Times New Roman"
      14 :                                  "Wingdings"
      15 :       Symbol italic (derived from Symbol)
</pre>
<br>
 The following picture shows how each font looks. The number on the left
 is the "text font code". In this picture precision 2 was selected.
End_Html
Begin_Macro(source)
fonts.C
End_Macro */


//______________________________________________________________________________
TAttText::TAttText()
{
   // AttText default constructor.
   //
   // Default text attributes are taken from the current style.

   if (!gStyle) {
      ResetAttText();
      return;
   }
   fTextAlign = gStyle->GetTextAlign();
   fTextAngle = gStyle->GetTextAngle();
   fTextColor = gStyle->GetTextColor();
   fTextFont  = gStyle->GetTextFont();
   fTextSize  = gStyle->GetTextSize();
}


//______________________________________________________________________________
TAttText::TAttText(Int_t align, Float_t angle, Color_t color, Style_t font, Float_t tsize)
{
   // AttText normal constructor.
   //
   // Text attributes are taken from the argument list.

   fTextAlign = align;
   fTextAngle = angle;
   fTextColor = color;
   fTextFont  = font;
   fTextSize  = tsize;
}


//______________________________________________________________________________
TAttText::~TAttText()
{
   // AttText destructor.
}


//______________________________________________________________________________
void TAttText::Copy(TAttText &atttext) const
{
   // Copy this text attributes to a new TAttText.

   atttext.fTextAlign  = fTextAlign;
   atttext.fTextAngle  = fTextAngle;
   atttext.fTextColor  = fTextColor;
   atttext.fTextFont   = fTextFont;
   atttext.fTextSize   = fTextSize;
}


//______________________________________________________________________________
void TAttText::Modify()
{
   // Change current text attributes if necessary.

   if (!gPad) return;

   // Do we need to change font?
   if (!gPad->IsBatch()) {
      gVirtualX->SetTextAngle(fTextAngle);
      Float_t wh = (Float_t)gPad->XtoPixel(gPad->GetX2());
      Float_t hh = (Float_t)gPad->YtoPixel(gPad->GetY1());
      Float_t tsize;
      if (wh < hh)  tsize = fTextSize*wh;
      else          tsize = fTextSize*hh;
      if (fTextFont%10 > 2) tsize = fTextSize;

      if (gVirtualX->GetTextFont() != fTextFont) {
            gVirtualX->SetTextFont(fTextFont);
            gVirtualX->SetTextSize(tsize);
      }
      if (gVirtualX->GetTextSize() != tsize)
            gVirtualX->SetTextSize(tsize);
      gVirtualX->SetTextAlign(fTextAlign);
      gVirtualX->SetTextColor(fTextColor);
   }
   gPad->SetAttTextPS(fTextAlign,fTextAngle,fTextColor,fTextFont,fTextSize);
}


//______________________________________________________________________________
void TAttText::ResetAttText(Option_t *)
{
   // Reset this text attributes to default values.

   fTextAlign  = 11;
   fTextAngle  = 0;
   fTextColor  = 1;
   fTextFont   = 62;
   fTextSize   = 0.05;
}


//______________________________________________________________________________
void TAttText::SaveTextAttributes(ostream &out, const char *name, Int_t alidef,
                                  Float_t angdef, Int_t coldef, Int_t fondef,
                                  Float_t sizdef)
{
   // Save text attributes as C++ statement(s) on output stream out.

   if (fTextAlign != alidef) {
      out<<"   "<<name<<"->SetTextAlign("<<fTextAlign<<");"<<endl;
   }
   if (fTextColor != coldef) {
      if (fTextColor > 228) {
         TColor::SaveColor(out, fTextColor);
         out<<"   "<<name<<"->SetTextColor(ci);" << endl;
      } else
         out<<"   "<<name<<"->SetTextColor("<<fTextColor<<");"<<endl;
   }
   if (fTextFont != fondef) {
      out<<"   "<<name<<"->SetTextFont("<<fTextFont<<");"<<endl;
   }
   if (fTextSize != sizdef) {
      out<<"   "<<name<<"->SetTextSize("<<fTextSize<<");"<<endl;
   }
   if (fTextAngle != angdef) {
      out<<"   "<<name<<"->SetTextAngle("<<fTextAngle<<");"<<endl;
   }
}


//______________________________________________________________________________
void TAttText::SetTextAttributes()
{
   // Invoke the DialogCanvas Text attributes.

   TVirtualPadEditor::UpdateTextAttributes(fTextAlign,fTextAngle,fTextColor,
                                           fTextFont,fTextSize);
}


//______________________________________________________________________________
void TAttText::SetTextSizePixels(Int_t npixels)
{
   // Set the text size in pixels.
   // If the font precision is greater than 2, the text size is set to npixels,
   // otherwise the text size is computed as a percent of the pad size.

   if (fTextFont%10 > 2) {
      fTextSize = Float_t(npixels);
   } else {
      TVirtualPad *pad = gROOT->GetSelectedPad();
      if (!pad) return;
      Float_t dy = pad->AbsPixeltoY(0) - pad->AbsPixeltoY(npixels);
      fTextSize = dy/(pad->GetY2() - pad->GetY1());
   }
}
