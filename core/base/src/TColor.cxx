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
#include "TROOT.h"
#include "TColor.h"
#include "TObjArray.h"
#include "TVirtualPad.h"
#include "TVirtualX.h"
#include "TError.h"
#include "TMathBase.h"
#include "TApplication.h"
#include <cmath>

ClassImp(TColor)

Bool_t  TColor::fgGrayscaleMode = kFALSE;
Bool_t  TColor::fgInitDone = kFALSE;
TArrayI TColor::fgPalette(0);

using std::floor;

//______________________________________________________________________________
/* Begin_Html
<center><h2>The color creation and management class</h2></center>

<ul>
<li><a href="#C00">Introduction</li></a>
<li><a href="#C01">Basic colors</li></a>
<li><a href="#C02">The color wheel</li></a>
<li><a href="#C03">Bright and dark colors</li></a>
<li><a href="#C04">Gray scale view of of canvas with colors</li></a>
<li><a href="#C05">Color palettes</li></a>
<li><a href="#C06">Color transparency</li></a>
</ul>

<a name="C00"></a><h3>Introduction</h3>

Colors are defined by their red, green and blue components, simply called the
RGB components. The colors are also known by the hue, light and saturation
components also known as the HLS components. When a new color is created the
components of both color systems are computed.
<p>
At initialization time, a table of colors is generated. An existing color can
be retrieved by its index:
<br><pre>
   TColor *color = gROOT->GetColor(10);
</pre>
<br>Then it can be manipulated. For example its RGB components can be modified:
<p>
<pre>
   color->SetRGB(0.1, 0.2, 0.3);
</pre>
<br>A new color can be created the following way:
<p>
<pre>
   Int_t ci = 1756; // color index
   TColor *color = new TColor(ci, 0.1, 0.2, 0.3);
</pre>
<p>
Two sets of colors are initialized;
<ul>
<li> The basic colors: colors with index from 0 to 50.
<li> The color wheel: colors with indices from 300 to 1000.
</ul>

<a name="C01"></a><h3>Basic colors</h3>
The following image displays the 50 basic colors.
End_Html
Begin_Macro(source)
{
   TCanvas *c = new TCanvas("c","Fill Area colors",0,0,500,200);
   c.DrawColorTable();
   return c;
}
End_Macro
Begin_Html

<a name="C02"></a><h3>The color wheel</h3>
The wheel contains the recommended 216 colors to be used in web applications.

The colors in the color wheel are created by <tt>TColor::CreateColorWheel</tt>.
<p>
Using this color set for your text, background or graphics will give your
application a consistent appearance across different platforms and browsers.
<p>
Colors are grouped by hue, the aspect most important in human perception.
Touching color chips have the same hue, but with different brightness and
vividness.
<p>
Colors of slightly different hues <b>clash</b>. If you intend to display
colors of the same hue together, you should pick them from the same group.
<p>
Each color chip is identified by a mnemonic (e.g. kYellow) and a number.
The keywords, kRed, kBlue, kYellow, kPink, etc are defined in the header file
<b>Rtypes.h</b> that is included in all ROOT other header files. It is better
to use these keywords in user code instead of hardcoded color numbers, e.g.:
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

<a name="C03"></a><h3>Bright and dark colors</h3>
The dark and bright color are used to give 3-D effects when drawing various
boxes (see TWbox, TPave, TPaveText, TPaveLabel, etc).
<ul>
   <li>The dark colors have an index = color_index+100
   <li>The bright colors have an index = color_index+150
   <li>Two static functions return the bright and dark color number
       corresponding to a color index. If the bright or dark color does not
       exist, they are created:
   <pre>
      Int_t dark   = TColor::GetColorDark(color_index);
      Int_t bright = TColor::GetColorBright(color_index);
   </pre>
</ul>

<a name="C04"></a><h3>Grayscale view of of canvas with colors</h3>
One can toggle between a grayscale preview and the regular colored mode using
<tt>TCanvas::SetGrayscale()</tt>. Note that in grayscale mode, access via RGB
will return grayscale values according to ITU standards (and close to b&w
printer grayscales), while access via HLS returns de-saturated grayscales. The
image below shows the ROOT color wheel in grayscale mode.

End_Html
Begin_Macro(source)
{
   TColorWheel *w = new TColorWheel();
   w->Draw();
   w->GetCanvas()->SetGrayscale();
   w->GetCanvas()->Modified();
   w->GetCanvas()->Update();
   return w->GetCanvas();
}
End_Macro
Begin_Html

<a name="C05"></a><h3>Color palettes</h3>
It is often very useful to represent a variable with a color map. The concept
of "color palette" allows to do that. One color palette is active at any time.
This "current palette" is set using:
<p>
<pre>
gStyle->SetPalette(...);
</pre>
<p>
This function has two parameters: the number of colors in the palette and an
array of containing the indices of colors in the palette. The following small
example demonstrates how to define and use the color palette:

End_Html
Begin_Macro(source)
{
   TCanvas *c1  = new TCanvas("c1","c1",0,0,600,400);
   TF2 *f1 = new TF2("f1","0.1+(1-(x-2)*(x-2))*(1-(y-2)*(y-2))",1,3,1,3);
   Int_t palette[5];
   palette[0] = 15;
   palette[1] = 20;
   palette[2] = 23;
   palette[3] = 30;
   palette[4] = 32;
   gStyle->SetPalette(5,palette);
   f1->Draw("colz");
   return c1;
}
End_Macro
Begin_Html

<p> To define more a complex palette with a continuous gradient of color, one
should use the static function <tt>TColor::CreateGradientColorTable()</tt>.
The following example demonstrates how to proceed:

End_Html
Begin_Macro(source)
{
   TCanvas *c2  = new TCanvas("c2","c2",0,0,600,400);
   TF2 *f2 = new TF2("f2","0.1+(1-(x-2)*(x-2))*(1-(y-2)*(y-2))",1,3,1,3);
   UInt_t Number = 3;
   Double_t Red[Number]    = { 1.00, 0.00, 0.00};
   Double_t Green[Number]  = { 0.00, 1.00, 0.00};
   Double_t Blue[Number]   = { 1.00, 0.00, 1.00};
   Double_t Length[Number] = { 0.00, 0.50, 1.00 };
   Int_t nb=50;
   TColor::CreateGradientColorTable(Number,Length,Red,Green,Blue,nb);
   f2->SetContour(nb);
   f2->Draw("surf1z");
   return c2;
}
End_Macro
Begin_Html

<p>The function <tt>TColor::CreateGradientColorTable()</tt> automatically 
calls </tt>gStyle->SetPalette()</tt>, so there is not need to add one.
<p>
After a call to <tt>TColor::CreateGradientColorTable()</tt> it is sometimes
useful to store the newly create palette for further use. In particular, it is
recommended to do if one wants to switch between several user define palettes.
To store a palette in an array it is enough to do:
<br>
<pre>
   Int_t MyPalette[100];
   Double_t r[]    = {0., 0.0, 1.0, 1.0, 1.0};
   Double_t g[]    = {0., 0.0, 0.0, 1.0, 1.0};
   Double_t b[]    = {0., 1.0, 0.0, 0.0, 1.0};
   Double_t stop[] = {0., .25, .50, .75, 1.0};
   Int_t FI = TColor::CreateGradientColorTable(5, stop, r, g, b, 100);
   for (int i=0;i<100;i++) MyPalette[i] = FI+i;
</pre>
<p>
Later on to reuse the palette <tt>MyPalette</tt> it will be enough to do
<p>
<pre>
   gStyle->SetPalette(100, MyPalette);
</pre>
<p>
As only one palette is active, one need to use <tt>TExec</tt> to be able to
display plots using different palettes on the same pad.
The following macro illustrate this feature.
End_Html
Begin_Macro(source)
../../../tutorials/graphs/multipalette.C
End_Macro 
Begin_Html

<a name="C06"></a><h3>Color transparency</h3>
To make a graphics object transparent it is enough to set its color to a 
transparent one. The color transparency is defined via its alpha component. The 
alpha value varies from <tt>0.</tt> (fully transparent) to <tt>1.</tt> (fully 
opaque). To set the alpha value of an existing color it is enough to do:
<pre>
   TColor *col26 = gROOT->GetColor(26);
   col26->SetAlpha(0.01);
</pre>
A new color can be created transparent the following way:
<pre>
   Int_t ci = 1756;
   TColor *color = new TColor(ci, 0.1, 0.2, 0.3, "", 0.5); // alpha = 0.5
</pre>
An example of tranparency usage with parallel coordinates can be found
in <tt>$ROOTSYS/tutorials/tree/parallelcoordtrans.C</tt>. Right now the 
transparency is implemented only for PDF output, SVG output, and for gif,
jpg and png outputs.

End_Html */


//______________________________________________________________________________
TColor::TColor(): TNamed()
{
   /* Begin_html
   Default constructor.
   End_html */

   fNumber = -1;
   fRed = fGreen = fBlue = fHue = fLight = fSaturation = -1;
   fAlpha = 1;
}


//______________________________________________________________________________
TColor::TColor(Int_t color, Float_t r, Float_t g, Float_t b, const char *name,
               Float_t a)
      : TNamed(name,"")
{
   /* Begin_html
   Normal color constructor. Initialize a color structure.
   Compute the RGB and HLS color components.
   End_html */

   TColor::InitializeColors();
   // do not enter if color number already exist
   TColor *col = gROOT->GetColor(color);
   if (col) {
      Warning("TColor", "color %d already defined", color);
      fNumber = col->GetNumber();
      fRed    = col->GetRed();
      fGreen  = col->GetGreen();
      fBlue   = col->GetBlue();
      fHue    = col->GetHue();
      fLight  = col->GetLight();
      fAlpha  = col->GetAlpha();
      fSaturation = col->GetSaturation();
      return;
   }

   fNumber = color;

   char aname[32];
   if (!name || !*name) {
      snprintf(aname,32, "Color%d", color);
      SetName(aname);
   }

   // enter in the list of colors
   TObjArray *lcolors = (TObjArray*)gROOT->GetListOfColors();
   lcolors->AddAtAndExpand(this, color);

   // fill color structure
   SetRGB(r, g, b);
   fAlpha = a;
}


//______________________________________________________________________________
TColor::~TColor()
{
   /* Begin_html
   Color destructor.
   End_html */

   gROOT->GetListOfColors()->Remove(this);
   if (gROOT->GetListOfColors()->GetEntries() == 0) {fgPalette.Set(0); fgPalette=0;}
}


//______________________________________________________________________________
TColor::TColor(const TColor &color) : TNamed(color)
{
   /* Begin_html
   Color copy constructor.
   End_html */

   ((TColor&)color).Copy(*this);
}


//______________________________________________________________________________
void TColor::InitializeColors()
{
   /* Begin_html
   Initialize colors used by the TCanvas based graphics (via TColor objects).
   This method should be called before the ApplicationImp is created (which
   initializes the GUI colors).
   End_html */

   if (fgInitDone)
      return;
   fgInitDone = kTRUE;
   if (gROOT->GetListOfColors()->First() == 0) {
      TColor *s0;
      Float_t r, g, b, h, l, s;
      Int_t   i;

      new TColor(kWhite,1,1,1,"background");
      new TColor(kBlack,0,0,0,"black");
      new TColor(2,1,0,0,"red");
      new TColor(3,0,1,0,"green");
      new TColor(4,0,0,1,"blue");
      new TColor(5,1,1,0,"yellow");
      new TColor(6,1,0,1,"magenta");
      new TColor(7,0,1,1,"cyan");
      new TColor(10,0.999,0.999,0.999,"white");
      new TColor(11,0.754,0.715,0.676,"editcol");

      // The color white above is defined as being nearly white.
      // Sets the associated dark color also to white.
      //TColor *c110 = gROOT->GetColor(110);
      TColor::GetColorDark(10);
      TColor *c110 = gROOT->GetColor(110);
      c110->SetRGB(0.999,0.999,.999);

      // Initialize Custom colors
      new TColor(20,0.8,0.78,0.67);
      new TColor(31,0.54,0.66,0.63);
      new TColor(41,0.83,0.81,0.53);
      new TColor(30,0.52,0.76,0.64);
      new TColor(32,0.51,0.62,0.55);
      new TColor(24,0.70,0.65,0.59);
      new TColor(21,0.8,0.78,0.67);
      new TColor(47,0.67,0.56,0.58);
      new TColor(35,0.46,0.54,0.57);
      new TColor(33,0.68,0.74,0.78);
      new TColor(39,0.5,0.5,0.61);
      new TColor(37,0.43,0.48,0.52);
      new TColor(38,0.49,0.6,0.82);
      new TColor(36,0.41,0.51,0.59);
      new TColor(49,0.58,0.41,0.44);
      new TColor(43,0.74,0.62,0.51);
      new TColor(22,0.76,0.75,0.66);
      new TColor(45,0.75,0.51,0.47);
      new TColor(44,0.78,0.6,0.49);
      new TColor(26,0.68,0.6,0.55);
      new TColor(28,0.53,0.4,0.34);
      new TColor(25,0.72,0.64,0.61);
      new TColor(27,0.61,0.56,0.51);
      new TColor(23,0.73,0.71,0.64);
      new TColor(42,0.87,0.73,0.53);
      new TColor(46,0.81,0.37,0.38);
      new TColor(48,0.65,0.47,0.48);
      new TColor(34,0.48,0.56,0.6);
      new TColor(40,0.67,0.65,0.75);
      new TColor(29,0.69,0.81,0.78);

      // Initialize some additional greyish non saturated colors
      new TColor(8, 0.35,0.83,0.33);
      new TColor(9, 0.35,0.33,0.85);
      new TColor(12,.3,.3,.3,"grey12");
      new TColor(13,.4,.4,.4,"grey13");
      new TColor(14,.5,.5,.5,"grey14");
      new TColor(15,.6,.6,.6,"grey15");
      new TColor(16,.7,.7,.7,"grey16");
      new TColor(17,.8,.8,.8,"grey17");
      new TColor(18,.9,.9,.9,"grey18");
      new TColor(19,.95,.95,.95,"grey19");
      new TColor(50, 0.83,0.35,0.33);

      // Initialize the Pretty Palette Spectrum Violet->Red
      //   The color model used here is based on the HLS model which
      //   is much more suitable for creating palettes than RGB.
      //   Fixing the saturation and lightness we can scan through the
      //   spectrum of visible light by using "hue" alone.
      //   In Root hue takes values from 0 to 360.
      Float_t  saturation = 1;
      Float_t  lightness = 0.5;
      Float_t  maxHue = 280;
      Float_t  minHue = 0;
      Int_t    maxPretty = 50;
      Float_t  hue;

      for (i=0 ; i<maxPretty ; i++) {
         hue = maxHue-(i+1)*((maxHue-minHue)/maxPretty);
         TColor::HLStoRGB(hue, lightness, saturation, r, g, b);
         new TColor(i+51, r, g, b);
      }

      // Initialize special colors for x3d
      for (i = 1; i < 8; i++) {
         s0 = gROOT->GetColor(i);
         s0->GetRGB(r,g,b);
         if (i == 1) { r = 0.6; g = 0.6; b = 0.6; }
         if (r == 1) r = 0.9; if (r == 0) r = 0.1;
         if (g == 1) g = 0.9; if (g == 0) g = 0.1;
         if (b == 1) b = 0.9; if (b == 0) b = 0.1;
         TColor::RGBtoHLS(r,g,b,h,l,s);
         TColor::HLStoRGB(h,0.6*l,s,r,g,b);
         new TColor(200+4*i-3,r,g,b);
         TColor::HLStoRGB(h,0.8*l,s,r,g,b);
         new TColor(200+4*i-2,r,g,b);
         TColor::HLStoRGB(h,1.2*l,s,r,g,b);
         new TColor(200+4*i-1,r,g,b);
         TColor::HLStoRGB(h,1.4*l,s,r,g,b);
         new TColor(200+4*i  ,r,g,b);
      }

      // Create the ROOT Color Wheel
      TColor::CreateColorWheel();
   }
   // If fgPalette.fN !=0 SetPalette has been called already
   // (from rootlogon.C for instance)

   if (!fgPalette.fN) SetPalette(1,0);
}


//______________________________________________________________________________
const char *TColor::AsHexString() const
{
   /* Begin_html
   Return color as hexadecimal string. This string can be directly passed
   to, for example, TGClient::GetColorByName(). String will be reused so
   copy immediately if needed.
   End_html */

   static TString tempbuf;

   Int_t r, g, b, a;
   r = Int_t(GetRed()   * 255);
   g = Int_t(GetGreen() * 255);
   b = Int_t(GetBlue()  * 255);
   a = Int_t(fAlpha     * 255);

   if (a != 255) {
      tempbuf.Form("#%02x%02x%02x%02x", a, r, g, b);
   } else {
      tempbuf.Form("#%02x%02x%02x", r, g, b);
   }
   return tempbuf;
}


//______________________________________________________________________________
void TColor::Copy(TObject &obj) const
{
   /* Begin_html
   Copy this color to obj.
   End_html */

   TNamed::Copy((TNamed&)obj);
   ((TColor&)obj).fRed   = fRed;
   ((TColor&)obj).fGreen = fGreen;
   ((TColor&)obj).fBlue  = fBlue;
   ((TColor&)obj).fHue   = fHue;
   ((TColor&)obj).fLight = fLight;
   ((TColor&)obj).fAlpha = fAlpha;
   ((TColor&)obj).fSaturation = fSaturation;
   ((TColor&)obj).fNumber = fNumber;
}


//______________________________________________________________________________
void TColor::CreateColorsGray()
{
   /* Begin_html
   Create the Gray scale colors in the Color Wheel
   End_html */

   if (gROOT->GetColor(kGray)) return;
   TColor *gray  = new TColor(kGray,204./255.,204./255.,204./255.);
   TColor *gray1 = new TColor(kGray+1,153./255.,153./255.,153./255.);
   TColor *gray2 = new TColor(kGray+2,102./255.,102./255.,102./255.);
   TColor *gray3 = new TColor(kGray+3, 51./255., 51./255., 51./255.);
   gray ->SetName("kGray");
   gray1->SetName("kGray+1");
   gray2->SetName("kGray+2");
   gray3->SetName("kGray+3");
}


//______________________________________________________________________________
void TColor::CreateColorsCircle(Int_t offset, const char *name, UChar_t *rgb)
{
   /* Begin_html
   Create the "circle" colors in the color wheel.
   End_html */

   TString colorname;
   for (Int_t n=0;n<15;n++) {
      Int_t colorn = offset+n-10;
      TColor *color = gROOT->GetColor(colorn);
      if (!color) {
         color = new TColor(colorn,rgb[3*n]/255.,rgb[3*n+1]/255.,rgb[3*n+2]/255.);
         color->SetTitle(color->AsHexString());
         if      (n>10) colorname.Form("%s+%d",name,n-10);
         else if (n<10) colorname.Form("%s-%d",name,10-n);
         else           colorname.Form("%s",name);
         color->SetName(colorname);
      }
   }
}


//______________________________________________________________________________
void TColor::CreateColorsRectangle(Int_t offset, const char *name, UChar_t *rgb)
{
   /* Begin_html
   Create the "rectangular" colors in the color wheel.
   End_html */

   TString colorname;
   for (Int_t n=0;n<20;n++) {
      Int_t colorn = offset+n-9;
      TColor *color = gROOT->GetColor(colorn);
      if (!color) {
         color = new TColor(colorn,rgb[3*n]/255.,rgb[3*n+1]/255.,rgb[3*n+2]/255.);
         color->SetTitle(color->AsHexString());
         if      (n>9) colorname.Form("%s+%d",name,n-9);
         else if (n<9) colorname.Form("%s-%d",name,9-n);
         else          colorname.Form("%s",name);
         color->SetName(colorname);
      }
   }
}


//______________________________________________________________________________
void TColor::CreateColorWheel()
{
   /* Begin_html
   Static function steering the creation of all colors in the color wheel.
   End_html */

   UChar_t magenta[46]= {255,204,255
                        ,255,153,255, 204,153,204
                        ,255,102,255, 204,102,204, 153,102,153
                        ,255, 51,255, 204, 51,204, 153, 51,153, 102, 51,102
                        ,255,  0,255, 204,  0,204, 153,  0,153, 102,  0,102,  51,  0, 51};

   UChar_t red[46]    = {255,204,204
                        ,255,153,153, 204,153,153
                        ,255,102,102, 204,102,102, 153,102,102
                        ,255, 51, 51, 204, 51, 51, 153, 51, 51, 102, 51, 51
                        ,255,  0,  0, 204,  0,  0, 153,  0,  0, 102,  0,  0,  51,  0,  0};

   UChar_t yellow[46] = {255,255,204
                        ,255,255,153, 204,204,153
                        ,255,255,102, 204,204,102, 153,153,102
                        ,255,255, 51, 204,204, 51, 153,153, 51, 102,102, 51
                        ,255,255,  0, 204,204,  0, 153,153,  0, 102,102,  0,  51, 51,  0};

   UChar_t green[46]  = {204,255,204
                        ,153,255,153, 153,204,153
                        ,102,255,102, 102,204,102, 102,153,102
                        , 51,255, 51,  51,204, 51,  51,153, 51,  51,102, 51
                        ,  0,255,  0,   0,204,  0,   0,153,  0,   0,102,  0,  0, 51,  0};

   UChar_t cyan[46]   = {204,255,255
                        ,153,255,255, 153,204,204
                        ,102,255,255, 102,204,204, 102,153,153
                        , 51,255,255,  51,204,204,  51,153,153,  51,102,102
                        ,  0,255,255,   0,204,204,   0,153,153,   0,102,102,   0, 51,  51};

   UChar_t blue[46]   = {204,204,255
                        ,153,153,255, 153,153,204
                        ,102,102,255, 102,102,204, 102,102,153
                        , 51, 51,255,  51, 51,204,  51, 51,153,  51, 51,102
                        ,  0,  0,255,   0,  0,204,   0,  0,153,   0,  0,102,   0,  0,  51};

   UChar_t pink[60] = {255, 51,153,  204,  0,102,  102,  0, 51,  153,  0, 51,  204, 51,102
                      ,255,102,153,  255,  0,102,  255, 51,102,  204,  0, 51,  255,  0, 51
                      ,255,153,204,  204,102,153,  153, 51,102,  153,  0,102,  204, 51,153
                      ,255,102,204,  255,  0,153,  204,  0,153,  255, 51,204,  255,  0,153};

   UChar_t orange[60]={255,204,153,  204,153,102,  153,102, 51,  153,102,  0,  204,153, 51
                      ,255,204,102,  255,153,  0,  255,204, 51,  204,153,  0,  255,204,  0
                      ,255,153, 51,  204,102,  0,  102, 51,  0,  153, 51,  0,  204,102, 51
                      ,255,153,102,  255,102,  0,  255,102, 51,  204, 51,  0,  255, 51,  0};

   UChar_t spring[60]={153,255, 51,  102,204,  0,   51,102,  0,   51,153,  0,  102,204, 51
                      ,153,255,102,  102,255,  0,  102,255, 51,   51,204,  0,   51,255, 0
                      ,204,255,153,  153,204,102,  102,153, 51,  102,153,  0,  153,204, 51
                      ,204,255,102,  153,255,  0,  204,255, 51,  153,204,  0,  204,255,  0};

   UChar_t teal[60] = {153,255,204,  102,204,153,   51,153,102,    0,153,102,   51,204,153
                      ,102,255,204,    0,255,102,   51,255,204,    0,204,153,    0,255,204
                      , 51,255,153,    0,204,102,    0,102, 51,    0,153, 51,   51,204,102
                      ,102,255,153,    0,255,153,   51,255,102,    0,204, 51,    0,255, 51};

   UChar_t azure[60] ={153,204,255,  102,153,204,   51,102,153,    0, 51,153,   51,102,204
                      ,102,153,255,    0,102,255,   51,102,255,    0, 51,204,    0, 51,255
                      , 51,153,255,    0,102,204,    0, 51,102,    0,102,153,   51,153,204
                      ,102,204,255,    0,153,255,   51,204,255,    0,153,204,    0,204,255};

   UChar_t violet[60]={204,153,255,  153,102,204,  102, 51,153,  102,  0,153,  153, 51,204
                      ,204,102,255,  153,  0,255,  204, 51,255,  153,  0,204,  204,  0,255
                      ,153, 51,255,  102,  0,204,   51,  0,102,   51,  0,153,  102, 51,204
                      ,153,102,255,  102,  0,255,  102, 51,255,   51,  0,204,   51,  0,255};

   TColor::CreateColorsCircle(kMagenta,"kMagenta",magenta);
   TColor::CreateColorsCircle(kRed,    "kRed",    red);
   TColor::CreateColorsCircle(kYellow, "kYellow", yellow);
   TColor::CreateColorsCircle(kGreen,  "kGreen",  green);
   TColor::CreateColorsCircle(kCyan,   "kCyan",   cyan);
   TColor::CreateColorsCircle(kBlue,   "kBlue",   blue);

   TColor::CreateColorsRectangle(kPink,  "kPink",  pink);
   TColor::CreateColorsRectangle(kOrange,"kOrange",orange);
   TColor::CreateColorsRectangle(kSpring,"kSpring",spring);
   TColor::CreateColorsRectangle(kTeal,  "kTeal",  teal);
   TColor::CreateColorsRectangle(kAzure, "kAzure", azure);
   TColor::CreateColorsRectangle(kViolet,"kViolet",violet);

   TColor::CreateColorsGray();
}


//______________________________________________________________________________
Int_t TColor::GetColorPalette(Int_t i)
{
   /* Begin_html
   Static function returning the color number i in current palette.
   End_html */

   Int_t ncolors = fgPalette.fN;
   if (ncolors == 0) return 0;
   Int_t icol    = i%ncolors;
   if (icol < 0) icol = 0;
   return fgPalette.fArray[icol];
}


//______________________________________________________________________________
Int_t TColor::GetNumberOfColors()
{
   /* Begin_html
   Static function returning number of colors in the color palette.
   End_html */

   return fgPalette.fN;
}


//______________________________________________________________________________
ULong_t TColor::GetPixel() const
{
   /* Begin_html
   Return pixel value corresponding to this color. This pixel value can
   be used in the GUI classes. This call does not work in batch mode since
   it needs to communicate with the graphics system.
   End_html */

   if (gVirtualX && !gROOT->IsBatch()) {
      if (gApplication) {
         TApplication::NeedGraphicsLibs();
         gApplication->InitializeGraphics();
      }
      return gVirtualX->GetPixel(fNumber);
   }

   return 0;
}


//______________________________________________________________________________
void TColor::HLS2RGB(Float_t hue, Float_t light, Float_t satur,
                     Float_t &r, Float_t &g, Float_t &b)
{
   /* Begin_html
   Static method to compute RGB from HLS. The l and s are between [0,1]
   and h is between [0,360]. The returned r,g,b triplet is between [0,1].
   End_html */

   Float_t rh, rl, rs, rm1, rm2;
   rh = rl = rs = 0;
   if (hue   > 0) rh = hue;   if (rh > 360) rh = 360;
   if (light > 0) rl = light; if (rl > 1)   rl = 1;
   if (satur > 0) rs = satur; if (rs > 1)   rs = 1;

   if (rl <= 0.5)
      rm2 = rl*(1.0 + rs);
   else
      rm2 = rl + rs - rl*rs;
   rm1 = 2.0*rl - rm2;

   if (!rs) { r = rl; g = rl; b = rl; return; }
   r = HLStoRGB1(rm1, rm2, rh+120);
   g = HLStoRGB1(rm1, rm2, rh);
   b = HLStoRGB1(rm1, rm2, rh-120);
}


//______________________________________________________________________________
Float_t TColor::HLStoRGB1(Float_t rn1, Float_t rn2, Float_t huei)
{
   /* Begin_html
   Static method. Auxiliary to HLS2RGB().
   End_html */

   Float_t hue = huei;
   if (hue > 360) hue = hue - 360;
   if (hue < 0)   hue = hue + 360;
   if (hue < 60 ) return rn1 + (rn2-rn1)*hue/60;
   if (hue < 180) return rn2;
   if (hue < 240) return rn1 + (rn2-rn1)*(240-hue)/60;
   return rn1;
}


//______________________________________________________________________________
void TColor::HLS2RGB(Int_t h, Int_t l, Int_t s, Int_t &r, Int_t &g, Int_t &b)
{
   /* Begin_html
   Static method to compute RGB from HLS. The h,l,s are between [0,255].
   The returned r,g,b triplet is between [0,255].
   End_html */

   Float_t hh, ll, ss, rr, gg, bb;

   hh = Float_t(h) * 360 / 255;
   ll = Float_t(l) / 255;
   ss = Float_t(s) / 255;

   TColor::HLStoRGB(hh, ll, ss, rr, gg, bb);

   r = (Int_t) (rr * 255);
   g = (Int_t) (gg * 255);
   b = (Int_t) (bb * 255);
}


//______________________________________________________________________________
void TColor::HSV2RGB(Float_t hue, Float_t satur, Float_t value,
                     Float_t &r, Float_t &g, Float_t &b)
{
   /* Begin_html
   Static method to compute RGB from HSV:
   <ul>
   <li> The hue value runs from 0 to 360.
   <li> The saturation is the degree of strength or purity and is from 0 to 1.
        Purity is how much white is added to the color, so S=1 makes the purest
        color (no white).
   <li> Brightness value also ranges from 0 to 1, where 0 is the black.
   </ul>
   The returned r,g,b triplet is between [0,1].
   End_html */

   Int_t i;
   Float_t f, p, q, t;

   if (satur==0) {
      // Achromatic (grey)
      r = g = b = value;
      return;
   }

   hue /= 60;   // sector 0 to 5
   i = (Int_t)floor(hue);
   f = hue-i;   // factorial part of hue
   p = value*(1-satur);
   q = value*(1-satur*f );
   t = value*(1-satur*(1-f));

   switch (i) {
      case 0:
         r = value;
         g = t;
         b = p;
         break;
      case 1:
         r = q;
         g = value;
         b = p;
         break;
      case 2:
         r = p;
         g = value;
         b = t;
         break;
      case 3:
         r = p;
         g = q;
         b = value;
         break;
      case 4:
         r = t;
         g = p;
         b = value;
         break;
      default:
         r = value;
         g = p;
         b = q;
         break;
   }
}


//______________________________________________________________________________
void TColor::ls(Option_t *) const
{
   /* Begin_html
   List this color with its attributes.
   End_html */

   printf("Color:%d  Red=%f Green=%f Blue=%f Name=%s\n",
          fNumber, fRed, fGreen, fBlue, GetName());
}


//______________________________________________________________________________
void TColor::Print(Option_t *) const
{
   /* Begin_html
   Dump this color with its attributes.
   End_html */

   ls();
}


//______________________________________________________________________________
void TColor::RGB2HLS(Float_t rr, Float_t gg, Float_t bb,
                     Float_t &hue, Float_t &light, Float_t &satur)
{
   /* Begin_html
   Static method to compute HLS from RGB. The r,g,b triplet is between
   [0,1], hue is between [0,360], light and satur are [0,1].
   End_html */

   Float_t rnorm, gnorm, bnorm, minval, maxval, msum, mdiff, r, g, b;
   minval = maxval =0 ;
   r = g = b = 0;
   if (rr > 0) r = rr; if (r > 1) r = 1;
   if (gg > 0) g = gg; if (g > 1) g = 1;
   if (bb > 0) b = bb; if (b > 1) b = 1;

   minval = r;
   if (g < minval) minval = g;
   if (b < minval) minval = b;
   maxval = r;
   if (g > maxval) maxval = g;
   if (b > maxval) maxval = b;

   rnorm = gnorm = bnorm = 0;
   mdiff = maxval - minval;
   msum  = maxval + minval;
   light = 0.5 * msum;
   if (maxval != minval) {
      rnorm = (maxval - r)/mdiff;
      gnorm = (maxval - g)/mdiff;
      bnorm = (maxval - b)/mdiff;
   } else {
      satur = hue = 0;
      return;
   }

   if (light < 0.5)
      satur = mdiff/msum;
   else
      satur = mdiff/(2.0 - msum);

   if (r == maxval)
      hue = 60.0 * (6.0 + bnorm - gnorm);
   else if (g == maxval)
      hue = 60.0 * (2.0 + rnorm - bnorm);
   else
      hue = 60.0 * (4.0 + gnorm - rnorm);

   if (hue > 360)
      hue = hue - 360;
}


//______________________________________________________________________________
void TColor::RGB2HSV(Float_t r, Float_t g, Float_t b,
                     Float_t &hue, Float_t &satur, Float_t &value)
{
   /* Begin_html
   Static method to compute HSV from RGB.
   <ul>
   <li> The input values:
   <ul>
   <li> r,g,b triplet is between [0,1].
   </ul>
   <li> The returned values:
   <ul>
   <li> The hue value runs from 0 to 360.
   <li> The saturation is the degree of strength or purity and is from 0 to 1.
        Purity is how much white is added to the color, so S=1 makes the purest
        color (no white).
   <li> Brightness value also ranges from 0 to 1, where 0 is the black.
   </ul>
   </ul>
   End_html */

   Float_t min, max, delta;

   min   = TMath::Min(TMath::Min(r, g), b);
   max   = TMath::Max(TMath::Max(r, g), b);
   value = max;

   delta = max - min;

   if (max != 0) {
      satur = delta/max;
   } else {
      satur = 0;
      hue   = -1;
      return;
   }

   if (r == max) {
      hue = (g-b)/delta;
   } else if (g == max) {
      hue = 2+(b-r)/delta;
   } else {
      hue = 4+(r-g)/delta;
   }

   hue *= 60;
   if (hue < 0) hue += 360;
}


//______________________________________________________________________________
void TColor::RGB2HLS(Int_t r, Int_t g, Int_t b, Int_t &h, Int_t &l, Int_t &s)
{
   /* Begin_html
   Static method to compute HLS from RGB. The r,g,b triplet is between
   [0,255], hue, light and satur are between [0,255].
   End_html */

   Float_t rr, gg, bb, hue, light, satur;

   rr = Float_t(r) / 255;
   gg = Float_t(g) / 255;
   bb = Float_t(b) / 255;

   TColor::RGBtoHLS(rr, gg, bb, hue, light, satur);

   h = (Int_t) (hue/360 * 255);
   l = (Int_t) (light * 255);
   s = (Int_t) (satur * 255);
}


//______________________________________________________________________________
void TColor::SetRGB(Float_t r, Float_t g, Float_t b)
{
   /* Begin_html
   Initialize this color and its associated colors.
   End_html */

   TColor::InitializeColors();
   fRed   = r;
   fGreen = g;
   fBlue  = b;

   if (fRed < 0) return;

   RGBtoHLS(r, g, b, fHue, fLight, fSaturation);

   Int_t nplanes = 16;
   if (gVirtualX) gVirtualX->GetPlanes(nplanes);
   if (nplanes == 0) nplanes = 16;

   // allocate color now (can be delayed when we have a large colormap)
#ifndef R__WIN32
   if (nplanes < 15)
#endif
      Allocate();

   if (fNumber > 50) return;

   // now define associated colors for WBOX shading
   Float_t dr, dg, db, lr, lg, lb;

   // set dark color
   HLStoRGB(fHue, 0.7*fLight, fSaturation, dr, dg, db);
   TColor *dark = gROOT->GetColor(100+fNumber);
   if (dark) {
      if (nplanes > 8) dark->SetRGB(dr, dg, db);
      else             dark->SetRGB(0.3,0.3,0.3);
   }

   // set light color
   HLStoRGB(fHue, 1.2*fLight, fSaturation, lr, lg, lb);
   TColor *light = gROOT->GetColor(150+fNumber);
   if (light) {
      if (nplanes > 8) light->SetRGB(lr, lg, lb);
      else             light->SetRGB(0.8,0.8,0.8);
   }
}


//______________________________________________________________________________
void TColor::Allocate()
{
   /* Begin_html
   Make this color known to the graphics system.
   End_html */

   if (gVirtualX && !gROOT->IsBatch())

      gVirtualX->SetRGB(fNumber, GetRed(), GetGreen(), GetBlue());
}


//______________________________________________________________________________
Int_t TColor::GetColor(const char *hexcolor)
{
   /* Begin_html
   Static method returning color number for color specified by
   hex color string of form: #rrggbb, where rr, gg and bb are in
   hex between [0,FF], e.g. "#c0c0c0".
   <br>
   If specified color does not exist it will be created with as
   name "#rrggbb" with rr, gg and bb in hex between [0,FF].
   End_html */

   if (hexcolor && *hexcolor == '#') {
      Int_t r, g, b;
      if (sscanf(hexcolor+1, "%02x%02x%02x", &r, &g, &b) == 3)
         return GetColor(r, g, b);
   }
   ::Error("TColor::GetColor(const char*)", "incorrect color string");
   return 0;
}


//______________________________________________________________________________
Int_t TColor::GetColor(Float_t r, Float_t g, Float_t b)
{
   /* Begin_html
   Static method returning color number for color specified by
   r, g and b. The r,g,b should be in the range [0,1].
   <br>
   If specified color does not exist it will be created
   with as name "#rrggbb" with rr, gg and bb in hex between
   [0,FF].
   End_html */

   Int_t rr, gg, bb;
   rr = Int_t(r * 255);
   gg = Int_t(g * 255);
   bb = Int_t(b * 255);

   return GetColor(rr, gg, bb);
}


//______________________________________________________________________________
Int_t TColor::GetColor(ULong_t pixel)
{
   /* Begin_html
   Static method returning color number for color specified by
   system dependent pixel value. Pixel values can be obtained, e.g.,
   from the GUI color picker.
   End_html */

   Int_t r, g, b;

   Pixel2RGB(pixel, r, g, b);

   return GetColor(r, g, b);
}


//______________________________________________________________________________
Int_t TColor::GetColor(Int_t r, Int_t g, Int_t b)
{
   /* Begin_html
   Static method returning color number for color specified by
   r, g and b. The r,g,b should be in the range [0,255].
   If the specified color does not exist it will be created
   with as name "#rrggbb" with rr, gg and bb in hex between
   [0,FF].
   End_html */

   TColor::InitializeColors();
   if (r < 0) r = 0;
   if (g < 0) g = 0;
   if (b < 0) b = 0;
   if (r > 255) r = 255;
   if (g > 255) g = 255;
   if (b > 255) b = 255;

   // Get list of all defined colors
   TObjArray *colors = (TObjArray*) gROOT->GetListOfColors();

   TColor *color = 0;

   // Look for color by name
   if ((color = (TColor*)colors->FindObject(Form("#%02x%02x%02x", r, g, b))))
      // We found the color by name, so we use that right away
      return color->GetNumber();

   Float_t rr, gg, bb;
   rr = Float_t(r)/255.;
   gg = Float_t(g)/255.;
   bb = Float_t(b)/255.;

   TIter next(colors);

   Int_t nplanes = 16;
   Float_t thres = 1.0/31.0;   // 5 bits per color : 0 - 0x1F !
   if (gVirtualX) gVirtualX->GetPlanes(nplanes);
   if (nplanes >= 24)
      thres = 1.0/255.0;       // 8 bits per color : 0 - 0xFF !

   // Loop over all defined colors
   while ((color = (TColor*)next())) {
      if (TMath::Abs(color->GetRed() - rr) > thres)
         continue;
      if (TMath::Abs(color->GetGreen() - gg) > thres)
         continue;
      if (TMath::Abs(color->GetBlue() - bb) > thres)
         continue;

      // We found a matching color in the color table
      return color->GetNumber();
   }

   // We didn't find a matching color in the color table, so we
   // add it. Note name is of the form "#rrggbb" where rr, etc. are
   // hexadecimal numbers.
   color = new TColor(colors->GetLast()+1, rr, gg, bb,
                      Form("#%02x%02x%02x", r, g, b));

   return color->GetNumber();
}


//______________________________________________________________________________
Int_t TColor::GetColorBright(Int_t n)
{
   /* Begin_html
   Static function: Returns the bright color number corresponding to n
   If the TColor object does not exist, it is created.
   The convention is that the bright color nb = n+150
   End_html */

   if (n < 0) return -1;

   // Get list of all defined colors
   TObjArray *colors = (TObjArray*) gROOT->GetListOfColors();
   Int_t ncolors = colors->GetSize();
   // Get existing color at index n
   TColor *color = 0;
   if (n < ncolors) color = (TColor*)colors->At(n);
   if (!color) return -1;

   //Get the rgb of the the new bright color corresponding to color n
   Float_t r,g,b;
   HLStoRGB(color->GetHue(), 1.2*color->GetLight(), color->GetSaturation(), r, g, b);

   //Build the bright color (unless the slot nb is already used)
   Int_t nb = n+150;
   TColor *colorb = 0;
   if (nb < ncolors) colorb = (TColor*)colors->At(nb);
   if (colorb) return nb;
   colorb = new TColor(nb,r,g,b);
   colorb->SetName(Form("%s_bright",color->GetName()));
   colors->AddAtAndExpand(colorb,nb);
   return nb;
}


//______________________________________________________________________________
Int_t TColor::GetColorDark(Int_t n)
{
   /* Begin_html
   Static function: Returns the dark color number corresponding to n
   If the TColor object does not exist, it is created.
   The convention is that the dark color nd = n+100
   End_html */

   if (n < 0) return -1;

   // Get list of all defined colors
   TObjArray *colors = (TObjArray*) gROOT->GetListOfColors();
   Int_t ncolors = colors->GetSize();
   // Get existing color at index n
   TColor *color = 0;
   if (n < ncolors) color = (TColor*)colors->At(n);
   if (!color) return -1;

   //Get the rgb of the the new dark color corresponding to color n
   Float_t r,g,b;
   HLStoRGB(color->GetHue(), 0.7*color->GetLight(), color->GetSaturation(), r, g, b);

   //Build the dark color (unless the slot nd is already used)
   Int_t nd = n+100;
   TColor *colord = 0;
   if (nd < ncolors) colord = (TColor*)colors->At(nd);
   if (colord) return nd;
   colord = new TColor(nd,r,g,b);
   colord->SetName(Form("%s_dark",color->GetName()));
   colors->AddAtAndExpand(colord,nd);
   return nd;
}


//______________________________________________________________________________
ULong_t TColor::Number2Pixel(Int_t ci)
{
   /* Begin_html
   Static method that given a color index number, returns the corresponding
   pixel value. This pixel value can be used in the GUI classes. This call
   does not work in batch mode since it needs to communicate with the
   graphics system.
   End_html */


   TColor::InitializeColors();
   TColor *color = gROOT->GetColor(ci);
   if (color)
      return color->GetPixel();
   else
      ::Warning("TColor::Number2Pixel", "color with index %d not defined", ci);

   return 0;
}


//______________________________________________________________________________
ULong_t TColor::RGB2Pixel(Float_t r, Float_t g, Float_t b)
{
   /* Begin_html
   Convert r,g,b to graphics system dependent pixel value.
   The r,g,b triplet must be [0,1].
   End_html */

   if (r < 0) r = 0;
   if (g < 0) g = 0;
   if (b < 0) b = 0;
   if (r > 1) r = 1;
   if (g > 1) g = 1;
   if (b > 1) b = 1;

   ColorStruct_t color;
   color.fRed   = UShort_t(r * 65535);
   color.fGreen = UShort_t(g * 65535);
   color.fBlue  = UShort_t(b * 65535);
   color.fMask  = kDoRed | kDoGreen | kDoBlue;
   gVirtualX->AllocColor(gVirtualX->GetColormap(), color);
   return color.fPixel;
}


//______________________________________________________________________________
ULong_t TColor::RGB2Pixel(Int_t r, Int_t g, Int_t b)
{
   /* Begin_html
   Convert r,g,b to graphics system dependent pixel value.
   The r,g,b triplet must be [0,255].
   End_html */

   if (r < 0) r = 0;
   if (g < 0) g = 0;
   if (b < 0) b = 0;
   if (r > 255) r = 255;
   if (g > 255) g = 255;
   if (b > 255) b = 255;

   ColorStruct_t color;
   color.fRed   = UShort_t(r * 257);  // 65535/255
   color.fGreen = UShort_t(g * 257);
   color.fBlue  = UShort_t(b * 257);
   color.fMask  = kDoRed | kDoGreen | kDoBlue;
   gVirtualX->AllocColor(gVirtualX->GetColormap(), color);
   return color.fPixel;
}


//______________________________________________________________________________
void TColor::Pixel2RGB(ULong_t pixel, Float_t &r, Float_t &g, Float_t &b)
{
   /* Begin_html
   Convert machine dependent pixel value (obtained via RGB2Pixel or
   via Number2Pixel() or via TColor::GetPixel()) to r,g,b triplet.
   The r,g,b triplet will be [0,1].
   End_html */

   ColorStruct_t color;
   color.fPixel = pixel;
   gVirtualX->QueryColor(gVirtualX->GetColormap(), color);
   r = (Float_t)color.fRed / 65535;
   g = (Float_t)color.fGreen / 65535;
   b = (Float_t)color.fBlue / 65535;
}


//______________________________________________________________________________
void TColor::Pixel2RGB(ULong_t pixel, Int_t &r, Int_t &g, Int_t &b)
{
   /* Begin_html
   Convert machine dependent pixel value (obtained via RGB2Pixel or
   via Number2Pixel() or via TColor::GetPixel()) to r,g,b triplet.
   The r,g,b triplet will be [0,255].
   End_html */

   ColorStruct_t color;
   color.fPixel = pixel;
   gVirtualX->QueryColor(gVirtualX->GetColormap(), color);
   r = color.fRed / 257;
   g = color.fGreen / 257;
   b = color.fBlue / 257;
}


//______________________________________________________________________________
const char *TColor::PixelAsHexString(ULong_t pixel)
{
   /* Begin_html
   Convert machine dependent pixel value (obtained via RGB2Pixel or
   via Number2Pixel() or via TColor::GetPixel()) to a hexadecimal string.
   This string can be directly passed to, for example,
   TGClient::GetColorByName(). String will be reused so copy immediately
   if needed.
   End_html */

   static TString tempbuf;
   Int_t r, g, b;
   Pixel2RGB(pixel, r, g, b);
   tempbuf.Form("#%02x%02x%02x", r, g, b);
   return tempbuf;
}


//______________________________________________________________________________
void TColor::SaveColor(std::ostream &out, Int_t ci)
{
   /* Begin_html
   Save a color with index > 228 as a C++ statement(s) on output stream out.
   End_html */

   char quote = '"';
   Float_t r,g,b;
   Int_t ri, gi, bi;
   TString cname;

   TColor *c = gROOT->GetColor(ci);
   if (c) c->GetRGB(r, g, b);
   else return;

   ri = (Int_t)(255*r);
   gi = (Int_t)(255*g);
   bi = (Int_t)(255*b);
   cname.Form("#%02x%02x%02x", ri, gi, bi);

   if (gROOT->ClassSaved(TColor::Class())) {
      out << std::endl;
   } else {
      out << std::endl;
      out << "   Int_t ci;   // for color index setting" << std::endl;
   }

   out<<"   ci = TColor::GetColor("<<quote<<cname.Data()<<quote<<");"<<std::endl;
}


//______________________________________________________________________________
Bool_t TColor::IsGrayscale()
{
   /* Begin_html
   Return whether all colors return grayscale values.
   End_html */

   return fgGrayscaleMode;
}


//______________________________________________________________________________
void TColor::SetGrayscale(Bool_t set /*= kTRUE*/)
{
   /* Begin_html
   Set whether all colors should return grayscale values.
   End_html */

   if (fgGrayscaleMode == set) return;

   fgGrayscaleMode = set;

   if (!gVirtualX || gROOT->IsBatch()) return;

   TColor::InitializeColors();
   TIter iColor(gROOT->GetListOfColors());
   TColor* color = 0;
   while ((color = (TColor*) iColor()))
      color->Allocate();
}


//______________________________________________________________________________
Int_t TColor::CreateGradientColorTable(UInt_t Number, Double_t* Stops,
                              Double_t* Red, Double_t* Green,
                              Double_t* Blue, UInt_t NColors, Float_t alpha)
{
   /* Begin_html
   Static function creating a color table with several connected linear gradients.
   <ul>
   <li>Number: The number of end point colors that will form the gradients.
               Must be at least 2.
   <li>Stops: Where in the whole table the end point colors should lie.
              Each entry must be on [0, 1], each entry must be greater than
              the previous entry.
   <li>Red, Green, Blue: The end point color values.
                         Each entry must be on [0, 1]
   <li>NColors: Total number of colors in the table. Must be at least 1.
   </ul>

   Returns a positive value on success and -1 on error.
   <p>
   The table is constructed by tracing lines between the given points in
   RGB space.  Each color value may have a value between 0 and 1.  The
   difference between consecutive "Stops" values gives the fraction of
   space in the whole table that should be used for the interval between
   the corresponding color values.
   <p>
   Normally the first element of Stops should be 0 and the last should be 1.
   If this is not true, fewer than NColors will be used in proportion with
   the total interval between the first and last elements of Stops.
   <p>
   This definition is similar to the povray-definition of gradient
   color tables.
   <p>
   For instance:
   <pre>
   UInt_t Number = 3;
   Double_t Red[3]   = { 0.0, 1.0, 1.0 };
   Double_t Green[3] = { 0.0, 0.0, 1.0 };
   Double_t Blue[3]  = { 1.0, 0.0, 1.0 };
   Double_t Stops[3] = { 0.0, 0.4, 1.0 };
   </pre>
   This defines a table in which there are three color end points:
   RGB = {0, 0, 1}, {1, 0, 0}, and {1, 1, 1} = blue, red, white
   The first 40% of the table is used to go linearly from blue to red.
   The remaining 60% of the table is used to go linearly from red to white.
   <p>
   If you define a very short interval such that less than one color fits
   in it, no colors at all will be allocated.  If this occurs for all
   intervals, ROOT will revert to the default palette.
   <p>
   Original code by Andreas Zoglauer (zog@mpe.mpg.de)
   End_html */

   TColor::InitializeColors();

   UInt_t g, c;
   UInt_t nPalette = 0;
   Int_t *palette = new Int_t[NColors+1];
   UInt_t nColorsGradient;
   TColor *color;
   Int_t highestIndex = 0;

   if(Number < 2 || NColors < 1){
      delete [] palette;
      return -1;
   }

   // Check if all RGB values are between 0.0 and 1.0 and
   // Stops goes from 0.0 to 1.0 in increasing order.
   for (c = 0; c < Number; c++) {
      if (Red[c] < 0 || Red[c] > 1.0 ||
          Green[c] < 0 || Green[c] > 1.0 ||
          Blue[c] < 0 || Blue[c] > 1.0 ||
          Stops[c] < 0 || Stops[c] > 1.0) {
         //Error("CreateGradientColorTable",
         //      "All RGB colors and stops have to be between 0.0 and 1.0");
         delete [] palette;
         return -1;
      }
      if (c >= 1) {
         if (Stops[c-1] > Stops[c]) {
            //Error("CreateGradientColorTable",
            //      "Stops have to be in increasing order");
            delete [] palette;
            return -1;
         }
      }
   }

   // Search for the highest color index not used in ROOT:
   // We do not want to overwrite some colors...
   TSeqCollection *colorTable = gROOT->GetListOfColors();
   if ((color = (TColor *) colorTable->Last()) != 0) {
      if (color->GetNumber() > highestIndex) {
         highestIndex = color->GetNumber();
      }
      while ((color = (TColor *) (colorTable->Before(color))) != 0) {
         if (color->GetNumber() > highestIndex) {
            highestIndex = color->GetNumber();
         }
      }
   }
   highestIndex++;

   // Now create the colors and add them to the default palette:

   // For each defined gradient...
   for (g = 1; g < Number; g++) {
      // create the colors...
      nColorsGradient = (Int_t) (floor(NColors*Stops[g]) - floor(NColors*Stops[g-1]));
      for (c = 0; c < nColorsGradient; c++) {
         new TColor(highestIndex,
                    Red[g-1] + c * (Red[g] - Red[g-1])/ nColorsGradient,
                    Green[g-1] + c * (Green[g] - Green[g-1])/ nColorsGradient,
                    Blue[g-1] + c * (Blue[g] - Blue[g-1])/ nColorsGradient,
                    "  ");
         gROOT->GetColor(highestIndex)->SetAlpha(alpha);
         palette[nPalette] = highestIndex;
         nPalette++;
         highestIndex++;
      }
   }

   TColor::SetPalette(nPalette, palette);
   delete [] palette;

   return highestIndex - NColors;
}


//______________________________________________________________________________
void TColor::SetPalette(Int_t ncolors, Int_t *colors, Float_t alpha)
{
   /* Begin_html
   Static function.
   The color palette is used by the histogram classes
    (see TH1::Draw options).
   For example TH1::Draw("col") draws a 2-D histogram with cells
   represented by a box filled with a color CI function of the cell content.
   if the cell content is N, the color CI used will be the color number
   in colors[N],etc. If the maximum cell content is > ncolors, all
   cell contents are scaled to ncolors.
   <p>
   <tt>if ncolors <= 0</tt> a default palette (see below) of 50 colors is
   defined. The colors defined in this palette are OK for coloring pads, labels.
   <p>
   <pre>
   index 0->9   : grey colors from light to dark grey
   index 10->19 : "brown" colors
   index 20->29 : "blueish" colors
   index 30->39 : "redish" colors
   index 40->49 : basic colors
   </pre>
   <p>
   <tt>if ncolors == 1 && colors == 0</tt>, then a Pretty Palette with a
   Spectrum Violet->Red is created with 50 colors. That's the default rain bow
   pallette.
   <p>
   Other prefined palettes with 255 colors are available when <tt>colors == 0</tt>. 
   The following value of <tt>ncolors</tt> give access to:
   <p>
   <pre>
   if ncolors = 51 and colors=0, a Deep Sea palette is used.
   if ncolors = 52 and colors=0, a Grey Scale palette is used.
   if ncolors = 53 and colors=0, a Dark Body Radiator palette is used.
   if ncolors = 54 and colors=0, a two-color hue palette palette is used.(dark blue through neutral gray to bright yellow) 
   if ncolors = 55 and colors=0, a Rain Bow palette is used.
   if ncolors = 56 and colors=0, an inverted Dark Body Radiator palette is used.
   </pre>
   (see TColor::CreateGradientColorTable for more details)
   <p>
   The color numbers specified in the palette can be viewed by selecting
   the item "colors" in the "VIEW" menu of the canvas toolbar.
   The color parameters can be changed via TColor::SetRGB.
   <p>
   Note that when drawing a 2D histogram <tt>h2</tt> with the option "COL" or 
   "COLZ" or with any "CONT" options using the color map, the number of colors 
   used is defined by the number of contours <tt>n</tt> specified with:
   <tt>h2->SetContour(n)</tt>
   End_html */

   Int_t i;
   static Int_t paletteType = 0;
   Int_t palette[50] = {19,18,17,16,15,14,13,12,11,20,
                        21,22,23,24,25,26,27,28,29,30, 8,
                        31,32,33,34,35,36,37,38,39,40, 9,
                        41,42,43,44,45,47,48,49,46,50, 2,
                         7, 6, 5, 4, 3, 2,1};
   // set default palette (pad type)
   if (ncolors <= 0) {
      ncolors = 50;
      fgPalette.Set(ncolors);
      for (i=0;i<ncolors;i++) fgPalette.fArray[i] = palette[i];
      paletteType = 1;
      return;
   }

   // set Pretty Palette Spectrum Violet->Red
   if (ncolors == 1 && colors == 0) {
      ncolors = 50;
      fgPalette.Set(ncolors);
      for (i=0;i<ncolors;i++) fgPalette.fArray[i] = 51+i;
      paletteType = 2;
      return;
   }

   // set Deep Sea palette
   if (ncolors == 51 && colors == 0) {
      TColor::InitializeColors();
      if (ncolors == fgPalette.fN && paletteType == 3) return;
      const Int_t nRGBs = 5;
      Double_t stops[nRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
      Double_t red[nRGBs]   = { 0.00, 0.09, 0.18, 0.09, 0.00 };
      Double_t green[nRGBs] = { 0.01, 0.02, 0.39, 0.68, 0.97 };
      Double_t blue[nRGBs]  = { 0.17, 0.39, 0.62, 0.79, 0.97 };
      TColor::CreateGradientColorTable(nRGBs, stops, red, green, blue, 255, alpha);
      paletteType = 3;
      return;
   }
   
   // set Grey Scale palette
   if (ncolors == 52 && colors == 0) {
      TColor::InitializeColors();
      if (ncolors == fgPalette.fN && paletteType == 4) return;
      const Int_t nRGBs = 3;
      Double_t stops[nRGBs] = { 0.00, 0.50, 1.00};
      Double_t red[nRGBs]   = { 0.00, 0.50, 1.00};
      Double_t green[nRGBs] = { 0.00, 0.50, 1.00};
      Double_t blue[nRGBs]  = { 0.00, 0.50, 1.00};
      TColor::CreateGradientColorTable(nRGBs, stops, red, green, blue, 255, alpha);
      paletteType = 4;
      return;
   }
   
   // set Dark Body Radiator palette
   if (ncolors == 53 && colors == 0) {
      TColor::InitializeColors();
      if (ncolors == fgPalette.fN && paletteType == 5) return;
      const Int_t nRGBs = 5;
      Double_t stops[nRGBs] = { 0.00, 0.25, 0.50, 0.75, 1.00};
      Double_t red[nRGBs]   = { 0.00, 0.50, 1.00, 1.00, 1.00};
      Double_t green[nRGBs] = { 0.00, 0.00, 0.55, 1.00, 1.00};
      Double_t blue[nRGBs]  = { 0.00, 0.00, 0.00, 0.00, 1.00};
      TColor::CreateGradientColorTable(nRGBs, stops, red, green, blue, 255, alpha);
      paletteType = 5;
      return;
   }
   
   // set two-color hue palette (dark blue through neutral gray to bright yellow)
   if (ncolors == 54 && colors == 0) {
      TColor::InitializeColors();
      if (ncolors == fgPalette.fN && paletteType == 6) return;
      const Int_t nRGBs = 3;
      Double_t stops[nRGBs] = { 0.00, 0.50, 1.00};
      Double_t red[nRGBs]   = { 0.00, 0.50, 1.00};
      Double_t green[nRGBs] = { 0.00, 0.50, 1.00};
      Double_t blue[nRGBs]  = { 0.50, 0.50, 0.00};
      TColor::CreateGradientColorTable(nRGBs, stops, red, green, blue, 255, alpha);
      paletteType = 6;
      return;
   }
   
   // set Rain Bow palette 
   if (ncolors == 55 && colors == 0) {
      TColor::InitializeColors();
      if (ncolors == fgPalette.fN && paletteType == 7) return;
      const Int_t nRGBs = 5;
      Double_t stops[nRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
      Double_t red[nRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
      Double_t green[nRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
      Double_t blue[nRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
      TColor::CreateGradientColorTable(nRGBs, stops, red, green, blue, 255, alpha);
      paletteType = 7;
      return;
   }
   
   // set Inverted Dark Body Radiator palette
   if (ncolors == 56 && colors == 0) {
      TColor::InitializeColors();
      if (ncolors == fgPalette.fN && paletteType == 5) return;
      const Int_t nRGBs = 5;
      Double_t stops[nRGBs] = { 0.00, 0.25, 0.50, 0.75, 1.00};
      Double_t red[nRGBs]   = { 1.00, 1.00, 1.00, 0.50, 0.00};
      Double_t green[nRGBs] = { 1.00, 1.00, 0.55, 0.00, 0.00};
      Double_t blue[nRGBs]  = { 1.00, 0.00, 0.00, 0.00, 0.00};
      TColor::CreateGradientColorTable(nRGBs, stops, red, green, blue, 255, alpha);
      paletteType = 5;
      return;
   }

   // set user defined palette
   fgPalette.Set(ncolors);
   if (colors)  for (i=0;i<ncolors;i++) fgPalette.fArray[i] = colors[i];
   else         for (i=0;i<ncolors;i++) fgPalette.fArray[i] = palette[i];
   paletteType = 4;
}

