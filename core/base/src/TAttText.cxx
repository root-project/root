// @(#)root/base:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream>
#include "Strlen.h"
#include "TROOT.h"
#include "TAttText.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TVirtualX.h"
#include "TError.h"
#include "TVirtualPadEditor.h"
#include "TColor.h"

ClassImp(TAttText);

/** \class TAttText
\ingroup Base
\ingroup GraphicsAtt

Text Attributes class.

This class is used (in general by secondary inheritance)
by many other classes (graphics, histograms). It holds all the text attributes.

## Text attributes
Text attributes are:

  - [Text Alignment](\ref ATTTEXT1)
  - [Text Angle](\ref ATTTEXT2)
  - [Text Color](\ref ATTTEXT3)
  - [Text Size](\ref ATTTEXT4)
  - [Text Font and Precision](\ref ATTTEXT5)
     - [Font quality and speed](\ref ATTTEXT51)
     - [How to use True Type Fonts](\ref ATTTEXT52)
     - [List of the currently supported fonts](\ref ATTTEXT53)

\anchor ATTTEXT1
## Text Alignment

The text alignment is an integer number (`align`) allowing to control
the horizontal and vertical position of the text string with respect
to the text position.
The text alignment of any class inheriting from `TAttText` can
be changed using the method `SetTextAlign` and retrieved using the
method `GetTextAlign`.

~~~ {.cpp}
   align = 10*HorizontalAlign + VerticalAlign
~~~

For horizontal alignment the following convention applies:

~~~ {.cpp}
   1=left adjusted, 2=centered, 3=right adjusted
~~~

For vertical alignment the following convention applies:

~~~ {.cpp}
   1=bottom adjusted, 2=centered, 3=top adjusted
~~~

For example:

~~~ {.cpp}
   align = 11 = left adjusted and bottom adjusted
   align = 32 = right adjusted and vertically centered
~~~

Begin_Macro(source)
textalign.C
End_Macro

Mnemonic constants are available:

~~~ {.cpp}
kHAlignLeft   = 10, kHAlignCenter = 20, kHAlignRight = 30,
kVAlignBottom = 1,  kVAlignCenter = 2,  kVAlignTop   = 3
~~~

They allow to write:

~~~ {.cpp}
object->SetTextAlign(kHAlignLeft+kVAlignTop);
~~~

\anchor ATTTEXT2
## Text Angle

Text angle in degrees.
The text angle of any class inheriting from `TAttText` can
be changed using the method `SetTextAngle` and retrieved using the
method `GetTextAngle`.
The following picture shows the text angle:

Begin_Macro(source)
textangle.C
End_Macro

\anchor ATTTEXT3
## Text Color

The text color is a color index (integer) pointing in the ROOT
color table.
The text color of any class inheriting from `TAttText` can
be changed using the method `SetTextColor` and retrieved using the
method `GetTextColor`.
The following table shows the first 50 default colors.

Begin_Macro
{
   TCanvas *c = new TCanvas("c","Fill Area colors",0,0,500,200);
   c->DrawColorTable();
   return c;
}
End_Macro

### Color transparency
`SetTextColorAlpha()`, allows to set a transparent color.
In the following example the text color of the text `text`
is set to blue with an opacity of 35% (i.e. a transparency of 65%).
(The color `kBlue` itself is internally stored as fully opaque.)

~~~ {.cpp}
text->SetTextColorAlpha(kBlue, 0.35);
~~~

The transparency is available on all platforms when the flag `OpenGL.CanvasPreferGL` is set to `1`
in `$ROOTSYS/etc/system.rootrc`, or on Mac with the Cocoa backend. On the file output
it is visible with PDF, PNG, Gif, JPEG, SVG, TeX ... but not PostScript.

Alternatively, you can call at the top of your script `gSytle->SetCanvasPreferGL();`.
Or if you prefer to activate GL for a single canvas `c`, then use `c->SetSupportGL(true);`.

\anchor ATTTEXT4
## Text Size

If the text precision (see next paragraph) is smaller than 3, the text
size (`textsize`) is a fraction of the current pad size. Therefore the
same `textsize` value can generate text outputs with different absolute
sizes in two different pads.
The text size in pixels (`charheight`) is computed the following way:

~~~ {.cpp}
   pad_width  = gPad->XtoPixel(gPad->GetX2());
   pad_height = gPad->YtoPixel(gPad->GetY1());
   if (pad_width < pad_height)  charheight = textsize*pad_width;
   else                         charheight = textsize*pad_height;
~~~

If the text precision is equal to 3, the text size doesn't depend on the pad's
dimensions. A given `textsize` value always generates the same absolute
size. The text size (`charheight`) is given in pixels:

~~~ {.cpp}
   charheight = textsize;
~~~

Note that to scale fonts to the same size as the old True Type package a
scale factor of `0.93376068` is apply to the text size before drawing.

The text size of any class inheriting from `TAttText` can
be changed using the method `SetTextSize` and retrieved using the
method `GetTextSize`.

\anchor ATTTEXT5
## Text Font and Precision

The text font code is combination of the font number and the precision.
~~~ {.cpp}
   Text font code = 10*fontnumber + precision
~~~
Font numbers must be between 1 and 14.

The precision can be:

  - `precision = 0` fast hardware fonts (steps in the size)
  - `precision = 1` scalable and rotatable hardware fonts (see below)
  - `precision = 2` scalable and rotatable hardware fonts
  - `precision = 3` scalable and rotatable hardware fonts. Text size
                           is given in pixels.

The text font and precision of any class inheriting from `TAttText` can
be changed using the method `SetTextFont` and retrieved using the
method `GetTextFont`.

\anchor ATTTEXT51
### Font quality and speed

When precision 0 is used, only the original non-scaled X11 system fonts are
used. The fonts have a minimum (4) and maximum (37) size in pixels. These
fonts are fast and are of good quality. Their size varies with large steps
and they cannot be rotated.
Precision 1 and 2 fonts have a different behaviour depending if the
True Type Fonts (TTF) are used or not. If TTF are used, you always get very good
quality scalable and rotatable fonts.
These days TTF fonts are rendered fast enough and can be used in all cases.

\anchor ATTTEXT52
### How to use True Type Fonts

TTF fonts are used by default. They can be deactivated via the following line
in the `.rootrc` file:

~~~ {.cpp}
   Unix.*.Root.UseTTFonts:     false
~~~

\anchor ATTTEXT53
### List of the currently supported fonts

~~~ {.cpp}
   Font number      TTF Names                   PostScript/PDF Names
       1 :       "Free Serif Italic"         "Times-Italic"
       2 :       "Free Serif Bold"           "Times-Bold"
       3 :       "Free Serif Bold Italic"    "Times-BoldItalic"
       4 :       "Tex Gyre Regular"          "Helvetica"
       5 :       "Tex Gyre Italic"           "Helvetica-Oblique"
       6 :       "Tex Gyre Bold"             "Helvetica-Bold"
       7 :       "Tex Gyre Bold Italic"      "Helvetica-BoldOblique"
       8 :       "Free Mono"                 "Courier"
       9 :       "Free Mono Oblique"         "Courier-Oblique"
      10 :       "Free Mono Bold"            "Courier-Bold"
      11 :       "Free Mono Bold Oblique"    "Courier-BoldOblique"
      12 :       "Symbol"                    "Symbol"
      13 :       "Free Serif"                "Times-Roman"
      14 :       "Wingdings"                 "ZapfDingbats"
~~~

The PostScript and PDF backends use the original PostScript-defined 13 fonts' styles
forming four type families (Courier, Helvetica, Times, Symbol) as listed in the
"Core Font Set" section of [this page](https://en.wikipedia.org/wiki/PostScript_fonts).
These fonts are always available and do not need to be loaded in the PS or PDF files
allowing to keep the files' sizes small.

On screen, text is rendered using free TTF fonts similar to the PDF ones. The corresponding
font files are coming with the ROOT distribution in `$ROOTSYS/fonts/Free*`.

Begin_Macro
fonts.C
End_Macro
*/

////////////////////////////////////////////////////////////////////////////////
/// AttText default constructor.
///
/// Default text attributes are taken from the current style.

TAttText::TAttText()
{
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

////////////////////////////////////////////////////////////////////////////////
/// AttText normal constructor.
///
/// Text attributes are taken from the argument list.

TAttText::TAttText(Int_t align, Float_t angle, Color_t color, Style_t font, Float_t tsize)
{
   fTextAlign = align;
   fTextAngle = angle;
   fTextColor = color;
   fTextFont  = font;
   fTextSize  = tsize;
}

////////////////////////////////////////////////////////////////////////////////
/// AttText destructor.

TAttText::~TAttText()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy this text attributes to a new TAttText.

void TAttText::Copy(TAttText &atttext) const
{
   atttext.fTextAlign  = fTextAlign;
   atttext.fTextAngle  = fTextAngle;
   atttext.fTextColor  = fTextColor;
   atttext.fTextFont   = fTextFont;
   atttext.fTextSize   = fTextSize;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the text in percent of the pad size.
///
/// If the font precision is greater than 2, the text size returned is the size in pixel
/// converted into percent of the pad size, otherwise the size returned is the same as the
/// size given as input parameter.

Float_t TAttText::GetTextSizePercent(Float_t size)
{
   Float_t rsize = size;
   if (fTextFont%10 > 2 && gPad) {
      UInt_t w = TMath::Abs(gPad->XtoAbsPixel(gPad->GetX2()) -
                            gPad->XtoAbsPixel(gPad->GetX1()));
      UInt_t h = TMath::Abs(gPad->YtoAbsPixel(gPad->GetY2()) -
                            gPad->YtoAbsPixel(gPad->GetY1()));
      if (w < h)
         rsize = rsize/w;
      else
         rsize = rsize/h;
   }
   return rsize;
}

////////////////////////////////////////////////////////////////////////////////
/// Change current text attributes if necessary.

void TAttText::Modify()
{
   if (!gPad) return;

   // Do we need to change font?
   if (!gPad->IsBatch()) {
      gVirtualX->SetTextAngle(fTextAngle);
      Float_t tsize;
      if (fTextFont%10 > 2) {
         tsize = fTextSize;
      } else {
         Float_t wh = (Float_t)gPad->XtoPixel(gPad->GetX2());
         Float_t hh = (Float_t)gPad->YtoPixel(gPad->GetY1());
         if (wh < hh)  tsize = fTextSize*wh;
         else          tsize = fTextSize*hh;
      }

      if (gVirtualX->GetTextFont() != fTextFont) {
         gVirtualX->SetTextFont(fTextFont);
         gVirtualX->SetTextSize(tsize);
      } else if (gVirtualX->GetTextSize() != tsize) {
         gVirtualX->SetTextSize(tsize);
      }
      gVirtualX->SetTextAlign(fTextAlign);
      gVirtualX->SetTextColor(fTextColor);
   }
   gPad->SetAttTextPS(fTextAlign,fTextAngle,fTextColor,fTextFont,fTextSize);
}

////////////////////////////////////////////////////////////////////////////////
/// Reset this text attributes to default values.

void TAttText::ResetAttText(Option_t *)
{
   fTextAlign  = 11;
   fTextAngle  = 0;
   fTextColor  = 1;
   fTextFont   = 62;
   fTextSize   = 0.05;
}

////////////////////////////////////////////////////////////////////////////////
/// Save text attributes as C++ statement(s) on output stream out.

void TAttText::SaveTextAttributes(std::ostream &out, const char *name, Int_t alidef,
                                  Float_t angdef, Int_t coldef, Int_t fondef,
                                  Float_t sizdef)
{
   if (fTextAlign != alidef)
      out << "   " << name << "->SetTextAlign(" << fTextAlign << ");\n";
   if (fTextColor != coldef)
      out << "   " << name << "->SetTextColor(" << TColor::SavePrimitiveColor(fTextColor) << ");\n";
   if (fTextFont != fondef)
      out << "   " << name << "->SetTextFont(" << fTextFont << ");\n";
   if (fTextSize != sizdef)
      out << "   " << name << "->SetTextSize(" << fTextSize << ");\n";
   if (fTextAngle != angdef)
      out << "   " << name << "->SetTextAngle(" << fTextAngle << ");\n";
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the DialogCanvas Text attributes.

void TAttText::SetTextAttributes()
{
   TVirtualPadEditor::UpdateTextAttributes(fTextAlign,fTextAngle,fTextColor,
                                           fTextFont,fTextSize);
}

////////////////////////////////////////////////////////////////////////////////
/// Set a transparent text color.
/// \param tcolor defines the text color
/// \param talpha defines the percentage of opacity from 0. (fully transparent) to 1. (fully opaque).
/// \note talpha is ignored (treated as 1) if the TCanvas has no GL support activated.

void TAttText::SetTextColorAlpha(Color_t tcolor, Float_t talpha)
{
   fTextColor = TColor::GetColorTransparent(tcolor, talpha);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the text size in pixels.
///
/// If the font precision is greater than 2, the text size is set to npixels,
/// otherwise the text size is computed as a percent of the pad size.

void TAttText::SetTextSizePixels(Int_t npixels)
{
   if (fTextFont%10 > 2) {
      fTextSize = Float_t(npixels);
   } else {
      TVirtualPad *pad = gROOT->GetSelectedPad();
      if (!pad) return;
      Float_t dy = pad->AbsPixeltoY(0) - pad->AbsPixeltoY(npixels);
      fTextSize = dy/(pad->GetY2() - pad->GetY1());
   }
}

void TAttText::SetTextColor(TColorNumber lcolor)
{
   SetTextColor(lcolor.number());
}
