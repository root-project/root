// @(#)root/base:$Id$
// Author: Rene Brun   12/05/95

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TAttMarker.h"

#include <iostream>
#include <cmath>
#include "TVirtualPad.h"
#include "TVirtualPadPainter.h"
#include "TVirtualPadEditor.h"
#include "TStyle.h"
#include "TColor.h"
#include "TPoint.h"


/** \class TAttMarker
\ingroup Base
\ingroup GraphicsAtt

Marker Attributes class.

This class is used (in general by secondary inheritance)
by many other classes (graphics, histograms). It holds all the markers
attributes.

## Marker attributes
The marker attributes are:

  - [Marker color](\ref ATTMARKER1)
  - [Marker style](\ref ATTMARKER2)
    - [Marker line width](\ref ATTMARKER21)
  - [Marker size](\ref M3)

\anchor ATTMARKER1
## Marker color
The marker color is a color index (integer) pointing in the ROOT color
table.
The marker color of any class inheriting from `TAttMarker` can
be changed using the method `SetMarkerColor` and retrieved using the
method `GetMarkerColor`.
The following table shows the first 50 default colors.

Begin_Macro
{
   TCanvas *c = new TCanvas("c","Fill Area colors",0,0,500,200);
   c->DrawColorTable();
   return c;
}
End_Macro

### Color transparency

`SetMarkerColorAlpha()`, allows to set a transparent color.
In the following example the marker color of the histogram `histo`
is set to blue with an opacity of 35% (i.e. a transparency of 65%).
(The color `kBlue` itself is internally stored as fully opaque.)

~~~ {.cpp}
histo->SetMarkerColorAlpha(kBlue, 0.35);
~~~

The transparency is available on all platforms when the flag `OpenGL.CanvasPreferGL` is set to `1`
in `$ROOTSYS/etc/system.rootrc`, or on Mac with the Cocoa backend. On the file output
it is visible with PDF, PNG, Gif, JPEG, SVG, TeX ... but not PostScript.

Alternatively, you can call at the top of your script `gSytle->SetCanvasPreferGL();`.
Or if you prefer to activate GL for a single canvas `c`, then use `c->SetSupportGL(true);`.

\anchor ATTMARKER2
## Marker style

The Marker style defines the markers' shape.
The marker style of any class inheriting from `TAttMarker` can
be changed using the method `SetMarkerStyle` and retrieved using the
method `GetMarkerStyle`.

The following list gives the currently supported markers (screen
and PostScript) style. Each marker style is identified by an integer number
(first column) corresponding to a marker shape (second column) and can be also
accessed via a global name (third column).

~~~ {.cpp}
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
       29                    full star            kFullStar
       30                    open star            kOpenStar
       31                    *                    kStar2
       32                    open triangle down   kOpenTriangleDown
       33                    full diamond         kFullDiamond
       34                    full cross           kFullCross
       35                    open diamond cross   kOpenDiamondCross
       36                    open square diagonal kOpenSquareDiagonal
       37                    open three triangle  kOpenThreeTriangles
       38                    octagon with cross   kOctagonCross
       39                    full three triangles kFullThreeTriangles
       40                    open four triangleX  kOpenFourTrianglesX
       41                    full four triangleX  kFullFourTrianglesX
       42                    open double diamond  kOpenDoubleDiamond
       43                    full double diamond  kFullDoubleDiamond
       44                    open four triangle+  kOpenFourTrianglesPlus
       45                    full four triangle+  kFullFourTrianglesPlus
       46                    open cross X         kOpenCrossX
       47                    full cross X         kFullCrossX
       48                    four squares X       kFourSquaresX
       49                    four squares+        kFourSquaresPlus
~~~

Begin_Macro
{
   TCanvas *c = new TCanvas("c","Marker types",0,0,500,200);
   TMarker::DisplayMarkerTypes();
}
End_Macro

\warning Non-symmetric symbols should be used carefully. See markerwarning.C

\anchor ATTMARKER21
### Marker line width

The line width of a marker is not actually a marker attribute since it does
only apply to open marker symbols and marker symbols consisting of lines. All
of these marker symbols are redefined with thicker lines by style numbers
starting from 50:

~~~ {.cpp}
   Marker numbers   Line width
      50 -  67         2
      68 -  85         3
      86 - 103         4
     104 - 121         5
   ...
~~~

Begin_Macro
{
   TCanvas *c = new TCanvas("c","Marker line widths",0,0,600,266);
   TMarker::DisplayMarkerLineWidths();
}
End_Macro

\anchor M3
## Marker size

Various marker sizes are shown in the figure below. The default marker size=1
is shown in the top left corner. Marker sizes smaller than 1 can be
specified. The marker size does not refer to any coordinate systems, it is an
absolute value. Therefore the marker size is not affected by any change
in TPad's scale. A marker size equal to 1 correspond to 8 pixels.
That is, a square marker with size 1 will be drawn with a side equal to 8
pixels on the screen.

The marker size of any class inheriting from `TAttMarker` can
be changed using the method `SetMarkerSize` and retrieved using the
method `GetMarkerSize`.

Begin_Macro
{
   auto c = new TCanvas("c","Marker sizes",0,0,500,200);
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
}
End_Macro

Note that the marker styles number 1 6 and 7 (the dots), cannot be scaled. They
are meant to be very fast to draw and are always drawn with the same number of
pixels; therefore `SetMarkerSize` does not apply on them. To have a
"scalable dot" a filled circle should be used instead, i.e. the marker style
number 20. By default (if `SetMarkerStyle` is not specified), the marker
style used is 1. That's the most common one to draw scatter plots.
*/

////////////////////////////////////////////////////////////////////////////////
/// TAttMarker default constructor.
///
/// Default text attributes are taking from the current style.

TAttMarker::TAttMarker()
{
   if (!gStyle) {
      fMarkerColor = 1;
      fMarkerStyle = kDot;
      fMarkerSize = 1;
   } else {
      fMarkerColor = gStyle->GetMarkerColor();
      fMarkerStyle = gStyle->GetMarkerStyle();
      fMarkerSize  = gStyle->GetMarkerSize();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// TAttMarker normal constructor.
///
/// Text attributes are taking from the argument list
///  - color : Marker Color Index
///  - style : Marker style (from 1 to 30)
///  - size  : marker size (float)

TAttMarker::TAttMarker(Color_t color, Style_t style, Size_t msize)
{
   fMarkerColor = color;
   fMarkerSize  = msize;
   fMarkerStyle = style;
}

////////////////////////////////////////////////////////////////////////////////
/// TAttMarker destructor.

TAttMarker::~TAttMarker()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy this marker attributes to a new TAttMarker.

void TAttMarker::Copy(TAttMarker &attmarker) const
{
   attmarker.fMarkerColor  = fMarkerColor;
   attmarker.fMarkerStyle  = fMarkerStyle;
   attmarker.fMarkerSize   = fMarkerSize;
}

////////////////////////////////////////////////////////////////////////////////
/// Internal helper function that returns the corresponding marker style with
/// line width 1 for the given style.

Style_t TAttMarker::GetMarkerStyleBase(Style_t style)
{
   if (style <= kFourSquaresPlus)
      return style;

   switch ((style - 50) % 18) {
   case 0: return kPlus;
   case 1: return kStar;
   case 2: return kMultiply;
   case 3: return kOpenCircle;
   case 4: return kOpenSquare;
   case 5: return kOpenTriangleUp;
   case 6: return kOpenDiamond;
   case 7: return kOpenCross;
   case 8: return kOpenStar;
   case 9: return kOpenTriangleDown;
   case 10: return kOpenDiamondCross;
   case 11: return kOpenSquareDiagonal;
   case 12: return kOpenThreeTriangles;
   case 13: return kOctagonCross;
   case 14: return kOpenFourTrianglesX;
   case 15: return kOpenDoubleDiamond;
   case 16: return kOpenFourTrianglesPlus;
   case 17: return kOpenCrossX;
   }

   return kDot;
}

////////////////////////////////////////////////////////////////////////////////
/// Internal helper function that returns the line width of the given marker
/// style (0 = filled marker)

Width_t TAttMarker::GetMarkerLineWidth(Style_t style)
{
   if (style >= 50)
      return ((style - 50) / 18) + 2;
   if (style == kPlus || style == kStar || style == kCircle || style == kMultiply || style == kFullDotSmall ||
       style == kOpenCircle || style == kOpenSquare || style == kOpenTriangleUp || style == kOpenDiamond ||
       style == kOpenCross || style == kOpenStar || style == kStar2 || style == kOpenTriangleDown ||
       style == kOpenDiamondCross || style == kOpenSquareDiagonal || style == kOpenThreeTriangles ||
       style == kOctagonCross || style == kOpenFourTrianglesX || style == kOpenDoubleDiamond ||
       style == kOpenFourTrianglesPlus || style == kOpenCrossX)
      return 1;

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Change current marker attributes if necessary.

void TAttMarker::Modify()
{
   if (gPad)
      ModifyOn(*gPad);
}

////////////////////////////////////////////////////////////////////////////////
/// Change current marker attributes if necessary on specified pad.

void TAttMarker::ModifyOn(TVirtualPad &pad)
{
   auto pp = pad.GetPainter();
   if (pp)
      pp->SetAttMarker(*this);
}


////////////////////////////////////////////////////////////////////////////////
/// Reset this marker attributes to the default values.

void TAttMarker::ResetAttMarker(Option_t *)
{
   fMarkerColor  = 1;
   fMarkerStyle  = kDot;
   fMarkerSize   = 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Save line attributes as C++ statement(s) on output stream out.

void TAttMarker::SaveMarkerAttributes(std::ostream &out, const char *name, Int_t coldef, Int_t stydef, Int_t sizdef)
{
   if (fMarkerColor != coldef)
      out << "   " << name << "->SetMarkerColor(" << TColor::SavePrimitiveColor(fMarkerColor) << ");\n";
   if (fMarkerStyle != stydef)
      out << "   " << name << "->SetMarkerStyle(" << fMarkerStyle << ");\n";
   if (fMarkerSize != sizdef)
      out << "   " << name << "->SetMarkerSize(" << fMarkerSize << ");\n";
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the DialogCanvas Marker attributes.

void TAttMarker::SetMarkerAttributes()
{
   TVirtualPadEditor::UpdateMarkerAttributes(fMarkerColor, fMarkerStyle, fMarkerSize);
}

////////////////////////////////////////////////////////////////////////////////
/// Set a transparent marker color.
/// \param mcolor defines the marker color
/// \param malpha defines the percentage of opacity from 0. (fully transparent) to 1. (fully opaque).
/// \note malpha is ignored (treated as 1) if the TCanvas has no GL support activated.

void TAttMarker::SetMarkerColorAlpha(Color_t mcolor, Float_t malpha)
{
   fMarkerColor = TColor::GetColorTransparent(mcolor, malpha);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the marker color.

void TAttMarker::SetMarkerColor(Color_t mcolor)
{
   fMarkerColor = mcolor;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the marker color.

void TAttMarker::SetMarkerColor(TColorNumber lcolor)
{
   SetMarkerColor(lcolor.number());
}

////////////////////////////////////////////////////////////////////////////////
/// Set the marker style.

void TAttMarker::SetMarkerStyle(Style_t mstyle)
{
   fMarkerStyle = mstyle;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the marker size.
/// Note that the marker styles number 1 6 and 7 (the dots), cannot be scaled.
/// They are meant to be very fast to draw and are always drawn with the same number of pixels;
/// therefore this method does not apply on them.

void TAttMarker::SetMarkerSize(Size_t msize)
{
   fMarkerSize  = msize;
}

////////////////////////////////////////////////////////////////////////////////
/// Return marker shape.
/// Depending from configured marker style different marker shapes are returned
/// For simple shape like circle just size is assigned, for other points vector is filled as well
/// For special applications (like GL) one can create set of triangles instead of complex filled shapes
/// This is required while GL not always able to correctly fill closed shape

TAttMarker::EMarkerShape TAttMarker::GetMarkerShape(Int_t &sz, std::vector<TPoint> &shape, Float_t scale, Bool_t prefer_triangles) const
{
   Int_t markerStyle = GetMarkerStyleBase(GetMarkerStyle());
   Int_t markerLineWidth = GetMarkerLineWidth(GetMarkerStyle());

   Float_t markerSizeReduced = scale * (GetMarkerSize() - std::floor(markerLineWidth/2.)/4.);
   const auto im = std::round(4*markerSizeReduced);
   const auto im2 = std::round(2*markerSizeReduced);

   auto addTriangle = [&shape](Int_t x1, Int_t y1, Int_t x2, Int_t y2, Int_t x3 = 0, Int_t y3 = 0) {
      shape.emplace_back(x1, y1);
      shape.emplace_back(x2, y2);
      shape.emplace_back(x3, y3);
   };

   auto addSquare = [&shape](Int_t x1, Int_t y1, Int_t x2, Int_t y2) {
      shape.emplace_back(x1, y1);
      shape.emplace_back(x1, y2);
      shape.emplace_back(x2, y2);

      shape.emplace_back(x1, y1);
      shape.emplace_back(x2, y2);
      shape.emplace_back(x2, y1);
   };

   sz = 0;
   shape.clear();

   switch (markerStyle) {
      case kDot:
         return kShapeDot;
      case kPlus:
         shape.resize(4);
         shape[0].fX = -im;  shape[0].fY =   0;
         shape[1].fX =  im;  shape[1].fY =   0;
         shape[2].fX =   0;  shape[2].fY = -im;
         shape[3].fX =   0;  shape[3].fY =  im;
         return kShapeSegments;
      case kStar:
      case kStar2: {
         const auto imx = std::round(0.707*4*markerSizeReduced);
         shape.resize(8);
         shape[0].fX = -im;  shape[0].fY = 0;
         shape[1].fX =  im;  shape[1].fY = 0;
         shape[2].fX = 0  ;  shape[2].fY = -im;
         shape[3].fX = 0  ;  shape[3].fY = im;
         shape[4].fX = -imx;  shape[4].fY = -imx;
         shape[5].fX =  imx;  shape[5].fY = imx;
         shape[6].fX = -imx;  shape[6].fY = imx;
         shape[7].fX =  imx;  shape[7].fY = -imx;
         return kShapeSegments;
      }
      case kCircle:
      case kOpenCircle:
         sz = im * 2;
         return kShapeCircle;
      case kMultiply: {
         const auto imx = std::round(0.707*4*markerSizeReduced);
         shape.reserve(4);
         shape.emplace_back(-imx, -imx);
         shape.emplace_back( imx,  imx);
         shape.emplace_back(-imx,  imx);
         shape.emplace_back( imx, -imx);
         return kShapeSegments;
      }
      case kFullDotSmall:
         shape.resize(4);
         shape[0].fX = -1;  shape[0].fY = 0;
         shape[1].fX =  1;  shape[1].fY = 0;
         shape[2].fX =  0;  shape[2].fY = -1;
         shape[3].fX =  0;  shape[3].fY = 1;
         return kShapeSegments;
      case kFullDotMedium:
         shape.resize(5);
         shape[0].fX = -1;  shape[0].fY = -1;
         shape[1].fX =  1;  shape[1].fY = -1;
         shape[2].fX =  1;  shape[2].fY =  1;
         shape[3].fX = -1;  shape[3].fY =  1;
         shape[4].fX = -1;  shape[4].fY = -1;
         return kShapeFilledArea;
      case kFullDotLarge:
      case kFullCircle:
         sz = im * 2;
         return kShapeFilledCircle;
      case kFullSquare:
         shape.resize(5);
         shape[0].fX = -im;  shape[0].fY = -im;
         shape[1].fX =  im;  shape[1].fY = -im;
         shape[2].fX =  im;  shape[2].fY = im;
         shape[3].fX = -im;  shape[3].fY = im;
         shape[4].fX = -im;  shape[4].fY = -im;
         return kShapeFilledArea;
      case kFullTriangleUp:
      case kOpenTriangleUp:
         shape.resize(4);
         shape[0].fX = -im;  shape[0].fY = im;
         shape[1].fX =  im;  shape[1].fY = im;
         shape[2].fX =   0;  shape[2].fY = -im;
         shape[3].fX = -im;  shape[3].fY = im;
         return markerStyle == kFullTriangleUp ? kShapeFilledArea : kShapePolyLine;
      case kFullTriangleDown:
      case kOpenTriangleDown:
         shape.resize(4);
         shape[0].fX =   0;  shape[0].fY = im;
         shape[1].fX =  im;  shape[1].fY = -im;
         shape[2].fX = -im;  shape[2].fY = -im;
         shape[3].fX =   0;  shape[3].fY = im;
         return markerStyle == kFullTriangleDown ? kShapeFilledArea : kShapePolyLine;
      case kOpenSquare:
         shape.resize(5);
         shape[0].fX = -im;  shape[0].fY = -im;
         shape[1].fX =  im;  shape[1].fY = -im;
         shape[2].fX =  im;  shape[2].fY = im;
         shape[3].fX = -im;  shape[3].fY = im;
         shape[4].fX = -im;  shape[4].fY = -im;
         return kShapePolyLine;
      case kOpenDiamond:
      case kFullDiamond: {
         shape.resize(5);
         const auto imx = std::round(2.66*markerSizeReduced);
         shape[0].fX =-imx;  shape[0].fY = 0;
         shape[1].fX =   0;  shape[1].fY = -im;
         shape[2].fX = imx;  shape[2].fY = 0;
         shape[3].fX =   0;  shape[3].fY = im;
         shape[4].fX =-imx;  shape[4].fY = 0;
         return markerStyle == kFullDiamond ? kShapeFilledArea : kShapePolyLine;
      }
      case kFullCross:
         if (prefer_triangles) {
            const auto imx = std::round(1.33*markerSizeReduced);
            shape.reserve(3 * 6);
            addSquare( -im, -imx,  -imx, imx);
            addSquare(-imx,  -im,   imx,  im);
            addSquare( imx, -imx,    im, imx);
            return kShapeTriangles;
         }
      case kOpenCross: {
         shape.resize(13);
         const auto imx = std::round(1.33*markerSizeReduced);
         shape[0].fX = -im;  shape[0].fY =-imx;
         shape[1].fX =-imx;  shape[1].fY =-imx;
         shape[2].fX =-imx;  shape[2].fY = -im;
         shape[3].fX = imx;  shape[3].fY = -im;
         shape[4].fX = imx;  shape[4].fY =-imx;
         shape[5].fX =  im;  shape[5].fY =-imx;
         shape[6].fX =  im;  shape[6].fY = imx;
         shape[7].fX = imx;  shape[7].fY = imx;
         shape[8].fX = imx;  shape[8].fY = im;
         shape[9].fX =-imx;  shape[9].fY = im;
         shape[10].fX=-imx;  shape[10].fY= imx;
         shape[11].fX= -im;  shape[11].fY= imx;
         shape[12].fX= -im;  shape[12].fY=-imx;
         return markerStyle == kFullCross ? kShapeFilledArea : kShapePolyLine;
      }
      case kFullStar:
         if (prefer_triangles) {
            const auto im1 = std::round(0.66*markerSizeReduced);
            const auto im3 = std::round(2.66*markerSizeReduced);
            const auto im4 = std::round(1.33*markerSizeReduced);
            shape.reserve(8 * 3);

            addTriangle( -im,  im4,  -im2, -im1,  -im4,  im4);
            addTriangle(-im2, -im1,  -im3,  -im,     0, -im2);
            addTriangle(   0, -im2,   im3,  -im,   im2, -im1);
            addTriangle( im2, -im1,    im,  im4,   im4,  im4);
            addTriangle( im4,  im4,     0,   im,  -im4,  im4);
            addTriangle(-im4,  im4,  -im2, -im1,     0, -im2);
            addTriangle(-im4,  im4,     0, -im2,   im2, -im1);
            addTriangle(-im4,  im4,   im2, -im1,   im4,  im4);
            return kShapeTriangles;
         }
      case kOpenStar: {
         const auto im1 = std::round(0.66*markerSizeReduced);
         const auto im3 = std::round(2.66*markerSizeReduced);
         const auto im4 = std::round(1.33*markerSizeReduced);
         shape.resize(11);
         shape[0].fX = -im;  shape[0].fY = im4;
         shape[1].fX =-im2;  shape[1].fY =-im1;
         shape[2].fX =-im3;  shape[2].fY = -im;
         shape[3].fX =   0;  shape[3].fY =-im2;
         shape[4].fX = im3;  shape[4].fY = -im;
         shape[5].fX = im2;  shape[5].fY =-im1;
         shape[6].fX =  im;  shape[6].fY = im4;
         shape[7].fX = im4;  shape[7].fY = im4;
         shape[8].fX =   0;  shape[8].fY = im;
         shape[9].fX =-im4;  shape[9].fY = im4;
         shape[10].fX= -im;  shape[10].fY= im4;
         return markerStyle == kFullStar ? kShapeFilledArea : kShapePolyLine;
      }
      case kOpenDiamondCross:
         shape.resize(8);
         shape[0].fX =-im;  shape[0].fY = 0;
         shape[1].fX =  0;  shape[1].fY = -im;
         shape[2].fX = im;  shape[2].fY = 0;
         shape[3].fX =  0;  shape[3].fY = im;
         shape[4].fX =-im;  shape[4].fY = 0;
         shape[5].fX = im;  shape[5].fY = 0;
         shape[6].fX =  0;  shape[6].fY = im;
         shape[7].fX =  0;  shape[7].fY =-im;
         return kShapePolyLine;
      case kOpenSquareDiagonal:
         shape.resize(8);
         shape[0].fX = -im;  shape[0].fY = -im;
         shape[1].fX =  im;  shape[1].fY = -im;
         shape[2].fX =  im;  shape[2].fY = im;
         shape[3].fX = -im;  shape[3].fY = im;
         shape[4].fX = -im;  shape[4].fY = -im;
         shape[5].fX =  im;  shape[5].fY = im;
         shape[6].fX = -im;  shape[6].fY = im;
         shape[7].fX =  im;  shape[7].fY = -im;
         return kShapePolyLine;
      case kOpenThreeTriangles:
         shape.resize(10);
         shape[0].fX =   0;  shape[0].fY =   0;
         shape[1].fX =-im2;  shape[1].fY =  im;
         shape[2].fX = im2;  shape[2].fY =  im;
         shape[3].fX =   0;  shape[3].fY =   0;
         shape[4].fX =-im2;  shape[4].fY = -im;
         shape[5].fX = -im;  shape[5].fY =   0;
         shape[6].fX =   0;  shape[6].fY =   0;
         shape[7].fX =  im;  shape[7].fY =   0;
         shape[8].fX = im2;  shape[8].fY =  -im;
         shape[9].fX =   0;  shape[9].fY =   0;
         return kShapePolyLine;
      case kOctagonCross:
         shape.resize(15);
         shape[0].fX = -im;  shape[0].fY = 0;
         shape[1].fX = -im;  shape[1].fY =-im2;
         shape[2].fX =-im2;  shape[2].fY = -im;
         shape[3].fX = im2;  shape[3].fY = -im;
         shape[4].fX =  im;  shape[4].fY =-im2;
         shape[5].fX =  im;  shape[5].fY = im2;
         shape[6].fX = im2;  shape[6].fY = im;
         shape[7].fX =-im2;  shape[7].fY = im;
         shape[8].fX = -im;  shape[8].fY = im2;
         shape[9].fX = -im;  shape[9].fY = 0;
         shape[10].fX = im;  shape[10].fY = 0;
         shape[11].fX =  0;  shape[11].fY = 0;
         shape[12].fX =  0;  shape[12].fY = -im;
         shape[13].fX =  0;  shape[13].fY = im;
         shape[14].fX =  0;  shape[14].fY = 0;
         return kShapePolyLine;
      case kFullThreeTriangles:
         shape.reserve(3 * 3);
         addTriangle( -im,   0,  -im2, im);
         addTriangle( im2,  im,    im,  0);
         addTriangle( im2, -im,  -im2, -im);
         return kShapeTriangles;
      case kOpenFourTrianglesX:
         shape.resize(13);
         shape[0].fX =     0;  shape[0].fY =    0;
         shape[1].fX =   im2;  shape[1].fY =   im;
         shape[2].fX =    im;  shape[2].fY =  im2;
         shape[3].fX =     0;  shape[3].fY =    0;
         shape[4].fX =    im;  shape[4].fY = -im2;
         shape[5].fX =   im2;  shape[5].fY =  -im;
         shape[6].fX =     0;  shape[6].fY =    0;
         shape[7].fX =  -im2;  shape[7].fY =  -im;
         shape[8].fX =   -im;  shape[8].fY = -im2;
         shape[9].fX =     0;  shape[9].fY =    0;
         shape[10].fX =   -im;  shape[10].fY =  im2;
         shape[11].fX =  -im2;  shape[11].fY =   im;
         shape[12].fX =     0;  shape[12].fY =  0;
         return kShapePolyLine;
      case kFullFourTrianglesX:
         shape.reserve(4 * 3);
         addTriangle( -im,  im2,  -im2,   im);
         addTriangle( im2,   im,    im,  im2);
         addTriangle(  im, -im2,   im2,  -im);
         addTriangle(-im2,  -im,  -im,  -im2);
         return kShapeTriangles;
      case kFullDoubleDiamond:
         if (prefer_triangles) {
            const auto im4 = std::round(markerSizeReduced);
            shape.reserve(8 * 3);
            addTriangle(   0,   im,   -im4,  im4);
            addTriangle(-im4,  im4,    -im,    0);
            addTriangle( -im,    0,   -im4, -im4);
            addTriangle(-im4, -im4,      0,  -im);
            addTriangle(   0,  -im,    im4, -im4);
            addTriangle( im4, -im4,     im,    0);
            addTriangle(  im,    0,    im4,  im4);
            addTriangle( im4,  im4,      0,   im);
            return kShapeTriangles;
         }
      case kOpenDoubleDiamond: {
         const auto im4 = std::round(markerSizeReduced);
         shape.resize(9);
         shape[0].fX=     0;   shape[0].fY= im;
         shape[1].fX=  -im4;   shape[1].fY= im4;
         shape[2].fX  = -im;   shape[2].fY = 0;
         shape[3].fX = -im4;   shape[3].fY = -im4;
         shape[4].fX =    0;   shape[4].fY = -im;
         shape[5].fX =  im4;   shape[5].fY = -im4;
         shape[6].fX =   im;   shape[6].fY = 0;
         shape[7].fX=   im4;   shape[7].fY= im4;
         shape[8].fX=     0;   shape[8].fY= im;
         return markerStyle == kFullDoubleDiamond ? kShapeFilledArea : kShapePolyLine;
      }
      case kOpenFourTrianglesPlus:
         shape.resize(11);
         shape[0].fX =    0;  shape[0].fY =    0;
         shape[1].fX =  im2;  shape[1].fY =   im;
         shape[2].fX = -im2;  shape[2].fY =   im;
         shape[3].fX =  im2;  shape[3].fY =  -im;
         shape[4].fX = -im2;  shape[4].fY =  -im;
         shape[5].fX =    0;  shape[5].fY =    0;
         shape[6].fX =   im;  shape[6].fY =  im2;
         shape[7].fX =   im;  shape[7].fY = -im2;
         shape[8].fX =  -im;  shape[8].fY =  im2;
         shape[9].fX =  -im;  shape[9].fY = -im2;
         shape[10].fX =    0; shape[10].fY =   0;
         return kShapePolyLine;
      case kFullFourTrianglesPlus:
         shape.reserve(4 * 3);
         addTriangle(-im2,   im,   im2,   im);
         addTriangle(  im,  im2,    im, -im2);
         addTriangle( im2,  -im,  -im2,  -im);
         addTriangle( -im, -im2,   -im,  im2);
         return kShapeTriangles;
      case kFullCrossX:
         if (prefer_triangles) {
            shape.reserve(6 * 3);
            addTriangle(-im2,   0,  -im,  im2,   -im2,  im);
            addTriangle(-im2,   0, -im2,   im,      0, im2);
            addTriangle(-im2, -im,  -im, -im2,    im2,  im);
            addTriangle(-im2, -im,  im2,   im,     im, im2);
            addTriangle( im2, -im,    0, -im2,    im2,   0);
            addTriangle( im2, -im,  im2,    0,     im,-im2);
            return kShapeTriangles;
         }
      case kOpenCrossX:
         shape.resize(13);
         shape[0].fX =    0;  shape[0].fY =  im2;
         shape[1].fX = -im2;  shape[1].fY =   im;
         shape[2].fX =  -im;  shape[2].fY =  im2;
         shape[3].fX = -im2;  shape[3].fY =    0;
         shape[4].fX =  -im;  shape[4].fY = -im2;
         shape[5].fX = -im2;  shape[5].fY =  -im;
         shape[6].fX =    0;  shape[6].fY = -im2;
         shape[7].fX =  im2;  shape[7].fY =  -im;
         shape[8].fX =   im;  shape[8].fY = -im2;
         shape[9].fX =  im2;  shape[9].fY =    0;
         shape[10].fX =  im;  shape[10].fY = im2;
         shape[11].fX = im2;  shape[11].fY =  im;
         shape[12].fX =   0;  shape[12].fY = im2;
         return markerStyle == kFullCrossX ? kShapeFilledArea : kShapePolyLine;
      case kFourSquaresX:
         shape.reserve(8 * 3);
         addTriangle(  -im2,   0,   -im,  im2,   -im2,   im);
         addTriangle(  -im2,   0,  -im2,   im,      0,  im2);
         addTriangle(   im2,   0,     0,  im2,    im2,   im);
         addTriangle(   im2,   0,   im2,   im,     im,  im2);
         addTriangle(  -im2, -im,   -im, -im2,   -im2,    0);
         addTriangle(  -im2, -im,  -im2,    0,      0, -im2);
         addTriangle(   im2, -im,     0, -im2,    im2,    0);
         addTriangle(   im2, -im,   im2,    0,     im, -im2);
         return kShapeTriangles;
      case kFourSquaresPlus: {
         const auto imx = std::round(1.33*markerSizeReduced);
         shape.reserve(4 * 2 * 3);
         addSquare(-imx,  imx, imx,   im);
         addSquare( imx, -imx,  im,  imx);
         addSquare( -im, -imx,-imx,  imx);
         addSquare(-imx,  -im, imx, -imx);
         return kShapeTriangles;
      }
   }

   return kShapeDot;
}
