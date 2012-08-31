// @(#)root/graf2d:$Id$
// Author: Olivier Couet, Timur Pocheptsov 23/01/2012


/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//#define NDEBUG

#include <stdexcept>
#include <cstring>
#include <cassert>

#include <Cocoa/Cocoa.h>

#include "QuartzFillArea.h"
#include "TColorGradient.h"
#include "QuartzMarker.h"
#include "CocoaPrivate.h"
#include "QuartzWindow.h"
#include "QuartzUtils.h"
#include "X11Drawable.h"
#include "QuartzText.h"
#include "QuartzLine.h"

#include "TGQuartz.h"
#include "TPoint.h"
#include "TColor.h"
#include "TStyle.h"
#include "TROOT.h"

ClassImp(TGQuartz)

//TODO:
//Originally, Olivier Couet suggested to have a separate module quartz with quartz-related graphics,
//to be used by both iOS and MacOSX code. Also, the separation of non-GUI and gui parts was suggested
//that's why we have TGQuartz and TGCocoa classes (TGCocoa is never used as it is, TGQuartz is
//created and initialzed by TROOT.
//Today it's clear that there is not need in any special quartz classes anymore -
//in my iOS applications/module I do not need anything from quartz module, also, the
//amount of code in quartz module is so small, that it can be merged back into cocoa module.

//At some point, I'll merge cocoa and quartz modules and cleanup all this
//mess and weird code we have in a quartz module.


namespace X11 = ROOT::MacOSX::X11;
namespace Quartz = ROOT::Quartz;

namespace {

//______________________________________________________________________________
void ConvertPointsROOTToCocoa(Int_t nPoints, const TPoint *xy, std::vector<TPoint> &dst, NSObject<X11Drawable> *drawable)
{
   assert(nPoints != 0 && "ConvertPointsROOTToCocoa, nPoints parameter is 0");
   assert(xy != 0 && "ConvertPointsROOTToCocoa, xy parameter is null");
   assert(drawable != 0 && "ConvertPointsROOTToCocoa, drawable parameter is null");
   
   dst.resize(nPoints);
   for (Int_t i = 0; i < nPoints; ++i) {
      dst[i].fX = xy[i].fX;
      dst[i].fY = SCoord_t(X11::LocalYROOTToCocoa(drawable, xy[i].fY));
   }
}

}

//______________________________________________________________________________
TGQuartz::TGQuartz()
{
   //Default ctor.
}


//______________________________________________________________________________
TGQuartz::TGQuartz(const char *name, const char *title)
            : TGCocoa(name, title)
{
   //Constructor.
}


//______________________________________________________________________________
void TGQuartz::DrawBox(Int_t x1, Int_t y1, Int_t x2, Int_t y2, EBoxMode mode)
{
   //Check some conditions first.
   if (fDirectDraw)//To avoid warnings from Quartz - no context at the moment!
      return;

   NSObject<X11Drawable> * const drawable = (NSObject<X11Drawable> *)GetSelectedDrawableChecked("DrawBox");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);

   const TColor * const fillColor = gROOT->GetColor(GetFillColor());
   if (!fillColor) {
      Error("DrawBox", "Fill color for index %d not found", GetFillColor());
      return;
   }

   //Go to low-left-corner system.
   y1 = Int_t(X11::LocalYROOTToCocoa(drawable, y1));
   y2 = Int_t(X11::LocalYROOTToCocoa(drawable, y2));

   if (const TColorGradient * const extendedColor = dynamic_cast<const TColorGradient *>(fillColor)) {
      //Draw a box with a gradient fill and a shadow.
      //Ignore all fill styles and EBoxMode, use a gradient fill.
      Quartz::DrawBoxGradient(ctx, x1, y1, x2, y2, extendedColor, kTRUE);//kTRUE == draw a shadow.
   } else {
      const bool isHollow = mode == kHollow || GetFillStyle() / 1000 == 2;
      unsigned patternIndex = 0;
      if (isHollow) {
         if (!Quartz::SetLineColor(ctx, GetLineColor())) {
            Error("DrawBox", "Can not find color for index %d", int(GetLineColor()));
            return;
         }
      } else {
         if (!Quartz::SetFillAreaParameters(ctx, &patternIndex)) {
            Error("DrawBox", "SetFillAreaParameters failed");
            return;
         }
      }

      Quartz::DrawBox(ctx, x1, y1, x2, y2, isHollow);
   }
}


//______________________________________________________________________________
void TGQuartz::DrawFillArea(Int_t n, TPoint *xy)
{
   //Comment from TVirtualX:

   // Draw a filled area through all points.
   // n         : number of points
   // xy        : array of points

   //End of comment.

   //Do some checks first.
   if (fDirectDraw)//To avoid warnings from Quartz - no context at the moment!
      return;

   NSObject<X11Drawable> * const drawable = (NSObject<X11Drawable> *)GetSelectedDrawableChecked("DrawFillArea");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;

   //Convert points to bottom-left system:
   ConvertPointsROOTToCocoa(n, xy, fConvertedPoints, drawable);
   
   const Quartz::CGStateGuard ctxGuard(ctx);

   const TColor * const fillColor = gROOT->GetColor(GetFillColor());
   if (!fillColor) {
      Error("DrawFillArea", "Could not find TColor for index %d", GetFillColor());
      return;
   }

   if (const TColorGradient * const extendedColor = dynamic_cast<const TColorGradient *>(fillColor)) {
      Quartz::DrawFillAreaGradient(ctx, n, &fConvertedPoints[0], extendedColor, kTRUE);//kTRUE == draw a shadow.
   } else {
      unsigned patternIndex = 0;
      if (!Quartz::SetFillAreaParameters(ctx, &patternIndex)) {
         Error("DrawFillArea", "SetFillAreaParameters failed");
         return;
      }

      Quartz::DrawFillArea(ctx, n, &fConvertedPoints[0], kFALSE);//The last argument - do not draw shadows.
   }
}


//______________________________________________________________________________
void TGQuartz::DrawCellArray(Int_t /*x1*/, Int_t /*y1*/, Int_t /*x2*/, Int_t /*y2*/, Int_t /*nx*/, Int_t /*ny*/, Int_t */*ic*/)
{
   //Noop.
}


//______________________________________________________________________________
void TGQuartz::DrawLine(Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   // Draw a line.
   // x1,y1        : begin of line
   // x2,y2        : end of line

   if (fDirectDraw)//To avoid warnings from Quartz - no context at the moment!
      return;

   //Do some checks first:
   assert(fSelectedDrawable > fPimpl->GetRootWindowID() && "DrawLine, bad drawable is selected");
   NSObject<X11Drawable> * const drawable = (NSObject<X11Drawable> *)GetSelectedDrawableChecked("DrawLine");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);
      
   if (!Quartz::SetLineColor(ctx, GetLineColor())) {
      Error("DrawLine", "Could not set line color for index %d", int(GetLineColor()));
      return;
   }
   
   Quartz::SetLineStyle(ctx, GetLineStyle());
   Quartz::SetLineWidth(ctx, GetLineWidth());

   Quartz::DrawLine(ctx, x1, X11::LocalYROOTToCocoa(drawable, y1), x2, X11::LocalYROOTToCocoa(drawable, y2));
}


//______________________________________________________________________________
void TGQuartz::DrawPolyLine(Int_t n, TPoint *xy)
{
   //Comment from TVirtualX:
   // Draw a line through all points.
   // n         : number of points
   // xy        : list of points
   //End of comment.

   //Some checks first.
   if (fDirectDraw)//To avoid warnings from Quartz - no context at the moment!
      return;

   NSObject<X11Drawable> * const drawable = (NSObject<X11Drawable> *)GetSelectedDrawableChecked("DrawPolyLine");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);
   
   if (!Quartz::SetLineColor(ctx, GetLineColor())) {
      Error("DrawPolyLine", "Could not find TColor for index %d", GetLineColor());
      return;
   }
   
   Quartz::SetLineStyle(ctx, GetLineStyle());
   Quartz::SetLineWidth(ctx, GetLineWidth());
   
   //Convert to bottom-left-corner system.
   ConvertPointsROOTToCocoa(n, xy, fConvertedPoints, drawable);

   Quartz::DrawPolyLine(ctx, n, &fConvertedPoints[0]);
}


//______________________________________________________________________________
void TGQuartz::DrawPolyMarker(Int_t n, TPoint *xy)
{
   //Comment from TVirtualX:
   // Draw PolyMarker
   // n         : number of points
   // xy        : list of points
   //End of comment.

   //Do some checks first.
   if (fDirectDraw)//To avoid warnings from Quartz - no context at the moment!
      return;

   NSObject<X11Drawable> * const drawable = (NSObject<X11Drawable> *)GetSelectedDrawableChecked("DrawPolyMarker");
   if (!drawable)
      return;
      
   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);

   if (!Quartz::SetFillColor(ctx, GetMarkerColor())) {
      Error("DrawPolyMarker", "Could not find TColor for index %d", GetMarkerColor());
      return;
   }
   
   Quartz::SetLineColor(ctx, GetMarkerColor());//Can not fail (for coverity).
   Quartz::SetLineStyle(ctx, 1);
   Quartz::SetLineWidth(ctx, 1);

   ConvertPointsROOTToCocoa(n, xy, fConvertedPoints, drawable);

   Quartz::DrawPolyMarker(ctx, n, &fConvertedPoints[0], GetMarkerSize(), GetMarkerStyle());
}


//______________________________________________________________________________
void TGQuartz::DrawText(Int_t x, Int_t y, Float_t /*angle*/, Float_t /*mgn*/, const char *text, ETextMode /*mode*/)
{
   if (fDirectDraw)//To avoid warnings from Quartz - no context at the moment!
      return;

   if (!text || !text[0])//Can this ever happen? TPad::PaintText does not check this.
      return;
   
   NSObject<X11Drawable> * const drawable = (NSObject<X11Drawable> *)GetSelectedDrawableChecked("DrawText");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;   
   const Quartz::CGStateGuard ctxGuard(ctx);

   //Before any core text drawing operations, reset text matrix.
   CGContextSetTextMatrix(ctx, CGAffineTransformIdentity);

   try {
      if (CTFontRef currentFont = fPimpl->fFontManager.SelectFont(GetTextFont(), GetTextSize())) {
         if (GetTextFont() / 10 == 12) {//Greek and math symbols.
            //This is a hack. Correct way is to extract glyphs from symbol.ttf,
            //find correct mapping, place this glyphs. This requires manual layout though (?),
            //and as usually, I have to many things to do, may be, one day I'll fix text rendering also.
            //This hack work only on MacOSX 10.7.3, does not work on iOS and I'm not sure about future/previous
            //versions of MacOSX.
            typedef std::vector<UniChar>::size_type size_type;

            std::vector<UniChar> unichars(std::strlen(text));
            for (size_type i = 0, len = unichars.size(); i < len; ++i)
               unichars[i] = 0xF000 + (unsigned char)text[i];
            
            Quartz::TextLine ctLine(unichars, currentFont, GetTextColor());
            ctLine.DrawLine(ctx, x, X11::LocalYROOTToCocoa(drawable, y));
         } else {
            const Quartz::TextLine ctLine(text, currentFont, GetTextColor());
            ctLine.DrawLine(ctx, x, X11::LocalYROOTToCocoa(drawable, y));
         }
      }
   } catch (const std::exception &e) {
      Error("DrawText", "Exception from Quartz::TextLine: %s", e.what());
   }
}

//______________________________________________________________________________
void TGQuartz::GetTextExtent(UInt_t &w, UInt_t &h, char *text)
{
   // Returns the size of the specified character string "mess".
   //
   // w    - the text width
   // h    - the text height
   // text - the string   

   if (!text || !text[0]) {
      w = 0;
      h = 0;
      return;
   }
   
   if (fPimpl->fFontManager.SelectFont(GetTextFont(), GetTextSize()))
      fPimpl->fFontManager.GetTextBounds(w, h, text);
}


//______________________________________________________________________________
Int_t TGQuartz::GetFontAscent() const
{
   // Returns the ascent of the current font (in pixels).
   // The ascent of a font is the distance from the baseline
   // to the highest position characters extend to.
   if (fPimpl->fFontManager.SelectFont(GetTextFont(), GetTextSize()))
      return Int_t(fPimpl->fFontManager.GetAscent());

   return 0;
}

//______________________________________________________________________________
Int_t TGQuartz::GetFontDescent() const
{
   // Returns the descent of the current font (in pixels.
   // The descent is the distance from the base line
   // to the lowest point characters extend to.
   if (fPimpl->fFontManager.SelectFont(GetTextFont(), GetTextSize()))
      return Int_t(fPimpl->fFontManager.GetDescent());

   return 0;
}


//______________________________________________________________________________
Float_t TGQuartz::GetTextMagnitude()
{
   // Returns the current font magnification factor
   return 0;
}

//______________________________________________________________________________
void TGQuartz::SetLineColor(Color_t cindex)
{
   // Set color index "cindex" for drawing lines.
   TAttLine::SetLineColor(cindex);
}


//______________________________________________________________________________
void TGQuartz::SetLineStyle(Style_t lstyle)
{
   // Set line style.   
   TAttLine::SetLineStyle(lstyle);
}


//______________________________________________________________________________
void TGQuartz::SetLineWidth(Width_t width)
{
   // Set the line width.
   
   TAttLine::SetLineWidth(width);
}


//______________________________________________________________________________
void TGQuartz::SetFillColor(Color_t cindex)
{
   // Set color index "cindex" for fill areas.

   TAttFill::SetFillColor(cindex);
}


//______________________________________________________________________________
void TGQuartz::SetFillStyle(Style_t style)
{
   // Set fill area style.   
   TAttFill::SetFillStyle(style);
}


//______________________________________________________________________________
void TGQuartz::SetMarkerColor(Color_t cindex)
{
   // Set color index "cindex" for markers.
   TAttMarker::SetMarkerColor(cindex);
}


//______________________________________________________________________________
void TGQuartz::SetMarkerSize(Float_t markersize)
{
   // Set marker size index.
   //
   // markersize - the marker scale factor
   TAttMarker::SetMarkerSize(markersize);
}


//______________________________________________________________________________
void TGQuartz::SetMarkerStyle(Style_t markerstyle)
{
   // Set marker style.

   TAttMarker::SetMarkerStyle(markerstyle);
}


//______________________________________________________________________________
void TGQuartz::SetTextAlign(Short_t talign)
{
   // Set the text alignment.
   //
   // talign = txalh horizontal text alignment
   // talign = txalv vertical text alignment

   TAttText::SetTextAlign(talign);
}


//______________________________________________________________________________
void TGQuartz::SetTextColor(Color_t cindex)
{
   // Set the color index "cindex" for text.

   TAttText::SetTextColor(cindex);
}


//______________________________________________________________________________
void TGQuartz::SetTextFont(Font_t fontnumber)
{
   // Set the current text font number.

   TAttText::SetTextFont(fontnumber);
}


//______________________________________________________________________________
void TGQuartz::SetTextSize(Float_t textsize)
{
   // Set the current text size to "textsize"
   
   TAttText::SetTextSize(textsize);
}


//______________________________________________________________________________
void TGQuartz::SetOpacity(Int_t /*percent*/)
{
   // Set opacity of the current window. This image manipulation routine
   // works by adding to a percent amount of neutral to each pixels RGB.
   // Since it requires quite some additional color map entries is it
   // only supported on displays with more than > 8 color planes (> 256
   // colors).
}


//______________________________________________________________________________
Int_t TGQuartz::SetTextFont(char * /*fontname*/, ETextSetMode /*mode*/)
{
   // Set text font to specified name "fontname".This function returns 0 if
   // the specified font is found, 1 if it is not.
   //
   // mode - loading flag
   //        mode = 0 search if the font exist (kCheck)
   //        mode = 1 search the font and load it if it exists (kLoad)
   
   return 0;
}

//______________________________________________________________________________
void *TGQuartz::GetSelectedDrawableChecked(const char *calledFrom) const
{
   assert(calledFrom != 0 && "GetSelectedDrawableChecked, calledFrom parameter is null");
   assert(fSelectedDrawable > fPimpl->GetRootWindowID() && "GetSelectedDrawableChecked, bad drawable is selected");
   
   NSObject<X11Drawable> *drawable = fPimpl->GetDrawable(fSelectedDrawable);
   if (!drawable.fIsPixmap) {
      //TPad/TCanvas ALWAYS draw only into a pixmap.
      Error(calledFrom, "Selected drawable is not a pixmap");
      return 0;
   }
   
   if (!drawable.fContext) {
      Error(calledFrom, "Context is null");
      return 0;
   }
   
   return drawable;
}
