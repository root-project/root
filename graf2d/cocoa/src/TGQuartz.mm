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
#include <iostream>
#include <cstring>
#include <cassert>
#include <limits>

#include <Cocoa/Cocoa.h>

#  include <ft2build.h>
#  include FT_FREETYPE_H
#  include FT_GLYPH_H

#include "QuartzFillArea.h"
#include "TColorGradient.h"
#include "QuartzMarker.h"
#include "CocoaPrivate.h"
#include "QuartzWindow.h"
#include "QuartzPixmap.h"
#include "QuartzUtils.h"
#include "X11Drawable.h"
#include "QuartzText.h"
#include "QuartzLine.h"
#include "CocoaUtils.h"
#include "TGQuartz.h"
#include "TString.h"
#include "TPoint.h"
#include "TColor.h"
#include "TStyle.h"
#include "TROOT.h"
#include "TMath.h"
#include "TEnv.h"
#include "TTF.h"

// To scale fonts to the same size as the TTF version
const Float_t kScale = 0.93376068;


namespace X11 = ROOT::MacOSX::X11;
namespace Quartz = ROOT::Quartz;
namespace Util = ROOT::MacOSX::Util;

namespace {

//______________________________________________________________________________
void ConvertPointsROOTToCocoa(Int_t nPoints, const TPoint *xy, std::vector<TPoint> &dst,
                              NSObject<X11Drawable> *drawable)
{
   assert(nPoints != 0 && "ConvertPointsROOTToCocoa, nPoints parameter is 0");
   assert(xy != 0 && "ConvertPointsROOTToCocoa, xy parameter is null");
   assert(drawable != 0 && "ConvertPointsROOTToCocoa, drawable parameter is null");

   const auto scaleFactor = drawable.fScaleFactor;

   dst.resize(nPoints);
   for (Int_t i = 0; i < nPoints; ++i) {
      dst[i].fX = SCoord_t(xy[i].fX * scaleFactor);
      dst[i].fY = SCoord_t(X11::LocalYROOTToCocoa(drawable, xy[i].fY) * scaleFactor);
   }
}

}

//______________________________________________________________________________
TGQuartz::TGQuartz()
            : fUseAA(true), fUseFAAA(false)
{
   //Default ctor.
   if (!TTFhandle::Init())
      Error("TGQuartz", "TTFhandle::Init() failed");

   SetAA();
}


//______________________________________________________________________________
TGQuartz::TGQuartz(const char *name, const char *title)
            : TGCocoa(name, title),
              fUseAA(true), fUseFAAA(false)
{
   //Constructor.
   if (!TTFhandle::Init())
      Error("TGQuartz", "TTFhandle::Init() failed");

   SetAA();
}

//______________________________________________________________________________
void TGQuartz::DrawBoxW(WinContext_t wctxt, Int_t x1, Int_t y1, Int_t x2, Int_t y2, EBoxMode mode)
{
   auto drawable0 = (NSObject<X11Drawable> * const) wctxt;
   if (!drawable0)
      return;

   //Check some conditions first.
   if ([drawable0 isDirectDraw]) {
      if (!drawable0.fIsPixmap) {
         QuartzView * const view = (QuartzView *)fPimpl->GetWindow(drawable0.fID).fContentView;
         if (!view) {
            ::Warning("DrawBoxW", "Invalid view/window for XOR-mode");
            return;
         }

         [view.fQuartzWindow addXorBox: view : x1 : y1 : x2 : y2];
      }
      return;
   }

   auto &attline = GetAttLine(wctxt);
   auto &attfill = GetAttFill(wctxt);

   auto drawable = (NSObject<X11Drawable> * const) GetPixmapDrawable(drawable0, "DrawBoxW");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);
   //AA flag is not a part of a state.
   const Quartz::CGAAStateGuard aaCtxGuard(ctx, fUseAA);

   //Go to low-left-corner system.
   y1 = Int_t(X11::LocalYROOTToCocoa(drawable, y1));
   y2 = Int_t(X11::LocalYROOTToCocoa(drawable, y2));

   if (const TColorGradient * const gradient = dynamic_cast<TColorGradient *>(gROOT->GetColor(attfill.GetFillColor()))) {
      //Draw a box with a gradient fill and a shadow.
      //Ignore all fill styles and EBoxMode, use a gradient fill.
      TPoint polygon[4];
      polygon[0].fX = x1, polygon[0].fY = y1;
      polygon[1].fX = x2, polygon[1].fY = y1;
      polygon[2].fX = x2, polygon[2].fY = y2;
      polygon[3].fX = x1, polygon[3].fY = y2;

      Quartz::DrawPolygonWithGradientFill(ctx, gradient, CGSizeMake(drawable.fWidth, drawable.fHeight),
                                          4, polygon, kFALSE); //kFALSE == don't draw a shadow.
   } else {
      const bool isHollow = mode == kHollow || attfill.GetFillStyle() / 1000 == 2;

      //Note! Pattern index (and its address) MUST live
      //long enough to be valid at the point of Quartz::DrawBox call!
      unsigned patternIndex = 0;
      if (isHollow) {
         if (!Quartz::SetLineColor(ctx, attline.GetLineColor())) {
            Error("DrawBoxW", "Can not find color for index %d", int(attline.GetLineColor()));
            return;
         }
      } else {
         if (!Quartz::SetFillAreaParameters(ctx, &patternIndex, attfill)) {
            Error("DrawBoxW", "SetFillAreaParameters failed");
            return;
         }
      }
      Quartz::SetLineStyle(ctx, attline.GetLineStyle());
      Quartz::SetLineWidth(ctx, attline.GetLineWidth());
      Quartz::DrawBox(ctx, x1, y1, x2, y2, isHollow);
   }
}


//______________________________________________________________________________
void TGQuartz::DrawBox(Int_t x1, Int_t y1, Int_t x2, Int_t y2, EBoxMode mode)
{
   DrawBoxW(GetSelectedContext(), x1, y1, x2, y2, mode);
}

//______________________________________________________________________________
void TGQuartz::DrawFillAreaW(WinContext_t wctxt, Int_t n, TPoint *xy)
{
   //Comment from TVirtualX:

   // Draw a filled area through all points.
   // n         : number of points
   // xy        : array of points

   auto drawable0 = (NSObject<X11Drawable> * const) wctxt;
   if (!drawable0)
      return;

   if (n < 3)
      return;
   // No fill area with direct drawing
   if ([drawable0 isDirectDraw])
      return;

   auto &attfill = GetAttFill(wctxt);

   auto drawable = (NSObject<X11Drawable> * const) GetPixmapDrawable(drawable0, "DrawFillAreaW");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;

   //Convert points to bottom-left system:
   ConvertPointsROOTToCocoa(n, xy, fConvertedPoints, drawable);

   const Quartz::CGStateGuard ctxGuard(ctx);
   //AA flag is not a part of a state.
   const Quartz::CGAAStateGuard aaCtxGuard(ctx, fUseFAAA);

   if (drawable.fScaleFactor > 1.) {
      // The CTM will be restored by 'ctxGuard'.
      CGContextScaleCTM(ctx, 1. / drawable.fScaleFactor, 1. / drawable.fScaleFactor);
   }

   const TColor * const fillColor = gROOT->GetColor(attfill.GetFillColor());
   if (!fillColor) {
      Error("DrawFillAreaW", "Could not find TColor for index %d", attfill.GetFillColor());
      return;
   }

   if (const TColorGradient * const gradient = dynamic_cast<const TColorGradient *>(fillColor)) {
      Quartz::DrawPolygonWithGradientFill(ctx, gradient, CGSizeMake(drawable.fWidth, drawable.fHeight),
                                          n, &fConvertedPoints[0], kFALSE);//kFALSE == don't draw a shadow.
   } else {
      unsigned patternIndex = 0;
      if (!Quartz::SetFillAreaParameters(ctx, &patternIndex, attfill)) {
         Error("DrawFillAreaW", "SetFillAreaParameters failed");
         return;
      }

      // kFALSE - do not draw shadows.
      // last argument - fill style
      Quartz::DrawFillArea(ctx, n, &fConvertedPoints[0], kFALSE, attfill);
   }
}


//______________________________________________________________________________
void TGQuartz::DrawFillArea(Int_t n, TPoint *xy)
{
   DrawFillAreaW(GetSelectedContext(), n, xy);
}


//______________________________________________________________________________
void TGQuartz::DrawCellArray(Int_t /*x1*/, Int_t /*y1*/, Int_t /*x2*/, Int_t /*y2*/,
                             Int_t /*nx*/, Int_t /*ny*/, Int_t */*ic*/)
{
   //Noop.
}

//______________________________________________________________________________
void TGQuartz::DrawLineW(WinContext_t wctxt, Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   auto drawable0 = (NSObject<X11Drawable> * const) wctxt;
   if (!drawable0)
      return;

   if ([drawable0 isDirectDraw]) {
      if (!drawable0.fIsPixmap) {
         QuartzView * const view = (QuartzView *)fPimpl->GetWindow(drawable0.fID).fContentView;
         if (!view) {
             ::Warning("DrawLineW", "Invalid view/window for XOR-mode");
             return;
         }

         [view.fQuartzWindow addXorLine: view : x1 : y1 : x2 : y2];
      }

      return;
   }

   auto &attline = GetAttLine(wctxt);

   auto drawable = (NSObject<X11Drawable> * const) GetPixmapDrawable(drawable0, "DrawLineW");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);
   //AA flag is not a part of a state.
   const Quartz::CGAAStateGuard aaCtxGuard(ctx, fUseAA);

   if (!Quartz::SetLineColor(ctx, attline.GetLineColor())) {
      Error("DrawLineW", "Could not set line color for index %d", int(attline.GetLineColor()));
      return;
   }

   Quartz::SetLineStyle(ctx, attline.GetLineStyle());
   Quartz::SetLineWidth(ctx, attline.GetLineWidth());

   Quartz::DrawLine(ctx, x1, X11::LocalYROOTToCocoa(drawable, y1), x2,
                    X11::LocalYROOTToCocoa(drawable, y2));
}

//______________________________________________________________________________
void TGQuartz::DrawLine(Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   // Draw a line.
   // x1,y1        : begin of line
   // x2,y2        : end of line

   assert(fSelectedDrawable > fPimpl->GetRootWindowID() && "DrawLine, bad drawable is selected");

   DrawLineW(GetSelectedContext(), x1, y1, x2, y2);
}


//______________________________________________________________________________
void TGQuartz::DrawPolyLineW(WinContext_t wctxt, Int_t n, TPoint *xy)
{
   auto drawable0 = (NSObject<X11Drawable> * const) wctxt;
   if (!drawable0)
      return;

   //Some checks first.
   if ([drawable0 isDirectDraw])
      return;

   auto &attline = GetAttLine(wctxt);

   auto drawable = (NSObject<X11Drawable> * const) GetPixmapDrawable(drawable0, "DrawPolyLineW");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);
   //AA flag is not a part of a state.
   const Quartz::CGAAStateGuard aaCtxGuard(ctx, fUseAA);

   if (!Quartz::SetLineColor(ctx, attline.GetLineColor())) {
      Error("DrawPolyLineW", "Could not find TColor for index %d", attline.GetLineColor());
      return;
   }

   Quartz::SetLineStyle(ctx, attline.GetLineStyle());
   Quartz::SetLineWidth(ctx, attline.GetLineWidth());

   //Convert to bottom-left-corner system.
   ConvertPointsROOTToCocoa(n, xy, fConvertedPoints, drawable);

   if (drawable.fScaleFactor > 1.)
      CGContextScaleCTM(ctx, 1. / drawable.fScaleFactor, 1. / drawable.fScaleFactor);

   Quartz::DrawPolyLine(ctx, n, &fConvertedPoints[0]);

   // CTM (current transformation matrix) is restored by 'ctxGuard's dtor.
}

//______________________________________________________________________________
void TGQuartz::DrawPolyLine(Int_t n, TPoint *xy)
{
   //Comment from TVirtualX:
   // Draw a line through all points.
   // n         : number of points
   // xy        : list of points
   //End of comment.

   assert(fSelectedDrawable > fPimpl->GetRootWindowID() && "DrawPolyLine, bad drawable is selected");

   DrawPolyLineW(GetSelectedContext(), n, xy);
}

//______________________________________________________________________________
void TGQuartz::DrawLinesSegmentsW(WinContext_t wctxt, Int_t n, TPoint *xy)
{
   for(Int_t i = 0; i < 2*n; i += 2)
      DrawPolyLineW(wctxt, 2, &xy[i]);
}

//______________________________________________________________________________
void  TGQuartz::DrawPolyMarkerW(WinContext_t wctxt, Int_t n, TPoint *xy)
{
   auto drawable0 = (NSObject<X11Drawable> * const) wctxt;
   if (!drawable0)
      return;

   //Do some checks first.
   if ([drawable0 isDirectDraw])
      return;

   auto &attmark = GetAttMarker(wctxt);

   auto drawable = (NSObject<X11Drawable> * const) GetPixmapDrawable(drawable0, "DrawPolyMarkerW");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);
   //AA flag is not a part of a state.
   const Quartz::CGAAStateGuard aaCtxGuard(ctx, fUseAA);

   if (!Quartz::SetFillColor(ctx, attmark.GetMarkerColor())) {
      Error("DrawPolyMarker", "Could not find TColor for index %d", attmark.GetMarkerColor());
      return;
   }

   Quartz::SetLineColor(ctx, attmark.GetMarkerColor());//Can not fail (for coverity).
   Quartz::SetLineStyle(ctx, 1);
   Quartz::SetLineWidth(ctx, TMath::Max(1, Int_t(TAttMarker::GetMarkerLineWidth(attmark.GetMarkerStyle()))));

   ConvertPointsROOTToCocoa(n, xy, fConvertedPoints, drawable);

   if (drawable.fScaleFactor > 1.)
      CGContextScaleCTM(ctx, 1. / drawable.fScaleFactor, 1. / drawable.fScaleFactor);

   Style_t markerstyle = TAttMarker::GetMarkerStyleBase(attmark.GetMarkerStyle());

   // The fast pixel markers need to be treated separately
   if (markerstyle == 1 || markerstyle == 6 || markerstyle == 7) {
       CGContextSetLineJoin(ctx, kCGLineJoinMiter);
       CGContextSetLineCap(ctx, kCGLineCapButt);
   } else {
       CGContextSetLineJoin(ctx, kCGLineJoinRound);
       CGContextSetLineCap(ctx, kCGLineCapRound);
   }

   Float_t MarkerSizeReduced = GetMarkerSize() - TMath::Floor(TAttMarker::GetMarkerLineWidth(attmark.GetMarkerStyle())/2.)/4.;
   Quartz::DrawPolyMarker(ctx, n, &fConvertedPoints[0], MarkerSizeReduced * drawable.fScaleFactor, markerstyle);

   CGContextSetLineJoin(ctx, kCGLineJoinMiter);
   CGContextSetLineCap(ctx, kCGLineCapButt);
}

//______________________________________________________________________________
void TGQuartz::DrawPolyMarker(Int_t n, TPoint *xy)
{
   //Comment from TVirtualX:
   // Draw PolyMarker
   // n         : number of points
   // xy        : list of points
   //End of comment.

   DrawPolyMarkerW(GetSelectedContext(), n, xy);
}


//______________________________________________________________________________

std::vector<UniChar> quartz_get_greek_unicars(const char *text)
{
   //This is a hack. Correct way is to extract glyphs from symbol.ttf,
   //find correct mapping, place this glyphs. This requires manual layout though (?),
   //and as usually, I have to many things to do, may be, one day I'll fix text rendering also.
   //This hack work only on MacOSX 10.7.3, does not work on iOS and I'm not sure about future/previous
   //versions of MacOSX.
   std::vector<UniChar> unichars(std::strlen(text));
   for (std::size_t i = 0; i < unichars.size(); ++i)
      unichars[i] = 0xF000 + (unsigned char)text[i];
   return unichars;
}

//______________________________________________________________________________
void TGQuartz::DrawTextW(WinContext_t wctxt, Int_t x, Int_t y, Float_t /* angle */ , Float_t /* mgn */,
                         const char *text, ETextMode /* mode */)
{
   auto drawable0 = (NSObject<X11Drawable> * const) wctxt;
   if (!drawable0)
      return;

   if ([drawable0 isDirectDraw])
      return;

   if (!text || !text[0])//Can this ever happen? TPad::PaintText does not check this.
      return;

   auto &atttext = GetAttText(wctxt);

   if (atttext.GetTextSize() < 1.5)//Do not draw anything, or CoreText will create some small (but not of size 0 font).
      return;

   auto drawable = (NSObject<X11Drawable> * const) GetPixmapDrawable(drawable0, "DrawTextW");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);

   //Before any core text drawing operations, reset text matrix.
   CGContextSetTextMatrix(ctx, CGAffineTransformIdentity);

   try {
      if (CTFontRef currentFont = fPimpl->fFontManager.SelectFont(atttext.GetTextFont(), kScale * atttext.GetTextSize())) {
         const unsigned fontIndex = atttext.GetTextFont() / 10;
         if (fontIndex == 12 || fontIndex == 15) {
            //Greek and math symbols.
            auto unichars = quartz_get_greek_unicars(text);
            Quartz::TextLine ctLine(unichars, currentFont, atttext.GetTextColor());
            ctLine.DrawLine(ctx, x, X11::LocalYROOTToCocoa(drawable, y), atttext);
         } else {
            const Quartz::TextLine ctLine(text, currentFont, atttext.GetTextColor());
            ctLine.DrawLine(ctx, x, X11::LocalYROOTToCocoa(drawable, y), atttext);
         }
      }
   } catch (const std::exception &e) {
      Error("DrawTextW", "Exception from Quartz::TextLine: %s", e.what());
   }
}

//______________________________________________________________________________
void TGQuartz::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                        const char *text, ETextMode mode)
{
   DrawTextW(GetSelectedContext(), x, y, angle, mgn, text, mode);
}

//______________________________________________________________________________
void TGQuartz::DrawTextW(WinContext_t wctxt, Int_t x, Int_t y, Float_t angle, Float_t /* mgn */,
                         const wchar_t *text, ETextMode mode)
{
   if (!text || !*text)
      return;

   if (!TTFhandle::Init()) {
      Error("DrawTextW", "wchar_t string to draw, but TTF initialization failed");
      return;
   }

   auto drawable0 = (NSObject<X11Drawable> * const) wctxt;
   if (!drawable0)
      return;

   if ([drawable0 isDirectDraw])
      return;

   auto drawable = (NSObject<X11Drawable> * const) GetPixmapDrawable(drawable0, "DrawTextW");
   if (!drawable)
      return;


   //Do not draw anything, or CoreText will create some small (but not of size 0 font).
   auto &att = GetAttText(wctxt);

   if (att.GetTextSize() < 1.5)//Do not draw anything, or CoreText will create some small (but not of size 0 font).
      return;

   TTFhandle::SetSmoothing(kTRUE);

   TTFhandle ttf;

   ttf.SetTextFont(att.GetTextFont());
   ttf.SetTextSize(att.GetTextSize());
   ttf.SetRotationMatrix(angle);
   ttf.PrepareString(text);
   ttf.LayoutGlyphs();

   Int_t txalh = att.GetTextAlign() / 10;
   Int_t txalv = att.GetTextAlign() % 10;
   FT_Vector   align_vect;                 ///< alignment vector

   // const EAlign align = EAlign(fTextAlign);
   // vertical alignment
   if (txalv == 3) // align == kTLeft || align == kTCenter || align == kTRight)
      align_vect.y = ttf.GetAscent();
   else if (txalv == 2) //  if (align == kMLeft || align == kMCenter || align == kMRight) {
      align_vect.y = ttf.GetAscent() / 2;
   else
      align_vect.y = 0;

   // horizontal alignment
   if (txalh == 3) // align == kTRight || align == kMRight || align == kBRight) {
      align_vect.x = ttf.GetWidth();
   else if (txalh == 2) // (align == kTCenter || align == kMCenter || align == kBCenter) {
      align_vect.x = ttf.GetWidth() / 2;
   else
      align_vect.x = 0;

   FT_Vector_Transform(&align_vect, ttf.GetRotMatrix());
   //This shift is from the original code.
   align_vect.x = align_vect.x >> 6;
   align_vect.y = align_vect.y >> 6;


   //This code is a modified (for Quartz) version of TG11TTF text drawing

   QuartzPixmap *dstPixmap = nil;
   if ([drawable isKindOfClass : [QuartzPixmap class]])
      dstPixmap = (QuartzPixmap *)drawable;
   else if ([drawable isKindOfClass : [QuartzView class]] || [drawable isKindOfClass : [QuartzWindow class]])
      dstPixmap = ((NSObject<X11Window> *)drawable).fBackBuffer;

   if (!dstPixmap) {
      //I can not read pixels from a window (I can, but this is too slow and unreliable).
      Error("DrawTextW", "fSelectedDrawable is neither QuartzPixmap nor a double buffered window");
      return;
   }

   //Comment from TGX11TTF:
   // compute the size and position of the XImage that will contain the text
   const Int_t xOff = TMath::Max(0, (Int_t) -ttf.GetBox().xMin);
   const Int_t yOff = TMath::Max(0, (Int_t) -ttf.GetBox().yMin);

   const Int_t w = ttf.GetBox().xMax + xOff;
   const Int_t h = ttf.GetBox().yMax + yOff;

   // If w or h is 0, very likely the string is only blank characters
   if (w <= 0 || h <= 0)
      return;

   const Int_t x1 = x - xOff - align_vect.x;
   const Int_t y1 = y + yOff + align_vect.y - h;

   UInt_t width = 0, height = 0;
   Int_t xy = 0;

   GetWindowSize((Drawable_t) drawable0.fID, xy, xy, width, height);

   // If string falls outside window, there is probably no need to draw it.
   if (x1 + w <= 0 || x1 >= (Int_t)width || y1 + h <= 0 || y1 >= (Int_t)height)
      return;

   //By default, all pixels are set to 0 (all components, that's what code in TGX11TTF also does here).
   Util::NSScopeGuard<QuartzPixmap> pixmap([[QuartzPixmap alloc] initWithW : w H : h scaleFactor : 1.f]);
   if (!pixmap.Get()) {
      Error("DrawTextW", "pixmap creation failed");
      return;
   }

   const unsigned char defaultBackgroundPixel[] = {255, 255, 255, 255};
   Util::ScopedArray<unsigned char> arrayGuard;
   if (mode == kClear) {
      //For this mode, TGX11TTF does some work to: a) preserve pixels under symbols
      //b) calculate (interpolate) pixel for glyphs.

      X11::Rectangle bbox(x1, y1, w, h);
      //We already check IsVisible, so, in principle, bbox at least has intersection with
      //the current selected drawable.
      if (X11::AdjustCropArea(dstPixmap, bbox))
         arrayGuard.Reset([dstPixmap readColorBits : bbox]);

      if (!arrayGuard.Get()) {
         Error("DrawTextW", "problem with reading background pixels");
         return;
      }

      // This is copy & paste from TGX11TTF:
      const Int_t xo = x1 < 0 ? -x1 : 0;
      const Int_t yo = y1 < 0 ? -y1 : 0;

      for (int yp = 0; yp < int(bbox.fHeight) && yo + yp < h; ++yp) {
         const unsigned char *srcBase = arrayGuard.Get() + bbox.fWidth * yp * 4;
         for (int xp = 0; xp < int(bbox.fWidth) && xo + xp < w; ++xp) {
            const unsigned char * const pixel = srcBase + xp * 4;
            [pixmap.Get() putPixel : pixel X : xo + xp Y : yo + yp];
         }
      }
   } else {
      //Find background color and set for all pixels.
      [pixmap.Get() addPixel : defaultBackgroundPixel];
   }

   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);

   CGContextSetRGBStrokeColor(ctx, 0., 0., 1., 1.);
   // paint the glyphs in the pixmap.
   for (UInt_t n = 0; n < ttf.GetNumGlyphs(); ++n) {
      if (auto bitmap = ttf.GetGlyphBitmap(n)) {
         const Int_t bx = bitmap->left + xOff;
         const Int_t by = h - bitmap->top - yOff;

         DrawFTGlyph(pixmap.Get(), &bitmap->bitmap, TGCocoa::GetPixel(att.GetTextColor()),
                     mode == kClear ? ULong_t(-1) : 0xffffff, bx, by);
      }
   }

   const X11::Rectangle copyArea(0, 0, w, h);
   const X11::Point dstPoint(x1, y1);
   [dstPixmap copy : pixmap.Get() area : copyArea withMask : nil clipOrigin : X11::Point() toPoint : dstPoint];
}

//______________________________________________________________________________
void TGQuartz::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                        const wchar_t *text, ETextMode mode)
{
   DrawTextW(GetSelectedContext(), x, y, angle, mgn, text, mode);
}

//______________________________________________________________________________
void TGQuartz::GetTextExtent(UInt_t &w, UInt_t &h, char *text)
{
   // Returns the size of the specified character string "mess".
   //
   // w    - the text width
   // h    - the text height
   // text - the string

   w = h = 0;

   if (!text || !*text)
      return;

   auto fontref = fPimpl->fFontManager.SelectFont(GetTextFont(), kScale*GetTextSize());
   if (!fontref)
      return;
   const unsigned fontIndex = GetTextFont() / 10;
   if (fontIndex == 12 || fontIndex == 15) {
      //Greek and math symbols.
      auto unichars = quartz_get_greek_unicars(text);
      fPimpl->fFontManager.GetTextBounds(fontref, w, h, unichars);
   } else {
      fPimpl->fFontManager.GetTextBounds(fontref, w, h, text);
   }
}

//______________________________________________________________________________
Bool_t TGQuartz::GetTextExtentA(Font_t font, Double_t size, UInt_t &w, UInt_t &h, const char *text)
{
   if (!text || !*text) {
      w = h = 0;
      return kTRUE;
   }

   auto fontref = fPimpl->fFontManager.SelectFont(font, kScale * size);
   if (!fontref)
      return kFALSE;

   const unsigned fontIndex = font / 10;
   if (fontIndex == 12 || fontIndex == 15) {
      //Greek and math symbols.
      auto unichars = quartz_get_greek_unicars(text);
      fPimpl->fFontManager.GetTextBounds(fontref, w, h, unichars);
   } else {
      fPimpl->fFontManager.GetTextBounds(fontref, w, h, text);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGQuartz::GetTextExtentA(Font_t, Double_t, UInt_t &w, UInt_t &h, const wchar_t *)
{
   // do not handle wchar, pad painter will switch to TTF
   w = h = 0;
   return kFALSE;
}


//______________________________________________________________________________
Int_t TGQuartz::GetFontAscent() const
{
   // Returns the ascent of the current font (in pixels).
   // The ascent of a font is the distance from the baseline
   // to the highest position characters extend to.
   if (auto fontref = fPimpl->fFontManager.SelectFont(GetTextFont(), kScale*GetTextSize()))
      return Int_t(fPimpl->fFontManager.GetAscent(fontref));

   return 0;
}

//______________________________________________________________________________
Int_t TGQuartz::GetFontAscent(const char *text) const
{
   // Returns the ascent of the current font (in pixels).
   // The ascent of a font is the distance from the baseline
   // to the highest position characters extend to.

   //In case of any problem we can always resort to the old version:
   if (!text || !*text)
      return GetFontAscent();

   if (auto fontref = fPimpl->fFontManager.SelectFont(GetTextFont(), kScale*GetTextSize())) {
      const unsigned fontIndex = GetTextFont() / 10;
      if (fontIndex == 12 || fontIndex == 15) {
         //Greek and math symbols.
         auto unichars = quartz_get_greek_unicars(text);
         return Int_t(fPimpl->fFontManager.GetAscent(fontref, unichars));
      } else
         return Int_t(fPimpl->fFontManager.GetAscent(fontref, text));
   }

   return 0;
}

//______________________________________________________________________________
Int_t TGQuartz::GetFontDescent() const
{
   // Returns the descent of the current font (in pixels.
   // The descent is the distance from the base line
   // to the lowest point characters extend to.
   if (auto fontref = fPimpl->fFontManager.SelectFont(GetTextFont(), kScale*GetTextSize()))
      return Int_t(fPimpl->fFontManager.GetDescent(fontref));

   return 0;
}

//______________________________________________________________________________
Int_t TGQuartz::GetFontDescent(const char *text) const
{
   // Returns the descent of the current font (in pixels.
   // The descent is the distance from the base line
   // to the lowest point characters extend to.

   //That's how it's tested in ROOT:
   if (!text || !*text)
      return GetFontDescent();

   if (auto fontref = fPimpl->fFontManager.SelectFont(GetTextFont(), kScale*GetTextSize())) {
      const unsigned fontIndex = GetTextFont() / 10;
      if (fontIndex == 12 || fontIndex == 15) {
         //Greek and math symbols.
         auto unichars = quartz_get_greek_unicars(text);
         return Int_t(fPimpl->fFontManager.GetDescent(fontref, unichars));
      } else
         return Int_t(fPimpl->fFontManager.GetDescent(fontref, text));
   }

   return 0;
}

//______________________________________________________________________________
Bool_t TGQuartz::GetFontAscentDescent(Font_t font, Double_t size, UInt_t &a, UInt_t &d, const char *text)
{
   a = d = 0;

   auto fontref = fPimpl->fFontManager.SelectFont(font, kScale * size);
   if (!fontref)
      return kFALSE;

   const unsigned fontIndex = font / 10;
   if (!text || !*text) {
      a = fPimpl->fFontManager.GetAscent(fontref);
      d = fPimpl->fFontManager.GetDescent(fontref);
   } else if (fontIndex == 12 || fontIndex == 15) {
      //Greek and math symbols.
      auto unichars = quartz_get_greek_unicars(text);
      a = fPimpl->fFontManager.GetAscent(fontref, unichars);
      d = fPimpl->fFontManager.GetDescent(fontref, unichars);
   } else {
      a = fPimpl->fFontManager.GetAscent(fontref, text);
      d = fPimpl->fFontManager.GetDescent(fontref, text);
   }

   return kTRUE;
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

   SetAttLine(GetSelectedContext(), *this);
}


//______________________________________________________________________________
void TGQuartz::SetLineStyle(Style_t lstyle)
{
   // Set line style.
   TAttLine::SetLineStyle(lstyle);

   SetAttLine(GetSelectedContext(), *this);
}


//______________________________________________________________________________
void TGQuartz::SetLineWidth(Width_t width)
{
   // Set the line width.

   TAttLine::SetLineWidth(width);

   SetAttLine(GetSelectedContext(), *this);
}


//______________________________________________________________________________
void TGQuartz::SetFillColor(Color_t cindex)
{
   // Set color index "cindex" for fill areas.

   TAttFill::SetFillColor(cindex);

   SetAttFill(GetSelectedContext(), *this);
}


//______________________________________________________________________________
void TGQuartz::SetFillStyle(Style_t style)
{
   // Set fill area style.
   TAttFill::SetFillStyle(style);

   SetAttFill(GetSelectedContext(), *this);
}


//______________________________________________________________________________
void TGQuartz::SetMarkerColor(Color_t cindex)
{
   // Set color index "cindex" for markers.
   TAttMarker::SetMarkerColor(cindex);

   SetAttMarker(GetSelectedContext(), *this);
}


//______________________________________________________________________________
void TGQuartz::SetMarkerSize(Float_t markersize)
{
   // Set marker size index.
   //
   // markersize - the marker scale factor
   TAttMarker::SetMarkerSize(markersize);

   SetAttMarker(GetSelectedContext(), *this);
}


//______________________________________________________________________________
void TGQuartz::SetMarkerStyle(Style_t markerstyle)
{
   // Set marker style.

   TAttMarker::SetMarkerStyle(markerstyle);

   SetAttMarker(GetSelectedContext(), *this);
}


//______________________________________________________________________________
void TGQuartz::SetTextAlign(Short_t talign)
{
   // Set the text alignment.

   TAttText::SetTextAlign(talign);

   SetAttText(GetSelectedContext(), *this);
}

//______________________________________________________________________________
void TGQuartz::SetTextColor(Color_t cindex)
{
   // Set the color index "cindex" for text.

   TAttText::SetTextColor(cindex);

   SetAttText(GetSelectedContext(), *this);
}


//______________________________________________________________________________
void TGQuartz::SetTextFont(Font_t fontNumber)
{
   // Set the current text font number.

   TAttText::SetTextFont(fontNumber);

   SetAttText(GetSelectedContext(), *this);
}

//______________________________________________________________________________
Int_t TGQuartz::SetTextFont(char * /* fontName */, ETextSetMode /* mode */)
{
   Error("SetTextFont", "Direct TTF font setting not supported");
   return 1;
}

//______________________________________________________________________________
void TGQuartz::SetTextSize(Float_t textsize)
{
   // Set the current text size to "textsize"

   TAttText::SetTextSize(textsize);

   SetAttText(GetSelectedContext(), *this);
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
void TGQuartz::SetOpacityW(WinContext_t /* wctxt */, Int_t /* percent */)
{
}

//______________________________________________________________________________
void TGQuartz::SetAttFill(WinContext_t wctxt, const TAttFill &att)
{
   att.Copy(GetAttFill(wctxt));

   // TODO: remove this after transition done
   TAttFill::SetFillColor(att.GetFillColor());
   TAttFill::SetFillStyle(att.GetFillStyle());
}

//______________________________________________________________________________
void TGQuartz::SetAttLine(WinContext_t wctxt, const TAttLine &att)
{
   att.Copy(GetAttLine(wctxt));

   // TODO: remove this after transition done
   TAttLine::SetLineColor(att.GetLineColor());
   TAttLine::SetLineStyle(att.GetLineStyle());
   TAttLine::SetLineWidth(att.GetLineWidth());
}

//______________________________________________________________________________
void TGQuartz::SetAttMarker(WinContext_t wctxt, const TAttMarker &att)
{
   att.Copy(GetAttMarker(wctxt));

   // TODO: remove this after transition done
   TAttMarker::SetMarkerColor(att.GetMarkerColor());
   TAttMarker::SetMarkerSize(att.GetMarkerSize());
   TAttMarker::SetMarkerStyle(att.GetMarkerStyle());
}

//______________________________________________________________________________
void TGQuartz::SetAttText(WinContext_t wctxt, const TAttText &att)
{
   att.Copy(GetAttText(wctxt));

   // TODO: remove this after transition done
   TAttText::SetTextAlign(att.GetTextAlign());
   TAttText::SetTextAngle(att.GetTextAngle());
   TAttText::SetTextColor(att.GetTextColor());
   TAttText::SetTextSize(att.GetTextSize());
   TAttText::SetTextFont(att.GetTextFont());
}

//TTF related part.

//______________________________________________________________________________
void TGQuartz::DrawFTGlyph(void *_pixmap, void *_source, ULong_t fore, ULong_t back, Int_t bx, Int_t by)
{
   //This function is a "remake" of TGX11FFT::DrawImage.

   //I'm using this code to reproduce the same text as generated by TGX11TTF.
   //It's quite sloppy, as in original version. I tried to make it not so ugly and
   //more or less readable.

   auto pixmap = (QuartzPixmap *)_pixmap;
   auto source = (FT_Bitmap *) _source;
   assert(pixmap != nil && "DrawFTGlyph, pixmap parameter is nil");
   assert(source != nil && "DrawFTGlyph, source parameter is null");

   if (TTFhandle::GetSmoothing()) {
      ColorStruct_t col[5];
      // background kClear, i.e. transparent, we take as background color
      // the average of the rgb values of all pixels covered by this character
      if (back == ULong_t(-1)) {
         const UInt_t maxDots = TMath::Min((UInt_t) 50000, source->width * source->rows);

         //In original code, they first have to extract
         //pixels and call XQueryColors.
         //I have only one loop here.
         ULong_t r = 0, g = 0, b = 0;
         UInt_t dotCnt = 0;
         for (unsigned y = 0; y < source->rows; y++) {
            for (unsigned x = 0; x < source->width; x++) {
               if (x + bx < pixmap.fWidth && y + by < pixmap.fHeight) {
                  const unsigned char * const pixels = pixmap.fData + (y + by) * pixmap.fWidth * 4 + (x + bx) * 4;
                  r += UShort_t(pixels[0] / 255. * 0xffff);
                  g += UShort_t(pixels[1] / 255. * 0xffff);
                  b += UShort_t(pixels[2] / 255. * 0xffff);
                  if (++dotCnt >= maxDots)
                     break;
               }
            }
         }

         if (dotCnt > 0) {
            r /= dotCnt;
            g /= dotCnt;
            b /= dotCnt;
         }

         col[0].fRed = (UShort_t) r;
         col[0].fGreen = (UShort_t) g;
         col[0].fBlue = (UShort_t) b;
      } else {
         // request background color
         col[0].fPixel = back;
         TGCocoa::QueryColor(kNone, col[0]);
      }

      // request foreground color
      col[4].fPixel = fore;
      TGCocoa::QueryColor(kNone, col[4]);//calculate fRed/fGreen/fBlue triple from fPixel.

      // interpolate between fore and background colors
      for (int x = 3; x > 0; --x) {
         col[x].fRed   = (col[4].fRed   * x + col[0].fRed   * (4 - x)) / 4;
         col[x].fGreen = (col[4].fGreen * x + col[0].fGreen * (4 - x)) / 4;
         col[x].fBlue  = (col[4].fBlue  * x + col[0].fBlue  * (4 - x)) / 4;
         TGCocoa::AllocColor(kNone, col[x]);//Calculate fPixel from fRed/fGreen/fBlue triplet.
      }

      // put smoothed character, character pixmap values are an index
      // into the 5 colors used for aliasing (4 = foreground, 0 = background)
      const unsigned char *s = source->buffer;
      for (unsigned y = 0; y < source->rows; ++y) {
         for (unsigned x = 0; x < source->width; ++x) {
            unsigned char d = (((*s++ & 0xff) + 10) * 5) / 256;
            if (d > 4)
               d = 4;
            if (d > 0) {
               const UChar_t pixel[] = {UChar_t(double(col[d].fRed) / 0xffff * 255),
                                        UChar_t(double(col[d].fGreen) / 0xffff * 255),
                                        UChar_t(double(col[d].fBlue) / 0xffff * 255), 255};
               [pixmap putPixel : pixel X : bx + x Y : by + y];
            }
         }
      }
   } else {
      // no smoothing, just put character using foreground color
      unsigned char rgba[4] = {};
      rgba[3] = 255;
      X11::PixelToRGB(fore, rgba);
      unsigned char d = 0;

      const unsigned char *row = source->buffer;
      for (unsigned y = 0; y < source->rows; ++y) {
         unsigned n = 0;
         const unsigned char *s = row;
         for (unsigned x = 0; x < source->width; ++x) {
            if (!n)
               d = *s++;

            if (TESTBIT(d,7 - n))
               [pixmap putPixel : rgba X : bx + x Y : by + y];

            if (++n == kBitsPerByte)
               n = 0;
         }

         row += source->pitch;
      }
   }
}

//Aux. functions.

//______________________________________________________________________________
void TGQuartz::SetAA()
{
   if (gEnv) {
      const TString value(TString(gEnv->GetValue("Cocoa.EnableAntiAliasing", "auto")).Strip());
      if (value == "auto") {
         [[NSScreen mainScreen] backingScaleFactor] > 1. ? fUseAA = true : fUseAA = false;
      } else if (value == "no")
         fUseAA = false;
      else {
         assert(value == "yes" && "SetAA, value must be 'yes', 'no' or 'auto'");
         fUseAA = true;
      }
      const TString valuefa(TString(gEnv->GetValue("Cocoa.EnableFillAreaAntiAliasing", "auto")).Strip());
      if (valuefa == "auto") {
         [[NSScreen mainScreen] backingScaleFactor] > 1. ? fUseFAAA = true : fUseFAAA = false;
      } else if (valuefa == "no")
         fUseFAAA = false;
      else {
         assert(valuefa == "yes" && "SetAA, value must be 'yes', 'no' or 'auto'");
         fUseFAAA = true;
      }
   }
}

//______________________________________________________________________________
TAttFill &TGQuartz::GetAttFill(WinContext_t wctxt)
{
   // attributes stored in direct drawable (view) and not in underlying pixmap
   auto drawable = (NSObject<X11Drawable> *) wctxt;
   if (!drawable || !drawable.attFill)
      return *this;
   return *drawable.attFill;
}

//______________________________________________________________________________
TAttLine &TGQuartz::GetAttLine(WinContext_t wctxt)
{
   // attributes stored in direct drawable (view) and not in underlying pixmap
   auto drawable = (NSObject<X11Drawable> *) wctxt;
   if (!drawable || !drawable.attLine)
      return *this;
   return *drawable.attLine;
}

//______________________________________________________________________________
TAttMarker &TGQuartz::GetAttMarker(WinContext_t wctxt)
{
   // attributes stored in direct drawable (view) and not in underlying pixmap
   auto drawable = (NSObject<X11Drawable> *) wctxt;
   if (!drawable || !drawable.attMarker)
      return *this;
   return *drawable.attMarker;
}

//______________________________________________________________________________
TAttText &TGQuartz::GetAttText(WinContext_t wctxt)
{
   // attributes stored in direct drawable (view) and not in underlying pixmap
   auto drawable = (NSObject<X11Drawable> *) wctxt;
   if (!drawable || !drawable.attText)
      return *this;
   return *drawable.attText;
}

//______________________________________________________________________________
void *TGQuartz::GetPixmapDrawable(void *drawable0, const char *calledFrom) const
{
   assert(calledFrom != 0 && "GetDrawableChecked, calledFrom parameter is null");

   if (!drawable0)
      return nullptr;

   auto drawable = (NSObject<X11Drawable> *) drawable0;
   if (!drawable.fIsPixmap) {
      //TPad/TCanvas ALWAYS draw only into a pixmap.
      if ([drawable isKindOfClass : [QuartzView class]]) {
         QuartzView *view = (QuartzView *)drawable;
         if (!view.fBackBuffer) {
            Error(calledFrom, "Selected window is not double buffered");
            return nullptr;
         }

         drawable = view.fBackBuffer;
      } else {
         Error(calledFrom, "Selected drawable is neither a pixmap, nor a double buffered window");
         return nullptr;
      }
   }

   if (!drawable.fContext) {
      Error(calledFrom, "Context is null");
      return nullptr;
   }

   return drawable;
}
