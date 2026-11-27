// @(#)root/cocoa:$Id$
// Author: Sergey Linev   25/11/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "QuartzDrawable.h"

#include "QuartzWindow.h"
#include "QuartzPixmap.h"
#include "QuartzUtils.h"
#include "QuartzFillArea.h"
#include "QuartzText.h"
#include "QuartzLine.h"
#include "QuartzMarker.h"
#include "CocoaPrivate.h"
#include "CocoaGuiTypes.h"
#include "TGCocoa.h"
#include "X11Buffer.h"

#include "TROOT.h"
#include "TColorGradient.h"
#include "TTFhandle.h"
#include "TError.h"
#include "TMath.h"

#include <Cocoa/Cocoa.h>

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


////////////////////////////////////////////////////////////////////////////////
/// Constructor

QuartzDrawable::QuartzDrawable(Int_t id, TGCocoa &cocoa, NSObject<X11Drawable> *drawable)
   : XDrawable(id), fCocoa(cocoa), fDrawable(drawable)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

QuartzDrawable::~QuartzDrawable()
{
}

////////////////////////////////////////////////////////////////////////////////
/// test if drawable is empty

Bool_t QuartzDrawable::IsNull() const
{
   return !fDrawable;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove pointer on drawable object

void QuartzDrawable::ClearDrawable()
{
   fDrawable = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Set fill attributes

void QuartzDrawable::SetAttFill(const TAttFill &att)
{
   fAttFill = att;
}

////////////////////////////////////////////////////////////////////////////////
/// Set line attributes

void QuartzDrawable::SetAttLine(const TAttLine &att)
{
   fAttLine = att;
}

////////////////////////////////////////////////////////////////////////////////
/// Set marker attributes for specified window

void QuartzDrawable::SetAttMarker(const TAttMarker &att)
{
   fAttMarker = att;
}

////////////////////////////////////////////////////////////////////////////////
/// Set text attributes for specified window

void QuartzDrawable::SetAttText(const TAttText &att)
{
   fAttText = att;
}

////////////////////////////////////////////////////////////////////////////////
/// Set window draw mode

void QuartzDrawable::SetDrawMode(TVirtualX::EDrawMode mode)
{
   fDrawMode = mode;

   // here XOR window and XOR operations can be removed if necessary
   if (fDrawable && [fDrawable isKindOfClass : [QuartzWindow class]])
      [fDrawable setDrawMode : mode];
}


///////////////////////////////////////////////////////////////////////////////
/// Retrun draw mode

TVirtualX::EDrawMode QuartzDrawable::GetDrawMode()
{
   if (fDrawable && [fDrawable isKindOfClass : [QuartzWindow class]])
      return [fDrawable getDrawMode];

   return fDrawMode;
}


////////////////////////////////////////////////////////////////////////////////
/// Clear window

void QuartzDrawable::ClearWindow()
{
   if (!fDrawable)
      return;

   if (fDrawable.fIsPixmap) {
      //Pixmaps are white by default.
      //This is bad - we can not have transparent sub-pads (in TCanvas)
      //because of this. But there is no way how gVirtualX can
      //obtain real pad's color and check for its transparency.
      CGContextRef pixmapCtx = fDrawable.fContext;
      assert(pixmapCtx != 0 && "ClearWindow, pixmap's context is null");
      //const Quartz::CGStateGuard ctxGuard(pixmapCtx);
      //CGContextSetRGBFillColor(pixmapCtx, 1., 1., 1., 1.);
      //CGContextFillRect(pixmapCtx, CGRectMake(0, 0, drawable.fWidth, drawable.fHeight));
      //Now we really clear!
      CGContextClearRect(pixmapCtx, CGRectMake(0, 0, fDrawable.fWidth, fDrawable.fHeight));
   } else {
      //For a window ClearArea with w == 0 and h == 0 means the whole window.
      fCocoa.ClearArea(fDrawable.fID, 0, 0, 0, 0);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Close window
/// Remove all references to actual window implementation

void QuartzDrawable::CloseWindow()
{
   if (!fDrawable)
      return;
   auto id = fDrawable.fID;
   fDrawable = nullptr;

   fCocoa.DestroyWindow(id);
}

////////////////////////////////////////////////////////////////////////////////
/// Update display for specified window.
///
///  \param [in] mode : (1) update (0) sync
///
/// Synchronise client and server once (not permanent).
/// Copy the pixmap gCws->fDrawing on the window gCws->fWindow
/// if the double buffer is on.

void QuartzDrawable::UpdateWindow(Int_t /* mode */)
{
   //Have no idea, why this can happen with ROOT - done by TGDNDManager :(
   if (!fDrawable || fDrawable.fIsPixmap == YES)
      return;

   auto window = (NSObject<X11Window> * const) fDrawable;

   if (QuartzPixmap * const pixmap = window.fBackBuffer) {
      assert([window.fContentView isKindOfClass : [QuartzView class]] && "UpdateWindow, content view is not a QuartzView");
      QuartzView *dstView = (QuartzView *)window.fContentView;

      if (dstView.fIsOverlapped)
         return;

      if (dstView.fContext) {
         //We can draw directly.
         const X11::Rectangle copyArea(0, 0, pixmap.fWidth, pixmap.fHeight);
         [dstView copy : pixmap area : copyArea withMask : nil clipOrigin : X11::Point() toPoint : X11::Point()];
      } else {
         //Have to wait.
         fCocoa.fPimpl->fX11CommandBuffer.AddUpdateWindow(dstView);
         fCocoa.Update(1);
      }
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Set opacity. This image manipulation routine works
/// by adding to a percent amount of neutral to each pixels RGB.
/// Since it requires quite some additional color map entries is it
/// only supported on displays with more than > 8 color planes (> 256
/// colors).

void QuartzDrawable::SetOpacity(Int_t /* percent */)
{
   // not necessary for Quartz
}

////////////////////////////////////////////////////////////////////////////////
/// Copy the pixmap wid at the position xpos, ypos in the specified window.

void QuartzDrawable::CopyPixmap(Int_t pixmapID, Int_t xpos, Int_t ypos)
{
   if (!fDrawable)
      return;

   assert(pixmapID > (Int_t)fCocoa.fPimpl->GetRootWindowID() &&
          "CopyPixmap, parameter 'pixmapID' is not a valid id");

   auto source = fCocoa.fPimpl->GetDrawable(pixmapID);
   assert([source isKindOfClass : [QuartzPixmap class]] &&
          "CopyPixmap, source is not a pixmap");
   auto pixmap = (QuartzPixmap *)source;

   NSObject<X11Drawable> * destination = nil;

   if (fDrawable.fIsPixmap) {
      destination = fDrawable;
   } else {
      auto window = (NSObject<X11Window> * const) fDrawable;
      if (window.fBackBuffer) {
         destination = window.fBackBuffer;
      } else {
         ::Warning("CopyPixmap", "Operation skipped, since destination"
                                 " window is not double buffered");
         return;
      }
   }

   const X11::Rectangle copyArea(0, 0, pixmap.fWidth, pixmap.fHeight);
   const X11::Point dstPoint(xpos, ypos);

   [destination copy : pixmap area : copyArea withMask : nil clipOrigin : X11::Point() toPoint : dstPoint];

}

////////////////////////////////////////////////////////////////////////////////
/// Return pixmap drawable

NSObject<X11Drawable> *QuartzDrawable::GetPixmapDrawable(const char *calledFrom) const
{
   assert(calledFrom != 0 && "GetPixmapDrawable, calledFrom parameter is null");

   if (!fDrawable)
      return nullptr;

   auto drawable = fDrawable;

   if (!drawable.fIsPixmap) {
      //TPad/TCanvas ALWAYS draw only into a pixmap.
      if ([drawable isKindOfClass : [QuartzView class]]) {
         QuartzView *view = (QuartzView *)drawable;
         if (!view.fBackBuffer) {
            ::Error(calledFrom, "Selected window is not double buffered");
            return nullptr;
         }

         drawable = view.fBackBuffer;
      } else {
         ::Error(calledFrom, "Selected drawable is neither a pixmap, nor a double buffered window");
         return nullptr;
      }
   }

   if (!drawable.fContext) {
      ::Error(calledFrom, "Context is null");
      return nullptr;
   }

   return drawable;
}

////////////////////////////////////////////////////////////////////////////////
/// Return view object which can be used for direct draw xor operations
/// Required only for support of old API, remove in ROOT7

QuartzView *QuartzDrawable::GetDirectView(const char *calledFrom) const
{
   QuartzView *view = nullptr;
   if (!fDrawable.fIsPixmap) {
      view = (QuartzView *)fCocoa.fPimpl->GetWindow(fDrawable.fID).fContentView;
      if (!view)
         ::Warning(calledFrom, "Invalid view/window for XOR-mode");
   }
   return view;
}


////////////////////////////////////////////////////////////////////////////////
/// Draw box

void QuartzDrawable::DrawBox(Int_t x1, Int_t y1, Int_t x2, Int_t y2, TVirtualX::EBoxMode mode)
{
   if (!fDrawable)
      return;

   //Check some conditions first.
   if ([fDrawable isDirectDraw]) {
      if (auto view = GetDirectView("DrawBox"))
         [view.fQuartzWindow addXorBox: view : x1 : y1 : x2 : y2];
      return;
   }

   auto drawable = GetPixmapDrawable("DrawBox");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);
   //AA flag is not a part of a state.
   const Quartz::CGAAStateGuard aaCtxGuard(ctx, fCocoa.fUseAA);

   //Go to low-left-corner system.
   y1 = Int_t(X11::LocalYROOTToCocoa(drawable, y1));
   y2 = Int_t(X11::LocalYROOTToCocoa(drawable, y2));

   if (const TColorGradient * const gradient = dynamic_cast<TColorGradient *>(gROOT->GetColor(fAttFill.GetFillColor()))) {
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
      const bool isHollow = mode == TVirtualX::kHollow || fAttFill.GetFillStyle() / 1000 == 2;

      //Note! Pattern index (and its address) MUST live
      //long enough to be valid at the point of Quartz::DrawBox call!
      unsigned patternIndex = 0;
      if (isHollow) {
         if (!Quartz::SetLineColor(ctx, fAttLine.GetLineColor())) {
            ::Error("DrawBox", "Can not find color for index %d", int(fAttLine.GetLineColor()));
            return;
         }
      } else {
         if (!Quartz::SetFillAreaParameters(ctx, &patternIndex, fAttFill)) {
            ::Error("DrawBox", "SetFillAreaParameters failed");
            return;
         }
      }
      Quartz::SetLineStyle(ctx, fAttLine.GetLineStyle());
      Quartz::SetLineWidth(ctx, fAttLine.GetLineWidth());
      Quartz::DrawBox(ctx, x1, y1, x2, y2, isHollow);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill area described by polygon on specified window
///
///  \param [in] n     number of points
///  \param [in] xy   list of points

void QuartzDrawable::DrawFillArea(Int_t n, TPoint *xy)
{
   if (!fDrawable || (n < 3))
      return;
   // No fill area with direct drawing
   if ([fDrawable isDirectDraw])
      return;

   auto drawable = GetPixmapDrawable("DrawFillArea");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;

   std::vector<TPoint> pnts;

   //Convert points to bottom-left system:
   ConvertPointsROOTToCocoa(n, xy, pnts, drawable);

   const Quartz::CGStateGuard ctxGuard(ctx);
   //AA flag is not a part of a state.
   const Quartz::CGAAStateGuard aaCtxGuard(ctx, fCocoa.fUseFAAA);

   if (drawable.fScaleFactor > 1.) {
      // The CTM will be restored by 'ctxGuard'.
      CGContextScaleCTM(ctx, 1. / drawable.fScaleFactor, 1. / drawable.fScaleFactor);
   }

   const TColor * const fillColor = gROOT->GetColor(fAttFill.GetFillColor());
   if (!fillColor) {
      Error("DrawFillAreaW", "Could not find TColor for index %d", fAttFill.GetFillColor());
      return;
   }

   if (const TColorGradient * const gradient = dynamic_cast<const TColorGradient *>(fillColor)) {
      Quartz::DrawPolygonWithGradientFill(ctx, gradient, CGSizeMake(drawable.fWidth, drawable.fHeight),
                                          pnts.size(), pnts.data(), kFALSE);//kFALSE == don't draw a shadow.
   } else {
      unsigned patternIndex = 0;
      if (!Quartz::SetFillAreaParameters(ctx, &patternIndex, fAttFill)) {
         Error("DrawFillAreaW", "SetFillAreaParameters failed");
         return;
      }

      // kFALSE - do not draw shadows.
      // last argument - fill style
      Quartz::DrawFillArea(ctx, pnts.size(), pnts.data(), kFALSE, fAttFill);
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Draw a line
///
///  \param [in] x1,y1        : begin of line
///  \param [in] x2,y2        : end of line

void QuartzDrawable::DrawLine(Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   if (!fDrawable)
      return;

   if ([fDrawable isDirectDraw]) {
      if (auto view = GetDirectView("DrawLine"))
         [view.fQuartzWindow addXorLine: view : x1 : y1 : x2 : y2];
      return;
   }

   auto drawable = GetPixmapDrawable("DrawLine");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);
   //AA flag is not a part of a state.
   const Quartz::CGAAStateGuard aaCtxGuard(ctx, fCocoa.fUseAA);

   if (!Quartz::SetLineColor(ctx, fAttLine.GetLineColor())) {
      ::Error("DrawLine", "Could not set line color for index %d", int(fAttLine.GetLineColor()));
      return;
   }

   Quartz::SetLineStyle(ctx, fAttLine.GetLineStyle());
   Quartz::SetLineWidth(ctx, fAttLine.GetLineWidth());

   Quartz::DrawLine(ctx, x1, X11::LocalYROOTToCocoa(drawable, y1), x2,
                    X11::LocalYROOTToCocoa(drawable, y2));

}


////////////////////////////////////////////////////////////////////////////////
/// Draw a line through all points
///
///  \param [in] n     number of points
///  \param [in] xy   list of points

void QuartzDrawable::DrawPolyLine(Int_t n, TPoint *xy)
{
   if (!fDrawable)
      return;

   //Some checks first.
   if ([fDrawable isDirectDraw]) {
      if (auto view = GetDirectView("DrawPolyLine"))
         [view.fQuartzWindow addXorPolyLine: view : n : xy : fAttLine ];
      return;
   }

   auto drawable = GetPixmapDrawable("DrawPolyLine");
   if (!drawable)
      return;

   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);
   //AA flag is not a part of a state.
   const Quartz::CGAAStateGuard aaCtxGuard(ctx, fCocoa.fUseAA);

   if (!Quartz::SetLineColor(ctx, fAttLine.GetLineColor())) {
      ::Error("DrawPolyLine", "Could not find TColor for index %d", fAttLine.GetLineColor());
      return;
   }

   Quartz::SetLineStyle(ctx, fAttLine.GetLineStyle());
   Quartz::SetLineWidth(ctx, fAttLine.GetLineWidth());

   std::vector<TPoint> pnts;

   //Convert to bottom-left-corner system.
   ConvertPointsROOTToCocoa(n, xy, pnts, drawable);

   if (drawable.fScaleFactor > 1.)
      CGContextScaleCTM(ctx, 1. / drawable.fScaleFactor, 1. / drawable.fScaleFactor);

   Quartz::DrawPolyLine(ctx, pnts.size(), pnts.data());

   // CTM (current transformation matrix) is restored by 'ctxGuard's dtor.
}

////////////////////////////////////////////////////////////////////////////////
/// Draws N segments between provided points
///
/// \param [in] n    number of segements
/// \param [in] xy   list of points, size 2*n

void QuartzDrawable::DrawLinesSegments(Int_t n, TPoint *xy)
{
   for(Int_t i = 0; i < 2*n; i += 2)
      DrawPolyLine(2, &xy[i]);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw n markers with the current attributes at position x, y
///
///  \param [in] n     number of markers to draw
///  \param [in] xy   x,y coordinates of markers

void QuartzDrawable::DrawPolyMarker(Int_t n, TPoint *xy)
{
   if (!fDrawable)
      return;

   if ([fDrawable isDirectDraw]) {
      if (auto view = GetDirectView("DrawPolyMarker"))
         [view.fQuartzWindow addXorMarker: view : n : xy : fAttMarker ];
      return;
   }

   auto drawable = GetPixmapDrawable("DrawPolyMarker");
   if (!drawable)
      return;

   std::vector<TPoint> pnts;

   ConvertPointsROOTToCocoa(n, xy, pnts, drawable);

   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);
   //AA flag is not a part of a state.
   const Quartz::CGAAStateGuard aaCtxGuard(ctx, fCocoa.fUseAA);

   Quartz::DrawPolyMarker(ctx, pnts.size(), pnts.data(), fAttMarker, drawable.fScaleFactor);
}


template<typename CharType>
void QuartzDrawable::DrawTextHelper(Int_t x, Int_t y, Float_t angle, [[maybe_unused]] Float_t mgn, const CharType *text, TVirtualX::ETextMode mode)
{
   if (!text || !*text || !fDrawable)
      return;

   if (!TTFhandle::Init()) {
      ::Error("DrawText", "char string to draw, but TTF initialization failed");
      return;
   }

   if ([fDrawable isDirectDraw])
      return;

   UInt_t width = 0, height = 0;
   Int_t xy = 0;
   fCocoa.GetWindowSize((Drawable_t) fDrawable.fID, xy, xy, width, height);

   //Do not draw anything when pixel size too small
   if (fAttText.GetTextSize() < 1.5)
      return;

   TTFhandle ttf;
   ttf.SetTextFont(fAttText.GetTextFont());
   ttf.SetTextSize(fAttText.GetTextSize());
   ttf.SetRotationMatrix(angle);
   ttf.PrepareString(text);
   ttf.LayoutGlyphs();

   if (ttf.ApplyAlignRotate(x, y, fAttText.GetTextAlign(), width, height))
      DrawTTFglyphs(x, y, ttf, mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a text string using current font
///
///  \param [in] mode       : drawing mode
///              - mode=0     : the background is not drawn (kClear)
///              - mode=1     : the background is drawn (kOpaque)
///  \param [in] x,y        : text position
///  \param [in] angle      : text angle
///  \param [in] mgn        : magnification factor
///  \param [in] text       : text string

void QuartzDrawable::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                         const char *text, TVirtualX::ETextMode mode)
{
   DrawTextHelper<char>(x, y, angle, mgn, text, mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a text string using current font
///
///  \param [in] mode       : drawing mode
///              - mode=0     : the background is not drawn (kClear)
///              - mode=1     : the background is drawn (kOpaque)
///  \param [in] x,y        : text position
///  \param [in] angle      : text angle
///  \param [in] mgn        : magnification factor
///  \param [in] text       : text string

void QuartzDrawable::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                         const wchar_t *text, TVirtualX::ETextMode mode)
{
   DrawTextHelper<wchar_t>(x, y, angle, mgn, text, mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw TTFhandle glyphs

void QuartzDrawable::DrawTTFglyphs(Int_t x1, Int_t y1, TTFhandle &ttf, TVirtualX::ETextMode mode)
{
   if (!fDrawable)
      return;

   if ([fDrawable isDirectDraw])
      return;

   auto drawable = GetPixmapDrawable("DrawTTFglyphs");
   if (!drawable)
      return;

   //This code is a modified (for Quartz) version of TG11TTF text drawing
   QuartzPixmap *dstPixmap = nil;
   if ([drawable isKindOfClass : [QuartzPixmap class]])
      dstPixmap = (QuartzPixmap *)drawable;
   else if ([drawable isKindOfClass : [QuartzView class]] || [drawable isKindOfClass : [QuartzWindow class]])
      dstPixmap = ((NSObject<X11Window> *)drawable).fBackBuffer;

   if (!dstPixmap) {
      //I can not read pixels from a window (I can, but this is too slow and unreliable).
      ::Error("DrawTTFglyphs", "fSelectedDrawable is neither QuartzPixmap nor a double buffered window");
      return;
   }

   Int_t w    = ttf.GetGlyphsWidth();
   Int_t h    = ttf.GetGlyphsHeight();

   //By default, all pixels are set to 0 (all components, that's what code in TGX11TTF also does here).
   Util::NSScopeGuard<QuartzPixmap> pixmap([[QuartzPixmap alloc] initWithW : w H : h scaleFactor : 1.f]);
   if (!pixmap.Get()) {
      ::Error("DrawTTFglyphs", "pixmap creation failed");
      return;
   }

   const unsigned char defaultBackgroundPixel[] = {255, 255, 255, 255};
   Util::ScopedArray<unsigned char> arrayGuard;
   if (mode == TVirtualX::kClear) {
      //For this mode, TGX11TTF does some work to: a) preserve pixels under symbols
      //b) calculate (interpolate) pixel for glyphs.

      X11::Rectangle bbox(x1, y1, w, h);
      //We already check IsVisible, so, in principle, bbox at least has intersection with
      //the current selected drawable.
      if (X11::AdjustCropArea(dstPixmap, bbox))
         arrayGuard.Reset([dstPixmap readColorBits : bbox]);

      if (!arrayGuard.Get()) {
         Error("DrawTTFglyphsW", "problem with reading background pixels");
         return;
      }

      // This is copy & paste from TGX11TTF:
      const Int_t xo = x1 < 0 ? -x1 : 0;
      const Int_t yo = y1 < 0 ? -y1 : 0;

      // while readColorBits returns RGBA coding swap values back before set pixel
      unsigned char swap_pixel[3];

      for (int yp = 0; yp < int(bbox.fHeight) && yo + yp < h; ++yp) {
         const unsigned char *srcBase = arrayGuard.Get() + bbox.fWidth * yp * 4;
         for (int xp = 0; xp < int(bbox.fWidth) && xo + xp < w; ++xp) {
            const unsigned char * const pixel = srcBase + xp * 4;
            swap_pixel[2] = pixel[0];
            swap_pixel[1] = pixel[1];
            swap_pixel[0] = pixel[2];
            [pixmap.Get() putPixel : swap_pixel X : xo + xp Y : yo + yp];
         }
      }
   } else {
      //Find background color and set for all pixels.
      [pixmap.Get() addPixel : defaultBackgroundPixel];
   }

   CGContextRef ctx = drawable.fContext;
   const Quartz::CGStateGuard ctxGuard(ctx);

      // 0 - backgroud, 4 - text color, 1..3 - interpolation for antialiasing
   unsigned char text_color[5][3];
   if (auto color = gROOT->GetColor(fAttText.GetTextColor())) {
      text_color[4][0] = color->GetRed() * 255;
      text_color[4][1] = color->GetGreen() * 255;
      text_color[4][2] = color->GetBlue() * 255;
   } else {
      // black as default text color
      text_color[4][0] = text_color[4][1] = text_color[4][2] = 0;
   }

   CGContextSetRGBStrokeColor(ctx, 0., 0., 1., 1.);
   // paint the glyphs in the pixmap.
   for (UInt_t nglyph = 0; nglyph < ttf.GetNumGlyphs(); nglyph++) {
      Int_t bx = 0, by = 0;
      UChar_t *buffer = nullptr;
      UInt_t width = 0, rows = 0, pitch = 0;
      if (!ttf.GetGlyphData(nglyph, bx, by, buffer, width, rows, pitch))
         continue;

      if (ttf.GetSmoothing()) {
         // background kClear, i.e. transparent, we take as background color
         // the average of the rgb values of all pixels covered by this character
         if (mode == TVirtualX::kClear) {
            const UInt_t maxDots = TMath::Min((UInt_t) 50000, width * rows);
            ULong_t sum[3] = {0, 0, 0};
            UInt_t dotCnt = 0;
            for (unsigned y = 0; y < rows; y++) {
               for (unsigned x = 0; x < width; x++) {
                  if (x + bx < pixmap.Get().fWidth && y + by < pixmap.Get().fHeight) {
                     auto pixels = pixmap.Get().fData + (y + by) * pixmap.Get().fWidth * 4 + (x + bx) * 4;
                     for (int k = 0; k < 3; ++k)
                        sum[k] += pixels[k];
                     if (++dotCnt >= maxDots)
                        break;
                  }
               }
            }
            for (int k = 0; k < 3; ++k)
               text_color[0][k] = sum[k] * (dotCnt > 0 ? 1. / dotCnt : 0.);
         } else {
            // use white as default background color
            text_color[0][0] = text_color[0][1] = text_color[0][2] = 0xff;
         }

         // interpolate between fore and background colors
         for (int x = 1; x < 4; ++x)
            for (int k = 0; k < 3; ++k)
               text_color[x][k] = (text_color[4][k] * x + text_color[0][k] * (4 - x)) / 4;

         // put smoothed character, character pixmap values are an index
         // into the 5 colors used for aliasing (4 = foreground, 0 = background)
         const unsigned char *s = buffer;
         for (unsigned y = 0; y < rows; ++y) {
            for (unsigned x = 0; x < width; ++x) {
               unsigned char d = TMath::Min((unsigned char) 4, (((*s++ & 0xff) + 10) * 5) / 256);
               if (d > 0)
                  [pixmap.Get() putPixel : text_color[d] X : bx + x Y : by + y];
            }
         }
      } else {
         UChar_t *row = buffer;
         for (unsigned y = 0; y < rows; y++) {
            unsigned n = 0;
            UChar_t d = 0;
            UChar_t *s = row;
            for (unsigned x = 0; x < width; x++) {
               if (n == 0) d = *s++;
               if (TESTBIT(d,7-n))
                  [pixmap.Get() putPixel : text_color[4] X : bx + x Y : by + y];
               if (++n == kBitsPerByte) n = 0;
            }
            row += pitch;
         }
      }
   }

   const X11::Rectangle copyArea(0, 0, w, h);
   const X11::Point dstPoint(x1, y1);
   [dstPixmap copy : pixmap.Get() area : copyArea withMask : nil clipOrigin : X11::Point() toPoint : dstPoint];
}

////////////////////////////////////////////////////////////////////////////////
/// Writes the specified window into GIF file. Returns 1 in case of success,
/// 0 otherwise.

Int_t QuartzDrawable::WriteGIF(const char *)
{
   ::Error("WriteGIF", "Not implemented for Quartz");
   return 0;
}
