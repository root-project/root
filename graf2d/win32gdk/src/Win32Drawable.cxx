// @(#)root/win32gdk:$Id$
// Author: Sergey Linev   23/11/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Win32Drawable.h"

#define ROOT_TGWin32cxx
#include "TGWin32.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <limits.h>
#include <process.h>

#include "TROOT.h"
#include "TApplication.h"
#include "TColor.h"
#include "TPoint.h"
#include "TTFhandle.h"
#include "TMath.h"
#include "TStorage.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TError.h"
#include "TException.h"
#include "TClassTable.h"
#include "KeySymbols.h"
#include "TString.h"
#include "TObjArray.h"
#include "TEnv.h"
#include "RStipples.h"
#include "GuiTypes.h"

#include "gifencode.h"

namespace {
//
// Markers
//
static int  gMarkerLineStyle = GDK_LINE_SOLID;
static int  gMarkerCapStyle  = GDK_CAP_ROUND;
static int  gMarkerJoinStyle = GDK_JOIN_ROUND;

//
// Keep style values for line GdkGC
//
static int  gCapStyle  = GDK_CAP_BUTT;
static int  gJoinStyle = GDK_JOIN_MITER;

const Int_t kBIGGEST_RGB_VALUE = 65535;

extern "C" {
void gdk_win32_draw_rectangle (GdkDrawable    *drawable,
                  GdkGC          *gc,
                  gint            filled,
                  gint            x,
                  gint            y,
                  gint            width,
                  gint            height);
void gdk_win32_draw_arc       (GdkDrawable    *drawable,
                  GdkGC          *gc,
                  gint            filled,
                  gint            x,
                  gint            y,
                  gint            width,
                  gint            height,
                  gint            angle1,
                  gint            angle2);
void gdk_win32_draw_polygon   (GdkDrawable    *drawable,
                  GdkGC          *gc,
                  gint            filled,
                  GdkPoint       *points,
                  gint            npoints);
void gdk_win32_draw_points    (GdkDrawable    *drawable,
                  GdkGC          *gc,
                  GdkPoint       *points,
                  gint            npoints);
void gdk_win32_draw_lines     (GdkDrawable    *drawable,
                  GdkGC          *gc,
                  GdkPoint       *points,
                  gint            npoints);
}

} // anonymous namespace


////////////////////////////////////////////////////////////////////////////////
/// constructor

Win32Drawable::Win32Drawable(Int_t id, TGWin32 &win32) : XDrawable(id), fWin32(win32)
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

Win32Drawable::~Win32Drawable()
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor


Bool_t Win32Drawable::IsNull() const
{
   return !fOpen;
}


////////////////////////////////////////////////////////////////////////////////
/// Set fill attributes for specified window

void Win32Drawable::SetAttFill(const TAttFill &att)
{
   Int_t cindex = att.GetFillColor();
   if (!gStyle->GetFillColor() && cindex > 1)
      cindex = 0;
   if (cindex >= 0)
      fWin32.SetColor(this, fGClist[kGCfill], cindex);
   fAttFill.SetFillColor(cindex);

   Int_t style = att.GetFillStyle() / 1000;
   Int_t fasi = att.GetFillStyle() % 1000;

   switch (style) {
      case 1:                     // solid
         fillHollow = 0;
         gdk_gc_set_fill(fGClist[kGCfill], GDK_SOLID);
         break;

      case 2:                     // pattern
         fillHollow = 1;
         break;

      case 3:                     // hatch
         fillHollow = 0;
         gdk_gc_set_fill(fGClist[kGCfill], GDK_STIPPLED);

         if (fasi != fillFasi) {
            if (fillPattern) {
               gdk_pixmap_unref(fillPattern);
               fillPattern = nullptr;
            }
            int stn = (fasi >= 1 && fasi <=25) ? fasi : 2;
            char pattern[32];
            for (int i = 0; i < 32; ++i)
               pattern[i] = ~gStipples[stn][i];
            fillPattern = gdk_bitmap_create_from_data(GDK_ROOT_PARENT(),
                                                            (const char *)&pattern, 16, 16);
            gdk_gc_set_stipple(fGClist[kGCfill], fillPattern);
            fillFasi = fasi;
         }
         break;

      default:
         fillHollow = 1;
   }

   fAttFill.SetFillStyle(att.GetFillStyle());
}


////////////////////////////////////////////////////////////////////////////////
/// Set line attributes for specified window.

void Win32Drawable::SetAttLine(const TAttLine &att)
{
   fWin32.SetColor(this, fGClist[kGCline], att.GetLineColor());
   fWin32.SetColor(this, fGClist[kGCdash], att.GetLineColor());

   if (fAttLine.GetLineStyle() != att.GetLineStyle()) { //set style index only if different
      if (att.GetLineStyle() <= 1)
         dashList.clear();
      else if (att.GetLineStyle() == 2)
         dashList = { 3, 3 };
      else if (att.GetLineStyle() == 3)
         dashList = { 1, 2 };
      else if (att.GetLineStyle() == 4) {
         dashList = { 3, 4, 1, 4} ;
      } else {
         TString st = (TString)gStyle->GetLineStyleString(att.GetLineStyle());
         auto tokens = st.Tokenize(" ");
         Int_t nt = tokens->GetEntries();
         dashList.resize(nt);
         for (Int_t j = 0; j < nt; ++j) {
            Int_t it;
            sscanf(tokens->At(j)->GetName(), "%d", &it);
            dashList[j] = (Int_t) (it/4);
         }
         delete tokens;
      }
      dashLength = 0;
      for (auto elem : dashList)
         dashLength += elem;
      dashOffset = 0;
      lineStyle = dashList.size() == 0 ? GDK_LINE_SOLID : GDK_LINE_ON_OFF_DASH;
   }

   lineWidth = att.GetLineWidth();
   if (lineWidth == 1 && lineStyle == GDK_LINE_SOLID)
      lineWidth = 0;

   if (lineStyle == GDK_LINE_SOLID) {
      gdk_gc_set_line_attributes(fGClist[kGCline], lineWidth,
                                 lineStyle,
                                 (GdkCapStyle) gCapStyle,
                                 (GdkJoinStyle) gJoinStyle);
   } else {
      gdk_gc_set_line_attributes(fGClist[kGCdash], lineWidth,
                                 lineStyle,
                                 (GdkCapStyle) gCapStyle,
                                 (GdkJoinStyle) gJoinStyle);
   }

   fAttLine = att;
}


////////////////////////////////////////////////////////////////////////////////
/// Set marker attributes

void Win32Drawable::SetAttMarker(const TAttMarker &att)
{
   fWin32.SetColor(this, fGClist[kGCmark], att.GetMarkerColor());

   Bool_t changed = (att.GetMarkerSize() != fAttMarker.GetMarkerSize()) ||
                    (att.GetMarkerStyle() != fAttMarker.GetMarkerStyle());

   fAttMarker = att;

   if (!changed)
      return;

   markerLineWidth = TAttMarker::GetMarkerLineWidth(att.GetMarkerStyle());
   markerType = att.GetMarkerShape(markerSize, markerShape, 1., TAttMarker::kDotAsLines);

   // The fast pixel markers need to be treated separately
   if (markerType == TAttMarker::kShapeSegments) {
      gdk_gc_set_line_attributes(fGClist[kGCmark], markerLineWidth, GDK_LINE_SOLID, GDK_CAP_BUTT, GDK_JOIN_MITER);
   } else {
      gdk_gc_set_line_attributes(fGClist[kGCmark], markerLineWidth,
                                 (GdkLineStyle) gMarkerLineStyle,
                                 (GdkCapStyle)  gMarkerCapStyle,
                                 (GdkJoinStyle) gMarkerJoinStyle);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set text attributes for speicfied window

void Win32Drawable::SetAttText(const TAttText &att)
{
   fWin32.SetColor(this, fGClist[kGCtext], att.GetTextColor());

   GdkGCValues values;
   gdk_gc_get_values(fGClist[kGCtext], &values);
   gdk_gc_set_foreground(fGClist[kGCinvt], &values.background);
   gdk_gc_set_background(fGClist[kGCinvt], &values.foreground);
   gdk_gc_set_background(fGClist[kGCtext], (GdkColor *) & fWin32.GetColor(0).color);

   fAttText = att;
}

////////////////////////////////////////////////////////////////////////////////
/// Set window draw mode

void Win32Drawable::SetDrawMode(TVirtualX::EDrawMode mode)
{
   GdkFunction func = GDK_COPY;
   if (mode == TVirtualX::kXor)
      func = GDK_XOR;
   else if (mode == TVirtualX::kInvert)
      func = GDK_INVERT;

   for (int i = 0; i < kMAXGC; i++) {
      if (fGClist[i])
         gdk_gc_set_function(fGClist[i], func);
   }

   drawMode = mode;
}

////////////////////////////////////////////////////////////////////////////////
/// Clear specified window.

void Win32Drawable::ClearWindow()
{
   if (!ispixmap && !double_buffer) {
      gdk_window_set_background(drawing, (GdkColor *) & fWin32.GetColor(0).color);
      gdk_window_clear(drawing);
      GdiFlush();
   } else {
      fWin32.SetColor(this, fGClist[kGCpxmp], 0);
      gdk_win32_draw_rectangle(drawing, fGClist[kGCpxmp], 1,
                         0, 0, width, height);
      fWin32.SetColor(this, fGClist[kGCpxmp], 1);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Close window

void Win32Drawable::CloseWindow()
{
   if (!shared) {
      if (ispixmap) {
         gdk_pixmap_unref(window);
      } else {
         gdk_window_destroy(window, kTRUE);
      }
   }

   if (fillPattern) {
      gdk_pixmap_unref(fillPattern);
      fillPattern = nullptr;
   }

   if (buffer) {
      gdk_pixmap_unref(buffer);
   }
   if (!new_colors.empty()) {
      gdk_colormap_free_colors((GdkColormap *) fWin32.fColormap,
                               (GdkColor *)new_colors.data(), new_colors.size());

      new_colors.clear();
   }

   for (int i = 0; i < kMAXGC; i++)
      gdk_gc_unref(fGClist[i]); //  gdk_gc_unref is equivalent to gdk_gc_destroy

   GdiFlush();
   fOpen = 0;

   fWin32.RemoveWindowHandle(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Update display.
/// mode : (1) update
///        (0) sync
///
/// Synchronise client and server once (not permanent).
/// Copy the pixmap gCws->drawing on the window gCws->window
/// if the double buffer is on.

void Win32Drawable::UpdateWindow(Int_t mode)
{
   if (double_buffer)
      gdk_window_copy_area(window, fGClist[kGCpxmp], 0, 0,
                           drawing, 0, 0, width, height);
   fWin32.Update(mode);
}


////////////////////////////////////////////////////////////////////////////////
///

static ULong_t GetPixelImage(Drawable_t id, Int_t x, Int_t y)
{
   if (!id) return 0;

   GdkImage *image = (GdkImage *)id;
   ULong_t pixel;

   if (image->depth == 1) {
      pixel = (((char *) image->mem)[y * image->bpl + (x >> 3)] & (1 << (7 - (x & 0x7)))) != 0;
   } else {
      UChar_t *pixelp = (UChar_t *) image->mem + y * image->bpl + x * image->bpp;
      switch (image->bpp) {
         case 1:
            pixel = *pixelp;
            break;
         // Windows is always LSB, no need to check image->byte_order.
         case 2:
            pixel = pixelp[0] | (pixelp[1] << 8);
            break;
         case 3:
            pixel = pixelp[0] | (pixelp[1] << 8) | (pixelp[2] << 16);
            break;
         case 4:
            pixel = pixelp[0] | (pixelp[1] << 8) | (pixelp[2] << 16);
            break;
      }
   }
   return pixel;
}


////////////////////////////////////////////////////////////////////////////////
/// Set opacity of a specified window. This image manipulation routine works
/// by adding to a percent amount of neutral to each pixels RGB.
/// Since it requires quite some additional color map entries is it
/// only supported on displays with more than > 8 color planes (> 256
/// colors)

void Win32Drawable::SetOpacity(Int_t percent)
{
   Int_t depth = gdk_visual_get_best_depth();

   if ((depth <= 8) || (percent <= 0)) return;
   if (percent > 100) percent = 100;

   // get pixmap from server as image
   GdkImage *image = gdk_image_get((GdkDrawable*)drawing, 0, 0,
                                   width, height);

   if (!image) return;

   std::vector<ULong_t> orgcolors;

   // collect different image colors
   for (UInt_t y = 0; y < height; y++) {
      for (UInt_t x = 0; x < width; x++) {
         ULong_t pixel = GetPixelImage((Drawable_t)image, x, y);
         if (std::find(orgcolors.begin(), orgcolors.end(), pixel) == orgcolors.end())
            orgcolors.emplace_back(pixel);
      }
   }
   if (orgcolors.empty()) {
      gdk_image_unref(image);
      return;
   }

   if (!new_colors.empty()) {
      gdk_colors_free((GdkColormap *)fWin32.fColormap, new_colors.data(), new_colors.size(), 0);
      new_colors.clear();
   }

   std::vector<GdkColor> xcol(orgcolors.size());

   for (std::size_t i = 0; i < orgcolors.size(); i++) {
      xcol[i].pixel = orgcolors[i];
      xcol[i].red = xcol[i].green = xcol[i].blue = 0;
   }

   GdkColorContext *cc = gdk_color_context_new(gdk_visual_get_system(), (GdkColormap *)fWin32.fColormap);
   gdk_color_context_query_colors(cc, xcol.data(), orgcolors.size());
   gdk_color_context_free(cc);

   // create new colors mixing:  "100-percent" of old color and "percent" of new background color
   XColor_t &bkgr = fWin32.GetColor(fAttFill.GetFillColor());

   for (std::size_t i = 0; i < orgcolors.size(); i++) {
      xcol[i].red   = (UShort_t) TMath::Min((Int_t) xcol[i].red * (100 - percent) / 100  + bkgr.color.red * percent / 100, kBIGGEST_RGB_VALUE);
      xcol[i].green = (UShort_t) TMath::Min((Int_t) xcol[i].green * (100 - percent) / 100  + bkgr.color.green * percent / 100, kBIGGEST_RGB_VALUE);
      xcol[i].blue  = (UShort_t) TMath::Min((Int_t) xcol[i].blue * (100 - percent) / 100  + bkgr.color.blue * percent / 100, kBIGGEST_RGB_VALUE);

      auto ret = gdk_color_alloc((GdkColormap *)fWin32.fColormap, &xcol[i]);

      if (!ret) {
         Warning("SetOpacityW",
                 "failed to allocate color %hd, %hd, %hd", xcol[i].red,
                 xcol[i].green, xcol[i].blue);
      // assumes that in case of failure xcol[i].pixel is not changed
      }
   }

   new_colors.resize(orgcolors.size());

   for (std::size_t i = 0; i < orgcolors.size(); i++)
      new_colors[i] = xcol[i].pixel;

   // put opaque colors in image
   for (UInt_t y = 0; y < height; y++) {
      for (UInt_t x = 0; x < width; x++) {
         ULong_t pixel = GetPixelImage((Drawable_t)image, x, y);
         auto iter = std::find(orgcolors.begin(), orgcolors.end(), pixel);
         if (iter != orgcolors.end()) {
            auto idx = iter - orgcolors.begin();
            fWin32.PutPixel((Drawable_t)image, x, y, new_colors[idx]);
         }
      }
   }

   // put image back in pixmap on server
   gdk_draw_image(drawing, fGClist[kGCpxmp], (GdkImage *)image,
                  0, 0, 0, 0, width, height);
   GdiFlush();

   // clean up
   gdk_image_unref(image);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy the pixmap wid at the position xpos, ypos

void Win32Drawable::CopyPixmap(Int_t wid, Int_t xpos, Int_t ypos)
{
   auto tws = fWin32.fWindows[wid];

   gdk_window_copy_area(drawing, tws->fGClist[kGCpxmp], xpos, ypos, tws->drawing,
                        0, 0, tws->width, tws->height);

   GdiFlush();
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a box
/// mode=0 hollow  (kHollow)
/// mode=1 solid   (kSolid)

void Win32Drawable::DrawBox(int x1, int y1, int x2, int y2, TVirtualX::EBoxMode mode)
{
   Int_t x = TMath::Min(x1, x2);
   Int_t y = TMath::Min(y1, y2);
   Int_t w = TMath::Abs(x2 - x1);
   Int_t h = TMath::Abs(y2 - y1);

   switch (mode) {
      case TVirtualX::kHollow:
         gdk_win32_draw_rectangle(drawing, fGClist[kGCline], 0, x, y, w, h);
         break;

      case TVirtualX::kFilled:
         gdk_win32_draw_rectangle(drawing, fGClist[kGCfill], 1, x, y, w, h);
         break;

      default:
         break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill area described by polygon
/// n         : number of points
/// xy(2,n)   : list of points

void Win32Drawable::DrawFillArea(int n, TPoint *xyt)
{
   std::vector<GdkPoint> xy(n);
   for (int i = 0; i < n; i++) {
      xy[i].x = xyt[i].fX;
      xy[i].y = xyt[i].fY;
   }

   if (fillHollow) {
      gdk_win32_draw_lines(drawing, fGClist[kGCfill], xy.data(), n);
   } else {
      gdk_win32_draw_polygon(drawing, fGClist[kGCfill], 1, xy.data(), n);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a line
/// x1,y1        : begin of line
/// x2,y2        : end of line

void Win32Drawable::DrawLine(Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   if (lineStyle == GDK_LINE_SOLID) {
      gdk_draw_line(drawing, fGClist[kGCline], x1, y1, x2, y2);
   } else {
      gdk_gc_set_dashes(fGClist[kGCdash], dashOffset, dashList.data(), dashList.size());
      gdk_draw_line(drawing, fGClist[kGCdash], x1, y1, x2, y2);
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Draw a line through all points
/// n         : number of points
/// xy        : list of points

void Win32Drawable::DrawPolyLine(int n, TPoint * xyt)
{
   if (n < 1)
      return;

   std::vector<GdkPoint> xy(n);
   for (int i = 0; i < n; i++) {
      xy[i].x = xyt[i].fX;
      xy[i].y = xyt[i].fY;
   }

   if (n > 1) {
      if (lineStyle == GDK_LINE_SOLID) {
         gdk_win32_draw_lines(drawing, fGClist[kGCline], xy.data(), n);
      } else {
         gdk_gc_set_dashes(fGClist[kGCdash], dashOffset, dashList.data(), dashList.size());
         gdk_win32_draw_lines(drawing, fGClist[kGCdash], xy.data(), n);

         // calculate length of line to update dash offset
         for (int i = 1; i < n; i++) {
            int dx = xy[i].x - xy[i - 1].x;
            int dy = xy[i].y - xy[i - 1].y;

            if (dx < 0) dx = -dx;
            if (dy < 0) dy = -dy;
            dashOffset += dx > dy ? dx : dy;
         }
         dashOffset %= dashLength;
      }
   } else {
      gdk_win32_draw_points(drawing,
                            fGClist[lineStyle == GDK_LINE_SOLID ? kGCline : kGCdash],
                            xy.data(), 1);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Draw N segments
/// n         : number of segments
/// xy        : list of points, 2*N size

void Win32Drawable::DrawLinesSegments(Int_t n, TPoint *xyt)
{
   for(Int_t i = 0; i < 2*n; i += 2)
      DrawPolyLine(2, &xyt[i]);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw n markers with the current attributes at position x, y
/// n    : number of markers to draw
/// xy   : x,y coordinates of markers

void Win32Drawable::DrawPolyMarker(Int_t n, TPoint *xyt)
{
   if (markerType == TAttMarker::kShapeDot) {
      std::vector<GdkPoint> xy(n);
      for (int i = 0; i < n; i++) {
         xy[i].x = xyt[i].fX;
         xy[i].y = xyt[i].fY;
      }

      gdk_win32_draw_points(drawing, fGClist[kGCmark], xy.data(), n);
      return;
   }
   int r = markerSize / 2;
   auto &shape0 = markerShape;
   std::vector<GdkPoint> shape(shape0.size());

   for (int m = 0; m < n; m++) {
      for (size_t i = 0; i < shape.size(); i++) {
         shape[i].x = xyt[m].fX + shape0[i].fX;
         shape[i].y = xyt[m].fY + shape0[i].fY;
      }
      switch(markerType) {
         case TAttMarker::kShapeDot:
            // handled before
            break;
         case TAttMarker::kShapeCircle:
            // hollow circle
            gdk_win32_draw_arc(drawing, fGClist[kGCmark], kFALSE, xyt[m].fX - r, xyt[m].fY - r,
                              markerSize, markerSize, 0, 23040);
            break;
         case TAttMarker::kShapeFilledCircle:
            // filled circle
            gdk_win32_draw_arc(drawing, fGClist[kGCmark], kTRUE, xyt[m].fX - r, xyt[m].fY - r,
                                 markerSize, markerSize, 0, 23040);
            break;
         case TAttMarker::kShapePolyLine:
            // hollow polygon
            gdk_win32_draw_lines(drawing, fGClist[kGCmark], shape.data(), shape.size());
            break;
         case TAttMarker::kShapeFilledArea:
            // fillded area
            gdk_win32_draw_polygon(drawing, fGClist[kGCmark], 1, shape.data(), shape.size());
            break;
         case TAttMarker::kShapeSegments:
            for (std::size_t i = 0; i < shape.size(); i += 2)
               gdk_draw_line(drawing, fGClist[kGCmark],
                              shape[i].x, shape[i].y, shape[i+1].x, shape[i+1].y);
            break;
         case TAttMarker::kShapeTriangles:
            // filled triangles
            for (std::size_t t = 0; t < shape.size(); t += 3)
               gdk_win32_draw_polygon(drawing, fGClist[kGCmark], 1, shape.data() + t, 3);
            break;
      }
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Draw text helper

template<typename CharType>
void Win32Drawable::DrawTextHelper(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                                   const CharType *text, TVirtualX::ETextMode mode)
{
   (void) mgn;
   UInt_t width = 0, height = 0;
   Int_t xy;
   fWin32.GetWindowSize((Drawable_t) drawing, xy, xy, width, height);

   TTFhandle ttf;
   ttf.SetSmoothing(fWin32.fDepth > 8);
   ttf.SetTextFont(fAttText.GetTextFont());
   ttf.SetTextSize(fAttText.GetTextSize());
   ttf.SetRotationMatrix(angle);
   ttf.PrepareString(text);

   ttf.LayoutGlyphs();

   if (ttf.ApplyAlignRotate(x, y, fAttText.GetTextAlign(), width, height))
      DrawTTFglyphs(x, y, ttf, mode);
}


////////////////////////////////////////////////////////////////////////////////
/// Draw text

void Win32Drawable::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn, const char *text, TVirtualX::ETextMode mode)
{
   DrawTextHelper<char>(x, y, angle, mgn, text, mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw text

void Win32Drawable::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn, const wchar_t *text, TVirtualX::ETextMode mode)
{
   DrawTextHelper<wchar_t>(x, y, angle, mgn, text, mode);
}


////////////////////////////////////////////////////////////////////////////////
/// Get the background of the current window in an XImage.
/// Used for TTF glyphs drawing

GdkImage *Win32Drawable::GetBackground(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   UInt_t width;
   UInt_t height;
   Int_t xy;
   fWin32.GetWindowSize((Drawable_t) drawing, xy, xy, width, height);

   if (x < 0) {
      w += x;
      x  = 0;
   }
   if (y < 0) {
      h += y;
      y  = 0;
   }

   if (x+w > width)  w = width - x;
   if (y+h > height) h = height - y;

   return gdk_image_get(drawing, x, y, w, h);
}


////////////////////////////////////////////////////////////////////////////////
/// Draw TTF glyphs

void Win32Drawable::DrawTTFglyphs(Int_t x1, Int_t y1, TTFhandle &ttf, TVirtualX::ETextMode mode)
{
   Int_t w    = ttf.GetGlyphsWidth();
   Int_t h    = ttf.GetGlyphsHeight();

   // create the XImage that will contain the text
   UInt_t depth = fWin32.fDepth;
   GdkImage *xim  = gdk_image_new(GDK_IMAGE_SHARED, gdk_visual_get_best(), w, h);

   // use malloc since Xlib will use free() in XDestroyImage
//   xim->data = (char *) malloc(xim->bytes_per_line * h);
//   memset(xim->data, 0, xim->bytes_per_line * h);

   ULong_t   pixel;
   ULong_t   back;
   GdkGCValues gcvals;

   gdk_gc_get_values(fGClist[kGCtext], &gcvals);

   // get the background
   if (mode == TVirtualX::kClear) {
      // if mode == kClear we need to get an image of the background
      GdkImage *bim = GetBackground(x1, y1, w, h);
      if (!bim) {
         Error("DrawTTFglyphsW", "error getting background image");
         return;
      }

      // and copy it into the text image
      Int_t xo = x1 < 0 ? -x1 : 0;
      Int_t yo = y1 < 0 ? -y1 : 0;

      for (int yp = 0; yp < (int) bim->height; yp++) {
         for (int xp = 0; xp < (int) bim->width; xp++) {
            pixel = GetPixelImage((Drawable_t)bim, xp, yp);
            fWin32.PutPixel((Drawable_t)xim, xo+xp, yo+yp, pixel);
         }
      }

      gdk_image_unref((GdkImage *)bim);

   } else {
      // if mode == kOpaque its simple, we just draw the background

      GdkImage *bim = GetBackground(x1, y1, w, h);
      if (!bim) {
         pixel = gcvals.background.pixel;
      } else {
         pixel = GetPixelImage((Drawable_t)bim, 0, 0);
      }
      Int_t xo = x1 < 0 ? -x1 : 0;
      Int_t yo = y1 < 0 ? -y1 : 0;

      for (int yp = 0; yp < h; yp++) {
         for (int xp = 0; xp < w; xp++) {
            fWin32.PutPixel((Drawable_t)xim, xo+xp, yo+yp, pixel);
         }
      }
      if (bim) {
         gdk_image_unref((GdkImage *)bim);
         mode = TVirtualX::kClear;
      } else {
         back = pixel;
      }
   }

   // paint the glyphs in the XImage
   for (UInt_t nglyph = 0; nglyph < ttf.GetNumGlyphs(); nglyph++) {
      Int_t bx = 0, by = 0;
      UChar_t *buffer = nullptr;
      UInt_t width = 0, rows = 0, pitch = 0;
      if (!ttf.GetGlyphData(nglyph, bx, by, buffer, width, rows, pitch))
         continue;

      if (ttf.GetSmoothing()) {

         GdkColor col[5];

         // background kClear, i.e. transparent, we take as background color
         // the average of the rgb values of all pixels covered by this character
         if (mode == TVirtualX::kClear) {
            const UInt_t ndots = TMath::Min((UInt_t) 50000, width * rows);
            std::vector<GdkColor> bcol(ndots);
            if (!bcol.size())
               return;

            for (UInt_t y = 0, dotcnt = 0; y < rows; y++)
               for (UInt_t x = 0; (x < width) && (dotcnt < bcol.size()); x++)
                  bcol[dotcnt++].pixel = GetPixelImage((Drawable_t)xim, bx + x, by + y);
            fWin32.QueryColors(fWin32.fColormap, bcol.data(), bcol.size());
            ULong_t r = 0, g = 0, b = 0;
            for (auto &bc : bcol) {
               r += bc.red;
               g += bc.green;
               b += bc.blue;
            }
            col[0].red   = (UShort_t) (r / bcol.size());
            col[0].green = (UShort_t) (g / bcol.size());
            col[0].blue  = (UShort_t) (b / bcol.size());
         } else {
            // query background color
            col[0].pixel = back;
            fWin32.QueryColors(fWin32.fColormap, &col[0], 1);
         }

         // query foreground color
         col[4].pixel = gcvals.foreground.pixel;
         fWin32.QueryColors(fWin32.fColormap, &col[4], 1);

         // calculate the 3 smooting colors
         // (interpolation between fore- and background colors)

         // interpolate between fore and backgound colors
         for (int x = 3; x > 0; x--) {
            col[x].red   = (col[4].red  *x + col[0].red  *(4-x)) /4;
            col[x].green = (col[4].green*x + col[0].green*(4-x)) /4;
            col[x].blue  = (col[4].blue *x + col[0].blue *(4-x)) /4;
            if (!fWin32.AllocColor(fWin32.fColormap, &col[x])) {
               ::Warning("DrawTTFGlyphs", "cannot allocate smoothing color");
               col[x].pixel = col[x+1].pixel;
            }
         }

         // put smoothed character, character pixmap values are an index
         // into the 5 colors used for aliasing (4 = foreground, 0 = background)
         UChar_t* row = buffer;
         for (unsigned y = 0; y < rows; y++) {
            UChar_t *s = row;
            for (unsigned x = 0; x < width; x++) {
               UChar_t d = (((*s++ & 0xff) + 10) * 5) / 256;
               if (d > 4)
                  d = 4;
               if (d > 0)
                  fWin32.PutPixel((Drawable_t)xim, bx + x, by + y, col[d].pixel);
            }
            row += pitch;
         }
      } else {
         // no smoothing, just put character using foreground color
         UChar_t* row = buffer;
         for (unsigned y = 0; y < rows; y++) {
            unsigned n = 0;
            UChar_t *s = row;
            UChar_t d;
            for (unsigned x = 0; x < width; x++) {
               if (n == 0)
                  d = *s++;
               if (TESTBIT(d, 7-n))
                  fWin32.PutPixel((Drawable_t)xim, bx + x, by + y, gcvals.foreground.pixel);
               if (++n == kBitsPerByte)
                  n = 0;
            }
            row += pitch;
         }
      }
   }

   // put the Ximage on the screen
   gdk_draw_image(drawing, fGClist[kGCpxmp], xim, 0, 0, x1, y1, w, h);

   gdk_image_unref(xim);
}


class TWin32GifEncode : public TGifEncode {
   private:
      GdkImage *fImage = nullptr;
   protected:
      void get_scline(int y, int width, unsigned char *buf) override
      {
         for (int x = 0; x < width; x++) {
            ULong_t pixel = GetPixelImage((Drawable_t)fImage, x, y);
            buf[x] = 0;

            auto iter = std::find(orgcolors.begin(), orgcolors.end(), pixel);
            if (iter != orgcolors.end()) {
               auto idx = iter - orgcolors.begin();
               if (idx < 256)
                  buf[x] = (unsigned char) idx;
            }
         }
      }
   public:
      std::vector<ULong_t> orgcolors;

      TWin32GifEncode(GdkImage *image) : fImage(image) {}
};

////////////////////////////////////////////////////////////////////////////////
/// Writes the specified window into GIF file.

Int_t Win32Drawable::WriteGIF(const char *name)
{
   GdkImage *image = gdk_image_get((GdkDrawable*)drawing, 0, 0,
                                   width, height);
   if (!image) {
      Error("WriteGIFW", "Cannot create image for writing GIF. Try in batch mode.");
      return 0;
   }

   TWin32GifEncode gif(image);

   /// Collect R G B of colors of the palette used by the image.
   /// The image pixels are changed to index values in these R G B arrays.
   /// This produces a colormap with only the used colors (so even on displays
   /// with more than 8 planes we will be able to create GIF's when the image
   /// contains no more than 256 different colors). If it does contain more
   /// colors we will have to use GIFquantize to reduce the number of colors.

   // collect different image colors
   for (UInt_t x = 0; x < width; x++) {
      for (UInt_t y = 0; y < height; y++) {
         ULong_t pixel = GetPixelImage((Drawable_t)image, x, y);
         if (std::find(gif.orgcolors.begin(), gif.orgcolors.end(), pixel) == gif.orgcolors.end())
            gif.orgcolors.emplace_back(pixel);
      }
   }

   auto ncolors = gif.orgcolors.size();

   if (ncolors > 256) {
      Error("WriteGIFW", "can not create GIF of image containing more than 256 colors");
      gdk_image_unref(image);
      return 0;
   }

   // get RGB values belonging to pixels
   std::vector<GdkColor> xcol(ncolors);

   for (std::size_t i = 0; i < ncolors; i++) {
      xcol[i].pixel = gif.orgcolors[i];
      xcol[i].red = GetRValue(xcol[i].pixel);
      xcol[i].green = GetGValue(xcol[i].pixel);
      xcol[i].blue = GetBValue(xcol[i].pixel);
   }

   GdkColorContext *cc = gdk_color_context_new(gdk_visual_get_system(), (GdkColormap *)fWin32.fColormap);
   gdk_color_context_query_colors(cc, xcol.data(), ncolors);
   gdk_color_context_free(cc);

   UShort_t maxcol = 0;
   for (std::size_t i = 0; i < ncolors; i++) {
      maxcol = TMath::Max(maxcol, xcol[i].red);
      maxcol = TMath::Max(maxcol, xcol[i].green);
      maxcol = TMath::Max(maxcol, xcol[i].blue);
   }
   if (maxcol == 0)
      maxcol = 255;

   std::vector<unsigned char> r(ncolors), b(ncolors), g(ncolors);

   for (std::size_t i = 0; i < ncolors; i++) {
      r[i] = (unsigned char) (xcol[i].red * 255 / maxcol);
      g[i] = (unsigned char) (xcol[i].green * 255 / maxcol);
      b[i] = (unsigned char) (xcol[i].blue * 255 / maxcol);
   }

   Int_t ret = 0;

   if (gif.OpenFile(name, "wb")) {
      auto len = gif.GIFencode(width, height, ncolors, r.data(), g.data(), b.data());
      if (len > 0)
         ret = 1;
      gif.CloseFile();
    } else {
      Error("WriteGIFW", "cannot write file: %s",name);
   }
   gdk_image_unref(image);

   return ret;
}
