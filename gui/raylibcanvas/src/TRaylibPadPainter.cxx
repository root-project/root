// Author:  Sergey Linev, GSI  06/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TRaylibPadPainter.h"

#include "TError.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TEnv.h"
#include "TMath.h"
#include "TPad.h"
#include "TPoint.h"
#include "TROOT.h"
#include "TColor.h"
#include "RStipples.h"

#include <raylib.h>
#include <raymath.h>
#include <vector>
#include <string>
#include <algorithm>
#include "earcut.hpp"

using namespace ROOT::Experimental;

// Scale factor aligning raylib text size with ROOT's TTF expectations
const Float_t kScale = 0.75f * 0.93376068f;

/** \class TRaylibPadPainter
    \ingroup raylibcanvas
    \brief Implement TVirtualPadPainter for raylib immediate-mode graphics

    Uses raylib drawing primitives called within BeginDrawing()/EndDrawing() block.
*/

// ============================ Font Table ================================

static const struct {
   const char *envKey;
   const char *fontName;
} gFontTable[] = {
   {"Root.TTFont.0",       "FreeSansBold.otf"},
   {"Root.TTFont.1",       "FreeSerifItalic.otf"},
   {"Root.TTFont.2",       "FreeSerifBold.otf"},
   {"Root.TTFont.3",       "FreeSerifBoldItalic.otf"},
   {"Root.TTFont.4",       "texgyreheros-regular.otf"},
   {"Root.TTFont.5",       "texgyreheros-italic.otf"},
   {"Root.TTFont.6",       "texgyreheros-bold.otf"},
   {"Root.TTFont.7",       "texgyreheros-bolditalic.otf"},
   {"Root.TTFont.8",       "FreeMono.otf"},
   {"Root.TTFont.9",       "FreeMonoOblique.otf"},
   {"Root.TTFont.10",      "FreeMonoBold.otf"},
   {"Root.TTFont.11",      "FreeMonoBoldOblique.otf"},
   {"Root.TTFont.12",      "symbol.ttf"},
   {"Root.TTFont.13",      "FreeSerif.otf"},
   {"Root.TTFont.14",      "wingding.ttf"},
   {"Root.TTFont.15",      "symbol.ttf"},
   {"Root.TTFont.STIXGen", "STIXGeneral.otf"},
   {"Root.TTFont.STIXGenIt", "STIXGeneralItalic.otf"},
   {"Root.TTFont.STIXGenBd", "STIXGeneralBol.otf"},
   {"Root.TTFont.STIXGenBdIt", "STIXGeneralBolIta.otf"},
   {"Root.TTFont.STIXSiz1Sym", "STIXSiz1Sym.otf"},
   {"Root.TTFont.STIXSiz1SymBd", "STIXSiz1SymBol.otf"},
   {"Root.TTFont.STIXSiz2Sym", "STIXSiz2Sym.otf"},
   {"Root.TTFont.STIXSiz2SymBd", "STIXSiz2SymBol.otf"},
   {"Root.TTFont.STIXSiz3Sym", "STIXSiz3Sym.otf"},
   {"Root.TTFont.STIXSiz3SymBd", "STIXSiz3SymBol.otf"},
   {"Root.TTFont.STIXSiz4Sym", "STIXSiz4Sym.otf"},
   {"Root.TTFont.STIXSiz4SymBd", "STIXSiz4SymBol.otf"},
   {"Root.TTFont.STIXSiz5Sym", "STIXSiz5Sym.otf"},
   {"Root.TTFont.ME",      "DroidSansFallback.ttf"},
   {"Root.TTFont.CJKMing", "DroidSansFallback.ttf"},
   {"Root.TTFont.CJKGothic", "DroidSansFallback.ttf"}
};


static constexpr int kFontTableSize = sizeof(gFontTable) / sizeof(gFontTable[0]);

// ======================== Static Font Cache =============================

struct FontEntry {
   int rlFontId = 0;   // font texture id from raylib
   int baseSize = 0;   // pixel size this font was loaded at
   std::string path;   // original font file path
};

Font TRaylibPadPainter::LoadFontForId([[maybe_unused]]Font_t fontnumber)
{
   return ::GetFontDefault();
/*

   int fontid = fontnumber / 10;
   if (fontid < 0 || fontid >= kFontTableSize)
      fontid = 0;

   static std::map<int, FontEntry> cache;
   auto iter = cache.find(fontid);
   if (iter == cache.end()) {
      const char *ttpath = gEnv->GetValue("Root.TTFontPath", TROOT::GetTTFFontDir());
      TString fname = gEnv->GetValue(gFontTable[fontid].envKey, gFontTable[fontid].fontName);
      const char *ttfont = gSystem->FindFile(ttpath, fname, kReadPermission);

      if (!ttfont) {
         Error("LoadFontForId",
               "Not found font %s in configured path %s", fname.Data(), ttpath);
         return ::GetFontDefault();
      }
      cache[fontid] = FontEntry{0, 64, ttfont};
   }

   auto &entry = cache[fontid];
   return LoadFontEx(entry.path.c_str(), entry.baseSize, nullptr, 0);
*/
}

// ======================== Color Conversion ==============================

Color TRaylibPadPainter::GetRaylibColor(Color_t id)
{
   auto c = gROOT->GetColor(id);
   if (c) {
      unsigned char r = (unsigned char)(c->GetRed() * 255);
      unsigned char g = (unsigned char)(c->GetGreen() * 255);
      unsigned char b = (unsigned char)(c->GetBlue() * 255);
      unsigned char a = (unsigned char)(c->GetAlpha() * 255);
      return (Color){r, g, b, a};
   }
   return BLACK;
}

float TRaylibPadPainter::GetCurrentAlpha() const
{
   Int_t style = GetAttFill().GetFillStyle();
   if (style >= 4000 && style <= 4255) {
      int opacity = style - 4000;
      return opacity / 100.0f;
   }
   auto col = gROOT->GetColor(GetAttFill().GetFillColor());
   if (col)
      return col->GetAlpha();
   return 1.0f;
}

// ========================== SetOpacity ==================================

void TRaylibPadPainter::SetOpacity(Int_t percent)
{
   fAttFill.SetFillStyle(4000 + percent);
   // EnableBlendMode(BLEND_ALPHA);
}

// ============================ Cursor ====================================

void TRaylibPadPainter::SetCursor(Int_t, ECursor cursor)
{
   MouseCursor rlCursor = MOUSE_CURSOR_ARROW;
   switch (cursor) {
      case kBottomLeft:   rlCursor = MOUSE_CURSOR_RESIZE_ALL; break;
      case kBottomRight:  rlCursor = MOUSE_CURSOR_RESIZE_ALL; break;
      case kTopLeft:      rlCursor = MOUSE_CURSOR_RESIZE_ALL; break;
      case kTopRight:     rlCursor = MOUSE_CURSOR_RESIZE_ALL; break;
      case kBottomSide:   rlCursor = MOUSE_CURSOR_RESIZE_NS; break;
      case kLeftSide:     rlCursor = MOUSE_CURSOR_RESIZE_EW; break;
      case kTopSide:      rlCursor = MOUSE_CURSOR_RESIZE_NS; break;
      case kRightSide:    rlCursor = MOUSE_CURSOR_RESIZE_EW; break;
      case kMove:         rlCursor = MOUSE_CURSOR_POINTING_HAND; break;
      case kCross:        rlCursor = MOUSE_CURSOR_CROSSHAIR; break;
      case kArrowHor:     rlCursor = MOUSE_CURSOR_RESIZE_EW; break;
      case kArrowVer:     rlCursor = MOUSE_CURSOR_RESIZE_NS; break;
      case kHand:         rlCursor = MOUSE_CURSOR_POINTING_HAND; break;
      case kRotate:       rlCursor = MOUSE_CURSOR_POINTING_HAND; break;
      case kPointer:      rlCursor = MOUSE_CURSOR_ARROW; break;
      case kArrowRight:   rlCursor = MOUSE_CURSOR_RESIZE_EW; break;
      case kCaret:        rlCursor = MOUSE_CURSOR_IBEAM; break;
      case kWatch:        rlCursor = MOUSE_CURSOR_ARROW; break;
      case kNoDrop:       rlCursor = MOUSE_CURSOR_NOT_ALLOWED; break;
      default:            rlCursor = MOUSE_CURSOR_ARROW; break;
   }
   SetMouseCursor(rlCursor);
}

// ========================== SaveImage ===================================

void TRaylibPadPainter::SaveImage(TVirtualPad *, const char *, Int_t /*gtype*/) const
{
   // if (fileName && !WindowShouldClose())
   //   ScreenShot(fileName, 0);
}

// ========================== DrawPixels ==================================

void TRaylibPadPainter::DrawPixels(const unsigned char *pixelData, UInt_t width, UInt_t height,
                                   Int_t dstX, Int_t dstY, Bool_t enableAlphaBlending)
{
   if (!pixelData || width == 0 || height == 0)
      return;

   Image img = {
        .data = (Color *) pixelData,
        .width = (int) width,
        .height = (int) height,
        .mipmaps = 1,
        .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
   };

   Texture2D tex = LoadTextureFromImage(img);
   UnloadImage(img);

   if (enableAlphaBlending)
      BeginBlendMode(BLEND_ALPHA_PREMULTIPLY);

   DrawTexture(tex, dstX, dstY, WHITE);

   if (enableAlphaBlending)
      EndBlendMode();
   UnloadTexture(tex);
}

// =========================== DrawLine ===================================

void TRaylibPadPainter::DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   auto &att = GetAttLine();
   Width_t lw = att.GetLineWidth();
   if (lw <= 0)
      return;

   Int_t px1 = gPad->XtoAbsPixel(x1);
   Int_t py1 = gPad->YtoAbsPixel(y1);
   Int_t px2 = gPad->XtoAbsPixel(x2);
   Int_t py2 = gPad->YtoAbsPixel(y2);

   Color col = GetRaylibColor(att.GetLineColor());

   if (lw > 1) {
      ::DrawLineEx((Vector2){(float)px1, (float)py1}, (Vector2){(float)px2, (float)py2}, (int)lw, col);
   } else {
      ::DrawLine(px1, py1, px2, py2, col);
   }
}

void TRaylibPadPainter::DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2)
{
   auto &att = GetAttLine();
   Width_t lw = att.GetLineWidth();
   if (lw <= 0)
      return;

   Int_t px1 = gPad->UtoAbsPixel(u1);
   Int_t py1 = gPad->VtoAbsPixel(v1);
   Int_t px2 = gPad->UtoAbsPixel(u2);
   Int_t py2 = gPad->VtoAbsPixel(v2);

   Color col = GetRaylibColor(att.GetLineColor());

   if (lw > 1) {
      ::DrawLineEx((Vector2){(float)px1, (float)py1}, (Vector2){(float)px2, (float)py2}, (int)lw, col);
   } else {
      ::DrawLine(px1, py1, px2, py2, col);
   }
}

// ============================ DrawBox ===================================

void TRaylibPadPainter::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, EBoxMode mode)
{
   if (mode == kHollow && GetAttLine().GetLineWidth() <= 0)
      return;

   Int_t px1 = gPad->XtoAbsPixel(x1);
   Int_t py1 = gPad->YtoAbsPixel(y1);
   Int_t px2 = gPad->XtoAbsPixel(x2);
   Int_t py2 = gPad->YtoAbsPixel(y2);

   int left   = std::min(px1, px2);
   int top    = std::min(py1, py2);
   int width  = std::abs(px2 - px1);
   int height = std::abs(py2 - py1);

   Rectangle rect = {(float)left, (float)top, (float)width, (float)height};

   if (mode == kHollow) {
      Color col = GetRaylibColor(GetAttLine().GetLineColor());
      Width_t lw = GetAttLine().GetLineWidth();
      DrawRectangleLinesEx(rect, (int)(lw > 0 ? lw : 1), col);
   } else {
      Color col = GetRaylibColor(GetAttFill().GetFillColor());
      float alpha = GetCurrentAlpha();
      if (alpha < 1.0f) {
         BeginBlendMode(BLEND_ALPHA_PREMULTIPLY);
         DrawRectangleRec(rect, Fade(col, alpha));
         EndBlendMode();
      } else {
         DrawRectangleRec(rect, col);
      }
   }
}

// ======================== DrawFillArea ==================================


void drawTriangleSafe(const Vector2 &a, const Vector2 &b, const Vector2 &c, Color color)
{
   // Ensure raylib reads it correctly regardless of your point layout orientation
   // (Uses the cross-product check logic)
   float cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
   if (cross < 0.0f) {
      DrawTriangle(a, b, c, color);
   } else {
      DrawTriangle(a, c, b, color);
   }
}

namespace mapbox {
namespace util {
template <>
struct nth<0, Vector2> { inline static float get(const Vector2& t) { return t.x; }; };
template <>
struct nth<1, Vector2> { inline static float get(const Vector2& t) { return t.y; }; };
}
}

void drawFilledPolygon(const std::vector<Vector2>& points, Color color)
{
   if (points.size() < 3) return; // A polygon must have at least 3 points

   // Earcut expects a nested vector where the first element is the main outline,
   // and subsequent elements are optional interior holes.
   std::vector<std::vector<Vector2>> polygon;
   polygon.push_back(points);

   // Run triangulation -> returns an array of indices (grouped by 3s for triangles)
   std::vector<uint32_t> indices = mapbox::earcut<uint32_t>(polygon);

   // Loop through the indices and draw each triangle
   for (size_t i = 0; i < indices.size(); i += 3)
      drawTriangleSafe(points[indices[i]], points[indices[i + 1]], points[indices[i + 2]], color);
}


void TRaylibPadPainter::DrawFillArea(Int_t nPoints, const Double_t *xs, const Double_t *ys)
{
   if (nPoints < 3 || GetAttFill().GetFillStyle() <= 0)
      return;

   Color col = GetRaylibColor(GetAttFill().GetFillColor());

   std::vector<Vector2> verts((size_t)nPoints);
   for (Int_t n = 0; n < nPoints; ++n) {
      verts[(size_t)n] = {(float)gPad->XtoAbsPixel(xs[n]), (float)gPad->YtoAbsPixel(ys[n])};
   }

   if (col.a < 255) {
      BeginBlendMode(BLEND_ALPHA_PREMULTIPLY);
      drawFilledPolygon(verts, col);
      EndBlendMode();
   } else {
      drawFilledPolygon(verts, col);
   }
}

void TRaylibPadPainter::DrawFillArea(Int_t nPoints, const Float_t *xs, const Float_t *ys)
{
   if (nPoints < 3 || GetAttFill().GetFillStyle() <= 0)
      return;

   Color col = GetRaylibColor(GetAttFill().GetFillColor());

   std::vector<Vector2> verts((size_t)nPoints);
   for (Int_t n = 0; n < nPoints; ++n)
      verts[(size_t)n] = {(float)gPad->XtoAbsPixel(xs[n]), (float)gPad->YtoAbsPixel(ys[n])};

   if (col.a < 255) {
      BeginBlendMode(BLEND_ALPHA_PREMULTIPLY);
      drawFilledPolygon(verts, col);
      EndBlendMode();
   } else {
      drawFilledPolygon(verts, col);
   }
}

// ========================= DrawPolyLine =================================


void DrawLineStripEx(const std::vector<Vector2> verts, int lw, Color col)
{
   if (lw <= 1)
      DrawLineStrip(verts.data(), verts.size(), col);
   else
      for (std::size_t i = 1; i < verts.size(); ++i)
         DrawLineEx(verts[i-1], verts[i], lw, col);
}


void TRaylibPadPainter::DrawPolyLine(Int_t nPoints, const Double_t *xs, const Double_t *ys)
{
   if (nPoints < 2 || GetAttLine().GetLineWidth() <= 0)
      return;

   Color col = GetRaylibColor(GetAttLine().GetLineColor());
   Width_t lw = GetAttLine().GetLineWidth();

   std::vector<Vector2> verts((size_t)nPoints);
   for (Int_t n = 0; n < nPoints; ++n)
      verts[n] = {(float)gPad->XtoAbsPixel(xs[n]), (float)gPad->YtoAbsPixel(ys[n])};

   DrawLineStripEx(verts, lw, col);
}

void TRaylibPadPainter::DrawPolyLine(Int_t nPoints, const Float_t *xs, const Float_t *ys)
{
   if (nPoints < 2 || GetAttLine().GetLineWidth() <= 0)
      return;

   Color col = GetRaylibColor(GetAttLine().GetLineColor());
   Width_t lw = GetAttLine().GetLineWidth();

   std::vector<Vector2> verts((size_t)nPoints);
   for (Int_t n = 0; n < nPoints; ++n)
      verts[n] = {(float)gPad->XtoAbsPixel(xs[n]), (float)gPad->YtoAbsPixel(ys[n])};

   DrawLineStripEx(verts, lw, col);
}

void TRaylibPadPainter::DrawPolyLineNDC(Int_t nPoints, const Double_t *u, const Double_t *v)
{
   if (nPoints < 2 || GetAttLine().GetLineWidth() <= 0)
      return;

   Color col = GetRaylibColor(GetAttLine().GetLineColor());
   Width_t lw = GetAttLine().GetLineWidth();

   std::vector<Vector2> verts((size_t)nPoints);
   for (Int_t n = 0; n < nPoints; ++n) {
      verts[(size_t)n] = {(float)gPad->UtoAbsPixel(u[n]), (float)gPad->VtoAbsPixel(v[n])};
   }

   if (lw > 1) {
      for (Int_t i = 1; i < nPoints; ++i)
         DrawLineEx(verts[(size_t)(i-1)], verts[(size_t)i], (int)lw, col);
   } else {
      DrawLineStrip(verts.data(), nPoints, col);
   }
}

// ======================= DrawPolyMarker =================================

template<typename T>
void drawPolyMarkerImpl(Int_t nPoints, const T *x, const T *y, const TAttMarker &attr, Color col)
{
   Int_t markerSize = 0;
   std::vector<TPoint> markerShape;
   auto markerType = attr.GetMarkerShape(markerSize, markerShape, 1., kTRUE);
   auto markerLineWidth = TAttMarker::GetMarkerLineWidth(attr.GetMarkerStyle());

   for (Int_t n = 0; n < nPoints; ++n) {
      Int_t px = gPad->XtoAbsPixel(x[n]);
      Int_t py = gPad->YtoAbsPixel(y[n]);

      switch (markerType) {
         case TAttMarker::kShapeDot:
            DrawPixelV((Vector2){(float)px, (float)py}, col);
            break;

         case TAttMarker::kShapeCircle:
         case TAttMarker::kShapeFilledCircle: {
            int radius = std::max(markerSize / 2, 1);
            Vector2 center{(float)px, (float)py};
            if (markerType == TAttMarker::kShapeFilledCircle)
               DrawCircleV(center, radius, col);
            else if (markerLineWidth > 1)
               DrawRing(center, radius, radius + markerLineWidth - 1, 0, 360, 60, col);
            else
               DrawCircleLinesV(center, radius, col);
            break;
         }

         case TAttMarker::kShapePolyLine: {
            std::vector<Vector2> verts(markerShape.size());
            for (size_t j = 0; j < markerShape.size(); ++j)
               verts[j] = {(float)(px + markerShape[j].fX), (float)(py + markerShape[j].fY)};
            DrawLineStripEx(verts, markerLineWidth, col);
            break;
         }

         case TAttMarker::kShapeFilledArea: {
            std::vector<Vector2> verts(markerShape.size());
            for (size_t j = 0; j < markerShape.size(); ++j)
               verts[j] = {(float)(px + markerShape[j].fX), (float)(py + markerShape[j].fY)};
            drawFilledPolygon(verts, col);
            break;
         }

         case TAttMarker::kShapeSegments: {
            for (size_t j = 0; j + 1 < markerShape.size(); j += 2) {
               Vector2 p1 = {(float)(px + markerShape[j].fX),   (float)(py + markerShape[j].fY)};
               Vector2 p2 = {(float)(px + markerShape[j+1].fX), (float)(py + markerShape[j+1].fY)};
               DrawLineStripEx({p1, p2}, markerLineWidth, col);
            }
            break;
         }

         case TAttMarker::kShapeTriangles: {
            for (size_t j = 0; j + 2 < markerShape.size(); j += 3) {
               Vector2 t1 = {(float)(px + markerShape[j].fX),   (float)(py + markerShape[j].fY)};
               Vector2 t2 = {(float)(px + markerShape[j+1].fX), (float)(py + markerShape[j+1].fY)};
               Vector2 t3 = {(float)(px + markerShape[j+2].fX), (float)(py + markerShape[j+2].fY)};
               drawTriangleSafe(t1, t2, t3, col);
            }
            break;
         }
      }
   }
}

void TRaylibPadPainter::DrawPolyMarker(Int_t nPoints, const Double_t *x, const Double_t *y)
{
   drawPolyMarkerImpl<Double_t>(nPoints, x, y, GetAttMarker(), GetRaylibColor(GetAttMarker().GetMarkerColor()));
}

void TRaylibPadPainter::DrawPolyMarker(Int_t nPoints, const Float_t *x, const Float_t *y)
{
   drawPolyMarkerImpl<Float_t>(nPoints, x, y, GetAttMarker(), GetRaylibColor(GetAttMarker().GetMarkerColor()));
}

// =========================== DrawText ===================================

// Helper: convert wchar_t to UTF-8 string
static std::string WCharToUtf8(const wchar_t *text)
{
   std::string utf8;
   for (const wchar_t *p = text; *p; ++p) {
      unsigned int cp = *p;
      if (cp < 0x80) {
         utf8 += (char)cp;
      } else if (cp < 0x800) {
         utf8 += (char)(0xC0 | (cp >> 6));
         utf8 += (char)(0x80 | (cp & 0x3F));
      } else {
         utf8 += (char)(0xE0 | (cp >> 12));
         utf8 += (char)(0x80 | ((cp >> 6) & 0x3F));
         utf8 += (char)(0x80 | (cp & 0x3F));
      }
   }
   return utf8;
}

// Helper: paint text at given pixel position
void TRaylibPadPainter::PaintTextHelper(int px, int py, const char *text)
{
   const TAttText &att = GetAttText();
   Font font = LoadFontForId(att.GetTextFont());
   if (font.texture.id == 0)
      return;

   Float_t textsize = att.GetTextSizePixels(*gPad);
   Int_t pixelsize = (Int_t)(textsize * kScale + 0.5);
   if (pixelsize <= 0) pixelsize = 10;

   Color col = GetRaylibColor(att.GetTextColor());

   const float spacing = 1.0f;

   // Measure text for alignment
   Vector2 textSize = MeasureTextEx(font, text, (float)pixelsize, spacing);

   Int_t txalh = att.GetTextAlign() / 10;
   Int_t txalv = att.GetTextAlign() % 10;

   float drawX = (float)px;
   float drawY = (float)py;

   // Horizontal alignment
   switch (txalh) {
      case 0: case 1: break;                              // left
      case 2: drawX -= textSize.x / 2.0f; break;          // center
      case 3: drawX -= textSize.x; break;                 // right
   }

   // Vertical alignment
   switch (txalv) {
      case 1: drawY -= textSize.y; break;                 // bottom
      case 2: drawY -= textSize.y / 2.0f; break;          // middle
      case 3: break;                                      // top
   }

   float angle = -att.GetTextAngle();

   if (angle == 0.0f) {
      DrawTextEx(font, text, (Vector2){drawX, drawY}, (float) pixelsize, spacing, col);
   } else {
      // Rotated text via DrawTextPro
      // Rectangle sourceRec = {0, 0, (float)font.texture.width, (float)font.texture.height};
      Vector2 origin = {0, 0};
      DrawTextPro(font, text, (Vector2){drawX, drawY}, origin, angle,
                  (float)pixelsize, spacing, col);
   }

   UnloadFont(font);
}

void TRaylibPadPainter::DrawText(Double_t x, Double_t y, const char *text, ETextMode /*mode*/)
{
   Int_t px = gPad->XtoAbsPixel(x);
   Int_t py = gPad->YtoAbsPixel(y);
   PaintTextHelper(px, py, text);
}

void TRaylibPadPainter::DrawText(Double_t x, Double_t y, const wchar_t *text, ETextMode /*mode*/)
{
   Int_t px = gPad->XtoAbsPixel(x);
   Int_t py = gPad->YtoAbsPixel(y);
   std::string utf8 = WCharToUtf8(text);
   PaintTextHelper(px, py, utf8.c_str());
}

void TRaylibPadPainter::DrawTextNDC(Double_t u, Double_t v, const char *text, ETextMode /*mode*/)
{
   Int_t px = gPad->UtoAbsPixel(u);
   Int_t py = gPad->VtoAbsPixel(v);
   PaintTextHelper(px, py, text);
}

void TRaylibPadPainter::DrawTextNDC(Double_t u, Double_t v, const wchar_t *text, ETextMode /*mode*/)
{
   Int_t px = gPad->UtoAbsPixel(u);
   Int_t py = gPad->VtoAbsPixel(v);
   std::string utf8 = WCharToUtf8(text);
   PaintTextHelper(px, py, utf8.c_str());
}

void TRaylibPadPainter::DrawTextUrl(Double_t x, Double_t y, const char *text, const char * /*url*/)
{
   Int_t px = gPad->XtoAbsPixel(x);
   Int_t py = gPad->YtoAbsPixel(y);
   PaintTextHelper(px, py, text);
}

// ======================== Text Measurement ===============================

void TRaylibPadPainter::GetTextExtent(Font_t fontid, Double_t size, UInt_t &w, UInt_t &h, const char *mess)
{
   Font rlFont = LoadFontForId(fontid);
   if (rlFont.texture.id == 0) {
      w = h = 0;
      return;
   }

   Int_t pixelsize = (Int_t)(size * kScale + 0.5);
   if (pixelsize <= 0)
      pixelsize = 10;

   Vector2 ts = MeasureTextEx(rlFont, mess, (float)pixelsize, 0.0f);
   w = (UInt_t)(ts.x + 0.5f);
   h = (UInt_t)(ts.y + 0.5f);

   UnloadFont(rlFont);
}

void TRaylibPadPainter::GetTextExtent(Font_t font, Double_t size, UInt_t &w, UInt_t &h, const wchar_t *mess)
{
   std::string utf8 = WCharToUtf8(mess);
   GetTextExtent(font, size, w, h, utf8.c_str());
}

void TRaylibPadPainter::GetTextAscentDescent(Font_t fontid, Double_t size, UInt_t &a, UInt_t &d, const char *mess)
{
   Font rlFont = LoadFontForId(fontid);
   if (rlFont.texture.id == 0) {
      a = d = 0;
      return;
   }

   Int_t pixelsize = (Int_t)(size * kScale + 0.5);
   if (pixelsize <= 0) pixelsize = 10;

   // Use bounding box from a measurement as ascent/descent approximation
   Vector2 ts = MeasureTextEx(rlFont, mess, (float)pixelsize, 0.0f);
   a = (UInt_t)(ts.y * 0.8 + 0.5f);   // ~80% ascent (typographic convention)
   d = (UInt_t)(ts.y * 0.2 + 0.5f);   // ~20% descent

   UnloadFont(rlFont);
}

void TRaylibPadPainter::GetTextAscentDescent(Font_t font, Double_t size, UInt_t &a, UInt_t &d, const wchar_t *mess)
{
   std::string utf8 = WCharToUtf8(mess);
   GetTextAscentDescent(font, size, a, d, utf8.c_str());
}

UInt_t TRaylibPadPainter::GetTextAdvance(Font_t fontid, Double_t size, const char *text, Bool_t /*kern*/)
{
   Font rlFont = LoadFontForId(fontid);
   if (rlFont.texture.id == 0)
      return 0;

   Int_t pixelsize = (Int_t)(size * kScale + 0.5);
   if (pixelsize <= 0) pixelsize = 10;

   Vector2 ts = MeasureTextEx(rlFont, text, (float)pixelsize, 0.0f);
   UInt_t advance = (UInt_t)(ts.x + 0.5f);

   UnloadFont(rlFont);
   return advance;
}

