// Author:  Sergey Linev, GSI  12/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGtk4PadPainter.h"
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
#include "TTFhandle.h"
#include "TImage.h"

#include <memory>
#include <map>

#include "Gtk4DrawArea.h"

#include <cairomm/context.h>
#include <cairomm/surface.h>
#include <cairo.h>



using namespace ROOT::Experimental;

// to scale fonts to the same size as in the TTF
const Float_t kScale = 0.93376068;

/** \class TGtk4PadPainter
    \ingroup gtk4canvas
    \brief Implement TVirtualPadPainter for Gtk4 graphics

   Uses Gtk4 DrawArea which creates Cairo::Context during update.
*/

//////////////////////////////////////////////////////////////////////////
/// Set opacity - similar to TVirtualPS usecase

void TGtk4PadPainter::SetOpacity(Int_t percent)
{
   fAttFill.SetFillStyle(4000 + percent);
}

////////////////////////////////////////////////////////////////////////////////
///Noop, for non-gl pad TASImage calls gVirtualX->CopyArea.

void TGtk4PadPainter::DrawPixels(const unsigned char * /*pixelData*/, UInt_t /*width*/, UInt_t /*height*/,
                             Int_t /*dstX*/, Int_t /*dstY*/, Bool_t /*enableAlphaBlending*/)
{

}


////////////////////////////////////////////////////////////////////////////////
/// Assign color to cairo context

void TGtk4PadPainter::SetGtk4Color(Color_t id)
{
   auto ctx = fDrawArea->GetContext();
   if (!ctx)
      return;

   auto c = gROOT->GetColor(id);
   if (c)
      ctx->set_source_rgba(c->GetRed(), c->GetGreen(), c->GetBlue(), c->GetAlpha());
   else
      ctx->set_source_rgb(0, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
/// Asign pen attributes to draw line

Bool_t TGtk4PadPainter::SetLinePen()
{
   auto ctx = fDrawArea->GetContext();
   if (!ctx)
      return kFALSE;

   auto &att = GetAttLine();
   if (att.GetLineWidth() <= 0)
      return kFALSE;

   SetGtk4Color(att.GetLineColor());
   ctx->set_line_width(att.GetLineWidth());

   auto style = att.GetLineStyle();

   TString patt;
   std::vector<double> pattern;

   if (style > 1)
      patt = gStyle->GetLineStyleString(style);

   if (patt.Length() > 2) {
      std::unique_ptr<TObjArray> tokens(patt.Tokenize(" "));
      for (Int_t j = 0; j < tokens->GetEntries(); j++) {
         int it = std::stoi(tokens->At(j)->GetName());
         pattern.emplace_back(0.1 * it * att.GetLineWidth());
      }
   }

   if (pattern.size() > 1)
      ctx->set_dash(pattern, 0);
   else
      ctx->unset_dash();

   return kTRUE;
}


////////////////////////////////////////////////////////////////////////////////
/// Asign fill attributes before drawin shape

Bool_t TGtk4PadPainter::SetFillBrush()
{
   auto ctx = fDrawArea->GetContext();
   if (!ctx)
      return kFALSE;

   auto &att = GetAttFill();
   fCustomPattern = 0;

   Int_t style = att.GetFillStyle() / 1000;

   if (style == 1) {
      SetGtk4Color(att.GetFillColor());
      return kTRUE;
   }

   if (style != 3)
      return kFALSE;

   Int_t fasi = att.GetFillStyle() % 1000;
   fCustomPattern = (fasi >= 1 && fasi <=25) ? fasi : 2;

   ctx->save();
   SetGtk4Color(att.GetFillColor());
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Complete fill operation - which may include custom pattern handling

void TGtk4PadPainter::ApplyFillBrush()
{
   auto ctx = fDrawArea->GetContext();

   if (fCustomPattern > 0) {

      unsigned char *bitmask_bytes = (unsigned char *) gStipples[fCustomPattern];

      const int width = 16, height = 16;
      auto format = Cairo::Surface::Format::A1;
      auto stride = Cairo::ImageSurface::format_stride_for_width(format, width);

      std::vector<unsigned char> padded_bytes(stride * height, 0);
      for (int row = 0; row < height; ++row) {
         padded_bytes[row * stride] = bitmask_bytes[row * 2];
         padded_bytes[row * stride + 1] = bitmask_bytes[row * 2 + 1];
      }

      auto mask_surface = Cairo::ImageSurface::create(padded_bytes.data(), format, width, height, stride);

      auto pattern = Cairo::SurfacePattern::create(mask_surface);
      pattern->set_extend(Cairo::Pattern::Extend::REPEAT);

      ctx->clip();

      ctx->mask(pattern);

      ctx->restore();

      fCustomPattern = 0;
   } else {

      ctx->fill();
   }

}



////////////////////////////////////////////////////////////////////////////////
/// Paint a simple line.

void TGtk4PadPainter::DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   if (!SetLinePen())
      return;

   const Int_t px1 = gPad->XtoAbsPixel(x1);
   const Int_t py1 = gPad->YtoAbsPixel(y1);
   const Int_t px2 = gPad->XtoAbsPixel(x2);
   const Int_t py2 = gPad->YtoAbsPixel(y2);

   auto ctx = fDrawArea->GetContext();
   ctx->move_to(px1, py1);
   ctx->line_to(px2, py2);
   ctx->stroke();
}


////////////////////////////////////////////////////////////////////////////////
/// Paint a simple line in normalized coordinates.

void TGtk4PadPainter::DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2)
{
   if (!SetLinePen())
      return;

   const Int_t px1 = gPad->UtoAbsPixel(u1);
   const Int_t py1 = gPad->VtoAbsPixel(v1);
   const Int_t px2 = gPad->UtoAbsPixel(u2);
   const Int_t py2 = gPad->VtoAbsPixel(v2);

   auto ctx = fDrawArea->GetContext();
   ctx->move_to(px1, py1);
   ctx->line_to(px2, py2);
   ctx->stroke();
}

////////////////////////////////////////////////////////////////////////////////
/// Paint a simple box.

void TGtk4PadPainter::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, EBoxMode mode)
{
   const Int_t px1 = gPad->XtoAbsPixel(x1);
   const Int_t py1 = gPad->YtoAbsPixel(y1);
   const Int_t px2 = gPad->XtoAbsPixel(x2);
   const Int_t py2 = gPad->YtoAbsPixel(y2);

   auto ctx = fDrawArea->GetContext();
   if (!ctx)
      return;

   if (mode == TVirtualPadPainter::kHollow) {
      if (SetLinePen()) {
         ctx->rectangle(TMath::Min(px1, px2), TMath::Min(py1, py2), TMath::Abs(px2 - px1), TMath::Abs(py2 - py1));
         ctx->stroke();
      }
   } else {
      if (SetFillBrush()) {
         ctx->rectangle(TMath::Min(px1, px2), TMath::Min(py1, py2), TMath::Abs(px2 - px1), TMath::Abs(py2 - py1));
         ApplyFillBrush();
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint filled area.

void TGtk4PadPainter::DrawFillArea(Int_t nPoints, const Double_t *xs, const Double_t *ys)
{
   if ((nPoints < 3) || !SetFillBrush())
      return;

   auto ctx = fDrawArea->GetContext();

   for (Int_t n = 0; n < nPoints; ++n) {
      auto px = gPad->XtoAbsPixel(xs[n]);
      auto py = gPad->YtoAbsPixel(ys[n]);
      if (n == 0)
         ctx->move_to(px, py);
      else
         ctx->line_to(px, py);
   }

   ctx->close_path();

   ApplyFillBrush();
}

////////////////////////////////////////////////////////////////////////////////
/// Paint filled area.

void TGtk4PadPainter::DrawFillArea(Int_t nPoints, const Float_t *xs, const Float_t *ys)
{
   if ((nPoints < 3) || !SetFillBrush())
      return;

   auto ctx = fDrawArea->GetContext();

   for (Int_t n = 0; n < nPoints; ++n) {
      auto px = gPad->XtoAbsPixel(xs[n]);
      auto py = gPad->YtoAbsPixel(ys[n]);
      if (n == 0)
         ctx->move_to(px, py);
      else
         ctx->line_to(px, py);
   }

   ctx->close_path();

   ApplyFillBrush();
}

////////////////////////////////////////////////////////////////////////////////
/// Paint Polyline.

void TGtk4PadPainter::DrawPolyLine(Int_t nPoints, const Double_t *xs, const Double_t *ys)
{
   if ((nPoints < 2) || !SetLinePen())
      return;

   auto ctx = fDrawArea->GetContext();

   for (Int_t n = 0; n < nPoints; ++n) {
      auto px = gPad->XtoAbsPixel(xs[n]);
      auto py = gPad->YtoAbsPixel(ys[n]);
      if (n == 0)
         ctx->move_to(px, py);
      else
         ctx->line_to(px, py);
   }

   ctx->stroke();
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polyline.

void TGtk4PadPainter::DrawPolyLine(Int_t nPoints, const Float_t *xs, const Float_t *ys)
{
   if ((nPoints < 2) || !SetLinePen())
      return;

   auto ctx = fDrawArea->GetContext();

   for (Int_t n = 0; n < nPoints; ++n) {
      auto px = gPad->XtoAbsPixel(xs[n]);
      auto py = gPad->YtoAbsPixel(ys[n]);
      if (n == 0)
         ctx->move_to(px, py);
      else
         ctx->line_to(px, py);
   }

   ctx->stroke();
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polyline in normalized coordinates.

void TGtk4PadPainter::DrawPolyLineNDC(Int_t nPoints, const Double_t *u, const Double_t *v)
{
   if ((nPoints < 2) || !SetLinePen())
      return;

   auto ctx = fDrawArea->GetContext();

   for (Int_t n = 0; n < nPoints; ++n) {
      auto px = gPad->UtoAbsPixel(u[n]);
      auto py = gPad->VtoAbsPixel(v[n]);
      if (n == 0)
         ctx->move_to(px, py);
      else
         ctx->line_to(px, py);
   }

   ctx->stroke();
}

////////////////////////////////////////////////////////////////////////////////
/// Helper function to paint markers

template<typename T>
void drawMarkersHelper(Int_t nPoints, const T *x, const T *y,
                       const TAttMarker &attr, Cairo::Context *ctx)
{
   if (!ctx || nPoints < 1)
      return;

   auto markerLineWidth = TAttMarker::GetMarkerLineWidth(attr.GetMarkerStyle());
   Int_t markerSize = 0;              ///< size of simple markers
   std::vector<TPoint> markerShape;   ///< marker shape points
   auto markerType = attr.GetMarkerShape(markerSize, markerShape);

   if ((markerType == TAttMarker::kShapeDot) || (markerLineWidth > 0)) {
      ctx->set_line_width(markerLineWidth);
   } else {
   }

   for (Int_t n = 0; n < nPoints; ++n) {
      Int_t px = gPad->XtoAbsPixel(x[n]);
      Int_t py = gPad->YtoAbsPixel(y[n]);
      Int_t cnt = 0;

      switch(markerType) {
         case TAttMarker::kShapeDot:
            ctx->rectangle(px + 0.5, py + 0.5, 1., 1.);
            ctx->fill();
            break;
         case TAttMarker::kShapeCircle:
            ctx->arc(px, py, markerSize / 2, 0.0, 2.0 * TMath::Pi());
            ctx->stroke();
            break;
         case TAttMarker::kShapeFilledCircle:
            // hollow or filled circle
            ctx->arc(px, py, markerSize / 2, 0.0, 2.0 * TMath::Pi());
            ctx->fill();
            break;
         case TAttMarker::kShapePolyLine:
            // hollow polygon
            for (auto &pnt : markerShape)
               if (cnt++)
                  ctx->line_to(px + pnt.fX, py + pnt.fY);
               else
                  ctx->move_to(px + pnt.fX, py + pnt.fY);
            ctx->stroke();
            break;
         case TAttMarker::kShapeFilledArea:
            // filled polygon
            for (auto &pnt : markerShape)
               if (cnt++)
                  ctx->line_to(px + pnt.fX, py + pnt.fY);
               else
                  ctx->move_to(px + pnt.fX, py + pnt.fY);
            ctx->close_path();
            ctx->fill();
            break;
         case TAttMarker::kShapeSegments:
            // segmented line
            for (size_t i = 0; i < markerShape.size() - 1; i += 2) {
               ctx->move_to(px + markerShape[i].fX, py + markerShape[i].fY);
               ctx->line_to(px + markerShape[i+1].fX, py + markerShape[i+1].fY);
               if (i == markerShape.size() - 2)
                  ctx->stroke();
               else
                  ctx->stroke_preserve();
            }
            break;
         case TAttMarker::kShapeTriangles:
            // filled triangles
            for (size_t i = 0; i < markerShape.size() - 2; i += 3) {
               ctx->move_to(px + markerShape[i].fX, py + markerShape[i].fY);
               ctx->line_to(px + markerShape[i+1].fX, py + markerShape[i+1].fY);
               ctx->line_to(px + markerShape[i+2].fX, py + markerShape[i+2].fY);
               ctx->close_path();
               if (i == markerShape.size() - 3)
                  ctx->fill();
               else
                  ctx->fill_preserve();
            }
            break;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker.

void TGtk4PadPainter::DrawPolyMarker(Int_t nPoints, const Double_t *x, const Double_t *y)
{
   SetGtk4Color(GetAttMarker().GetMarkerColor());

   drawMarkersHelper<Double_t>(nPoints, x, y,
                              GetAttMarker(), fDrawArea->GetContext());
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker.

void TGtk4PadPainter::DrawPolyMarker(Int_t nPoints, const Float_t *x, const Float_t *y)
{
   SetGtk4Color(GetAttMarker().GetMarkerColor());

   drawMarkersHelper<Float_t>(nPoints, x, y,
                              GetAttMarker(), fDrawArea->GetContext());
}

////////////////////////////////////////////////////////////////////////////////
/// Produce image

void TGtk4PadPainter::SaveImage(TVirtualPad * /* pad */, const char * /* fileName */, Int_t /* gtype */) const
{
}

////////////////////////////////////////////////////////////////////////////////
/// Render TTF glyphs on drawable area

void TGtk4PadPainter::DrawTTFglyphs(Int_t px, Int_t py, TTFhandle &ttf, ETextMode /* mode */)
{
   // position inside the pad is provided, therefore shift it to global image
   px += fPad->UtoAbsPixel(0);
   py += fPad->VtoAbsPixel(1);

   auto ctx = fDrawArea->GetContext();

   SetGtk4Color(GetAttText().GetTextColor());

   for (UInt_t nglyph = 0; nglyph < ttf.GetNumGlyphs(); nglyph++) {
      Int_t bx = 0, by = 0;
      UChar_t *buffer = nullptr;
      UInt_t width = 0, rows = 0, pitch = 0;
      if (!ttf.GetGlyphData(nglyph, bx, by, buffer, width, rows, pitch))
         continue;

      // Cairo's A8 surfaces need their own row stride (padding), which
       // usually differs from FreeType's bmp.pitch - copy row by row
      auto stride = Cairo::ImageSurface::format_stride_for_width(Cairo::Surface::Format::A8, width);
      std::vector<unsigned char> data(stride * rows, 0);

      for (unsigned int row = 0; row < rows; ++row)
         std::memcpy(data.data() + row * stride, buffer + row * pitch, width);

      auto surface =
         Cairo::ImageSurface::create(data.data(), Cairo::Surface::Format::A8, width, rows, stride);

      ctx->mask(surface, px + bx, py + by);
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Render TTF glyphs on drawable area

void TGtk4PadPainter::DrawImage(TImage *img, Int_t px, Int_t py, Int_t)
{
   // position inside the pad is provided, therefore shift it to global image
   px += fPad->UtoAbsPixel(0);
   py += fPad->VtoAbsPixel(1);

   auto width = img->GetWidth();
   auto height = img->GetHeight();
   auto argbBuffer = (unsigned char*) img->GetArgbArray();

   auto ctx = fDrawArea->GetContext();

   int stride = Cairo::ImageSurface::format_stride_for_width(Cairo::Surface::Format::ARGB32, width);

    // 2. Wrap your raw memory pointer into a Cairo Surface.
    // This borrows the memory pointer without making a deep copy.
    Cairo::RefPtr<Cairo::ImageSurface> surface =
        Cairo::ImageSurface::create(argbBuffer, Cairo::Surface::Format::ARGB32, width, height, stride);

    // 3. Set the surface as the source pattern in the context
    // (0, 0) specifies the top-left offset to place the image
    ctx->set_source(surface, px, py);

    // 4. Paint the pattern onto the context canvas
    ctx->paint();
}
