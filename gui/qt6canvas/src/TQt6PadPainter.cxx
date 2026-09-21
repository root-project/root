// Author:  Sergey Linev, GSI  26/06/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TQt6PadPainter.h"
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

#include "QPaintWidget.h"

#include <QRect>
#include <QImage>
#include <QPainter>

using namespace ROOT::Experimental;


/** \class TQt6PadPainter
    \ingroup qt6canvas
    \brief Implement TVirtualPadPainter for Qt6 graphics

   Uses QPainter object which only exists inside paintEvent of Qt.
*/

//////////////////////////////////////////////////////////////////////////
/// Set opacity - similar to TVirtualPS usecase

void TQt6PadPainter::SetOpacity(Int_t percent)
{
   fAttFill.SetFillStyle(4000 + percent);
}


////////////////////////////////////////////////////////////////////////////////
///Noop, for non-gl pad TASImage calls gVirtualX->CopyArea.

void TQt6PadPainter::DrawPixels(const unsigned char * /*pixelData*/, UInt_t /*width*/, UInt_t /*height*/,
                             Int_t /*dstX*/, Int_t /*dstY*/, Bool_t /*enableAlphaBlending*/)
{

}


////////////////////////////////////////////////////////////////////////////////
/// Paint a simple line.

void TQt6PadPainter::DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   auto painter = fPaintWidget->getPainter();
   if (!painter || GetAttLine().GetLineWidth() <= 0)
      return;

   const Int_t px1 = gPad->XtoAbsPixel(x1);
   const Int_t py1 = gPad->YtoAbsPixel(y1);
   const Int_t px2 = gPad->XtoAbsPixel(x2);
   const Int_t py2 = gPad->YtoAbsPixel(y2);

   painter->setPen(GetLinePen());

   painter->setRenderHint(QPainter::Antialiasing);

   painter->drawLine(QPoint(px1, py1), QPoint(px2, py2));
}


////////////////////////////////////////////////////////////////////////////////
/// Paint a simple line in normalized coordinates.

void TQt6PadPainter::DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2)
{
   auto painter = fPaintWidget->getPainter();
   if (!painter || GetAttLine().GetLineWidth() <= 0)
      return;

   const Int_t px1 = gPad->UtoAbsPixel(u1);
   const Int_t py1 = gPad->VtoAbsPixel(v1);
   const Int_t px2 = gPad->UtoAbsPixel(u2);
   const Int_t py2 = gPad->VtoAbsPixel(v2);

   painter->setPen(GetLinePen());

   painter->setRenderHint(QPainter::Antialiasing);

   painter->drawLine(QPoint(px1, py1), QPoint(px2, py2));
}

////////////////////////////////////////////////////////////////////////////////
/// Paint a simple box.

void TQt6PadPainter::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, EBoxMode mode)
{
   if (GetAttLine().GetLineWidth() <= 0 && mode == TVirtualPadPainter::kHollow)
      return;

   auto painter = fPaintWidget->getPainter();
   if (!painter)
      return;

   const Int_t px1 = gPad->XtoAbsPixel(x1);
   const Int_t py1 = gPad->YtoAbsPixel(y1);
   const Int_t px2 = gPad->XtoAbsPixel(x2);
   const Int_t py2 = gPad->YtoAbsPixel(y2);

   if (mode == TVirtualPadPainter::kHollow) {
      // draw only border
      painter->setPen(GetLinePen());
      painter->setRenderHint(QPainter::Antialiasing);
      painter->setBrush(Qt::NoBrush);
   } else {
      // draw only fill
      painter->setPen(Qt::NoPen);
      painter->setBrush(GetFillBrush());
   }

   QRect rectangle(TMath::Min(px1, px2), TMath::Min(py1, py2), TMath::Abs(px2 - px1), TMath::Abs(py2 - py1));
   painter->drawRect(rectangle);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint filled area.

void TQt6PadPainter::DrawFillArea(Int_t nPoints, const Double_t *xs, const Double_t *ys)
{
   auto painter = fPaintWidget->getPainter();

   if (!painter || (GetAttFill().GetFillStyle() <= 0) || (nPoints < 3))
      return;

   QList<QPointF> points;
   for (Int_t n = 0; n < nPoints; ++n) {
      auto px = gPad->XtoAbsPixel(xs[n]);
      auto py = gPad->YtoAbsPixel(ys[n]);
      points.push_back({(qreal) px, (qreal) py});
   }

   painter->setPen(Qt::NoPen);
   painter->setBrush(GetFillBrush());
   painter->drawPolygon(points);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint filled area.

void TQt6PadPainter::DrawFillArea(Int_t nPoints, const Float_t *xs, const Float_t *ys)
{
   auto painter = fPaintWidget->getPainter();

   if (!painter || (GetAttFill().GetFillStyle() <= 0) || (nPoints < 3))
      return;

   QList<QPointF> points;
   for (Int_t n = 0; n < nPoints; ++n) {
      auto px = gPad->XtoAbsPixel(xs[n]);
      auto py = gPad->YtoAbsPixel(ys[n]);
      points.push_back({(qreal) px, (qreal) py});
   }

   painter->setPen(Qt::NoPen);
   painter->setBrush(GetFillBrush());
   painter->drawPolygon(points);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint Polyline.

void TQt6PadPainter::DrawPolyLine(Int_t nPoints, const Double_t *xs, const Double_t *ys)
{
   auto painter = fPaintWidget->getPainter();
   if (!painter || (GetAttLine().GetLineWidth() <= 0) || (nPoints < 2))
      return;

   QList<QPointF> points;
   for (Int_t n = 0; n < nPoints; ++n) {
      auto px = gPad->XtoAbsPixel(xs[n]);
      auto py = gPad->YtoAbsPixel(ys[n]);
      points.push_back({(qreal) px, (qreal) py});
   }

   painter->setPen(GetLinePen());

   painter->setRenderHint(QPainter::Antialiasing);

   painter->drawPolyline(points);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polyline.

void TQt6PadPainter::DrawPolyLine(Int_t nPoints, const Float_t *xs, const Float_t *ys)
{
   auto painter = fPaintWidget->getPainter();
   if (!painter || (GetAttLine().GetLineWidth() <= 0) || (nPoints < 2))
      return;

   QList<QPointF> points;
   for (Int_t n = 0; n < nPoints; ++n) {
      auto px = gPad->XtoAbsPixel(xs[n]);
      auto py = gPad->YtoAbsPixel(ys[n]);
      points.push_back({(qreal) px, (qreal) py});
   }

   painter->setPen(GetLinePen());

   painter->setRenderHint(QPainter::Antialiasing);

   painter->drawPolyline(points);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polyline in normalized coordinates.

void TQt6PadPainter::DrawPolyLineNDC(Int_t nPoints, const Double_t *u, const Double_t *v)
{
   auto painter = fPaintWidget->getPainter();
   if (!painter || (GetAttLine().GetLineWidth() <= 0) || (nPoints < 2))
      return;

   QList<QPointF> points;
   for (Int_t n = 0; n < nPoints; ++n) {
      auto px = gPad->UtoAbsPixel(u[n]);
      auto py = gPad->VtoAbsPixel(v[n]);
      points.push_back({(qreal) px, (qreal) py});
   }

   painter->setPen(GetLinePen());

   painter->setRenderHint(QPainter::Antialiasing);

   painter->drawPolyline(points);
}

////////////////////////////////////////////////////////////////////////////////
/// Helper function to paint markers

template<typename T>
void drawMarkersHelper(Int_t nPoints, const T *x, const T *y,
                       const TAttMarker &attr, const QColor &col, QPainter *painter)
{
   if (!painter || nPoints < 1)
      return;

   auto markerLineWidth = TAttMarker::GetMarkerLineWidth(attr.GetMarkerStyle());
   Int_t markerSize = 0;              ///< size of simple markers
   std::vector<TPoint> markerShape;   ///< marker shape points
   auto markerType = attr.GetMarkerShape(markerSize, markerShape);

   painter->setRenderHint(QPainter::Antialiasing);

   if ((markerType == TAttMarker::kShapeDot) || (markerLineWidth > 0)) {
      QPen customPen;
      customPen.setColor(col);
      customPen.setWidth(markerLineWidth);
      customPen.setStyle(Qt::SolidLine);
      painter->setPen(customPen);
      painter->setBrush(Qt::NoBrush);
   } else {
      QBrush customBrush(col);
      painter->setBrush(customBrush);
      painter->setPen(Qt::NoPen);
   }

   for (Int_t n = 0; n < nPoints; ++n) {
      Int_t px = gPad->XtoAbsPixel(x[n]);
      Int_t py = gPad->YtoAbsPixel(y[n]);

      QList<QPointF> points;
      for (auto &pnt : markerShape)
         points.push_back({(qreal) (px + pnt.fX), (qreal) (py + pnt.fY)});

      switch(markerType) {
         case TAttMarker::kShapeDot:
            painter->drawPoint(px, py);
            break;
         case TAttMarker::kShapeCircle:
         case TAttMarker::kShapeFilledCircle:
            // hollow or filled circle
            painter->drawEllipse({px, py}, markerSize / 2, markerSize / 2);
            break;
         case TAttMarker::kShapePolyLine:
            // hollow polygon
            painter->drawPolyline(points);
            break;
         case TAttMarker::kShapeFilledArea:
            // filled polygon
            painter->drawPolygon(points);
            break;
         case TAttMarker::kShapeSegments:
            // segmented line
            painter->drawLines(points);
            break;
         case TAttMarker::kShapeTriangles:
            // filled triangles
            for (int i = 0; i < points.size() - 2; i += 3) {
               // Construct a temporary triangle polygon
               QPolygonF triangle;
               triangle << points[i] << points[i + 1] << points[i + 2];
               // Draw and fill the triangle
               painter->drawPolygon(triangle);
            }
            break;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker.

void TQt6PadPainter::DrawPolyMarker(Int_t nPoints, const Double_t *x, const Double_t *y)
{
   drawMarkersHelper<Double_t>(nPoints, x, y,
                              GetAttMarker(), GetQColor(GetAttMarker().GetMarkerColor()), fPaintWidget->getPainter());
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker.

void TQt6PadPainter::DrawPolyMarker(Int_t nPoints, const Float_t *x, const Float_t *y)
{
   drawMarkersHelper<Float_t>(nPoints, x, y,
                              GetAttMarker(), GetQColor(GetAttMarker().GetMarkerColor()), fPaintWidget->getPainter());
}

////////////////////////////////////////////////////////////////////////////////
/// Produce image

void TQt6PadPainter::SaveImage(TVirtualPad * /* pad */, const char * /* fileName */, Int_t /* gtype */) const
{
}

////////////////////////////////////////////////////////////////////////////////
/// Return QColor created from specified TColor

QColor TQt6PadPainter::GetQColor(Color_t id)
{
   auto c = gROOT->GetColor(id);
   if (c)
      return QColor((int)(c->GetRed() * 255), (int)(c->GetGreen() * 255), (int)(c->GetBlue() * 255), (int)(c->GetAlpha() * 255));
   return QColor(0, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
/// Return QPen for lines drawing

QPen TQt6PadPainter::GetLinePen()
{
   auto &att = GetAttLine();

   auto style = att.GetLineStyle();

   QPen customPen;
   customPen.setColor(GetQColor(att.GetLineColor()));
   customPen.setWidth(att.GetLineWidth());
   customPen.setStyle(Qt::SolidLine);

   TString patt;

   if (style > 1)
      patt = gStyle->GetLineStyleString(style);

   if (patt.Length() > 2) {
      QList<qreal> pattern;
      std::unique_ptr<TObjArray> tokens(patt.Tokenize(" "));
      for (Int_t j = 0; j < tokens->GetEntries(); j++) {
         int it = std::stoi(tokens->At(j)->GetName());
         pattern.push_back(0.25 * it);
      }
      if (pattern.size() > 1) {
         customPen.setStyle(Qt::CustomDashLine);
         customPen.setDashPattern(pattern);
      }
   }

   return customPen;
}

////////////////////////////////////////////////////////////////////////////////
/// Return QBrush for fill drawing drawing

QBrush TQt6PadPainter::GetFillBrush()
{
   auto &att = GetAttFill();

   Int_t style = att.GetFillStyle() / 1000;

   if (style == 1)
      return QBrush(GetQColor(att.GetFillColor()));

   if (style == 3) {
      Int_t fasi  = att.GetFillStyle() % 1000;
      Int_t stn = (fasi >= 1 && fasi <=25) ? fasi : 2;
      QBitmap bitmap = QBitmap::fromData(QSize(16, 16), (uchar *)gStipples[stn]);
      QImage image = bitmap.toImage();
      image.setColor(0, qRgba(0, 0, 0, 0)); // transparent
      image.setColor(1, GetQColor(att.GetFillColor()).rgba());
      return QBrush(QPixmap::fromImage(image.copy()));
   }

   return QBrush(Qt::NoBrush);
}


////////////////////////////////////////////////////////////////////////////////
/// Render TTF glyphs on drawable area

void TQt6PadPainter::DrawTTFglyphs(Int_t px, Int_t py, TTFhandle &ttf, [[maybe_unused]] ETextMode mode)
{
   // position inside the pad is provided, therefore shift it to global image
   px += fPad->UtoAbsPixel(0);
   py += fPad->VtoAbsPixel(1);

   const TAttText &att = GetAttText();

   auto painter = fPaintWidget->getPainter();

   auto textColor = GetQColor(att.GetTextColor());

   for (UInt_t nglyph = 0; nglyph < ttf.GetNumGlyphs(); nglyph++) {
      Int_t bx = 0, by = 0;
      UChar_t *buffer = nullptr;
      UInt_t width = 0, rows = 0, pitch = 0;
      if (!ttf.GetGlyphData(nglyph, bx, by, buffer, width, rows, pitch))
         continue;

      QImage colorFill(width, rows, QImage::Format_ARGB32);
      colorFill.fill(textColor);

      QPainter maskPainter(&colorFill);

      QImage maskImage(
         buffer,
         width,
         rows,
         pitch,
         QImage::Format_Alpha8
      );

      maskPainter.setCompositionMode(QPainter::CompositionMode_DestinationIn);
      maskPainter.drawImage(0, 0, maskImage);
      maskPainter.end();

      painter->drawImage(QPoint(px + bx, py + by), colorFill);
   }
}

void TQt6PadPainter::DrawImage(TImage *img, Int_t px, Int_t py, Int_t)
{
   // position inside the pad is provided, therefore shift it to global image
   px += fPad->UtoAbsPixel(0);
   py += fPad->VtoAbsPixel(1);

   auto width = img->GetWidth();
   auto height = img->GetHeight();
   auto argbBuffer = (const uchar*) img->GetArgbArray();

   auto painter = fPaintWidget->getPainter();

   int bytesPerLine = width * 4;

    // 1. Wrap the raw memory array into a QImage.
    // This constructor does NOT copy the pixel data; it points directly to your buffer.
    QImage image(
        argbBuffer,
        width,
        height,
        bytesPerLine,
        QImage::Format_ARGB32
    );

    // 2. Draw the image using QPainter
    // (0, 0) is the top-left coordinate where the image will be drawn
    painter->drawImage(px, py, image);
}


