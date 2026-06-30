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
#include "TROOT.h"
#include "TColor.h"
#include "RStipples.h"

#include <memory>

#include "QPaintWidget.h"

#include <QFont>
#include <QFontDatabase>
#include <QRect>
#include <QPainter>

/** \class TQt6PadPainter
    \ingroup qt6canvas
    \brief Implement TVirtualPadPainter for Qt6 graphics
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
   for (Int_t n = 0; n < nPoints; ++n)
      points.push_back({gPad->XtoAbsPixel(xs[n]), gPad->YtoAbsPixel(ys[n])});

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
   for (Int_t n = 0; n < nPoints; ++n)
      points.push_back({gPad->XtoAbsPixel(xs[n]), gPad->YtoAbsPixel(ys[n])});

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
   for (Int_t n = 0; n < nPoints; ++n)
      points.push_back({gPad->XtoAbsPixel(xs[n]), gPad->YtoAbsPixel(ys[n])});

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
   for (Int_t n = 0; n < nPoints; ++n)
      points.push_back({gPad->XtoAbsPixel(xs[n]), gPad->YtoAbsPixel(ys[n])});

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
   for (Int_t n = 0; n < nPoints; ++n)
      points.push_back({gPad->UtoAbsPixel(u[n]), gPad->VtoAbsPixel(v[n])});

   painter->setPen(GetLinePen());

   painter->setRenderHint(QPainter::Antialiasing);

   painter->drawPolyline(points);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker.

void TQt6PadPainter::DrawPolyMarker(Int_t nPoints, const Double_t *x, const Double_t *y)
{
   if (nPoints < 1)
      return;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker.

void TQt6PadPainter::DrawPolyMarker(Int_t nPoints, const Float_t *x, const Float_t *y)
{
   if (nPoints < 1)
      return;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint text.

void TQt6PadPainter::DrawText(Double_t x, Double_t y, const char *text, ETextMode /*mode*/)
{
   const Int_t px = gPad->XtoAbsPixel(x);
   const Int_t py = gPad->YtoAbsPixel(y);

   PaintQString(px, py, text);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint text with url

void TQt6PadPainter::DrawTextUrl(Double_t x, Double_t y, const char *text, const char * /* url */)
{
   const Int_t px = gPad->XtoAbsPixel(x);
   const Int_t py = gPad->YtoAbsPixel(y);

   PaintQString(px, py, text);
}

////////////////////////////////////////////////////////////////////////////////
/// Special version working with wchar_t and required by TMathText.

void TQt6PadPainter::DrawText(Double_t x, Double_t y, const wchar_t *text, ETextMode /*mode*/)
{
   const Int_t px = gPad->XtoAbsPixel(x);
   const Int_t py = gPad->YtoAbsPixel(y);

   PaintQString(px, py, QString::fromWCharArray(text));
}

////////////////////////////////////////////////////////////////////////////////
/// Paint text in normalized coordinates.

void TQt6PadPainter::DrawTextNDC(Double_t u, Double_t v, const char *text, ETextMode /*mode*/)
{
   const Int_t px = gPad->UtoAbsPixel(u);
   const Int_t py = gPad->VtoAbsPixel(v);

   PaintQString(px, py, text);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint text in normalized coordinates.

void TQt6PadPainter::DrawTextNDC(Double_t  u, Double_t v, const wchar_t *text, ETextMode /*mode*/)
{
   const Int_t px = gPad->UtoAbsPixel(u);
   const Int_t py = gPad->VtoAbsPixel(v);

   PaintQString(px, py, QString::fromWCharArray(text));
}


////////////////////////////////////////////////////////////////////////////////
/// Produce image

void TQt6PadPainter::SaveImage(TVirtualPad *pad, const char *fileName, Int_t /* gtype */) const
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
         Int_t it = std::stoi(tokens->At(j)->GetName());
         pattern.push_back(0.25 * it / att.GetLineWidth());
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
/// Return font family for specified ROOT font id
/// If necessary, register TTF font to Qt first

QString TQt6PadPainter::GetFontFamily(Font_t fontnumber)
{
   // TODO: make special generic method, used from several places
   static const char *fonttable[][2] = {
     { "Root.TTFont.0", "FreeSansBold.otf" },
     { "Root.TTFont.1", "FreeSerifItalic.otf" },
     { "Root.TTFont.2", "FreeSerifBold.otf" },
     { "Root.TTFont.3", "FreeSerifBoldItalic.otf" },
     { "Root.TTFont.4", "texgyreheros-regular.otf" },
     { "Root.TTFont.5", "texgyreheros-italic.otf" },
     { "Root.TTFont.6", "texgyreheros-bold.otf" },
     { "Root.TTFont.7", "texgyreheros-bolditalic.otf" },
     { "Root.TTFont.8", "FreeMono.otf" },
     { "Root.TTFont.9", "FreeMonoOblique.otf" },
     { "Root.TTFont.10", "FreeMonoBold.otf" },
     { "Root.TTFont.11", "FreeMonoBoldOblique.otf" },
     { "Root.TTFont.12", "symbol.ttf" },
     { "Root.TTFont.13", "FreeSerif.otf" },
     { "Root.TTFont.14", "wingding.ttf" },
     { "Root.TTFont.15", "symbol.ttf" },
     { "Root.TTFont.STIXGen", "STIXGeneral.otf" },
     { "Root.TTFont.STIXGenIt", "STIXGeneralItalic.otf" },
     { "Root.TTFont.STIXGenBd", "STIXGeneralBol.otf" },
     { "Root.TTFont.STIXGenBdIt", "STIXGeneralBolIta.otf" },
     { "Root.TTFont.STIXSiz1Sym", "STIXSiz1Sym.otf" },
     { "Root.TTFont.STIXSiz1SymBd", "STIXSiz1SymBol.otf" },
     { "Root.TTFont.STIXSiz2Sym", "STIXSiz2Sym.otf" },
     { "Root.TTFont.STIXSiz2SymBd", "STIXSiz2SymBol.otf" },
     { "Root.TTFont.STIXSiz3Sym", "STIXSiz3Sym.otf" },
     { "Root.TTFont.STIXSiz3SymBd", "STIXSiz3SymBol.otf" },
     { "Root.TTFont.STIXSiz4Sym", "STIXSiz4Sym.otf" },
     { "Root.TTFont.STIXSiz4SymBd", "STIXSiz4SymBol.otf" },
     { "Root.TTFont.STIXSiz5Sym", "STIXSiz5Sym.otf" },
     { "Root.TTFont.ME", "DroidSansFallback.ttf" },
     { "Root.TTFont.CJKMing", "DroidSansFallback.ttf" },
     { "Root.TTFont.CJKGothic", "DroidSansFallback.ttf" }
   };

   int fontid = fontnumber / 10;
   if (fontid < 0 || fontid > 31)
      fontid = 0;

   static std::map<int, QString> registeredFonts;

   auto iter = registeredFonts.find(fontid);
   if (iter != registeredFonts.end())
      return iter->second;

   const char *ttpath = gEnv->GetValue("Root.TTFontPath",
                                        TROOT::GetTTFFontDir());

   TString fname = gEnv->GetValue(fonttable[fontid][0], fonttable[fontid][1]);

   const char *ttfont = gSystem->FindFile(ttpath, fname, kReadPermission);

   if (!ttfont) {
      ::Error("TQt6PadPainter::GetFontFamily", "Not found font %s in configured path %s", fname.Data(), ttpath);
      return "";
   }

   int qtId = QFontDatabase::addApplicationFont(ttfont);
   if (qtId == -1) {
      ::Error("TQt6PadPainter::GetFontFamily", "No able to add font %s to QFontDataBase", ttfont);
      return "";
   }

   QString fontFamily = QFontDatabase::applicationFontFamilies(qtId).at(0);

   registeredFonts[fontid] = fontFamily;

   return fontFamily;
}

////////////////////////////////////////////////////////////////////////////////
/// Actual text painting image

void TQt6PadPainter::PaintQString(int x, int y, const QString &s)
{
   auto painter = fPaintWidget->getPainter();
   if (!painter)
      return;

   const TAttText &att = GetAttText();
   auto family = GetFontFamily(att.GetTextFont());
   if (family.isEmpty())
      return;

   painter->setFont(QFont(family, att.GetTextSizePixels(*gPad)));

   painter->setPen(GetQColor(att.GetTextColor()));

   Int_t txalh = att.GetTextAlign() / 10;
   Int_t txalv = att.GetTextAlign() % 10;

   auto fm = painter->fontMetrics();

   switch (txalh) {
      case 0:
      case 1: break; //left
      case 2: x -= fm.horizontalAdvance(s) / 2; break; //center
      case 3: x -= fm.horizontalAdvance(s); break; //right
   }

   switch (txalv) {
      case 1: break; //bottom
      case 2: y += fm.height() / 2; break; // middle
      case 3: y += fm.height(); break; //top
   }

   if (att.GetTextAngle() == 0) {
      // Just draw text
      painter->drawText(x, y, s);
   } else {
      // Draw with rotation
      painter->save();
      painter->translate(x, y);
      painter->rotate(-att.GetTextAngle());
      painter->drawText(0, 0, s);
      painter->restore();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Returns text extent

void TQt6PadPainter::GetTextExtent(Font_t font, Double_t size, UInt_t &w, UInt_t &h, const char *mess)
{
   auto family = GetFontFamily(font);
   if (family.isEmpty())
      return;

   QFontMetrics fm(QFont(family, size));
   QRect rect = fm.boundingRect(mess);

   w = rect.width();
   h = rect.height();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns text extent

void TQt6PadPainter::GetTextExtent(Font_t font, Double_t size, UInt_t &w, UInt_t &h, const wchar_t *mess)
{
   auto family = GetFontFamily(font);
   if (family.isEmpty())
      return;

   QFontMetrics fm(QFont(family, size));
   QRect rect = fm.boundingRect(QString::fromWCharArray(mess));

   w = rect.width();
   h = rect.height();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns text accent / descent

void TQt6PadPainter::GetTextAscentDescent(Font_t font, Double_t size, UInt_t &a, UInt_t &d, const char *mess)
{
   auto family = GetFontFamily(font);
   if (family.isEmpty())
      return;

   QFontMetrics fm(QFont(family, size));
   QRect rect = fm.boundingRect(mess);

   a = -rect.top();
   d = rect.bottom();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns text accent / descent

void TQt6PadPainter::GetTextAscentDescent(Font_t font, Double_t size, UInt_t &a, UInt_t &d, const wchar_t *mess)
{
   auto family = GetFontFamily(font);
   if (family.isEmpty())
      return;

   QFontMetrics fm(QFont(family, size));
   QRect rect = fm.boundingRect(QString::fromWCharArray(mess));

   a = -rect.top();
   d = rect.bottom();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns text advance

UInt_t TQt6PadPainter::GetTextAdvance(Font_t font, Double_t size, const char *text, Bool_t)
{
   auto family = GetFontFamily(font);
   if (family.isEmpty())
      return 0;

   QFontMetrics fm(QFont(family, size));
   return fm.horizontalAdvance(QString(text));
}
