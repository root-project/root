// $Id$
// Author:  Sergey Linev  10/04/2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TWebPadPainter.h"
#include "TCanvas.h"
#include "TObjString.h"
#include "TPoint.h"
#include "TError.h"
#include "TImage.h"
#include "TROOT.h"
#include "TMath.h"
#include "TPad.h"
#include "TWebVirtualX.h"
#include "TWebCanvas.h"


ClassImp(TWebPadPainter)

/** \class TWebPadPainter
\ingroup gpad

Implement TVirtualPadPainter which abstracts painting operations.
*/



////////////////////////////////////////////////////////////////////////////////
///Empty ctor. We need it only because of explicit copy ctor.

TWebPadPainter::TWebPadPainter() :
   fAttr(0),
   fAttrChanged(kFALSE),
   fPainting(0),
   fCw(0),
   fCh(0),
   fKx(1),
   fKy(1)
{
}

////////////////////////////////////////////////////////////////////////////////
///Destructor. We should destruct all primitives.

TWebPadPainter::~TWebPadPainter()
{
   ResetPainting();
   if (fAttr) { delete fAttr; fAttr = 0; }
}


void TWebPadPainter::SetWebCanvasSize(UInt_t w, UInt_t h)
{
   fCw = w;
   fCh = h;
   fKx = (w>0) ? 1./w : 1.;
   fKy = (h>0) ? 1./h : 1.;
}


TWebPainting *TWebPadPainter::TakePainting()
{
   TWebPainting *res = fPainting;
   fPainting = 0;
   return res;
}

void TWebPadPainter::ResetPainting()
{
   if (fPainting) delete fPainting;
   fPainting = 0;
   if (fAttr) fAttrChanged = kTRUE;
}

TWebPainterAttributes *TWebPadPainter::Attr()
{
   fAttrChanged = kTRUE;
   if (!fAttr) fAttr = new TWebPainterAttributes();
   return fAttr;
}

Float_t *TWebPadPainter::Reserve(Int_t sz)
{
   if (!fPainting) fPainting = new TWebPainting();
   return fPainting->Reserve(sz);
}

void TWebPadPainter::GetAttributes(Int_t attrmask)
{
   if (gVirtualX) {
      if (attrmask & 1) {
         SetLineColor(gVirtualX->GetLineColor());
         SetLineStyle(gVirtualX->GetLineStyle());
         SetLineWidth(gVirtualX->GetLineWidth());
      }
      if (attrmask & 2) {
         SetFillColor(gVirtualX->GetFillColor());
         SetFillStyle(gVirtualX->GetFillStyle());
      }
      if (attrmask & 4) {
         SetMarkerColor(gVirtualX->GetMarkerColor());
         SetMarkerSize(gVirtualX->GetMarkerSize());
         SetMarkerStyle(gVirtualX->GetMarkerStyle());
      }
      if (attrmask & 8) {
         SetTextAlign(gVirtualX->GetTextAlign());
         SetTextAngle(gVirtualX->GetTextAngle());
         SetTextColor(gVirtualX->GetTextColor());
         SetTextFont(gVirtualX->GetTextFont());
         SetTextSize(gVirtualX->GetTextSize());
      }
   }
}


void TWebPadPainter::StoreOperation(const char* opt, TObject* obj, Int_t attrmask)
{
   if (!fPainting) fPainting = new TWebPainting();

   GetAttributes(attrmask);

   if (fAttrChanged) {
      fPainting->Add(fAttr->Clone(), "attr");
      fAttrChanged = kFALSE;
   }
   if (!obj) obj = new TObjString("any");
   fPainting->Add(obj, opt);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a gVirtualX Pixmap.

Int_t TWebPadPainter::CreateDrawable(UInt_t /* w */, UInt_t /* h */)
{
   // return gVirtualX->OpenPixmap(Int_t(w), Int_t(h));
   return TWebVirtualX::WebId; // one should return positive value to overcome problem
}


////////////////////////////////////////////////////////////////////////////////
/// Clear the current gVirtualX window.

void TWebPadPainter::ClearDrawable()
{
   // gVirtualX->ClearWindow();
}


////////////////////////////////////////////////////////////////////////////////
/// Copy a gVirtualX pixmap.

void TWebPadPainter::CopyDrawable(Int_t, Int_t, Int_t)
{
   // gVirtualX->CopyPixmap(id, px, py);
}


////////////////////////////////////////////////////////////////////////////////
/// Close the current gVirtualX pixmap.

void TWebPadPainter::DestroyDrawable(Int_t /* device */)
{
   // gVirtualX->ClosePixmap();
}


////////////////////////////////////////////////////////////////////////////////
/// Select the window in which the graphics will go.

void TWebPadPainter::SelectDrawable(Int_t /* device */)
{
   // gVirtualX->SelectWindow(device);
}

////////////////////////////////////////////////////////////////////////////////
///Noop, for non-gl pad TASImage calls gVirtualX->CopyArea.

void TWebPadPainter::DrawPixels(const unsigned char * /*pixelData*/, UInt_t /*width*/, UInt_t /*height*/,
                             Int_t /*dstX*/, Int_t /*dstY*/, Bool_t /*enableAlphaBlending*/)
{

}


////////////////////////////////////////////////////////////////////////////////
/// Paint a simple line.

void TWebPadPainter::DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   GetAttributes(1);

   if (GetLineWidth()<=0) return;

   StoreOperation("line", 0, 1);

   Float_t *buf = Reserve(4);
   buf[0] = x1;
   buf[1] = y1;
   buf[2] = x2;
   buf[3] = y2;
}


////////////////////////////////////////////////////////////////////////////////
/// Paint a simple line in normalized coordinates.

void TWebPadPainter::DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2)
{
   GetAttributes(1);

   if (GetLineWidth()<=0) return;

   StoreOperation("linendc", 0, 1);

   Float_t *buf = Reserve(4);
   buf[0] = u1;
   buf[1] = v1;
   buf[2] = u2;
   buf[3] = v2;
}


////////////////////////////////////////////////////////////////////////////////
/// Paint a simple box.

void TWebPadPainter::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, EBoxMode mode)
{
   // printf("PAINT BOX %4.2f %4.2f %4.2f %4.2f mode %d linecol %d\n", x1, y1, x2, y2, (int) mode, gVirtualX->GetLineColor());

   // GetAttributes(1); // get only line attributes

   if (GetLineWidth()<=0 && mode == TVirtualPadPainter::kHollow) return;

   if (mode == TVirtualPadPainter::kHollow)
      StoreOperation("rect", 0, 1); // only border
   else
      StoreOperation("box", 0, 2); // only fill

   Float_t *buf = Reserve(4);
   buf[0] = x1;
   buf[1] = y1;
   buf[2] = x2;
   buf[3] = y2;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint filled area.

void TWebPadPainter::DrawFillArea(Int_t nPoints, const Double_t *xs, const Double_t *ys)
{
   if (nPoints < 3) {
      ::Error("TWebPadPainter::DrawFillArea", "invalid number of points %d", nPoints);
      return;
   }

   StoreOperation("fillarea", new TObjString(TString::Itoa(nPoints, 10)), 2);

   Float_t *buf = Reserve(nPoints*2);
   for (Int_t n=0;n<nPoints;++n) {
      buf[n*2] = xs[n];
      buf[n*2+1] = ys[n];
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Paint filled area.

void TWebPadPainter::DrawFillArea(Int_t nPoints, const Float_t *xs, const Float_t *ys)
{
   if (nPoints < 3) {
      ::Error("TWebPadPainter::DrawFillArea", "invalid number of points %d", nPoints);
      return;
   }

   StoreOperation("fillarea", new TObjString(TString::Itoa(nPoints, 10)), 2);

   Float_t *buf = Reserve(nPoints*2);
   for (Int_t n=0;n<nPoints;++n) {
      buf[n*2] = xs[n];
      buf[n*2+1] = ys[n];
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint Polyline.

void TWebPadPainter::DrawPolyLine(Int_t nPoints, const Double_t *xs, const Double_t *ys)
{

   GetAttributes(1); // get only line attributes

   if (GetLineWidth()<=0) return;

   if (nPoints < 2) {
      ::Error("TWebPadPainter::DrawPolyLine", "invalid number of points");
      return;
   }

   StoreOperation("polyline", new TObjString(TString::Itoa(nPoints, 10)), 1);

   Float_t *buf = Reserve(nPoints*2);
   for (Int_t n=0;n<nPoints;++n) {
      buf[n*2] = xs[n];
      buf[n*2+1] = ys[n];
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Paint polyline.

void TWebPadPainter::DrawPolyLine(Int_t nPoints, const Float_t *xs, const Float_t *ys)
{
   GetAttributes(1); // get only line attributes

   if (GetLineWidth()<=0) return;

   if (nPoints < 2) {
      ::Error("TWebPadPainter::DrawPolyLine", "invalid number of points");
      return;
   }

   StoreOperation("polyline", new TObjString(TString::Itoa(nPoints, 10)), 1);

   Float_t *buf = Reserve(nPoints*2);
   for (Int_t n=0;n<nPoints;++n) {
      buf[n*2] = xs[n];
      buf[n*2+1] = ys[n];
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Paint polyline in normalized coordinates.

void TWebPadPainter::DrawPolyLineNDC(Int_t nPoints, const Double_t *u, const Double_t *v)
{
   GetAttributes(1); // get only line attributes

   if (GetLineWidth()<=0) return;

   if (nPoints < 2) {
      ::Error("TWebPadPainter::DrawPolyLineNDC", "invalid number of points %d", nPoints);
      return;
   }

   StoreOperation("polylinendc", new TObjString(TString::Itoa(nPoints, 10)), 1);

   Float_t *buf = Reserve(nPoints*2);
   for (Int_t n=0;n<nPoints;++n) {
      buf[n*2] = u[n];
      buf[n*2+1] = v[n];
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker.

void TWebPadPainter::DrawPolyMarker(Int_t nPoints, const Double_t *x, const Double_t *y)
{
   if (nPoints < 1) {
      ::Error("TWebPadPainter::DrawPolyMarker", "invalid number of points %d", nPoints);
      return;
   }

   StoreOperation("polymarker", new TObjString(TString::Itoa(nPoints, 10)), 5);

   Float_t *buf = Reserve(nPoints*2);
   for (Int_t n=0;n<nPoints;++n) {
      buf[n*2] = x[n];
      buf[n*2+1] = y[n];
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker.

void TWebPadPainter::DrawPolyMarker(Int_t nPoints, const Float_t *x, const Float_t *y)
{
   if (nPoints < 1) {
      ::Error("TWebPadPainter::DrawPolyMarker", "invalid number of points %d", nPoints);
      return;
   }

   StoreOperation("polymarker", new TObjString(TString::Itoa(nPoints, 10)), 5);

   Float_t *buf = Reserve(nPoints*2);
   for (Int_t n=0;n<nPoints;++n) {
      buf[n*2] = x[n];
      buf[n*2+1] = y[n];
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Paint text.

void TWebPadPainter::DrawText(Double_t x, Double_t y, const char *text, ETextMode /*mode*/)
{
   StoreOperation("text", new TObjString(text), 8);

   Float_t *buf = Reserve(2);
   buf[0] = x;
   buf[1] = y;
}


////////////////////////////////////////////////////////////////////////////////
/// Special version working with wchar_t and required by TMathText.

void TWebPadPainter::DrawText(Double_t x, Double_t y, const wchar_t * /*text*/, ETextMode /*mode*/)
{
   StoreOperation("text", new TObjString("wchar_t"), 8);

   Float_t *buf = Reserve(2);
   buf[0] = x;
   buf[1] = y;
}


////////////////////////////////////////////////////////////////////////////////
/// Paint text in normalized coordinates.

void TWebPadPainter::DrawTextNDC(Double_t u, Double_t v, const char *text, ETextMode /*mode*/)
{
   StoreOperation("textndc", new TObjString(text), 8);

   Float_t *buf = Reserve(2);
   buf[0] = u;
   buf[1] = v;
}


////////////////////////////////////////////////////////////////////////////////
/// Save the image displayed in the canvas pointed by "pad" into a binary file.

void TWebPadPainter::SaveImage(TVirtualPad * /*pad*/, const char * /*fileName*/, Int_t /*type*/) const
{
}


////////////////////////////////////////////////////////////////////////////////
/// Paint text in normalized coordinates.

void TWebPadPainter::DrawTextNDC(Double_t  u , Double_t v, const wchar_t * /*text*/, ETextMode /*mode*/)
{
   StoreOperation("textndc", new TObjString("wchar_t"), 8);

   Float_t *buf = Reserve(2);
   buf[0] = u;
   buf[1] = v;
}


////////////////////////////////////////////////////////////////////////////////////
/// Reimplement X function

void TWebPadPainter::DrawFillArea(Int_t np, TPoint *xy)
{
   if (np < 3) {
      ::Error("TWebPadPainter::DrawFillArea", "invalid number of points %d", np);
      return;
   }

   StoreOperation("fillarea", new TObjString(TString::Itoa(np, 10)), 2);

   Float_t *buf = Reserve(np*2);
   for (Int_t n=0;n<np;++n) {
      buf[n*2] = gPad->PixeltoX(xy[n].fX);
      buf[n*2+1] = gPad->PixeltoY(xy[n].fY);
   }

}

