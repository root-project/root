// Author:  Sergey Linev, GSI  23/10/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TWebPS.h"

TWebPS::~TWebPS()
{
   ResetPainting();
}

TWebPainting *TWebPS::TakePainting()
{
   TWebPainting *res = fPainting;
   fPainting = nullptr;
   return res;
}

void TWebPS::ResetPainting()
{
   if (fPainting) delete fPainting;
   fPainting = nullptr;
}


Float_t *TWebPS::StoreOperation(const std::string &oper, unsigned attrkind, int opersize)
{
   if (!fPainting) fPainting = new TWebPainting();

   if (attrkind & attrLine)
      fPainting->AddOper(std::string("lattr:") +
                         std::to_string((int)GetLineColor()) + ":" +
                         std::to_string((int)GetLineStyle()) + ":" +
                         std::to_string((int)GetLineWidth()));

   if (attrkind & attrFill)
      fPainting->AddOper(std::string("fattr:") +
                         std::to_string((int)GetFillColor()) + ":" +
                         std::to_string((int)GetFillStyle()));

   if (attrkind & attrMarker)
      fPainting->AddOper(std::string("mattr:") +
                         std::to_string((int)GetMarkerColor()) + ":" +
                         std::to_string((int)GetMarkerStyle()) + ":" +
                         std::to_string((int)GetMarkerSize()));

   if (attrkind & attrText)
      fPainting->AddOper(std::string("tattr:") +
                         std::to_string((int)GetTextColor()) + ":" +
                         std::to_string((int)GetTextFont()) + ":" +
                         std::to_string((int)(GetTextSize()>=1 ? GetTextSize() : -1000*GetTextSize())) + ":" +
                         std::to_string((int)GetTextAlign()) + ":" +
                         std::to_string((int)GetTextAngle()));

   fPainting->AddOper(oper);

   return fPainting->Reserve(opersize);
}

void TWebPS::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   Float_t *buf = (GetFillStyle() > 0) ? StoreOperation("bbox", attrFill, 4) : StoreOperation("rect", attrLine, 4);

   buf[0] = x1;
   buf[1] = y1;
   buf[2] = x2;
   buf[3] = y2;
}

void TWebPS::DrawPolyMarker(Int_t nPoints, Float_t *x, Float_t *y)
{
   if (nPoints < 1) return;

   Float_t *buf = StoreOperation(std::string("pmark:") + std::to_string(nPoints), attrLine | attrMarker, nPoints*2);

   for (Int_t n=0;n<nPoints;++n) {
      buf[n*2] = x[n];
      buf[n*2+1] = y[n];
   }
}

void TWebPS::DrawPolyMarker(Int_t nPoints, Double_t *x, Double_t *y)
{
   if (nPoints < 1) return;

   Float_t *buf = StoreOperation(std::string("pmark:") + std::to_string(nPoints), attrLine | attrMarker, nPoints*2);

   for (Int_t n=0;n<nPoints;++n) {
      buf[n*2] = x[n];
      buf[n*2+1] = y[n];
   }
}

void TWebPS::DrawPS(Int_t nPoints, Float_t *xw, Float_t *yw)
{
   Float_t *buf = nullptr;
   if (nPoints < 0) {
      nPoints = -nPoints;
      if ((GetFillStyle() <= 0) || (nPoints < 2))  return;
      buf = StoreOperation("pfill:" + std::to_string(nPoints), attrFill, nPoints*2);
   } else {
      if ((GetLineWidth() <= 0) || (nPoints < 2))  return;
      buf = StoreOperation("pline:" + std::to_string(nPoints), attrLine, nPoints*2);
   }
   for (Int_t n=0;n<nPoints;++n) {
      buf[n*2] = xw[n];
      buf[n*2+1] = yw[n];
   }
}

void TWebPS::DrawPS(Int_t nPoints, Double_t *xw, Double_t *yw)
{
   Float_t *buf = nullptr;
   if (nPoints < 0) {
      nPoints = -nPoints;
      if ((GetFillStyle() <= 0) || (nPoints < 2))  return;
      buf = StoreOperation("pfill:" + std::to_string(nPoints), attrFill, nPoints*2);
   } else {
      if ((GetLineWidth() <= 0) || (nPoints < 2))  return;
      buf = StoreOperation("pline:" + std::to_string(nPoints), attrLine, nPoints*2);
   }
   for (Int_t n=0;n<nPoints;++n) {
      buf[n*2] = xw[n];
      buf[n*2+1] = yw[n];
   }
}

void TWebPS::Text(Double_t x, Double_t y, const char *str)
{
   Float_t *buf = StoreOperation(std::string("text:") + str, attrText, 2);
   buf[0] = x;
   buf[1] = y;
}


void TWebPS::Text(Double_t x, Double_t y, const wchar_t *)
{
   Float_t *buf = StoreOperation(std::string("text:") + "wchar_t", attrText, 2);
   buf[0] = x;
   buf[1] = y;
}
