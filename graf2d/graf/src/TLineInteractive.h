// @(#)root/graf:$Id$
// Author: Sergey Linev   25/09/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLineInteractive
#define ROOT_TLineInteractive

#include "TVirtualPad.h"

#include "TMath.h"
#include "TLine.h"

class TLineInteractive : public TVirtualPad::TInteractive {

protected:
   Double_t oldX1 = 0., oldY1 = 0., oldX2 = 0., oldY2 = 0.;
   Double_t newX1 = 0., newY1 = 0., newX2 = 0., newY2 = 0.;
   Bool_t isNDC = kFALSE;
   Int_t dx1 = 0, dx2 = 0, dy1 = 0, dy2 = 0;
   Int_t selectPoint = 0;

public:

   TLineInteractive(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Bool_t ndc = kFALSE)
   {
      newX1 = oldX1 = x1;
      newY1 = oldY1 = y1;
      newX2 = oldX2 = x2;
      newY2 = oldY2 = y2;
      isNDC = ndc;
   }

   void SelectPoint(TVirtualPad &parent, Int_t px, Int_t py)
   {
      constexpr Int_t kMaxDiff = 20;

      Int_t px1, py1, px2, py2;
      if (isNDC) {
         px1 = parent.UtoAbsPixel(newX1);
         py1 = parent.VtoAbsPixel(newY1);
         px2 = parent.UtoAbsPixel(newX2);
         py2 = parent.VtoAbsPixel(newY2);
      } else {
         px1 = parent.XtoAbsPixel(parent.XtoPad(newX1));
         py1 = parent.YtoAbsPixel(parent.YtoPad(newY1));
         px2 = parent.XtoAbsPixel(parent.XtoPad(newX2));
         py2 = parent.YtoAbsPixel(parent.YtoPad(newY2));
      }

      if (abs(px1 - px) + abs(py1 - py) < kMaxDiff) {
         selectPoint = 1;
         parent.SetCursor(kPointer);
      } else if (abs(px2 - px) + abs(py2 - py) < kMaxDiff) {
         selectPoint = 2;
         parent.SetCursor(kPointer);
      } else {
         selectPoint = 3;
         dx1 = px1 - px;
         dx2 = px2 - px;
         dy1 = py1 - py;
         dy2 = py2 - py;
         parent.SetCursor(kMove);
      }
   }

   Double_t GetXCoord(TVirtualPad &parent, Int_t px)
   {
      if (!isNDC)
         return parent.PadtoX(parent.AbsPixeltoX(px));
      Double_t ww = parent.GetWw();
      Double_t wndc = parent.GetAbsWNDC();
      return ww > 0 && wndc > 0 ? (px / ww - parent.GetAbsXlowNDC()) / wndc : 0.;
   }

   Double_t GetYCoord(TVirtualPad &parent, Int_t py)
   {
      if (!isNDC)
         return parent.PadtoY(parent.AbsPixeltoY(py));

      Double_t wh = parent.GetWh();
      Double_t hndc = parent.GetAbsHNDC();
      return wh > 0 && hndc > 0 ? ((1. - py / wh) - parent.GetAbsYlowNDC()) / hndc : 0.;
   }

   void ChangeCoordinate(TVirtualPad &parent, Int_t px, Int_t py, Bool_t vert = kFALSE, Bool_t horiz = kFALSE)
   {
      Bool_t m = selectPoint == 3;
      if (selectPoint & 1) {
         newX1 = GetXCoord(parent, px + (m ? dx1 : 0));
         newY1 = GetYCoord(parent, py + (m ? dy1 : 0));
      }
      if (selectPoint & 2) {
         newX2 = GetXCoord(parent, px + (m ? dx2 : 0));
         newY2 = GetYCoord(parent, py + (m ? dy2 : 0));
      }

      if (vert) {
         if (selectPoint & 1)
            newX2 = newX1;
         else
            newX1 = newX2;
      }
      if (horiz) {
         if (selectPoint & 1)
            newY2 = newY1;
         else
            newY1 = newY2;
      }
   }

   void PaintLine(TVirtualPad &parent)
   {
      if (isNDC) {
         Double_t xx[2] = {newX1, newX2};
         Double_t yy[2] = {newY1, newY2};
         parent.PaintPolyLineNDC(2, xx, yy, "iline");
      } else {
         Double_t xx[2] = {parent.XtoPad(newX1), parent.XtoPad(newX2)};
         Double_t yy[2] = {parent.YtoPad(newY1), parent.YtoPad(newY2)};
         parent.PaintPolyLine(2, xx, yy, "iline");
      }
   }

   char GetGuideChar() const
   {
      if ((selectPoint == 1) || (selectPoint == 2)) {
         static const char GUIDES[2][2][2] = {{{'4', '1'}, {'3', '2'}}, {{'2', '3'}, {'1', '4'}}};
         int x_idx = newX1 > newX2 ? 1 : 0;
         int y_idx = newY1 > newY2 ? 1 : 0;
         return GUIDES[selectPoint - 1][x_idx][y_idx];
      }
      return selectPoint == 3 ? 'i' : '\0';
   }

   void Apply(TLine *l, Bool_t usenew = kTRUE)
   {
      l->SetX1(usenew ? newX1 : oldX1);
      l->SetY1(usenew ? newY1 : oldY1);
      l->SetX2(usenew ? newX2 : oldX2);
      l->SetY2(usenew ? newY2 : oldY2);
   }

};

#endif