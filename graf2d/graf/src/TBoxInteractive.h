
// @(#)root/graf:$Id$
// Author: Sergey Linev   18/09/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualPad.h"

#ifndef ROOT_TBoxInteractive
#define ROOT_TBoxInteractive

class TBoxInteractive : public TVirtualPad::TInteractive {

   protected:

      enum EMode { pNone = 0, pA, pB, pC, pD, pTop, pL, pR, pBot, pINSIDE } fMode = pNone;

   public:
      Bool_t isBox = kTRUE;
      Int_t px1 = 0, px2 = 0, py1 = 0, py2 = 0, dpx1 = 0, dpy2 = 0;
      Double_t oldX1 = 0., oldY1 = 0., oldX2 = 0., oldY2 = 0.;
      Double_t newX1 = 0., newY1 = 0., newX2 = 0., newY2 = 0.;

      TBoxInteractive(Bool_t bx = kTRUE, Double_t x1 = 0., Double_t y1 = 0., Double_t x2 = 0., Double_t y2 = 0.);

      void CalcPixelCoord(const TVirtualPad &parent, Double_t x1, Double_t y1, Double_t x2, Double_t y2);

      Bool_t SelectCorner(Int_t px, Int_t py, Bool_t canX = kTRUE, Bool_t canY = kTRUE);

      Bool_t SelectDiamondCorner(Int_t px, Int_t py, Bool_t testInside = kTRUE);

      Bool_t IsResizing() const;

      Bool_t IsOpaque(const TVirtualPad &parent) const;

      Bool_t ProcessMouseMove(const TVirtualPad &parent, Int_t px, Int_t py, Bool_t canX = kTRUE, Bool_t canY = kTRUE, Double_t aspectRatio = 0.);

      virtual void PaintOutline(TVirtualPad &parent);

      void PaintDiamondCorners(TVirtualPad &parent, const char *id = "diamond");

      void ApplyChanges(TVirtualPad &parent);

      void SetCursor(TVirtualPad &parent, Bool_t is_down) const;

      char GetGuideChar() const;
};

#endif