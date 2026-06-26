// Author:  Sergey Linev, GSI  26/06/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQt6PadPainter
#define ROOT_TQt6PadPainter

#include "TPadPainterBase.h"


class TQt6Canvas;

class TQt6PadPainter : public TPadPainterBase {

friend class TQt6Canvas;

protected:

public:

   TQt6PadPainter() {} // NOLINT: not allowed to use = default because of TObject::kIsOnHeap detection, see ROOT-10300

   Bool_t   HasTTFonts() const override { return kTRUE; }

   void     SetOpacity(Int_t percent) override;

   //2. "Off-screen management" part.
   Int_t    CreateDrawable(UInt_t, UInt_t) override { return -1; }
   void     ClearDrawable() override {}
   void     CopyDrawable(Int_t, Int_t, Int_t) override {}
   void     DestroyDrawable(Int_t) override {}
   void     SelectDrawable(Int_t) override {}
   void     SetDoubleBuffer(Int_t /* device */, Int_t /* mode */) override {}

   //jpg, png, bmp, gif output.
   void     SaveImage(TVirtualPad *, const char *, Int_t) const override;

   //TASImage support (noop for a non-gl pad).
   void     DrawPixels(const unsigned char *pixelData, UInt_t width, UInt_t height,
                       Int_t dstX, Int_t dstY, Bool_t enableAlphaBlending) override;

   void     DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2) override;
   void     DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2) override;

   void     DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, EBoxMode mode) override;
   //TPad needs double and float versions.
   void     DrawFillArea(Int_t n, const Double_t *x, const Double_t *y) override;
   void     DrawFillArea(Int_t n, const Float_t *x, const Float_t *y) override;

   //TPad needs both double and float versions of DrawPolyLine.
   void     DrawPolyLine(Int_t n, const Double_t *x, const Double_t *y) override;
   void     DrawPolyLine(Int_t n, const Float_t *x, const Float_t *y) override;
   void     DrawPolyLineNDC(Int_t n, const Double_t *u, const Double_t *v) override;

   //TPad needs both versions.
   void     DrawPolyMarker(Int_t n, const Double_t *x, const Double_t *y) override;
   void     DrawPolyMarker(Int_t n, const Float_t *x, const Float_t *y) override;

   void     DrawText(Double_t x, Double_t y, const char *text, ETextMode mode) override;
   void     DrawText(Double_t x, Double_t y, const wchar_t *text, ETextMode mode) override;
   void     DrawTextNDC(Double_t u, Double_t v, const char *text, ETextMode mode) override;
   void     DrawTextNDC(Double_t u, Double_t v, const wchar_t *text, ETextMode mode) override;

   void     DrawTextUrl(Double_t x, Double_t y, const char *text, const char *url) override;

   Bool_t   IsSupportAlpha() const override { return kTRUE; }

private:
   //Let's make this clear:
   TQt6PadPainter(const TQt6PadPainter &rhs) = delete;
   TQt6PadPainter(TQt6PadPainter && rhs) = delete;
   TQt6PadPainter & operator = (const TQt6PadPainter &rhs) = delete;
   TQt6PadPainter & operator = (TQt6PadPainter && rhs) = delete;

   ClassDefOverride(TQt6PadPainter, 0) // Pad painter on Qt6 canvas
};

#endif
