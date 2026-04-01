// @(#)root/gpad:$Id$
// Author:  Olivier Couet, Timur Pocheptsov  06/05/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPadPainter
#define ROOT_TPadPainter

#include "TPadPainterBase.h"

/*
TPadPainter is implementation of TVirtualPadPainter interface for TVirtualX.
gVirtualX can be X11 or GDI, but pad painter can be gVirtualX (X11 or GDI),
or gl pad painter.
*/

class TVirtualPad;

class TPadPainter : public TPadPainterBase {
   WinContext_t   fWinContext;

public:
   TPadPainter();

   void     SetOpacity(Int_t percent) override;
   Float_t  GetTextMagnitude() const override;

   //Overall attributes
   void      SetAttFill(const TAttFill &att) override;
   void      SetAttLine(const TAttLine &att) override;
   void      SetAttMarker(const TAttMarker &att) override;
   void      SetAttText(const TAttText &att) override;

   //2. "Off-screen management" part.
   Int_t    CreateDrawable(UInt_t w, UInt_t h) override;
   void     ClearDrawable() override;
   Int_t    ResizeDrawable(Int_t device, UInt_t w, UInt_t h) override;
   void     CopyDrawable(Int_t device, Int_t px, Int_t py) override;
   void     DestroyDrawable(Int_t device) override;
   void     SelectDrawable(Int_t device) override;
   void     UpdateDrawable(Int_t mode) override;
   void     SetDrawMode(Int_t device, Int_t mode) override;


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

   void     DrawSegments(Int_t n, Double_t *x, Double_t *y) override;
   void     DrawSegmentsNDC(Int_t n, Double_t *u, Double_t *v) override;

   //TPad needs both versions.
   void     DrawPolyMarker(Int_t n, const Double_t *x, const Double_t *y) override;
   void     DrawPolyMarker(Int_t n, const Float_t *x, const Float_t *y) override;

   void     DrawText(Double_t x, Double_t y, const char *text, ETextMode mode) override;
   void     DrawText(Double_t x, Double_t y, const wchar_t *text, ETextMode mode) override;
   void     DrawTextNDC(Double_t u, Double_t v, const char *text, ETextMode mode) override;
   void     DrawTextNDC(Double_t u, Double_t v, const wchar_t *text, ETextMode mode) override;

   //jpg, png, bmp, gif output.
   void     SaveImage(TVirtualPad *pad, const char *fileName, Int_t type) const override;

   Bool_t   IsNative() const override { return kTRUE; }

   Bool_t   IsCocoa() const override;

   Bool_t   IsSupportAlpha() const override;

private:
   //Let's make this clear:
   TPadPainter(const TPadPainter &) = delete;
   TPadPainter(TPadPainter &&) = delete;
   TPadPainter &operator=(const TPadPainter &) = delete;
   TPadPainter &operator=(TPadPainter &&) = delete;

   ClassDefOverride(TPadPainter, 0) //TPad painting
};

#endif
