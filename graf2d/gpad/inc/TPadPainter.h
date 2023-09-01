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

#include "TVirtualPadPainter.h"

/*
TVirtualPadPainter is an attempt to abstract
painting operation furthermore. gVirtualX can
be X11 or GDI, but pad painter can be gVirtualX (X11 or GDI),
or gl pad painter.
*/

class TVirtualPad;

class TPadPainter : public TVirtualPadPainter {
public:
   TPadPainter();
   //Final overriders for TVirtualPadPainter pure virtual functions.
   //1. Part, which simply delegates to TVirtualX.

   //Line attributes.
   Color_t  GetLineColor() const override;
   Style_t  GetLineStyle() const override;
   Width_t  GetLineWidth() const override;

   void     SetLineColor(Color_t lcolor) override;
   void     SetLineStyle(Style_t lstyle) override;
   void     SetLineWidth(Width_t lwidth) override;

   //Fill attributes.
   Color_t  GetFillColor() const override;
   Style_t  GetFillStyle() const override;
   Bool_t   IsTransparent() const override;

   void     SetFillColor(Color_t fcolor) override;
   void     SetFillStyle(Style_t fstyle) override;
   void     SetOpacity(Int_t percent) override;

   //Text attributes.
   Short_t  GetTextAlign() const override;
   Float_t  GetTextAngle() const override;
   Color_t  GetTextColor() const override;
   Font_t   GetTextFont()  const override;
   Float_t  GetTextSize()  const override;
   Float_t  GetTextMagnitude() const override;

   void     SetTextAlign(Short_t align) override;
   void     SetTextAngle(Float_t tangle) override;
   void     SetTextColor(Color_t tcolor) override;
   void     SetTextFont(Font_t tfont) override;
   void     SetTextSize(Float_t tsize) override;
   void     SetTextSizePixels(Int_t npixels) override;

   //2. "Off-screen management" part.
   Int_t    CreateDrawable(UInt_t w, UInt_t h) override;
   void     ClearDrawable() override;
   void     CopyDrawable(Int_t device, Int_t px, Int_t py) override;
   void     DestroyDrawable(Int_t device) override;
   void     SelectDrawable(Int_t device) override;

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

   //jpg, png, bmp, gif output.
   void     SaveImage(TVirtualPad *pad, const char *fileName, Int_t type) const override;

private:
   //Let's make this clear:
   TPadPainter(const TPadPainter &) = delete;
   TPadPainter(TPadPainter &&) = delete;
   TPadPainter &operator=(const TPadPainter &) = delete;
   TPadPainter &operator=(TPadPainter &&) = delete;

   ClassDefOverride(TPadPainter, 0) //TPad painting
};

#endif
