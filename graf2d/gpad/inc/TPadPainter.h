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

#ifndef ROOT_TVirtualPadPainter
#include "TVirtualPadPainter.h"
#endif

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
   Color_t  GetLineColor() const;
   Style_t  GetLineStyle() const;
   Width_t  GetLineWidth() const;

   void     SetLineColor(Color_t lcolor);
   void     SetLineStyle(Style_t lstyle);
   void     SetLineWidth(Width_t lwidth);
   //Fill attributes.
   Color_t  GetFillColor() const;
   Style_t  GetFillStyle() const;
   Bool_t   IsTransparent() const;

   void     SetFillColor(Color_t fcolor);
   void     SetFillStyle(Style_t fstyle);
   void     SetOpacity(Int_t percent);
   //Text attributes.
   Short_t  GetTextAlign() const;
   Float_t  GetTextAngle() const;
   Color_t  GetTextColor() const;
   Font_t   GetTextFont()  const;
   Float_t  GetTextSize()  const;
   Float_t  GetTextMagnitude() const;

   void     SetTextAlign(Short_t align);
   void     SetTextAngle(Float_t tangle);
   void     SetTextColor(Color_t tcolor);
   void     SetTextFont(Font_t tfont);
   void     SetTextSize(Float_t tsize);
   void     SetTextSizePixels(Int_t npixels);
   //2. "Off-screen management" part.
   Int_t    CreateDrawable(UInt_t w, UInt_t h);
   void     ClearDrawable();
   void     CopyDrawable(Int_t id, Int_t px, Int_t py);
   void     DestroyDrawable();
   void     SelectDrawable(Int_t device);

   //TASImage support (noop for a non-gl pad).
   void     DrawPixels(const unsigned char *pixelData, UInt_t width, UInt_t height,
                       Int_t dstX, Int_t dstY, Bool_t enableAlphaBlending);


   void     DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2);
   void     DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2);

   void     DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, EBoxMode mode);
   //TPad needs double and float versions.
   void     DrawFillArea(Int_t n, const Double_t *x, const Double_t *y);
   void     DrawFillArea(Int_t n, const Float_t *x, const Float_t *y);

   //TPad needs both double and float versions of DrawPolyLine.
   void     DrawPolyLine(Int_t n, const Double_t *x, const Double_t *y);
   void     DrawPolyLine(Int_t n, const Float_t *x, const Float_t *y);
   void     DrawPolyLineNDC(Int_t n, const Double_t *u, const Double_t *v);

   //TPad needs both versions.
   void     DrawPolyMarker(Int_t n, const Double_t *x, const Double_t *y);
   void     DrawPolyMarker(Int_t n, const Float_t *x, const Float_t *y);

   void     DrawText(Double_t x, Double_t y, const char *text, ETextMode mode);
   void     DrawText(Double_t x, Double_t y, const wchar_t *text, ETextMode mode);
   void     DrawTextNDC(Double_t u, Double_t v, const char *text, ETextMode mode);
   void     DrawTextNDC(Double_t u, Double_t v, const wchar_t *text, ETextMode mode);

   //jpg, png, bmp, gif output.
   void     SaveImage(TVirtualPad *pad, const char *fileName, Int_t type) const;

private:
   //Let's make this clear:
   TPadPainter(const TPadPainter &rhs) = delete;
   TPadPainter(TPadPainter && rhs) = delete;
   TPadPainter & operator = (const TPadPainter &rhs) = delete;
   TPadPainter & operator = (TPadPainter && rhs) = delete;

   ClassDef(TPadPainter, 0) //Abstract interface for painting in TPad
};

#endif
