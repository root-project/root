// @(#)root/base:$Id$
// Author: Timur Pocheptsov   6/5/2009

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualPadPainter
#define ROOT_TVirtualPadPainter

#include "Rtypes.h"
#include "GuiTypes.h"

class TVirtualPad;
class TVirtualPS;
class TAttFill;
class TAttLine;
class TAttMarker;
class TAttText;

class TVirtualPadPainter {
public:
   enum EBoxMode  {kHollow, kFilled};
   enum ETextMode {
// clang++ <v20 (-Wshadow) complains about shadowing Getline.h global enum EGetLineMode. Let's silence warning:
#if defined(__clang__) && __clang_major__ < 20
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
#endif
      kClear,
#if defined(__clang__) && __clang_major__ < 20
#pragma clang diagnostic pop
#endif
      kOpaque};

   virtual ~TVirtualPadPainter();

   //Line attributes to be set up in TPad.
   virtual Color_t  GetLineColor() const = 0;
   virtual Style_t  GetLineStyle() const = 0;
   virtual Width_t  GetLineWidth() const = 0;

   virtual void     SetLineColor(Color_t lcolor) = 0;
   virtual void     SetLineStyle(Style_t lstyle) = 0;
   virtual void     SetLineWidth(Width_t lwidth) = 0;

   //Fill attributes to be set up in TPad.
   virtual Color_t  GetFillColor() const = 0;
   virtual Style_t  GetFillStyle() const = 0;
   virtual Bool_t   IsTransparent() const = 0;

   virtual void     SetFillColor(Color_t fcolor) = 0;
   virtual void     SetFillStyle(Style_t fstyle) = 0;
   virtual void     SetOpacity(Int_t percent) = 0;

   //Text attributes.
   virtual Short_t  GetTextAlign() const = 0;
   virtual Float_t  GetTextAngle() const = 0;
   virtual Color_t  GetTextColor() const = 0;
   virtual Font_t   GetTextFont() const = 0;
   virtual Float_t  GetTextSize() const = 0;
   virtual Float_t  GetTextMagnitude() const = 0;

   virtual void     SetTextAlign(Short_t align=11) = 0;
   virtual void     SetTextAngle(Float_t tangle=0) = 0;
   virtual void     SetTextColor(Color_t tcolor=1) = 0;
   virtual void     SetTextFont(Font_t tfont=62) = 0;
   virtual void     SetTextSize(Float_t tsize=1) = 0;
   virtual void     SetTextSizePixels(Int_t npixels) = 0;

   //Marker attributes
   virtual Color_t  GetMarkerColor() const { return 0; }
   virtual Style_t  GetMarkerStyle() const { return 0; }
   virtual Size_t   GetMarkerSize()  const { return 0; }

   virtual void     SetMarkerColor(Color_t /* mcolor */ = 1) {}
   virtual void     SetMarkerStyle(Style_t /* mstyle */ = 1) {}
   virtual void     SetMarkerSize(Size_t /* msize */ = 1) {}

   virtual void      SetAttFill(const TAttFill &att);
   virtual void      SetAttLine(const TAttLine &att);
   virtual void      SetAttMarker(const TAttMarker &att);
   virtual void      SetAttText(const TAttText &att);

   virtual const TAttFill &GetAttFill() const;
   virtual const TAttLine &GetAttLine() const;
   virtual const TAttMarker &GetAttMarker() const;
   virtual const TAttText &GetAttText() const;

   //This part is an interface to X11 pixmap management and to save sub-pads off-screens for OpenGL.
   //Currently, must be implemented only for X11/GDI
   virtual Int_t    CreateDrawable(UInt_t w, UInt_t h) = 0;//gVirtualX->OpenPixmap
   virtual void     ClearDrawable() = 0;//gVirtualX->Clear()
   virtual Int_t    ResizeDrawable(Int_t /* device */, UInt_t /* w */, UInt_t /* h */) { return 0; } //gVirtualX->ResizePixmap
   virtual void     CopyDrawable(Int_t device, Int_t px, Int_t py) = 0;
   virtual void     DestroyDrawable(Int_t device) = 0;//gVirtualX->CloseWindow
   virtual void     SelectDrawable(Int_t device) = 0;//gVirtualX->SelectWindow
   virtual void     UpdateDrawable(Int_t /* mode */) {}
   virtual void     SetDrawMode(Int_t /* device */, Int_t /* mode */) {}
   virtual void     SetDoubleBuffer(Int_t device, Int_t mode);
   virtual void     SetCursor(Int_t win, ECursor cursor);


   //TASImage support.
   virtual void     DrawPixels(const unsigned char *pixelData, UInt_t width, UInt_t height,
                               Int_t dstX, Int_t dstY, Bool_t enableAlphaBlending) = 0;
   //
   //These functions are not required by X11/GDI.
   virtual void     InitPainter();
   virtual void     InvalidateCS();
   virtual void     LockPainter();
   virtual void     NewPage() {}

   //Now, drawing primitives.
   virtual void     DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2) = 0;
   virtual void     DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2) = 0;

   virtual void     DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, EBoxMode mode) = 0;

   virtual void     DrawFillArea(Int_t n, const Double_t *x, const Double_t *y) = 0;
   virtual void     DrawFillArea(Int_t n, const Float_t *x, const Float_t *y) = 0;

   virtual void     DrawPolyLine(Int_t n, const Double_t *x, const Double_t *y) = 0;
   virtual void     DrawPolyLine(Int_t n, const Float_t *x, const Float_t *y) = 0;
   virtual void     DrawPolyLineNDC(Int_t n, const Double_t *u, const Double_t *v) = 0;

   virtual void     DrawSegments(Int_t n, Double_t *x, Double_t *y);
   virtual void     DrawSegmentsNDC(Int_t n, Double_t *u, Double_t *v);

   virtual void     DrawPolyMarker(Int_t n, const Double_t *x, const Double_t *y) = 0;
   virtual void     DrawPolyMarker(Int_t n, const Float_t *x, const Float_t *y) = 0;

   virtual void     DrawText(Double_t x, Double_t y, const char *text, ETextMode mode) = 0;
   virtual void     DrawText(Double_t x, Double_t y, const wchar_t *text, ETextMode mode) = 0;
   virtual void     DrawTextNDC(Double_t u, Double_t v, const char *text, ETextMode mode) = 0;
   virtual void     DrawTextNDC(Double_t u, Double_t v, const wchar_t *text, ETextMode mode) = 0;

   virtual void     DrawTextUrl(Double_t x, Double_t y, const char *text, const char *url);

   //gif, jpg, png, bmp output.
   virtual void     SaveImage(TVirtualPad *pad, const char *fileName, Int_t type) const = 0;

   virtual void     OnPad(TVirtualPad *) {}

   virtual Bool_t   IsNative() const { return kFALSE; }
   virtual Bool_t   IsCocoa() const { return kFALSE; }
   virtual TVirtualPS *GetPS() const { return nullptr; }
   virtual Bool_t   IsSupportAlpha() const { return kFALSE; }

   static TVirtualPadPainter *PadPainter(Option_t *opt = "");

   ClassDef(TVirtualPadPainter, 0)//Painter interface for pad.
};

#endif
