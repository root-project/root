// Author:  Sergey Linev, GSI  10/04/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWebPadPainter
#define ROOT_TWebPadPainter

#include "TVirtualPadPainter.h"

#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttText.h"
#include "TAttMarker.h"

#include "TWebPainting.h"

#include <memory>

class TVirtualPad;
class TWebCanvas;
class TPoint;

/*
TWebPadPainter tries to support old Paint methods of the ROOT classes.
Main classes (like histograms or graphs) should be painted on JavaScript side
*/

class TWebPadPainter : public TVirtualPadPainter, public TAttLine, public TAttFill, public TAttText, public TAttMarker {

friend class TWebCanvas;

protected:

   TWebPainting *fPainting{nullptr};      ///!< object to store all painting, owned by TWebPS object

   enum { attrLine = 0x1, attrFill = 0x2, attrMarker = 0x4, attrText = 0x8, attrAll = 0xf };

   Float_t *StoreOperation(const std::string &oper, unsigned attrkind, int opersize = 0);

public:
   TWebPadPainter() = default;
   virtual ~TWebPadPainter() = default;

   void SetPainting(TWebPainting *p) { fPainting = p; }


   //Final overriders for TVirtualPadPainter pure virtual functions.
   //1. Part, which simply catch attributes.
   //Line attributes.
   Color_t  GetLineColor() const { return TAttLine::GetLineColor(); }
   Style_t  GetLineStyle() const { return TAttLine::GetLineStyle(); }
   Width_t  GetLineWidth() const { return TAttLine::GetLineWidth(); }

   void     SetLineColor(Color_t lcolor) { TAttLine::SetLineColor(lcolor); }
   void     SetLineStyle(Style_t lstyle) { TAttLine::SetLineStyle(lstyle); }
   void     SetLineWidth(Width_t lwidth) { TAttLine::SetLineWidth(lwidth); }

   //Fill attributes.
   Color_t  GetFillColor() const { return TAttFill::GetFillColor(); }
   Style_t  GetFillStyle() const { return TAttFill::GetFillStyle(); }
   Bool_t   IsTransparent() const { return TAttFill::IsTransparent(); }

   void     SetFillColor(Color_t fcolor)  { TAttFill::SetFillColor(fcolor); }
   void     SetFillStyle(Style_t fstyle) { TAttFill::SetFillStyle(fstyle); }
   void     SetOpacity(Int_t percent) { TAttFill::SetFillStyle(4000 + percent); }

   //Text attributes.
   Short_t  GetTextAlign() const { return TAttText::GetTextAlign(); }
   Float_t  GetTextAngle() const { return TAttText::GetTextAngle(); }
   Color_t  GetTextColor() const { return TAttText::GetTextColor(); }
   Font_t   GetTextFont()  const { return TAttText::GetTextFont(); }
   Float_t  GetTextSize()  const { return TAttText::GetTextSize(); }
   Float_t  GetTextMagnitude() const { return  0; }

   void     SetTextAlign(Short_t align) { TAttText::SetTextAlign(align); }
   void     SetTextAngle(Float_t tangle) { TAttText::SetTextAngle(tangle); }
   void     SetTextColor(Color_t tcolor) { TAttText::SetTextColor(tcolor); }
   void     SetTextFont(Font_t tfont) { TAttText::SetTextFont(tfont); }
   void     SetTextSize(Float_t tsize) { TAttText::SetTextSize(tsize); }
   void     SetTextSizePixels(Int_t npixels) { TAttText::SetTextSizePixels(npixels); }

   //2. "Off-screen management" part.
   Int_t    CreateDrawable(UInt_t, UInt_t) { return -1; }
   void     ClearDrawable() {}
   void     CopyDrawable(Int_t, Int_t, Int_t) {}
   void     DestroyDrawable(Int_t) {}
   void     SelectDrawable(Int_t) {}
   //jpg, png, bmp, gif output.
   void     SaveImage(TVirtualPad *, const char *, Int_t) const {}


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

   // reimplementing some direct X11 calls, which are not fetched
   void     DrawFillArea(Int_t n, TPoint *xy);

private:
   //Let's make this clear:
   TWebPadPainter(const TWebPadPainter &rhs) = delete;
   TWebPadPainter(TWebPadPainter && rhs) = delete;
   TWebPadPainter & operator = (const TWebPadPainter &rhs) = delete;
   TWebPadPainter & operator = (TWebPadPainter && rhs) = delete;

   ClassDef(TWebPadPainter, 0) //Abstract interface for painting in TPad
};

#endif
