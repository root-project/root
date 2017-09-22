// $Id$
// Author:  Sergey Linev  10/04/2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWebPadPainter
#define ROOT_TWebPadPainter

#include "TVirtualPadPainter.h"

#include "TWebPainting.h"

class TVirtualPad;
class TWebCanvas;
class TPoint;

/*
TWebPadPainter tries to support old Paint methods of the ROOT classes.
Main classes (like histograms or graphs) should be painted on JavaScript side
*/

class TWebPadPainter : public TVirtualPadPainter {

friend class TWebCanvas;

protected:
   TWebPainterAttributes *fAttr; ///!< current attributes
   Bool_t fAttrChanged;          ///!< flag that attributes are changed after last paint operation
   TWebPainting *fPainting;      ///!< object to store all painting
   UInt_t fCw, fCh;              ///!< canvas dimensions, need for back pixel conversion
   Float_t fKx, fKy;             ///!< coefficient to recalculate pixel coordinates

   TWebPainterAttributes *Attr();

   void GetAttributes(Int_t attrmask = 0xf);

   void StoreOperation(const char* opt, TObject* obj = 0, Int_t attrmask = 0xf);

   Float_t *Reserve(Int_t sz);

public:
   TWebPadPainter();
   virtual ~TWebPadPainter();

   TWebPainting *TakePainting();
   void ResetPainting();

   void SetWebCanvasSize(UInt_t w, UInt_t h);

   //Final overriders for TVirtualPadPainter pure virtual functions.
   //1. Part, which simply delegates to TVirtualX.
   //Line attributes.
   Color_t  GetLineColor() const { return fAttr ? fAttr->GetLineColor() : 0; }
   Style_t  GetLineStyle() const { return fAttr ? fAttr->GetLineStyle() : 0; }
   Width_t  GetLineWidth() const { return fAttr ? fAttr->GetLineWidth() : 0; }

   void     SetLineColor(Color_t lcolor) { if (GetLineColor()!=lcolor) Attr()->SetLineColor(lcolor); }
   void     SetLineStyle(Style_t lstyle) { if (GetLineStyle()!=lstyle) Attr()->SetLineStyle(lstyle); }
   void     SetLineWidth(Width_t lwidth) { if (GetLineWidth()!=lwidth) Attr()->SetLineWidth(lwidth); }

   //Fill attributes.
   Color_t  GetFillColor() const { return fAttr ? fAttr->GetFillColor() : 0; }
   Style_t  GetFillStyle() const { return fAttr ? fAttr->GetFillStyle() : 0; }
   Bool_t   IsTransparent() const { return fAttr ? fAttr->IsTransparent() : kFALSE; }

   void     SetFillColor(Color_t fcolor)  { if (GetFillColor()!=fcolor) Attr()->SetFillColor(fcolor); }
   void     SetFillStyle(Style_t fstyle) { if (GetFillStyle()!=fstyle) Attr()->SetFillStyle(fstyle); }
   void     SetOpacity(Int_t percent) { if (GetFillStyle()!=4000+percent) Attr()->SetFillStyle(4000 + percent); }

   //Text attributes.
   Short_t  GetTextAlign() const { return fAttr ? fAttr->GetTextAlign() : 0; }
   Float_t  GetTextAngle() const { return fAttr ? fAttr->GetTextAngle() : 0; }
   Color_t  GetTextColor() const { return fAttr ? fAttr->GetTextColor() : 0; }
   Font_t   GetTextFont()  const { return fAttr ? fAttr->GetTextFont() : 0; }
   Float_t  GetTextSize()  const { return fAttr ? fAttr->GetTextSize() : 0; }
   Float_t  GetTextMagnitude() const { return  0; }

   void     SetTextAlign(Short_t align) { if (GetTextAlign()!=align) Attr()->SetTextAlign(align); }
   void     SetTextAngle(Float_t tangle) { if (GetTextAngle()!=tangle) Attr()->SetTextAngle(tangle); }
   void     SetTextColor(Color_t tcolor) { if (GetTextColor()!=tcolor) Attr()->SetTextColor(tcolor); }
   void     SetTextFont(Font_t tfont) { if (GetTextFont()!=tfont) Attr()->SetTextFont(tfont); }
   void     SetTextSize(Float_t tsize) { if (GetTextSize()!=tsize) Attr()->SetTextSize(tsize); }
   void     SetTextSizePixels(Int_t npixels) { Attr()->SetTextSizePixels(npixels); }

   //MISSING in base class - Marker attributes

   Color_t   GetMarkerColor() const { return fAttr ? fAttr->GetMarkerColor() : 0; }
   Size_t    GetMarkerSize() const { return fAttr ? fAttr->GetMarkerSize() : 0; }
   Style_t   GetMarkerStyle() const { return fAttr ? fAttr->GetMarkerStyle() : 0; }

   void      SetMarkerColor(Color_t cindex) { if (GetMarkerColor()!=cindex) Attr()->SetMarkerColor(cindex); }
   void      SetMarkerSize(Float_t markersize) { if (GetMarkerSize()!=markersize) Attr()->SetMarkerSize(markersize); }
   void      SetMarkerStyle(Style_t markerstyle) { if (GetMarkerStyle()!=markerstyle) Attr()->SetMarkerStyle(markerstyle); }

   //2. "Off-screen management" part.
   Int_t    CreateDrawable(UInt_t w, UInt_t h);
   void     ClearDrawable();
   void     CopyDrawable(Int_t id, Int_t px, Int_t py);
   void     DestroyDrawable(Int_t device);
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
