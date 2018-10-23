// Author:  Sergey Linev, GSI  23/10/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWebPS
#define ROOT_TWebPS

#include "TVirtualPS.h"

#include "TWebPadPainter.h"

class TWebPS : public TVirtualPS {
   TWebPadPainter &fPainter;
public:
   TWebPS(TWebPadPainter &p) : fPainter(p) {}

   //Redirect calls to WebPainter
   //Line attributes.
   Color_t  GetLineColor() const { return fPainter.GetLineColor(); }
   Style_t  GetLineStyle() const { return fPainter.GetLineStyle(); }
   Width_t  GetLineWidth() const { return fPainter.GetLineWidth(); }

   void     SetLineColor(Color_t lcolor) { fPainter.SetLineColor(lcolor); }
   void     SetLineStyle(Style_t lstyle) { fPainter.SetLineStyle(lstyle); }
   void     SetLineWidth(Width_t lwidth) { fPainter.SetLineWidth(lwidth); }

   //Fill attributes.
   Color_t  GetFillColor() const { return fPainter.GetFillColor(); }
   Style_t  GetFillStyle() const { return fPainter.GetFillStyle(); }
   Bool_t   IsTransparent() const { return fPainter.IsTransparent(); }

   void     SetFillColor(Color_t fcolor)  { fPainter.SetFillColor(fcolor); }
   void     SetFillStyle(Style_t fstyle) { fPainter.SetFillStyle(fstyle); }
   void     SetOpacity(Int_t percent) { fPainter.SetOpacity(percent); }

   //Text attributes.
   Short_t  GetTextAlign() const { return fPainter.GetTextAlign(); }
   Float_t  GetTextAngle() const { return fPainter.GetTextAngle(); }
   Color_t  GetTextColor() const { return fPainter.GetTextColor(); }
   Font_t   GetTextFont()  const { return fPainter.GetTextFont(); }
   Float_t  GetTextSize()  const { return fPainter.GetTextSize(); }
   Float_t  GetTextMagnitude() const { return fPainter.GetTextMagnitude(); }

   void     SetTextAlign(Short_t align) { fPainter.SetTextAlign(align); }
   void     SetTextAngle(Float_t tangle) { fPainter.SetTextAngle(tangle); }
   void     SetTextColor(Color_t tcolor) { fPainter.SetTextColor(tcolor); }
   void     SetTextFont(Font_t tfont) { fPainter.SetTextFont(tfont); }
   void     SetTextSize(Float_t tsize) { fPainter.SetTextSize(tsize); }
   void     SetTextSizePixels(Int_t npixels) { fPainter.SetTextSizePixels(npixels); }

   //MISSING in base class - Marker attributes

   Color_t   GetMarkerColor() const { return fPainter.GetMarkerColor(); }
   Size_t    GetMarkerSize() const { return fPainter.GetMarkerSize(); }
   Style_t   GetMarkerStyle() const { return fPainter.GetMarkerStyle(); }

   void      SetMarkerColor(Color_t cindex) { fPainter.SetMarkerColor(cindex); }
   void      SetMarkerSize(Float_t markersize) { fPainter.SetMarkerSize(markersize); }
   void      SetMarkerStyle(Style_t markerstyle) { fPainter.SetMarkerStyle(markerstyle); }


   virtual void CellArrayBegin(Int_t W, Int_t H, Double_t x1, Double_t x2, Double_t y1, Double_t y2) {}
   virtual void CellArrayFill(Int_t r, Int_t g, Int_t b) {}
   virtual void CellArrayEnd()  {}
   virtual void Close(Option_t *opt = "")  {}
   virtual void DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2)  {}
   virtual void
   DrawFrame(Double_t xl, Double_t yl, Double_t xt, Double_t yt, Int_t mode, Int_t border, Int_t dark, Int_t light)  {}
   virtual void DrawPolyMarker(Int_t n, Float_t *x, Float_t *y)  {}
   virtual void DrawPolyMarker(Int_t n, Double_t *x, Double_t *y)  {}
   virtual void DrawPS(Int_t n, Float_t *xw, Float_t *yw)  {}
   virtual void DrawPS(Int_t n, Double_t *xw, Double_t *yw)  {}
   virtual void NewPage()  {}
   virtual void Open(const char *filename, Int_t type = -111)  {}
   virtual void Text(Double_t x, Double_t y, const char *string)  {}
   virtual void Text(Double_t x, Double_t y, const wchar_t *string)  {}
   virtual void SetColor(Float_t r, Float_t g, Float_t b)  {}

   ClassDef(TWebPS, 0) // Redirection of VirtualPS to Web painter
};

#endif
