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
