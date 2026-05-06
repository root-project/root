// @(#)root/postscript:$Id: TPDF.h,v 1.0
// Author: Olivier Couet

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPDF
#define ROOT_TPDF


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPDF                                                                 //
//                                                                      //
// PDF driver.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualPS.h"
#include <vector>
#include <string>


class TPoints;

class TPDF : public TVirtualPS {

protected:
   Float_t fRed = 0.;                   ///< Per cent of red
   Float_t fGreen = 0.;                 ///< Per cent of green
   Float_t fBlue = 0.;                  ///< Per cent of blue
   Float_t fAlpha = 1.;                 ///< Per cent of transparency
   std::vector<float> fAlphas;          ///< List of alpha values used
   Float_t fXsize = 0.;                 ///< Page size along X
   Float_t fYsize = 0.;                 ///< Page size along Y
   Int_t fType = 0;                     ///< Workstation type used to know if the PDF is open
   Int_t fPageFormat = 0;               ///< Page format (A4, Letter etc ...)
   Int_t fPageOrientation = 0;          ///< Page orientation (Portrait, Landscape)
   Int_t fStartStream = 0;              ///< Stream start
   Float_t fLineScale = 0.;             ///< Line width scale factor
   std::vector<Int_t> fObjPos;          ///< Objects position
   Int_t fNbPage = 0;                   ///< Number of pages
   Int_t fCurrentPage = 0;              ///< Object number of the current page
   std::vector<int> fPageObjects;       ///< Page object numbers
   std::vector<std::string> fUrls;      ///< URLs
   std::vector<float> fRectX1;          ///< x1 /Rect coordinates for url annots
   std::vector<float> fRectY1;          ///< y1 /Rect coordinates for url annots
   std::vector<float> fRectX2;          ///< x2 /Rect coordinates for url annots
   std::vector<float> fRectY2;          ///< y2 /Rect coordinates for url annots
   Bool_t fObjectIsOpen = kFALSE;       ///< True if an object is opened
   Bool_t fPageNotEmpty = kFALSE;       ///< True if the current page is not empty
   Bool_t fCompress = kFALSE;           ///< True when fBuffer must be compressed
   Bool_t fRange = kFALSE;              ///< True when a range has been defined
   Bool_t fUrl = kFALSE;                ///< True when the text has an URL
   Int_t fNbUrl = 1;                    ///< Number of URLs in the current page
   Double_t fA = 1.;                    ///< "a" value of the Current Transformation Matrix (CTM)
   Double_t fB = 0.;                    ///< "b" value of the Current Transformation Matrix (CTM)
   Double_t fC = 0.;                    ///< "c" value of the Current Transformation Matrix (CTM)
   Double_t fD = 1.;                    ///< "d" value of the Current Transformation Matrix (CTM)
   Double_t fE = 0.;                    ///< "e" value of the Current Transformation Matrix (CTM)
   Double_t fF = 0.;                    ///< "f" value of the Current Transformation Matrix (CTM)

   static Int_t fgLineJoin; ///< Appearance of joining lines
   static Int_t fgLineCap;  ///< Appearance of line caps

   void EnsureBufferSize(Int_t required_size);

public:
   TPDF();
   TPDF(const char *filename, Int_t type=-111);
   ~TPDF() override;

   void CellArrayBegin(Int_t W, Int_t H, Double_t x1, Double_t x2, Double_t y1, Double_t y2) override;
   void CellArrayFill(Int_t r, Int_t g, Int_t b) override;
   void CellArrayEnd() override;
   void Close(Option_t *opt = "") override;
   Double_t CMtoPDF(Double_t u) { return Int_t(0.5 + 72 * u / 2.54); }
   void ComputeRect(const char* chars, Double_t fontsize, Double_t a, Double_t b, Double_t c, Double_t d, Double_t e, Double_t f);
   void DrawBox(Double_t x1, Double_t y1,Double_t x2, Double_t  y2) override;
   void DrawFrame(Double_t xl, Double_t yl, Double_t xt, Double_t  yt,
                  Int_t mode, Int_t border, Int_t dark, Int_t light) override;
   void DrawHatch(Float_t dy, Float_t angle, Int_t n, Float_t *x, Float_t *y);
   void DrawHatch(Float_t dy, Float_t angle, Int_t n, Double_t *x, Double_t *y);
   void DrawPolyLine(Int_t n, TPoints *xy);
   void DrawPolyLineNDC(Int_t n, TPoints *uv);
   void DrawPolyMarker(Int_t n, Float_t *x, Float_t *y) override;
   void DrawPolyMarker(Int_t n, Double_t *x, Double_t *y) override;
   void DrawPS(Int_t n, Float_t *xw, Float_t *yw) override;
   void DrawPS(Int_t n, Double_t *xw, Double_t *yw) override;
   void LineTo(Double_t x, Double_t y);
   void MoveTo(Double_t x, Double_t y);
   void EndObject();
   void FontEncode();
   void NewObject(Int_t n);
   void NewPage() override;
   void Off();
   void On();
   void Open(const char *filename, Int_t type = -111) override;
   void PatternEncode();
   void PrintFast(Int_t nch, const char *string = "") override;
   void PrintStr(const char *string = "") override;
   void Range(Float_t xrange, Float_t yrange);
   void SetAlpha(Float_t alpha = 1.);
   void SetColor(Int_t color = 1);
   void SetColor(Float_t r, Float_t g, Float_t b) override;
   void SetFillColor( Color_t cindex = 1) override;
   void SetFillPatterns(Int_t ipat, Int_t color);
   void SetLineColor( Color_t cindex = 1) override;
   void SetLineJoin(Int_t linejoin = 0);
   void SetLineCap(Int_t linecap = 0);
   void SetLineScale(Float_t scale = 1) {fLineScale = scale;}
   void SetLineStyle(Style_t linestyle = 1) override;
   void SetLineWidth(Width_t linewidth = 1) override;
   void SetMarkerColor(Color_t cindex = 1) override;
   void SetTextColor(Color_t cindex = 1) override;
   void Text(Double_t x, Double_t y, const char *string) override;
   void Text(Double_t, Double_t, const wchar_t *) override;
   void TextUrl(Double_t x, Double_t y, const char *string, const char *url) override;
   void TextNDC(Double_t u, Double_t v, const char *string);
   void TextNDC(Double_t, Double_t, const wchar_t *);
   void WriteCompressedBuffer();
   void WriteCM(Double_t a, Double_t b, Double_t c, Double_t d, Double_t e, Double_t f, Bool_t acc = kTRUE);
   void WriteReal(Float_t r, Bool_t space = kTRUE) override;
   void WriteUrlObjects();
   Double_t UtoPDF(Double_t u);
   Double_t VtoPDF(Double_t v);
   Double_t XtoPDF(Double_t x);
   Double_t YtoPDF(Double_t y);

   ClassDefOverride(TPDF, 1); // PDF driver
};

#endif
