// @(#)root/postscript:$Id$
// Author: O.Couet   16/07/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPostScript
#define ROOT_TPostScript


#include "TVirtualPS.h"

class TPoints;

class TPostScript : public TVirtualPS {

protected:
   Float_t fX1v = 0.;             ///< X bottom left corner of paper
   Float_t fY1v = 0.;             ///< Y bottom left corner of paper
   Float_t fX2v = 0.;             ///< X top right corner of paper
   Float_t fY2v = 0.;             ///< Y top right corner of paper
   Float_t fX1w = 0.;             ///<
   Float_t fY1w = 0.;             ///<
   Float_t fX2w = 0.;             ///<
   Float_t fY2w = 0.;             ///<
   Float_t fDXC = 0.;             ///<
   Float_t fDYC = 0.;             ///<
   Float_t fXC = 0.;              ///<
   Float_t fYC = 0.;              ///<
   Float_t fFX = 0.;              ///<
   Float_t fFY = 0.;              ///<
   Float_t fXVP1 = 0.;            ///<
   Float_t fXVP2 = 0.;            ///<
   Float_t fYVP1 = 0.;            ///<
   Float_t fYVP2 = 0.;            ///<
   Float_t fXVS1 = 0.;            ///<
   Float_t fXVS2 = 0.;            ///<
   Float_t fYVS1 = 0.;            ///<
   Float_t fYVS2 = 0.;            ///<
   Float_t fXsize = 0.;           ///< Page size along X
   Float_t fYsize = 0.;           ///< Page size along Y
   Float_t fMaxsize = 0.;         ///< Largest dimension of X and Y
   Float_t fRed = 0.;             ///< Per cent of red
   Float_t fGreen = 0.;           ///< Per cent of green
   Float_t fBlue = 0.;            ///< Per cent of blue
   Float_t fWidth = 0.;           ///< Current line width
   Int_t   fStyle = 1;            ///< Current line style
   Float_t fLineScale = 0.;       ///< Line width scale factor
   Int_t   fSave = 0;             ///< Number of gsave for restore
   Int_t   fNXzone = 0;           ///< Number of zones along X
   Int_t   fNYzone = 0;           ///< Number of zones along Y
   Int_t   fIXzone = 0;           ///< Current zone along X
   Int_t   fIYzone = 0;           ///< Current zone along Y
   Float_t fMarkerSizeCur = 0.;   ///< current transformed value of marker size
   Int_t   fNpages = 0;           ///< number of pages
   Int_t   fType = 0;             ///< PostScript workstation type
   Int_t   fMode = 0;             ///< PostScript mode
   Int_t   fClip = 0;             ///< Clipping mode
   Bool_t  fBoundingBox = kFALSE; ///< True for Encapsulated PostScript
   Bool_t  fClear = kFALSE;       ///< True when page must be cleared
   Bool_t  fClipStatus = kFALSE;  ///< Clipping Indicator
   Bool_t  fRange = kFALSE;       ///< True when a range has been defined
   Bool_t  fZone = kFALSE;        ///< Zone indicator
   char    fPatterns[32];         ///< Indicate if pattern n is defined
   Int_t   fNbinCT = 0;           ///< Number of entries in the current Cell Array
   Int_t   fNbCellW = 0;          ///< Number of boxes per line
   Int_t   fNbCellLine = 0;       ///< Number of boxes in the current line
   Int_t   fMaxLines = 0;         ///< Maximum number of lines in a PS array
   Int_t   fLastCellRed = 0;      ///< Last red value
   Int_t   fLastCellGreen = 0;    ///< Last green value
   Int_t   fLastCellBlue = 0;     ///< Last blue value
   Int_t   fNBSameColorCell = 0;  ///< Number of boxes with the same color
   TString fFileName;             ///< PS file name
   Bool_t  fFontEmbed = kFALSE;   ///< True is FontEmbed has been called
   Bool_t  fMustEmbed[29];        ///< flag to embed font

   static Int_t fgLineJoin;       ///< Appearance of joining lines
   static Int_t fgLineCap;        ///< Appearance of line caps

public:
   TPostScript();
   TPostScript(const char *filename, Int_t type=-111);
   ~TPostScript() override;

   void  CellArrayBegin(Int_t W, Int_t H, Double_t x1, Double_t x2,
                                          Double_t y1, Double_t y2) override;
   void  CellArrayFill(Int_t r, Int_t g, Int_t b) override;
   void  CellArrayEnd() override;
   void  Close(Option_t *opt="") override;
   Int_t CMtoPS(Double_t u) {return Int_t(0.5 + 72*u/2.54);}
   void  DefineMarkers();
   void  DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2) override;
   void  DrawFrame(Double_t xl, Double_t yl, Double_t xt, Double_t yt, Int_t mode, Int_t border, Int_t dark,
                  Int_t light) override;
   void  DrawHatch(Float_t dy, Float_t angle, Int_t n, Float_t *x, Float_t *y);
   void  DrawHatch(Float_t dy, Float_t angle, Int_t n, Double_t *x, Double_t *y);
   void  DrawPolyLine(Int_t n, TPoints *xy);
   void  DrawPolyLineNDC(Int_t n, TPoints *uv);
   void  DrawPolyMarker(Int_t n, Float_t *x, Float_t *y) override;
   void  DrawPolyMarker(Int_t n, Double_t *x, Double_t *y) override;
   void  DrawPS(Int_t n, Float_t *xw, Float_t *yw) override;
   void  DrawPS(Int_t n, Double_t *xw, Double_t *yw) override;
   bool  FontEmbedType1(const char *filename);
   bool  FontEmbedType2(const char *filename);
   bool  FontEmbedType42(const char *filename);
   void  FontEmbed();
   void  FontEncode();
   void  Initialize();
   void  NewPage() override;
   void  Off();
   void  On();
   void  Open(const char *filename, Int_t type=-111) override;
   void  SaveRestore(Int_t flag);
   void  SetFillColor(Color_t cindex=1) override;
   void  SetFillPatterns(Int_t ipat, Int_t color);
   void  SetLineColor(Color_t cindex=1) override;
   void  SetLineJoin(Int_t linejoin=0);
   void  SetLineCap(Int_t linecap=0);
   void  SetLineStyle(Style_t linestyle = 1) override;
   void  SetLineWidth(Width_t linewidth = 1) override;
   void  SetLineScale(Float_t scale=3) {fLineScale = scale;}
   void  SetMarkerColor(Color_t cindex=1) override;
   void  SetTextColor(Color_t cindex=1) override;
   void  SetWidth(Width_t linewidth = 1);
   void  SetStyle(Style_t linestyle = 1);
   void  MovePS(Int_t x, Int_t y);
   void  Range(Float_t xrange, Float_t yrange);
   void  SetColor(Int_t color = 1);
   void  SetColor(Float_t r, Float_t g, Float_t b) override;
   void  Text(Double_t x, Double_t y, const char *string) override;
   void  TextUrl(Double_t x, Double_t y, const char *string, const char *url) override;
   void  Text(Double_t x, Double_t y, const wchar_t *string) override;
   void  TextNDC(Double_t u, Double_t v, const char *string);
   void  TextNDC(Double_t u, Double_t v, const wchar_t *string);
   Int_t UtoPS(Double_t u);
   Int_t VtoPS(Double_t v);
   Int_t XtoPS(Double_t x);
   Int_t YtoPS(Double_t y);
   void  Zone();

   ClassDefOverride(TPostScript,1)  //PostScript driver
};

#endif
