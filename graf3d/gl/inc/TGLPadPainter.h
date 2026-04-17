// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  06/05/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLPadPainter
#define ROOT_TGLPadPainter

#include "TPadPainterBase.h"
#include "TGLFontManager.h"
#include "TGLPadUtils.h"
#include "TPoint.h"
#include "GuiTypes.h"

#include <vector>

class TLinearGradient;
class TRadialGradient;
/*
The _main_ purpose of TGLPadPainter is to enable 2d gl raphics
inside standard TPad/TCanvas.
*/
class TGLPadPainter : public TPadPainterBase {
private:
   Rgl::Pad::PolygonStippleSet fSSet;
   Rgl::Pad::Tesselator        fTess;
   Rgl::Pad::MarkerPainter     fMarker;
   Rgl::Pad::GLLimits          fLimits;

   WinContext_t   fWinContext; // context of selected drawable
   TAttFill       fGlFillAtt;  // fill attributes used for GL

   std::vector<Double_t>       fVs;//Vertex buffer for tesselator.

   TGLFontManager              fFM;
   TGLFont                     fF;

   Int_t                       fVp[4];

   std::vector<TPoint>         fPoly;
   Bool_t                      fIsHollowArea;

   Bool_t                      fLocked;

   template<class Char_t>
   void DrawTextHelper(Double_t x, Double_t y, const Char_t *text, ETextMode mode);
public:
   TGLPadPainter();

   void     SetOpacity(Int_t percent) override;
   Float_t  GetTextMagnitude() const override;


   void      OnPad(TVirtualPad *) override;

   // Overwrite only attributes setters
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

   void     InitPainter() override;
   void     InvalidateCS() override;
   void     LockPainter() override;

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
   void     DrawText(Double_t, Double_t, const wchar_t *, ETextMode) override;
   void     DrawTextNDC(Double_t x, Double_t y, const char *text, ETextMode mode) override;
   void     DrawTextNDC(Double_t, Double_t, const wchar_t *, ETextMode) override;

   //jpg, png, gif and bmp output.
   void     SaveImage(TVirtualPad *pad, const char *fileName, Int_t type) const override;

   //TASImage support.
   void     DrawPixels(const unsigned char *pixelData, UInt_t width, UInt_t height,
                       Int_t dstX, Int_t dstY, Bool_t enableBlending) override;

   Bool_t IsNative() const override { return kTRUE; }

   Bool_t   IsCocoa() const override;

   Bool_t   IsSupportAlpha() const override { return kTRUE; }

private:

   //Attention! GL_PROJECTION will become
   //the current matrix after these calls.
   void     SaveProjectionMatrix()const;
   void     RestoreProjectionMatrix()const;

   //Attention! GL_MODELVIEW will become the
   //current matrix after these calls.
   void     SaveModelviewMatrix()const;
   void     RestoreModelviewMatrix()const;

   void     SaveViewport();
   void     RestoreViewport();

   void     DrawPolyMarker();

   //Aux. functions for a gradient and solid fill:
   void DrawPolygonWithGradient(Int_t n, const Double_t *x, const Double_t *y);
   //
   void DrawGradient(const TLinearGradient *gradient, Int_t n, const Double_t *x, const Double_t *y);
   void DrawGradient(const TRadialGradient *gradient, Int_t n, const Double_t *x, const Double_t *y);
   //
   void DrawTesselation(Int_t n, const Double_t *x, const Double_t *y);

   TGLPadPainter(const TGLPadPainter &rhs);
   TGLPadPainter & operator = (const TGLPadPainter &rhs);

   ClassDefOverride(TGLPadPainter, 0)
};

#endif

