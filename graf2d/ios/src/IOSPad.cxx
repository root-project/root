// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 14/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <algorithm>
#include <stdexcept>

#include <CoreText/CTStringAttributes.h>
#include <CoreText/CTFont.h>
#include <CoreText/CTLine.h>

#include "TViewer3DPad.h"
#include "TMultiGraph.h"
#include "TVirtualX.h"
#include "TString.h"
#include "THStack.h"
#include "TFrame.h"
#include "TPoint.h"
#include "TColor.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TView.h"
#include "TROOT.h"
#include "TMath.h"
#include "TH1.h"

//Internal (module) includes.
#include "IOSGraphicUtils.h"
#include "IOSFillPatterns.h"
#include "IOSPad.h"

namespace ROOT {
namespace iOS {

//______________________________________________________________________________
Pad::Pad(UInt_t w, UInt_t h)
{
   fViewW = w;
   fViewH = h;

   fX1 = 0.;
   fX2 = 1.;
   fY1 = 0.;
   fY2 = 1.;

   fUxmin = 0.;
   fUymin = 0.;
   fUxmax = 0.;
   fUymax = 0.;

   fTheta = 30;
   fPhi = 50;

   fAspectRatio = 0.;

   fTickx = gStyle->GetPadTickX();
   fTicky = gStyle->GetPadTickY();

   fLogx = gStyle->GetOptLogx();
   fLogy = gStyle->GetOptLogy();
   fLogz = gStyle->GetOptLogz();

   fBorderSize = 0;
   fBorderMode = 0;

   fGridx = gStyle->GetPadGridX();
   fGridy = gStyle->GetPadGridY();

   fAbsCoord = kFALSE;
   fEditable = kTRUE;
   fFixedAspectRatio = kFALSE;

   fFrame = 0;
   fView = 0;

   fViewer3D = 0;
   fSelectionIsValid = kFALSE;
   fSelectionAreaWidth = w;
   fSelected = 0;
   fParentOfSelected = 0;
   fInSelectionMode = kFALSE;
   fInHighlightMode = kFALSE;
   fObjectID = 1;

   fContains3DObject = kFALSE;

   cd();

   // Set pad parameters and Compute conversion coefficients
   SetPad("", "", 0., 0., 1., 1., 0, 0, 0);

   Range(0., 0., 1., 1.);
}

//______________________________________________________________________________
Pad::~Pad()
{
   delete fFrame;
   delete fViewer3D;
   //Absolutely not clear, if pad owns view or not,
   //because I've seen code wich delete's and creates view outside pad
   //and ignores pad.
   //At the same time, there is a code in a pad, which can delete fView.
   //What a mess!
   delete fView;
}

//______________________________________________________________________________
const char *Pad::GetName() const
{
   return "iOSPad";
}

//______________________________________________________________________________
const char *Pad::GetTitle() const
{
   return "iOSPad";
}

//______________________________________________________________________________
void Pad::Clear(Option_t *)
{
   fSelectionIsValid = kFALSE;
   fSelected = 0;
   fParentOfSelected = 0;
   fSelectables.clear();
   fParentPainters.clear();
   fSelectionBuffer.clear();
   fObjectID = 1;

   fPrimitives.SetOwner(kFALSE);
   fPrimitives.Clear();

   fContains3DObject = kFALSE;

   Range(0., 0., 1., 1.);
}

//______________________________________________________________________________
void Pad::SetViewWH(UInt_t viewW, UInt_t viewH)
{
   fViewW = viewW;
   fViewH = viewH;

   ResizePad();

   fPainter.SetTransform(fViewW, GetX1(), GetX2(), fViewH, GetY1(), GetY2());
}

//______________________________________________________________________________
UInt_t Pad::GetWw() const
{
   return fViewW;
}

//______________________________________________________________________________
UInt_t Pad::GetWh() const
{
   return fViewH;
}

//______________________________________________________________________________
void Pad::SetFixedAspectRatio(Bool_t fixed)
{
   // Fix pad aspect ratio to current value if fixed is true.
   if (fixed) {
      if (!fFixedAspectRatio) {
         if (fHNDC != 0.)
            fAspectRatio = fWNDC / fHNDC;
         else {
            //cannot fix aspect ratio, height of pad is 0
            return;
         }
         fFixedAspectRatio = kTRUE;
      }
   } else {
      fFixedAspectRatio = kFALSE;
      fAspectRatio = 0;
   }
}

//______________________________________________________________________________
Double_t Pad::PadtoX(Double_t x) const
{
   // Convert x from pad to X.
   if (fLogx && x < 50)
      return Double_t(TMath::Exp(2.302585092994 * x));

   return x;
}

//______________________________________________________________________________
Double_t Pad::PadtoY(Double_t y) const
{
   // Convert y from pad to Y.
   if (fLogy && y < 50)
      return Double_t(TMath::Exp(2.302585092994 * y));

   return y;
}

//______________________________________________________________________________
Double_t Pad::XtoPad(Double_t x) const
{
   // Convert x from X to pad.
   if (fLogx) {
      if (x > 0)
         x = TMath::Log10(x);
      else
         x = fUxmin;
   }

   return x;
}

//______________________________________________________________________________
Double_t Pad::YtoPad(Double_t y) const
{
   // Convert y from Y to pad.
   if (fLogy) {
      if (y > 0)
         y = TMath::Log10(y);
      else
         y = fUymin;
   }

   return y;
}

//______________________________________________________________________________
Int_t Pad::UtoPixel(Double_t u) const
{
   Double_t val;
   if (fAbsCoord)
      val = fUtoAbsPixelk + u * fUtoPixel;
   else
      val = u * fUtoPixel;

   if (val < -kMaxPixel)
      return -kMaxPixel;

   if (val >  kMaxPixel)
      return  kMaxPixel;

   return Int_t(val);
}

//______________________________________________________________________________
Int_t Pad::VtoPixel(Double_t v) const
{
   Double_t val;
   if (fAbsCoord)
      val = fVtoAbsPixelk + v * fVtoPixel;
   else
      val = fVtoPixelk + v * fVtoPixel;

   if (val < -kMaxPixel)
      return -kMaxPixel;

   if (val >  kMaxPixel)
      return  kMaxPixel;

   return Int_t(val);
}

//______________________________________________________________________________
Int_t Pad::XtoAbsPixel(Double_t x) const
{
   const Double_t val = fXtoAbsPixelk + x * fXtoPixel;
   if (val < -kMaxPixel)
      return -kMaxPixel;
   if (val >  kMaxPixel)
      return kMaxPixel;

   return Int_t(val);
}

//______________________________________________________________________________
Int_t Pad::YtoAbsPixel(Double_t y) const
{
   const Double_t val = fYtoAbsPixelk + y*fYtoPixel;
   if (val < -kMaxPixel)
      return -kMaxPixel;
   if (val >  kMaxPixel)
      return kMaxPixel;

   return Int_t(val);
}

//______________________________________________________________________________
Int_t Pad::XtoPixel(Double_t x) const
{
   Double_t val;
   if (fAbsCoord)
      val = fXtoAbsPixelk + x * fXtoPixel;
   else
      val = fXtoPixelk + x * fXtoPixel;

   if (val < -kMaxPixel)
      return -kMaxPixel;
   if (val >  kMaxPixel)
      return  kMaxPixel;

   return Int_t(val);
}

//______________________________________________________________________________
Int_t Pad::YtoPixel(Double_t y) const
{
   Double_t val;
   if (fAbsCoord)
      val = fYtoAbsPixelk + y * fYtoPixel;
   else
      val = fYtoPixelk + y * fYtoPixel;

   if (val < -kMaxPixel)
      return -kMaxPixel;
   if (val > kMaxPixel)
      return kMaxPixel;

   return Int_t(val);
}

//______________________________________________________________________________
Double_t Pad::PixeltoX(Int_t px)
{
   if (fAbsCoord)
      return fAbsPixeltoXk + px * fPixeltoX;
   else
      return fPixeltoXk + px * fPixeltoX;
}

//______________________________________________________________________________
Double_t Pad::PixeltoY(Int_t py)
{
   if (fAbsCoord)
      return fAbsPixeltoYk + py * fPixeltoY;
   else
      return fPixeltoYk + py * fPixeltoY;
}

//______________________________________________________________________________
void Pad::SetLogx(Int_t value)
{
   // Set Lin/Log scale for X
   //   value = 0 X scale will be linear
   //   value = 1 X scale will be logarithmic (base 10)
   //   value > 1 reserved for possible support of base e or other
   fLogx = value;
   delete fView;
   fView = 0;
}


//______________________________________________________________________________
void Pad::SetLogy(Int_t value)
{
   // Set Lin/Log scale for Y
   //   value = 0 Y scale will be linear
   //   value = 1 Y scale will be logarithmic (base 10)
   //   value > 1 reserved for possible support of base e or other

   fLogy = value;
   delete fView;
   fView=0;
}


//______________________________________________________________________________
void Pad::SetLogz(Int_t value)
{
   // Set Lin/Log scale for Z

   fLogz = value;
   delete fView;
   fView=0;
}

//______________________________________________________________________________
void Pad::SetPad(Double_t xlow, Double_t ylow, Double_t xup, Double_t yup)
{
   // Set canvas range for pad and resize the pad. If the aspect ratio
   // was fixed before the call it will be un-fixed.

   // Reorder points to make sure xlow,ylow is bottom left point and
   // xup,yup is top right point.
   if (xup < xlow)
      std::swap(xup, xlow);

   if (yup < ylow)
      std::swap(yup, ylow);

   fXlowNDC = xlow;
   fYlowNDC = ylow;
   fWNDC = xup - xlow;
   fHNDC = yup - ylow;

   SetFixedAspectRatio(kFALSE);

   ResizePad();
}

//______________________________________________________________________________
void Pad::SetPad(const char *, const char *, Double_t xlow, Double_t ylow, Double_t xup,
                 Double_t yup, Color_t color, Short_t bordersize, Short_t bordermode)
{
   // Set all pad parameters.
   SetFillStyle(1001);
   SetBottomMargin(gStyle->GetPadBottomMargin());
   SetTopMargin(gStyle->GetPadTopMargin());
   SetLeftMargin(gStyle->GetPadLeftMargin());
   SetRightMargin(gStyle->GetPadRightMargin());

   if (color >= 0)
      SetFillColor(color);
   else
      SetFillColor(gStyle->GetPadColor());

   if (bordersize <  0)
      fBorderSize = gStyle->GetPadBorderSize();
   else
      fBorderSize = bordersize;

   if (bordermode < -1)
      fBorderMode = gStyle->GetPadBorderMode();
   else
      fBorderMode = bordermode;

   SetPad(xlow, ylow, xup, yup);
}

//______________________________________________________________________________
void Pad::SetVertical(Bool_t vert)
{
   // Set pad vertical (default) or horizontal
   if (vert)
      ResetBit(kHori);
   else
      SetBit(kHori);
}

//______________________________________________________________________________
void Pad::GetRange(Double_t &x1, Double_t &y1, Double_t &x2, Double_t &y2)
{
   // Return pad world coordinates range.
   x1 = fX1;
   y1 = fY1;
   x2 = fX2;
   y2 = fY2;
}

//______________________________________________________________________________
void Pad::Range(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   // Set world coordinate system for the pad.
   // Emits signal "RangeChanged()", in the slot get the range
   // via GetRange().

   if ((x1 >= x2) || (y1 >= y2)) {
      //Error("Range", "illegal world coordinates range: x1=%f, y1=%f, x2=%f, y2=%f",x1,y1,x2,y2);
      return;
   }

   fUxmin = x1;
   fUxmax = x2;
   fUymin = y1;
   fUymax = y2;

   if (fX1 == x1 && fY1 == y1 && fX2 == x2 && fY2 == y2) return;

   fX1  = x1;
   fY1  = y1;
   fX2  = x2;
   fY2  = y2;

   // compute pad conversion coefficients
   ResizePad();

   fPainter.SetTransform(GetWw(), GetX1(), GetX2(), GetWh(), GetY1(), GetY2());
}

//______________________________________________________________________________
void Pad::GetRangeAxis(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax)
{
   // Return pad axis coordinates range.
   xmin = fUxmin;
   ymin = fUymin;
   xmax = fUxmax;
   ymax = fUymax;
}

//______________________________________________________________________________
void Pad::RangeAxis(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax)
{
   // Set axis coordinate system for the pad.
   // The axis coordinate system is a subset of the world coordinate system
   // xmin,ymin is the origin of the current coordinate system,
   // xmax is the end of the X axis, ymax is the end of the Y axis.
   // By default a margin of 10 per cent is left on all sides of the pad
   // Emits signal "RangeAxisChanged()", in the slot get the axis range
   // via GetRangeAxis().

   if ((xmin >= xmax) || (ymin >= ymax)) {
      //Error("RangeAxis", "illegal axis coordinates range: xmin=%f, ymin=%f, xmax=%f, ymax=%f", xmin, ymin, xmax, ymax);
      return;
   }

   fUxmin  = xmin;
   fUymin  = ymin;
   fUxmax  = xmax;
   fUymax  = ymax;
}

//______________________________________________________________________________
void Pad::GetPadPar(Double_t &xlow, Double_t &ylow, Double_t &xup, Double_t &yup)
{
   // Return lower and upper bounds of the pad in NDC coordinates.
   xlow = fXlowNDC;
   ylow = fYlowNDC;
   xup  = fXlowNDC+fWNDC;
   yup  = fYlowNDC+fHNDC;
}

//______________________________________________________________________________
void Pad::SetFillStyle(Style_t fillStyle)
{
   // Overrride TAttFill::FillStyle for TPad because we want to handle style=0
   // as style 4000.

   if (!fillStyle)
      fillStyle = 4000;

   TAttFill::SetFillStyle(fillStyle);
}

//______________________________________________________________________________
void Pad::SetEditable(Bool_t mode)
{
   // Set pad editable yes/no
   // If a pad is not editable:
   // - one cannot modify the pad and its objects via the mouse.
   // - one cannot add new objects to the pad

   fEditable = mode;
}

//______________________________________________________________________________
TVirtualPad *Pad::cd(Int_t)
{
   gPad = this;
   fPainter.SetTransform(GetWw(), GetX1(), GetX2(), GetWh(), GetY1(), GetY2());

   return this;
}

//______________________________________________________________________________
void Pad::SetView(TView *view)
{
   // Set the current TView. Delete previous view if view=0.
   // This code was taken from original TPad and it dies after
   // attempt to free memory - view was already deleted by THistPainter.
   // So, I simply assume, that pad does not own fView.
   /*
   if (!view)
      delete fView;
   */
   fView = view;
}

//______________________________________________________________________________
void Pad::ResizePad(Option_t *)
{
   // Compute pad conversion coefficients.
   //
   //   Conversion from x to px & y to py
   //   =================================
   //
   //       x - xmin     px - pxlow              xrange  = xmax-xmin
   //       --------  =  ----------      with
   //        xrange        pxrange               pxrange = pxmax-pxmin
   //
   //               pxrange(x-xmin)
   //   ==>  px =   ---------------  + pxlow   = fXtoPixelk + fXtoPixel * x
   //                    xrange
   //
   //   ==>  fXtoPixelk = pxlow - pxrange*xmin/xrange
   //        fXtoPixel  = pxrange/xrange
   //           where  pxlow   = fAbsXlowNDC*fCw
   //                  pxrange = fAbsWNDC*fCw
   //
   //
   //       y - ymin     py - pylow              yrange  = ymax-ymin
   //       --------  =  ----------      with
   //        yrange        pyrange               pyrange = pymax-pymin
   //
   //               pyrange(y-ymin)
   //   ==>  py =   ---------------  + pylow   = fYtoPixelk + fYtoPixel * y
   //                    yrange
   //
   //   ==>  fYtoPixelk = pylow - pyrange*ymin/yrange
   //        fYtoPixel  = pyrange/yrange
   //           where  pylow   = (1-fAbsYlowNDC)*fCh
   //                  pyrange = -fAbsHNDC*fCh
   //
   //-  Conversion from px to x & py to y
   //   =================================
   //
   //             xrange(px-pxlow)
   //   ==>  x =  ----------------  + xmin  = fPixeltoXk + fPixeltoX * px
   //                 pxrange
   //-
   //   ==>  fPixeltoXk = xmin - pxlow*xrange/pxrange
   //        fPixeltoX  = xrange/pxrange
   //
   //             yrange(py-pylow)
   //   ==>  y =  ----------------  + ymin  = fPixeltoYk + fPixeltoY * py
   //                 pyrange
   //-
   //   ==>  fPixeltoYk = ymin - pylow*yrange/pyrange
   //        fPixeltoY  = yrange/pyrange
   //
   //-----------------------------------------------------------------------
   //
   //  Computation of the coefficients in case of LOG scales
   //- =====================================================
   //
   //   A, Conversion from pixel coordinates to world coordinates
   //
   //       Log(x) - Log(xmin)      Log(x/xmin)       px - pxlow
   //  u = --------------------- =  -------------  =  -----------
   //      Log(xmax) - Log(xmin)    Log(xmax/xmin)     pxrange
   //
   //  ==> Log(x/xmin) = u*Log(xmax/xmin)
   //      x = xmin*exp(u*Log(xmax/xmin)
   //   Let alfa = Log(xmax/xmin)/fAbsWNDC
   //
   //      x = xmin*exp(-alfa*pxlow) + exp(alfa*px)
   //      x = fPixeltoXk*exp(fPixeltoX*px)
   //  ==> fPixeltoXk = xmin*exp(-alfa*pxlow)
   //      fPixeltoX  = alfa
   //
   //       Log(y) - Log(ymin)      Log(y/ymin)       pylow - py
   //  v = --------------------- =  -------------  =  -----------
   //      Log(ymax) - Log(ymin)    Log(ymax/ymin)     pyrange
   //
   //   Let beta = Log(ymax/ymin)/pyrange
   //      Log(y/ymin) = beta*pylow - beta*py
   //      y/ymin = exp(beta*pylow - beta*py)
   //      y = ymin*exp(beta*pylow)*exp(-beta*py)
   //  ==> y = fPixeltoYk*exp(fPixeltoY*py)
   //      fPixeltoYk = ymin*exp(beta*pylow)
   //      fPixeltoY  = -beta
   //
   //-  B, Conversion from World coordinates to pixel coordinates
   //
   //  px = pxlow + u*pxrange
   //     = pxlow + Log(x/xmin)/alfa
   //     = pxlow -Log(xmin)/alfa  + Log(x)/alfa
   //     = fXtoPixelk + fXtoPixel*Log(x)
   //  ==> fXtoPixelk = pxlow -Log(xmin)/alfa
   //  ==> fXtoPixel  = 1/alfa
   //
   //  py = pylow - Log(y/ymin)/beta
   //     = fYtoPixelk + fYtoPixel*Log(y)
   //  ==> fYtoPixelk = pylow - Log(ymin)/beta
   //      fYtoPixel  = 1/beta

   // Recompute subpad positions in case pad has been moved/resized
   fAbsXlowNDC  = fXlowNDC;
   fAbsYlowNDC  = fYlowNDC;
   fAbsWNDC     = fWNDC;
   fAbsHNDC     = fHNDC;
   Double_t ww = fViewW;
   Double_t wh = fViewH;
   Double_t pxlow   = fAbsXlowNDC * ww;
   Double_t pylow   = (1-fAbsYlowNDC) * wh;
   Double_t pxrange = fAbsWNDC * ww;
   Double_t pyrange = -fAbsHNDC * wh;
   // Linear X axis
   Double_t rounding = 0.00005;
   Double_t xrange  = fX2 - fX1;
   fXtoAbsPixelk = rounding + pxlow - pxrange*fX1/xrange;      //origin at left
   fXtoPixelk = rounding +  -pxrange*fX1/xrange;
   fXtoPixel  = pxrange/xrange;
   fAbsPixeltoXk = fX1 - pxlow*xrange/pxrange;
   fPixeltoXk = fX1;
   fPixeltoX  = xrange/pxrange;
   // Linear Y axis
   Double_t yrange  = fY2 - fY1;
   fYtoAbsPixelk = rounding + pylow - pyrange*fY1/yrange;      //origin at top
   fYtoPixelk = rounding +  -pyrange - pyrange*fY1/yrange;
   fYtoPixel  = pyrange/yrange;
   fAbsPixeltoYk = fY1 - pylow*yrange/pyrange;
   fPixeltoYk = fY1;
   fPixeltoY  = yrange/pyrange;
   // Coefficients to convert from pad NDC coordinates to pixel coordinates

   fUtoAbsPixelk = rounding + pxlow;
   fUtoPixelk = rounding;
   fUtoPixel  = pxrange;
   fVtoAbsPixelk = rounding + pylow;
   fVtoPixelk = -pyrange;
   fVtoPixel  = pyrange;
   gVirtualX->SetLineWidth(-1);
   gVirtualX->SetTextSize(-1);
   if (fView)
      fView->ResizePad();
}

//______________________________________________________________________________
TFrame *Pad::GetFrame()
{
   // Get frame.
   //Original TPad has a COMPLETE MESS here. I'm trying to fix this.
   if (!fFrame) {
      fFrame = new TFrame(0., 0., 1., 1.);

      Int_t framecolor = GetFrameFillColor();

      if (!framecolor)
         framecolor = GetFillColor();

      fFrame->SetFillColor(framecolor);
      fFrame->SetFillStyle(GetFrameFillStyle());
      fFrame->SetLineColor(GetFrameLineColor());
      fFrame->SetLineStyle(GetFrameLineStyle());
      fFrame->SetLineWidth(GetFrameLineWidth());
      fFrame->SetBorderSize(GetFrameBorderSize());
      fFrame->SetBorderMode(GetFrameBorderMode());
   }

   return fFrame;
}

//______________________________________________________________________________
TObject *Pad::GetPrimitive(const char *name) const
{
   // Final-overrider for TVirtualPad.

   TIter next(&fPrimitives);
   TObject *obj = 0;

   while ((obj=next())) {
      if (!strcmp(name, obj->GetName()))
         return obj;

      TObject *found = obj->FindObject(name);
      if (found)
         return found;
   }

   return 0;
}

//______________________________________________________________________________
void Pad::Paint(Option_t *)
{
   cd();

   //First, fill pad's area with white
   //- it's required in case this pad
   //has some fill pattern.
   gVirtualX->SetFillStyle(1001);
   gVirtualX->SetFillColor(0);
   PaintBox(fX1,fY1,fX2,fY2);
   //
   PaintBorder(GetFillColor(), kTRUE);

   TObjOptLink *lnk = (TObjOptLink*)GetListOfPrimitives()->FirstLink();
   TObject *obj;

   while (lnk) {
      obj = lnk->GetObject();
      obj->Paint(lnk->GetOption());
      lnk = (TObjOptLink*)lnk->Next();

      //This is the special case, which can not
      //be processed in a generic way at the moment.
      if (obj->InheritsFrom("TF2"))
         fContains3DObject = kTRUE;
   }

   Modified(kFALSE);
}

//______________________________________________________________________________
void Pad::PaintForSelection()
{
   fInSelectionMode = kTRUE;
   fPainter.SetPainterMode(Painter::kPaintToSelectionBuffer);
   //
   fObjectID = 1;
   fSelectables.clear();
   fParentPainters.clear();

   TObjOptLink *lnk = (TObjOptLink*)GetListOfPrimitives()->FirstLink();
   TObject *obj;

   while (lnk) {
      obj = lnk->GetObject();
      obj->Paint(lnk->GetOption());
      lnk = (TObjOptLink*)lnk->Next();
   }
   //
   fPainter.SetPainterMode(Painter::kPaintToView);
   fInSelectionMode = kFALSE;
}

//______________________________________________________________________________
void Pad::PaintShadowForSelected() const
{
   fInHighlightMode = kTRUE;
   fPainter.SetPainterMode(Painter::kPaintShadow);

   if (fParentOfSelected) {
      fParentOfSelected->Paint(GetSelectedParentDrawOption());
   } else if (fSelected) {
      fSelected->Paint(GetSelectedDrawOption());
   }

   fPainter.SetPainterMode(Painter::kPaintToView);
   fInHighlightMode = kFALSE;
}

//______________________________________________________________________________
void Pad::PaintSelected() const
{
   fInHighlightMode = kTRUE;

   fPainter.SetPainterMode(Painter::kPaintSelected);

   if (fParentOfSelected)
      fParentOfSelected->Paint(GetSelectedParentDrawOption());
   else if (fSelected)
      fSelected->Paint(GetSelectedDrawOption());

   fPainter.SetPainterMode(Painter::kPaintToView);
   fInHighlightMode = kFALSE;
}

//______________________________________________________________________________
void Pad::PaintBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Option_t *option)
{
   // Paint box in CurrentPad World coordinates.
   //
   // if option[0] = 's' the box is forced to be paint with style=0
   // if option[0] = 'l' the box contour is drawn

   Int_t style0 = gVirtualX->GetFillStyle();
   Int_t style  = style0;

   if (option[0] == 's') {
      gVirtualX->SetFillStyle(0);
      style = 0;
   }

   if (style) {
      if (style > 3000 && style < 4000) {
         if (style < 3026) {
            // draw stipples with fFillColor foreground
            fPainter.DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kFilled);
         }

         if (style >= 3100 && style < 4000) {
            Double_t xb[4], yb[4];
            xb[0] = x1; xb[1] = x1; xb[2] = x2; xb[3] = x2;
            yb[0] = y1; yb[1] = y2; yb[2] = y2; yb[3] = y1;
            PaintFillAreaHatches(4, xb, yb, style);
            return;
         }
         //special case for TAttFillCanvas
         if (gVirtualX->GetFillColor() == 10) {
            gVirtualX->SetFillColor(1);
            fPainter.DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kFilled);
            gVirtualX->SetFillColor(10);
         }
      } else if (style >= 4000 && style <= 4100) {
         // For style >=4000 we make the window transparent.
         // From 4000 to 4100 the window is 100% transparent to 100% opaque
         //ignore this style option when this is the canvas itself
         fPainter.DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kFilled);
      } else {
         fPainter.DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kFilled);
      }

      if (option[0] == 'l') fPainter.DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kHollow);
   } else {
      fPainter.DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kHollow);
      if (option[0] == 's') gVirtualX->SetFillStyle(style0);
   }
}

//______________________________________________________________________________
void Pad::PaintFillArea(Int_t, Float_t *, Float_t *, Option_t *)
{
   // Paint fill area in CurrentPad World coordinates.
   throw std::runtime_error("Dummy version for floats");
}

//______________________________________________________________________________
void Pad::PaintFillArea(Int_t nn, Double_t *xx, Double_t *yy, Option_t *)
{
   // Paint fill area in CurrentPad World coordinates.

   if (nn < 3) return;
   Int_t n=0;
   Double_t xmin,xmax,ymin,ymax;
   if (TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }

   Int_t nc = 2*nn+1;

   std::vector<Double_t> x(nc);
   std::vector<Double_t> y(nc);

   n = ClipPolygon(nn, xx, yy, nc, &x[0], &y[0], xmin, ymin, xmax, ymax);
   if (!n)
      return;

   // Paint the fill area with hatches
   Int_t fillstyle = gVirtualX->GetFillStyle();

   if (fillstyle >= 3100 && fillstyle < 4000) {
      PaintFillAreaHatches(nn, &x[0], &y[0], fillstyle);
      return;
   }

   fPainter.DrawFillArea(n, &x[0], &y[0]);
}

//______________________________________________________________________________
void Pad::PaintPadFrame(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax)
{
   //Paint histogram/graph frame.
   //Original TPad has a COMPLETE MESS here, I can
   //not understand, how it was possible to write.
   //Trying to fix it at least to something sane.
   if (!fFrame)
      GetFrame();

   fFrame->SetX1(xmin);
   fFrame->SetX2(xmax);
   fFrame->SetY1(ymin);
   fFrame->SetY2(ymax);

   if (gROOT->GetForceStyle())
      fFrame->UseCurrentStyle();

   if (!GetListOfPrimitives()->FindObject(fFrame)) {
      GetListOfPrimitives()->AddFirst(fFrame);
      //fFrame->SetBit(kMustCleanup);
   }

   fFrame->Paint();
}

//______________________________________________________________________________
void Pad::PaintLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   // Paint line in CurrentPad World coordinates.

   Double_t x[2], y[2];
   x[0] = x1;   x[1] = x2;   y[0] = y1;   y[1] = y2;

   //If line is totally clipped, return
   if (TestBit(TGraph::kClipFrame)) {
      if (Clip(x,y,fUxmin,fUymin,fUxmax,fUymax) == 2) return;
   } else {
      if (Clip(x,y,fX1,fY1,fX2,fY2) == 2) return;
   }

   fPainter.DrawLine(x[0], y[0], x[1], y[1]);
}

//______________________________________________________________________________
void Pad::PaintLineNDC(Double_t u1, Double_t v1,Double_t u2, Double_t v2)
{
   const Double_t xRange = GetX2() - GetX1();
   const Double_t yRange = GetY2() - GetY1();

   fPainter.DrawLine(GetX1() + u1 * xRange, GetY1() + v1 * yRange,
                     GetX1() + u2 * xRange, GetY1() + v2 * yRange);
}

//______________________________________________________________________________
void Pad::PaintLine3D(Float_t *p1, Float_t *p2)
{
   // Paint 3-D line in the CurrentPad.

   if (!fView) return;

   // convert from 3-D to 2-D pad coordinate system
   Double_t xpad[6];
   Double_t temp[3];
   Int_t i;
   for (i=0;i<3;i++) temp[i] = p1[i];
   fView->WCtoNDC(temp, &xpad[0]);
   for (i=0;i<3;i++) temp[i] = p2[i];
   fView->WCtoNDC(temp, &xpad[3]);
   PaintLine(xpad[0],xpad[1],xpad[3],xpad[4]);
}

//______________________________________________________________________________
void Pad::PaintLine3D(Double_t *p1, Double_t *p2)
{
   // Paint 3-D line in the CurrentPad.

   //take into account perspective view
   if (!fView) return;
   // convert from 3-D to 2-D pad coordinate system
   Double_t xpad[6];
   Double_t temp[3];
   Int_t i;
   for (i=0;i<3;i++) temp[i] = p1[i];
   fView->WCtoNDC(temp, &xpad[0]);
   for (i=0;i<3;i++) temp[i] = p2[i];
   fView->WCtoNDC(temp, &xpad[3]);
   PaintLine(xpad[0],xpad[1],xpad[3],xpad[4]);
}


//______________________________________________________________________________
void Pad::PaintPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *)
{
   // Paint polyline in CurrentPad World coordinates.

   if (n < 2) return;

   Double_t xmin,xmax,ymin,ymax;
   if (TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }
   Int_t i, i1=-1,np=1;
   for (i=0; i<n-1; i++) {
      Double_t x1=x[i];
      Double_t y1=y[i];
      Double_t x2=x[i+1];
      Double_t y2=y[i+1];
      Int_t iclip = Clip(&x[i],&y[i],xmin,ymin,xmax,ymax);
      if (iclip == 2) {
         i1 = -1;
         continue;
      }
      np++;
      if (i1 < 0) i1 = i;
      if (iclip == 0 && i < n-2) continue;
      fPainter.DrawPolyLine(np, &x[i1], &y[i1]);
      if (iclip) {
         x[i] = x1;
         y[i] = y1;
         x[i+1] = x2;
         y[i+1] = y2;
      }
      i1 = -1;
      np = 1;
   }
}

//______________________________________________________________________________
void Pad::PaintPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
   // Paint polyline in CurrentPad World coordinates.
   //
   //  If option[0] == 'C' no clipping

   if (n < 2) return;

   Double_t xmin,xmax,ymin,ymax;
   Bool_t mustClip = kTRUE;
   if (TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
      if (option && (option[0] == 'C')) mustClip = kFALSE;
   }

   Int_t i, i1=-1, np=1, iclip=0;

   for (i=0; i < n-1; i++) {
      Double_t x1=x[i];
      Double_t y1=y[i];
      Double_t x2=x[i+1];
      Double_t y2=y[i+1];
      if (mustClip) {
         iclip = Clip(&x[i],&y[i],xmin,ymin,xmax,ymax);
         if (iclip == 2) {
            i1 = -1;
            continue;
         }
      }
      np++;
      if (i1 < 0) i1 = i;
      if (iclip == 0 && i < n-2) continue;

      fPainter.DrawPolyLine(np, &x[i1], &y[i1]);
      if (iclip) {
         x[i] = x1;
         y[i] = y1;
         x[i+1] = x2;
         y[i+1] = y2;
      }
      i1 = -1;
      np = 1;
   }
}

//______________________________________________________________________________
void Pad::PaintPolyLineNDC(Int_t n, Double_t *x, Double_t *y, Option_t *)
{
   if (n <= 0) //Check from original TPad.
      return;

   fPainter.DrawPolyLineNDC(n, x, y);
}

//______________________________________________________________________________
void Pad::PaintPolyLine3D(Int_t n, Double_t *p)
{
   // Paint 3-D polyline in the CurrentPad.
   if (!fView) return;

   // Loop on each individual line
   for (Int_t i = 1; i < n; i++)
      PaintLine3D(&p[3*i-3], &p[3*i]);
}


//______________________________________________________________________________
void Pad::PaintPolyMarker(Int_t nn, Float_t *x, Float_t *y, Option_t *)
{
   // Paint polymarker in CurrentPad World coordinates.

   Int_t n = TMath::Abs(nn);
   Double_t xmin,xmax,ymin,ymax;
   if (nn > 0 || TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }
   Int_t i,i1=-1,np=0;
   for (i=0; i<n; i++) {
      if (x[i] >= xmin && x[i] <= xmax && y[i] >= ymin && y[i] <= ymax) {
         np++;
         if (i1 < 0) i1 = i;
         if (i < n-1) continue;
      }
      if (np == 0) continue;

      fPainter.DrawPolyMarker(np, &x[i1], &y[i1]);

      i1 = -1;
      np = 0;
   }
}


//______________________________________________________________________________
void Pad::PaintPolyMarker(Int_t nn, Double_t *x, Double_t *y, Option_t *)
{
   // Paint polymarker in CurrentPad World coordinates.

   Int_t n = TMath::Abs(nn);
   Double_t xmin,xmax,ymin,ymax;
   if (nn > 0 || TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }
   Int_t i,i1=-1,np=0;
   for (i=0; i<n; i++) {
      if (x[i] >= xmin && x[i] <= xmax && y[i] >= ymin && y[i] <= ymax) {
         np++;
         if (i1 < 0) i1 = i;
         if (i < n-1) continue;
      }
      if (np == 0) continue;

      fPainter.DrawPolyMarker(np, &x[i1], &y[i1]);

      i1 = -1;
      np = 0;
   }
}

//______________________________________________________________________________
void Pad::PaintText(Double_t x, Double_t y, const char *text)
{
   // Paint text in CurrentPad World coordinates.
   fPainter.DrawText(x, y, text, TVirtualPadPainter::kClear);
}

//______________________________________________________________________________
void Pad::PaintText(Double_t, Double_t, const wchar_t *)
{
}

//______________________________________________________________________________
void Pad::PaintTextNDC(Double_t u, Double_t v, const char *text)
{
   // Paint text in CurrentPad NDC coordinates.
   const Double_t xRange = GetX2() - GetX1();
   const Double_t yRange = GetY2() - GetY1();
   fPainter.DrawText(gPad->GetX1() + u * xRange, gPad->GetY1() + v * yRange, text, TVirtualPadPainter::kClear);
}

//______________________________________________________________________________
void PaintTextNDC(Double_t, Double_t, const wchar_t *)
{
}


//______________________________________________________________________________
TH1F *Pad::DrawFrame(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax, const char *title)
{
   //  Draw a pad frame
   //
   //  Compute real pad range taking into account all margins
   //  Use services of TH1F class
   Pad *padsav = (Pad*)gPad;
   if (this !=  padsav) {
      Warning("DrawFrame","Drawframe must be called for the current pad only");
      return padsav->DrawFrame(xmin,ymin,xmax,ymax,title);
   }

   cd();

   TH1F *hframe = (TH1F*)FindObject("hframe");
   if (hframe) delete hframe;
   Int_t nbins = 1000;
   //if log scale in X, use variable bin size linear with log(x)
   //this gives a better precision when zooming on the axis
   if (fLogx && xmin > 0 && xmax > xmin) {
      Double_t xminl = TMath::Log(xmin);
      Double_t xmaxl = TMath::Log(xmax);
      Double_t dx = (xmaxl-xminl)/nbins;
      Double_t *xbins = new Double_t[nbins+1];
      xbins[0] = xmin;
      for (Int_t i=1;i<=nbins;i++) {
         xbins[i] = TMath::Exp(xminl+i*dx);
      }
      hframe = new TH1F("hframe",title,nbins,xbins);
      delete [] xbins;
   } else {
      hframe = new TH1F("hframe",title,nbins,xmin,xmax);
   }
   hframe->SetBit(TH1::kNoStats);
   hframe->SetBit(kCanDelete);
   hframe->SetMinimum(ymin);
   hframe->SetMaximum(ymax);
   hframe->GetYaxis()->SetLimits(ymin,ymax);
   hframe->SetDirectory(0);
   hframe->Draw(" ");

   Update();
   if (padsav) padsav->cd();
   return hframe;
}

//______________________________________________________________________________
void Pad::RedrawAxis(Option_t *option)
{
   //  Redraw the frame axis
   //  Redrawing axis may be necessary in case of superimposed histograms
   //  when one or more histograms have a fill color
   //  Instead of calling this function, it may be more convenient
   //  to call directly h1->Draw("sameaxis") where h1 is the pointer
   //  to the first histogram drawn in the pad.
   //
   //  By default, if the pad has the options gridx or/and gridy activated,
   //  the grid is not drawn by this function.
   //  if option="g" is specified, this will force the drawing of the grid
   //  on top of the picture

   // get first histogram in the list of primitives
   TString opt = option;
   opt.ToLower();


   cd();

   TIter next(&fPrimitives);
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom(TH1::Class())) {
         TH1 *hobj = (TH1*)obj;
         if (opt.Contains("g")) hobj->DrawCopy("sameaxig");
         else                   hobj->DrawCopy("sameaxis");
         return;
      }
      if (obj->InheritsFrom(TMultiGraph::Class())) {
         TMultiGraph *mg = (TMultiGraph*)obj;
         if (mg) mg->GetHistogram()->DrawCopy("sameaxis");
         return;
      }
      if (obj->InheritsFrom(TGraph::Class())) {
         TGraph *g = (TGraph*)obj;
         if (g) g->GetHistogram()->DrawCopy("sameaxis");
         return;
      }
      if (obj->InheritsFrom(THStack::Class())) {
         THStack *hs = (THStack*)obj;
         if (hs) hs->GetHistogram()->DrawCopy("sameaxis");
         return;
      }
   }
}

//______________________________________________________________________________
void Pad::GetTextExtent(UInt_t &w, UInt_t &h, const char *text)
{
   fPainter.GetTextExtent(w, h, text);
}

//______________________________________________________________________________
void Pad::SetContext(CGContextRef ctx)
{
   fPainter.SetContext(ctx);
}

//______________________________________________________________________________
const char *Pad::GetSelectedParentDrawOption()const
{
   //Linear search :) But list is very short.
   TObjOptLink *lnk = (TObjOptLink*)GetListOfPrimitives()->FirstLink();
   while (lnk) {
      TObject *obj = lnk->GetObject();
      if (obj == fParentOfSelected)
         return lnk->GetOption();
      lnk = (TObjOptLink*)lnk->Next();
   }

   return "";
}

//______________________________________________________________________________
const char *Pad::GetSelectedDrawOption()const
{
   //Linear search :) But list is very short.
   TObjOptLink *lnk = (TObjOptLink*)GetListOfPrimitives()->FirstLink();
   while (lnk) {
      TObject *obj = lnk->GetObject();
      if (obj == fSelected)
         return lnk->GetOption();
      lnk = (TObjOptLink*)lnk->Next();
   }

   return "";
}

//______________________________________________________________________________
TObject *Pad::FindObject(const char *name) const
{
   return fPrimitives.FindObject(name);
}


//______________________________________________________________________________
TObject *Pad::FindObject(const TObject *obj) const
{
   return fPrimitives.FindObject(obj);
}

//___________________________________________________________
Int_t Pad::Clip(Float_t *x, Float_t *y, Float_t xclipl, Float_t yclipb, Float_t xclipr, Float_t yclipt)
{
   // Clipping routine: Cohen Sutherland algorithm.
   //
   //   If Clip ==2 the segment is outside the boundary.
   //   If Clip ==1 the segment has one point outside the boundary.
   //   If Clip ==0 the segment is inside the boundary.
   //
   // _Input parameters:
   //
   //  x[2], y[2] : Segment coordinates
   //  xclipl, yclipb, xclipr, yclipt : Clipping boundary
   //
   // _Output parameters:
   //
   //  x[2], y[2] : New segment coordinates

   const Float_t kP=10000;
   Int_t clip = 0;

   for (Int_t i=0;i<2;i++) {
      if (TMath::Abs(xclipl-x[i]) <= TMath::Abs(xclipr-xclipl)/kP) x[i] = xclipl;
      if (TMath::Abs(xclipr-x[i]) <= TMath::Abs(xclipr-xclipl)/kP) x[i] = xclipr;
      if (TMath::Abs(yclipb-y[i]) <= TMath::Abs(yclipt-yclipb)/kP) y[i] = yclipb;
      if (TMath::Abs(yclipt-y[i]) <= TMath::Abs(yclipt-yclipb)/kP) y[i] = yclipt;
   }

   // Compute the first endpoint codes.
   Int_t code1 = ClippingCode(x[0],y[0],xclipl,yclipb,xclipr,yclipt);
   Int_t code2 = ClippingCode(x[1],y[1],xclipl,yclipb,xclipr,yclipt);

   Double_t xt=0, yt=0;
   Int_t clipped = 0; //this variable could be used in a future version
   while(code1 + code2) {
      clipped = 1;

      // The line lies entirely outside the clipping boundary
      if (code1&code2) {
         clip = 2;
         return clip;
      }

      // The line is subdivided into several parts
      Int_t ic = code1;
      if (ic == 0) ic = code2;
      if (ic & 0x1) {
         yt = y[0] + (y[1]-y[0])*(xclipl-x[0])/(x[1]-x[0]);
         xt = xclipl;
      }
      if (ic & 0x2) {
         yt = y[0] + (y[1]-y[0])*(xclipr-x[0])/(x[1]-x[0]);
         xt = xclipr;
      }
      if (ic & 0x4) {
         xt = x[0] + (x[1]-x[0])*(yclipb-y[0])/(y[1]-y[0]);
         yt = yclipb;
      }
      if (ic & 0x8) {
         xt = x[0] + (x[1]-x[0])*(yclipt-y[0])/(y[1]-y[0]);
         yt = yclipt;
      }
      if (ic == code1) {
         x[0]  = xt;
         y[0]  = yt;
         code1 = ClippingCode(xt,yt,xclipl,yclipb,xclipr,yclipt);
      } else {
         x[1]  = xt;
         y[1]  = yt;
         code2 = ClippingCode(xt,yt,xclipl,yclipb,xclipr,yclipt);
      }
   }
   clip = clipped;
   return clip;
}


//___________________________________________________________
Int_t Pad::Clip(Double_t *x, Double_t *y, Double_t xclipl, Double_t yclipb, Double_t xclipr, Double_t yclipt)
{
   // Clipping routine: Cohen Sutherland algorithm.
   //
   //   If Clip ==2 the segment is outside the boundary.
   //   If Clip ==1 the segment has one point outside the boundary.
   //   If Clip ==0 the segment is inside the boundary.
   //
   // _Input parameters:
   //
   //  x[2], y[2] : Segment coordinates
   //  xclipl, yclipb, xclipr, yclipt : Clipping boundary
   //
   // _Output parameters:
   //
   //  x[2], y[2] : New segment coordinates

   const Double_t kP=10000;
   Int_t clip = 0;

   for (Int_t i=0;i<2;i++) {
      if (TMath::Abs(xclipl-x[i]) <= TMath::Abs(xclipr-xclipl)/kP) x[i] = xclipl;
      if (TMath::Abs(xclipr-x[i]) <= TMath::Abs(xclipr-xclipl)/kP) x[i] = xclipr;
      if (TMath::Abs(yclipb-y[i]) <= TMath::Abs(yclipt-yclipb)/kP) y[i] = yclipb;
      if (TMath::Abs(yclipt-y[i]) <= TMath::Abs(yclipt-yclipb)/kP) y[i] = yclipt;
   }

   // Compute the first endpoint codes.
   Int_t code1 = 0;
   if (x[0] < xclipl) code1 = code1 | 0x1;
   if (x[0] > xclipr) code1 = code1 | 0x2;
   if (y[0] < yclipb) code1 = code1 | 0x4;
   if (y[0] > yclipt) code1 = code1 | 0x8;
   Int_t code2 = 0;
   if (x[1] < xclipl) code2 = code2 | 0x1;
   if (x[1] > xclipr) code2 = code2 | 0x2;
   if (y[1] < yclipb) code2 = code2 | 0x4;
   if (y[1] > yclipt) code2 = code2 | 0x8;

   Double_t xt=0, yt=0;
   Int_t clipped = 0; //this variable could be used in a future version
   while(code1 + code2) {
      clipped = 1;

      // The line lies entirely outside the clipping boundary
      if (code1&code2) {
         clip = 2;
         return clip;
      }

      // The line is subdivided into several parts
      Int_t ic = code1;
      if (ic == 0) ic = code2;
      if (ic & 0x1) {
         yt = y[0] + (y[1]-y[0])*(xclipl-x[0])/(x[1]-x[0]);
         xt = xclipl;
      }
      if (ic & 0x2) {
         yt = y[0] + (y[1]-y[0])*(xclipr-x[0])/(x[1]-x[0]);
         xt = xclipr;
      }
      if (ic & 0x4) {
         xt = x[0] + (x[1]-x[0])*(yclipb-y[0])/(y[1]-y[0]);
         yt = yclipb;
      }
      if (ic & 0x8) {
         xt = x[0] + (x[1]-x[0])*(yclipt-y[0])/(y[1]-y[0]);
         yt = yclipt;
      }
      if (ic == code1) {
         x[0]  = xt;
         y[0]  = yt;
         code1 = ClippingCode(xt,yt,xclipl,yclipb,xclipr,yclipt);
      } else {
         x[1]  = xt;
         y[1]  = yt;
         code2 = ClippingCode(xt,yt,xclipl,yclipb,xclipr,yclipt);
      }
   }
   clip = clipped;
   return clip;
}


//___________________________________________________________
Int_t Pad::ClippingCode(Double_t x, Double_t y, Double_t xcl1, Double_t ycl1, Double_t xcl2, Double_t ycl2)
{
   // Compute the endpoint codes for TPad::Clip.

   Int_t code = 0;
   if (x < xcl1) code = code | 0x1;
   if (x > xcl2) code = code | 0x2;
   if (y < ycl1) code = code | 0x4;
   if (y > ycl2) code = code | 0x8;
   return code;
}


//___________________________________________________________
Int_t Pad::ClipPolygon(Int_t n, Double_t *x, Double_t *y, Int_t nn, Double_t *xc, Double_t *yc, Double_t xclipl, Double_t yclipb, Double_t xclipr, Double_t yclipt)
{
   // Clip polygon using the Sutherland-Hodgman algorithm.
   //
   // Input parameters:
   //
   //  n: Number of points in the polygon to be clipped
   //  x[n], y[n] : Polygon do be clipped vertices
   //  xclipl, yclipb, xclipr, yclipt : Clipping boundary
   //
   // Output parameters:
   //
   // nn: number of points in xc and yc
   // xc, yc: clipped polygon vertices. The Int_t returned by this function is
   //         the number of points in the clipped polygon. These vectors must
   //         be allocated by the calling function. A size of 2*n for each is
   //         enough.
   //
   // Sutherland and Hodgman's polygon-clipping algorithm uses a divide-and-conquer
   // strategy: It solves a series of simple and identical problems that, when
   // combined, solve the overall problem. The simple problem is to clip a polygon
   // against a single infinite clip edge. Four clip edges, each defining one boundary
   // of the clip rectangle, successively clip a polygon against a clip rectangle.
   //
   // Steps of Sutherland-Hodgman's polygon-clipping algorithm:
   //
   // * Polygons can be clipped against each edge of the window one at a time.
   //   Windows/edge intersections, if any, are easy to find since the X or Y coordinates
   //   are already known.
   // * Vertices which are kept after clipping against one window edge are saved for
   //   clipping against the remaining edges.
   // * Note that the number of vertices usually changes and will often increases.
   //
   // The clip boundary determines a visible and invisible region. The edges from
   // vertex i to vertex i+1 can be one of four types:
   //
   // * Case 1 : Wholly inside visible region - save endpoint
   // * Case 2 : Exit visible region - save the intersection
   // * Case 3 : Wholly outside visible region - save nothing
   // * Case 4 : Enter visible region - save intersection and endpoint

   Int_t nc, nc2;
   Double_t x1, y1, x2, y2, slope; // Segment to be clipped

   std::vector<Double_t> xc2(nn);
   std::vector<Double_t> yc2(nn);

   // Clip against the left boundary
   x1 = x[n-1]; y1 = y[n-1];
   nc2 = 0;
   Int_t i;
   for (i=0; i<n; i++) {
      x2 = x[i]; y2 = y[i];
      if (x1 == x2) {
         slope = 0;
      } else {
         slope = (y2-y1)/(x2-x1);
      }
      if (x1 >= xclipl) {
         if (x2 < xclipl) {
            xc2[nc2] = xclipl; yc2[nc2++] = slope*(xclipl-x1)+y1;
         } else {
            xc2[nc2] = x2; yc2[nc2++] = y2;
         }
      } else {
         if (x2 >= xclipl) {
            xc2[nc2] = xclipl; yc2[nc2++] = slope*(xclipl-x1)+y1;
            xc2[nc2] = x2; yc2[nc2++] = y2;
         }
      }
      x1 = x2; y1 = y2;
   }

   // Clip against the top boundary
   x1 = xc2[nc2-1]; y1 = yc2[nc2-1];
   nc = 0;
   for (i=0; i<nc2; i++) {
      x2 = xc2[i]; y2 = yc2[i];
      if (y1 == y2) {
         slope = 0;
      } else {
         slope = (x2-x1)/(y2-y1);
      }
      if (y1 <= yclipt) {
         if (y2 > yclipt) {
            xc[nc] = x1+(yclipt-y1)*slope; yc[nc++] = yclipt;
         } else {
            xc[nc] = x2; yc[nc++] = y2;
         }
      } else {
         if (y2 <= yclipt) {
            xc[nc] = x1+(yclipt-y1)*slope; yc[nc++] = yclipt;
            xc[nc] = x2; yc[nc++] = y2;
         }
      }
      x1 = x2; y1 = y2;
   }

   // Clip against the right boundary
   x1 = xc[nc-1]; y1 = yc[nc-1];
   nc2 = 0;
   for (i=0; i<nc; i++) {
      x2 = xc[i]; y2 = yc[i];
      if (x1 == x2) {
         slope = 0;
      } else {
         slope = (y2-y1)/(x2-x1);
      }
      if (x1 <= xclipr) {
         if (x2 > xclipr) {
            xc2[nc2] = xclipr; yc2[nc2++] = slope*(xclipr-x1)+y1;
         } else {
            xc2[nc2] = x2; yc2[nc2++] = y2;
         }
      } else {
         if (x2 <= xclipr) {
            xc2[nc2] = xclipr; yc2[nc2++] = slope*(xclipr-x1)+y1;
            xc2[nc2] = x2; yc2[nc2++] = y2;
         }
      }
      x1 = x2; y1 = y2;
   }

   // Clip against the bottom boundary
   x1 = xc2[nc2-1]; y1 = yc2[nc2-1];
   nc = 0;
   for (i=0; i<nc2; i++) {
      x2 = xc2[i]; y2 = yc2[i];
      if (y1 == y2) {
         slope = 0;
      } else {
         slope = (x2-x1)/(y2-y1);
      }
      if (y1 >= yclipb) {
         if (y2 < yclipb) {
            xc[nc] = x1+(yclipb-y1)*slope; yc[nc++] = yclipb;
         } else {
            xc[nc] = x2; yc[nc++] = y2;
         }
      } else {
         if (y2 >= yclipb) {
            xc[nc] = x1+(yclipb-y1)*slope; yc[nc++] = yclipb;
            xc[nc] = x2; yc[nc++] = y2;
         }
      }
      x1 = x2; y1 = y2;
   }

   if (nc < 3) nc =0;
   return nc;
}

//______________________________________________________________________________
void Pad::PaintBorder(Color_t color, Bool_t tops)
{
   // Paint the pad border.
   // Draw first  a box as a normal filled box
   if(color >= 0) {
      TAttLine::Modify();  //Change line attributes only if necessary
      TAttFill::Modify();  //Change fill area attributes only if necessary

      PaintBox(fX1,fY1,fX2,fY2);
   }

   if (color < 0)
      color = -color;
   // then paint 3d frame (depending on bordermode)
   if (IsTransparent())
      return;

   // Paint a 3D frame around the pad.
   if (!fBorderMode)
      return;

   Int_t bordersize = fBorderSize;
   if (bordersize <= 0)
      bordersize = 2;

   const Double_t realBsX = bordersize / (GetAbsWNDC() * GetWw()) * (fX2 - fX1);
   const Double_t realBsY = bordersize / (GetAbsHNDC() * GetWh()) * (fY2 - fY1);

   Short_t px1,py1,px2,py2;
   Double_t xl, xt, yl, yt;

   // GetDarkColor() and GetLightColor() use GetFillColor()
   Color_t oldcolor = GetFillColor();
   SetFillColor(color);
   TAttFill::Modify();
   Color_t light = 0, dark = 0;
   if (color != 0) {
      light = TColor::GetColorBright(color);
      dark  = TColor::GetColorDark(color);
   }

   // Compute real left bottom & top right of the box in pixels
   px1 = XtoPixel(fX1);   py1 = YtoPixel(fY1);
   px2 = XtoPixel(fX2);   py2 = YtoPixel(fY2);
   if (px1 < px2) {xl = fX1; xt = fX2; }
   else           {xl = fX2; xt = fX1;}
   if (py1 > py2) {yl = fY1; yt = fY2;}
   else           {yl = fY2; yt = fY1;}

   Double_t frameXs[7] = {}, frameYs[7] = {};

   // Draw top&left part of the box
   frameXs[0] = xl;           frameYs[0] = yl;
   frameXs[1] = xl + realBsX; frameYs[1] = yl + realBsY;
   frameXs[2] = frameXs[1];   frameYs[2] = yt - realBsY;
   frameXs[3] = xt - realBsX; frameYs[3] = frameYs[2];
   frameXs[4] = xt;           frameYs[4] = yt;
   frameXs[5] = xl;           frameYs[5] = yt;
   frameXs[6] = xl;           frameYs[6] = yl;

   if (fBorderMode == -1) gVirtualX->SetFillColor(dark);
   else                   gVirtualX->SetFillColor(light);
   fPainter.DrawFillArea(7, frameXs, frameYs);

   // Draw bottom&right part of the box
   frameXs[0] = xl;              frameYs[0] = yl;
   frameXs[1] = xl + realBsX;    frameYs[1] = yl + realBsY;
   frameXs[2] = xt - realBsX;    frameYs[2] = frameYs[1];
   frameXs[3] = frameXs[2];      frameYs[3] = yt - realBsY;
   frameXs[4] = xt;              frameYs[4] = yt;
   frameXs[5] = xt;              frameYs[5] = yl;
   frameXs[6] = xl;              frameYs[6] = yl;

   if (fBorderMode == -1) gVirtualX->SetFillColor(light);
   else                   gVirtualX->SetFillColor(dark);
   fPainter.DrawFillArea(7, frameXs, frameYs);

   gVirtualX->SetFillColor(-1);
   SetFillColor(oldcolor);

   if (!tops)
      return;
}

//______________________________________________________________________________
void Pad::PaintFillAreaHatches(Int_t nn, Double_t *xx, Double_t *yy, Int_t FillStyle)
{
   //   This function paints hatched fill area arcording to the FillStyle value
   // The convention for the Hatch is the following:
   //
   //            FillStyle = 3ijk
   //
   //    i (1-9) : specify the space between each hatch
   //              1 = minimum  9 = maximum
   //              the final spacing is i*GetHatchesSpacing(). The hatches spacing
   //              is set by SetHatchesSpacing()
   //
   //    j (0-9) : specify angle between 0 and 90 degrees
   //
   //              0 = 0
   //              1 = 10
   //              2 = 20
   //              3 = 30
   //              4 = 45
   //              5 = Not drawn
   //              6 = 60
   //              7 = 70
   //              8 = 80
   //              9 = 90
   //
   //    k (0-9) : specify angle between 90 and 180 degrees
   //              0 = 180
   //              1 = 170
   //              2 = 160
   //              3 = 150
   //              4 = 135
   //              5 = Not drawn
   //              6 = 120
   //              7 = 110
   //              8 = 100
   //              9 = 90

   static Double_t ang1[10] = {0., 10., 20., 30., 45.,5., 60., 70., 80., 90.};
   static Double_t ang2[10] = {180.,170.,160.,150.,135.,5.,120.,110.,100., 90.};

   Int_t fasi  = FillStyle%1000;
   Int_t idSPA = (Int_t)(fasi/100);
   Int_t iAng2 = (Int_t)((fasi-100*idSPA)/10);
   Int_t iAng1 = fasi%10;
   Double_t dy = 0.003*(Double_t)(idSPA)*gStyle->GetHatchesSpacing();
   Int_t lw = gStyle->GetHatchesLineWidth();
   Short_t lws = 0;
   Int_t   lss = 0;
   Int_t   lcs = 0;

   // Save the current line attributes

   lws = gVirtualX->GetLineWidth();
   lss = gVirtualX->GetLineStyle();
   lcs = gVirtualX->GetLineColor();

   // Change the current line attributes to draw the hatches
   gVirtualX->SetLineStyle(1);
   gVirtualX->SetLineWidth(Short_t(lw));
   gVirtualX->SetLineColor(gVirtualX->GetFillColor());

   // Draw the hatches
   if (ang1[iAng1] != 5.) PaintHatches(dy, ang1[iAng1], nn, xx, yy);
   if (ang2[iAng2] != 5.) PaintHatches(dy, ang2[iAng2], nn, xx, yy);

   // Restore the line attributes
   gVirtualX->SetLineStyle(lss);
   gVirtualX->SetLineWidth(lws);
   gVirtualX->SetLineColor(lcs);
}

//______________________________________________________________________________
void Pad::PaintHatches(Double_t dy, Double_t angle,
                        Int_t nn, Double_t *xx, Double_t *yy)
{
   // This routine draw hatches inclined with the
   // angle "angle" and spaced of "dy" in normalized device
   // coordinates in the surface defined by n,xx,yy.

   Int_t i, i1, i2, nbi, m, inv;
   Double_t ratiox, ratioy, ymin, ymax, yrot, ycur;
   const Double_t angr  = TMath::Pi()*(180-angle)/180.;
   const Double_t epsil = 0.0001;
   const Int_t maxnbi = 100;
   Double_t xli[maxnbi], xlh[2], ylh[2], xt1, xt2, yt1, yt2;
   Double_t ll, x, y, x1, x2, y1, y2, a, b, xi, xip, xin, yi, yip;

   Double_t rwxmin = gPad->GetX1();
   Double_t rwxmax = gPad->GetX2();
   Double_t rwymin = gPad->GetY1();
   Double_t rwymax = gPad->GetY2();
   ratiox = 1/(rwxmax-rwxmin);
   ratioy = 1/(rwymax-rwymin);

   Double_t sina = TMath::Sin(angr), sinb;
   Double_t cosa = TMath::Cos(angr), cosb;
   if (TMath::Abs(cosa) <= epsil) cosa=0.;
   if (TMath::Abs(sina) <= epsil) sina=0.;
   sinb = -sina;
   cosb = cosa;

   // Values needed to compute the hatches in TRUE normalized space (NDC)
   Int_t iw = gPad->GetWw();
   Int_t ih = gPad->GetWh();
   Double_t x1p,y1p,x2p,y2p;
   gPad->GetPadPar(x1p,y1p,x2p,y2p);
   iw  = (Int_t)(iw*x2p)-(Int_t)(iw*x1p);
   ih  = (Int_t)(ih*y2p)-(Int_t)(ih*y1p);
   Double_t wndc  = TMath::Min(1.,(Double_t)iw/(Double_t)ih);
   Double_t hndc  = TMath::Min(1.,(Double_t)ih/(Double_t)iw);

   // Search ymin and ymax
   ymin = 1.;
   ymax = 0.;
   for (i=1; i<=nn; i++) {
      x    = wndc*ratiox*(xx[i-1]-rwxmin);
      y    = hndc*ratioy*(yy[i-1]-rwymin);
      yrot = sina*x+cosa*y;
      if (yrot > ymax) ymax = yrot;
      if (yrot < ymin) ymin = yrot;
   }
   ymax = (Double_t)((Int_t)(ymax/dy))*dy;

   for (ycur=ymax; ycur>=ymin; ycur=ycur-dy) {
      nbi = 0;
      for (i=2; i<=nn+1; i++) {
         i2 = i;
         i1 = i-1;
         if (i == nn+1) i2=1;
         x1  = wndc*ratiox*(xx[i1-1]-rwxmin);
         y1  = hndc*ratioy*(yy[i1-1]-rwymin);
         x2  = wndc*ratiox*(xx[i2-1]-rwxmin);
         y2  = hndc*ratioy*(yy[i2-1]-rwymin);
         xt1 = cosa*x1-sina*y1;
         yt1 = sina*x1+cosa*y1;
         xt2 = cosa*x2-sina*y2;
         yt2 = sina*x2+cosa*y2;

         // Line segment parallel to oy
         if (xt1 == xt2) {
            if (yt1 < yt2) {
               yi  = yt1;
               yip = yt2;
            } else {
               yi  = yt2;
               yip = yt1;
            }
            if ((yi <= ycur) && (ycur < yip)) {
               nbi++;
               if (nbi >= maxnbi) return;
               xli[nbi-1] = xt1;
            }
            continue;
         }

         // Line segment parallel to ox
         if (yt1 == yt2) {
            if (yt1 == ycur) {
               nbi++;
               if (nbi >= maxnbi) return;
               xli[nbi-1] = xt1;
               nbi++;
               if (nbi >= maxnbi) return;
               xli[nbi-1] = xt2;
            }
            continue;
         }

         // Other line segment
         a = (yt1-yt2)/(xt1-xt2);
         b = (yt2*xt1-xt2*yt1)/(xt1-xt2);
         if (xt1 < xt2) {
            xi  = xt1;
            xip = xt2;
         } else {
            xi  = xt2;
            xip = xt1;
         }
         xin = (ycur-b)/a;
         if  ((xi <= xin) && (xin < xip) &&
              (TMath::Min(yt1,yt2) <= ycur) &&
              (ycur < TMath::Max(yt1,yt2))) {
            nbi++;
            if (nbi >= maxnbi) return;
            xli[nbi-1] = xin;
         }
      }

      // Sorting of the x coordinates intersections
      inv = 0;
      m   = nbi-1;
L30:
      for (i=1; i<=m; i++) {
         if (xli[i] < xli[i-1]) {
            inv++;
            ll       = xli[i-1];
            xli[i-1] = xli[i];
            xli[i]   = ll;
         }
      }
      m--;
      if (inv == 0) goto L50;
      inv = 0;
      goto L30;

      // Draw the hatches
L50:
      if (nbi%2 != 0) continue;

      for (i=1; i<=nbi; i=i+2) {
         // Rotate back the hatches
         xlh[0] = cosb*xli[i-1]-sinb*ycur;
         ylh[0] = sinb*xli[i-1]+cosb*ycur;
         xlh[1] = cosb*xli[i]  -sinb*ycur;
         ylh[1] = sinb*xli[i]  +cosb*ycur;
         // Convert hatches' positions from true NDC to WC
         xlh[0] = (xlh[0]/wndc)*(rwxmax-rwxmin)+rwxmin;
         ylh[0] = (ylh[0]/hndc)*(rwymax-rwymin)+rwymin;
         xlh[1] = (xlh[1]/wndc)*(rwxmax-rwxmin)+rwxmin;
         ylh[1] = (ylh[1]/hndc)*(rwymax-rwymin)+rwymin;
         gPad->PaintLine(xlh[0], ylh[0], xlh[1], ylh[1]);
      }
   }
}

//______________________________________________________________________________
TVirtualViewer3D *Pad::GetViewer3D(Option_t *)
{
   if (!fViewer3D)
      fViewer3D = new TViewer3DPad(*this);
   return fViewer3D;
}

//______________________________________________________________________________
void Pad::ExecuteRotateView(Int_t evType, Int_t px, Int_t py)
{
   fView->ExecuteRotateView(evType, px, py);
}

//______________________________________________________________________________
void Pad::ExecuteEventAxis(Int_t event, Int_t px, Int_t py, TAxis *axis)
{
   //This copy is a copy&paste&cut_half_of_guts from original TPad.
   //I do not understand at the moment, why this code is here and not in TAxis.

   static Int_t axisNumber;
   static Double_t ratio1, ratio2;
   static Int_t px1old, py1old, px2old, py2old;
   Int_t bin1, bin2;
   Double_t temp, xmin,xmax;

   switch (event) {

   case kButton1Down:
      axisNumber = 1;
      if (!strcmp(axis->GetName(),"xaxis")) {
         axisNumber = 1;
         if (!IsVertical()) axisNumber = 2;
      }
      if (!strcmp(axis->GetName(),"yaxis")) {
         axisNumber = 2;
         if (!IsVertical()) axisNumber = 1;
      }
      if (!strcmp(axis->GetName(),"zaxis")) {
         axisNumber = 3;
      }
         if (axisNumber == 1) {
            ratio1 = (AbsPixeltoX(px) - GetUxmin())/(GetUxmax() - GetUxmin());
            px1old = XtoAbsPixel(GetUxmin()+ratio1*(GetUxmax() - GetUxmin()));
            py1old = YtoAbsPixel(GetUymin());
            px2old = px1old;
            py2old = YtoAbsPixel(GetUymax());
         } else if (axisNumber == 2) {
            ratio1 = (AbsPixeltoY(py) - GetUymin())/(GetUymax() - GetUymin());
            py1old = YtoAbsPixel(GetUymin()+ratio1*(GetUymax() - GetUymin()));
            px1old = XtoAbsPixel(GetUxmin());
            px2old = XtoAbsPixel(GetUxmax());
            py2old = py1old;
         } else {
            ratio1 = (AbsPixeltoY(py) - GetUymin())/(GetUymax() - GetUymin());
            py1old = YtoAbsPixel(GetUymin()+ratio1*(GetUymax() - GetUymin()));
            px1old = XtoAbsPixel(GetUxmax());
            px2old = XtoAbsPixel(GetX2());
            py2old = py1old;
         }
      // No break !!!

   case kButton1Motion:
         if (axisNumber == 1) {
            ratio2 = (AbsPixeltoX(px) - GetUxmin())/(GetUxmax() - GetUxmin());
            px2old = XtoAbsPixel(GetUxmin()+ratio2*(GetUxmax() - GetUxmin()));
         } else {
            ratio2 = (AbsPixeltoY(py) - GetUymin())/(GetUymax() - GetUymin());
            py2old = YtoAbsPixel(GetUymin()+ratio2*(GetUymax() - GetUymin()));
         }
   break;

   case kButton1Up:

      if (1) {
         if (axisNumber == 1) {
            ratio2 = (AbsPixeltoX(px) - GetUxmin())/(GetUxmax() - GetUxmin());
            xmin = GetUxmin() +ratio1*(GetUxmax() - GetUxmin());
            xmax = GetUxmin() +ratio2*(GetUxmax() - GetUxmin());
         } else if (axisNumber == 2) {
            ratio2 = (AbsPixeltoY(py) - GetUymin())/(GetUymax() - GetUymin());
            xmin = GetUymin() +ratio1*(GetUymax() - GetUymin());
            xmax = GetUymin() +ratio2*(GetUymax() - GetUymin());
         } else {
            ratio2 = (AbsPixeltoY(py) - GetUymin())/(GetUymax() - GetUymin());
            xmin = ratio1;
            xmax = ratio2;
         }
         if (xmin > xmax) {
            temp   = xmin;
            xmin   = xmax;
            xmax   = temp;
            temp   = ratio1;
            ratio1 = ratio2;
            ratio2 = temp;
         }

         // xmin and xmax need to be adjusted in case of CONT4.

         if (!strcmp(axis->GetName(),"xaxis")) axisNumber = 1;
         if (!strcmp(axis->GetName(),"yaxis")) axisNumber = 2;
         if (ratio2 - ratio1 > 0.05) {
            //update object owning this axis
            TH1 *hobj1 = (TH1*)axis->GetParent();
            bin1 = axis->FindFixBin(xmin);
            bin2 = axis->FindFixBin(xmax);
            if (axisNumber == 1) axis->SetRange(bin1,bin2);
            if (axisNumber == 2 && hobj1) {
               if (hobj1->GetDimension() == 1) {
                  if (hobj1->GetNormFactor() != 0) {
                     Double_t norm = hobj1->GetSumOfWeights()/hobj1->GetNormFactor();
                     xmin *= norm;
                     xmax *= norm;
                  }
                  hobj1->SetMinimum(xmin);
                  hobj1->SetMaximum(xmax);
                  hobj1->SetBit(TH1::kIsZoomed);
               } else {
                  axis->SetRange(bin1,bin2);
               }
            }
         }
      }
      break;
   }
}

//______________________________________________________________________________
Bool_t Pad::SelectionIsValid() const
{
   return fSelectionIsValid;
}

//______________________________________________________________________________
void Pad::InvalidateSelection(Bool_t invalidateBufferOnly)
{
   fSelectionIsValid = kFALSE;
   if (!invalidateBufferOnly) {
      fSelected = 0;
      fParentOfSelected = 0;
   }
}

//______________________________________________________________________________
void Pad::SetSelectionBuffer(UInt_t w, UInt_t h, unsigned char *buff)
{
   fSelectionAreaWidth = w;
   fSelectionIsValid = kTRUE;
   fSelectionBuffer.assign(buff, buff + w * h * 4);
}

//______________________________________________________________________________
void Pad::Pick(Int_t px, Int_t py)
{
   if (fContains3DObject) {/* && fView) {
      const Double_t dx = 0.05 * (fUxmax - fUxmin);
      if ((AbsPixeltoX(px) > fUxmin + dx) && (AbsPixeltoX(px) < fUxmax - dx)) {
         TObjOptLink *lnk = (TObjOptLink*)GetListOfPrimitives()->FirstLink();
         while (lnk) {
            TObject *obj = lnk->GetObject();
            if (obj->InheritsFrom("TF2")) {//The only 3D object we can have now.
               fSelected = obj;
               fParentOfSelected = obj;
               return;
            }
         }
      } else {
         fSelected = 0;
         fParentOfSelected = 0;
         return;
      }*/
      fSelected = 0;
      fParentOfSelected = 0;
      return;
   }

   const UInt_t offset = (py * fSelectionAreaWidth + px) * 4;
   const unsigned red = fSelectionBuffer[offset + 1];
   const unsigned green = fSelectionBuffer[offset + 2];
   const unsigned blue = fSelectionBuffer[offset + 3];

   GraphicUtils::IDEncoder enc(10, 255);
   const UInt_t id = enc.ColorToId(red, green, blue);
   if (id > 0 && id <= fSelectables.size()) {
      const ObjectPair_t &found = fSelectables[id - 1];
      //fSelected = fSelectables[id];
      fSelected = found.first;
      fParentOfSelected = found.second;
   } else {
      fSelected = 0;
      fParentOfSelected = 0;
   }
}

//______________________________________________________________________________
void Pad::Unpick()
{
   fSelected = 0;
   fParentOfSelected = 0;
}

//______________________________________________________________________________
TObject *Pad::ObjectInPoint(Int_t px, Int_t py)
{
   const UInt_t offset = (py * fSelectionAreaWidth + px) * 4;
   const unsigned red = fSelectionBuffer[offset + 1];
   const unsigned green = fSelectionBuffer[offset + 2];
   const unsigned blue = fSelectionBuffer[offset + 3];

   GraphicUtils::IDEncoder enc(10, 255);
   const UInt_t id = enc.ColorToId(red, green, blue);
   if (id > 0 && id <= fSelectables.size()) {
      const ObjectPair_t &found = fSelectables[id - 1];
      return found.first;
   }

   return 0;
}

//______________________________________________________________________________
void Pad::PushTopLevelSelectable(TObject *top)
{
   if (!fInSelectionMode)
      return;
   Parent_t newTopLevel(top, fObjectID);
   ObjectPair_t newPair(top, top);
   fPainter.SetCurrentObjectID(fObjectID);
   fObjectID++;

   fParentPainters.push_back(newTopLevel);
   fSelectables.push_back(newPair);
}

//______________________________________________________________________________
void Pad::PushSelectableObject(TObject *obj)
{
   if (!fInSelectionMode)
      return;

   ObjectPair_t newPair(obj, 0);
   if (fParentPainters.size())
      newPair.second = fParentPainters.back().first;
   fSelectables.push_back(newPair);
   fPainter.SetCurrentObjectID(fObjectID);
   fObjectID++;
}

//______________________________________________________________________________
void Pad::PopTopLevelSelectable()
{
   if (!fInSelectionMode)
      return;

   if (fParentPainters.size())
      fParentPainters.pop_back();
   if (fParentPainters.size())
      fPainter.SetCurrentObjectID(fParentPainters.back().second);
}

//______________________________________________________________________________
TObject *Pad::GetSelected()const
{
   return fSelected;
}

//______________________________________________________________________________
void Pad::PaintThumbnail()
{
   cd();

   fPainter.SetPainterMode(Painter::kPaintThumbnail);

   gVirtualX->SetFillStyle(1001);
   gVirtualX->SetFillColor(0);

   PaintBox(fX1,fY1,fX2,fY2);
   PaintBorder(GetFillColor(), kTRUE);

   TObjOptLink *lnk = (TObjOptLink*)GetListOfPrimitives()->FirstLink();
   TObject *obj;

   while (lnk) {
      obj = lnk->GetObject();
      obj->Paint(lnk->GetOption());
      lnk = (TObjOptLink*)lnk->Next();
   }

   fPainter.SetPainterMode(Painter::kPaintToView);
}

//______________________________________________________________________________
void Pad::SetPaintOption(const TObject *obj, const char *option)
{
   TObjOptLink *lnk = (TObjOptLink*)GetListOfPrimitives()->FirstLink();

   while (lnk) {
      TObject *nestedObj = lnk->GetObject();
      if (obj == nestedObj) {
         lnk->SetOption(option);
         break;
      }
      lnk = (TObjOptLink*)lnk->Next();
   }
}

}
}
