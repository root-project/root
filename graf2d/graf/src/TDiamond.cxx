// @(#)root/graf:$Id$
// Author: Rene Brun   22/06/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <cstdlib>

#include <iostream>
#include "TBufferFile.h"
#include "TROOT.h"
#include "TDiamond.h"
#include "TVirtualPad.h"
#include "TVirtualPadPainter.h"
#include "TCanvasImp.h"
#include "TMath.h"


/** \class TDiamond
\ingroup BasicGraphics

Draw a Diamond.

A diamond is defined by:

- Its central left coordinates x1,y1
- Its top central coordinates x2,y2

A diamond has line attributes (see TAttLine) and fill area attributes (see TAttFill).

Like for the class TPaveText, a TDiamond may have one or more line(s) of text inside.

Begin_Macro(source)
../../../tutorials/visualisation/graphics/diamond.C
End_Macro
*/

////////////////////////////////////////////////////////////////////////////////
/// Diamond default constructor.

TDiamond::TDiamond(): TPaveText()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Diamond standard constructor.

TDiamond::TDiamond(Double_t x1, Double_t y1,Double_t x2, Double_t  y2)
     :TPaveText(x1,y1,x2,y2)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Diamond destructor.

TDiamond::~TDiamond()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TDiamond::TDiamond(const TDiamond &diamond) : TPaveText()
{
   TBufferFile b(TBuffer::kWrite);
   TDiamond *p = (TDiamond*)(&diamond);
   p->Streamer(b);
   b.SetReadMode();
   b.SetBufferOffset(0);
   Streamer(b);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute distance from point px,py to a diamond.
///
///  Compute the closest distance of approach from point px,py to the
///  edges of this diamond.
///  The distance is computed in pixels units.

Int_t TDiamond::DistancetoPrimitive(Int_t px, Int_t py)
{
   return TPaveText::DistancetoPrimitive(px,py);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this diamond with its current attributes.

void TDiamond::Draw(Option_t *option)
{
   AppendPad(option);

}

////////////////////////////////////////////////////////////////////////////////
/// Execute action corresponding to one event.
///
/// This member function is called when a Diamond object is clicked.
///
/// If the mouse is clicked inside the diamond, the diamond is moved.
///
/// If the mouse is clicked on the 4 tops (pL,pR,pTop,pBot), the diamond is
/// rescaled.

void TDiamond::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   if (!gPad || !gPad->IsEditable()) return;

   auto &parent = *gPad;

   const Int_t kMaxDiff = 5;
   const Int_t kMinSize = 20;

   static Int_t px1, px2, py1, py2, dpx1, dpy2, px1p, px2p, py1p, py2p;
   static enum { pNone, pTop, pL, pR, pBot, pINSIDE } mode = pNone;
   static Double_t oldX1, oldY1, oldX2, oldY2;
   static Bool_t hasOld = kFALSE;
   Bool_t opaque  = parent.OpaqueMoving();
   Bool_t ropaque = parent.OpaqueResizing();

   auto paint_or_set = [&parent,this](Bool_t paint)
   {
      auto x1 = parent.AbsPixeltoX(px1);
      auto y1 = parent.AbsPixeltoY(py1);
      auto x2 = parent.AbsPixeltoX(px2);
      auto y2 = parent.AbsPixeltoY(py2);
      if (!paint) {
         SetX1(parent.PadtoX(x1));
         SetY1(parent.PadtoY(y1));
         SetX2(parent.PadtoX(x2));
         SetY2(parent.PadtoY(y2));
      } else {
         auto pp = parent.GetPainter();
         Double_t arrx[5] = { x1, (x1+x2) / 2, x2, (x1+x2) / 2, x1 };
         Double_t arry[5] = { (y1+y2)/2, y2, (y1+y2)/2, y1, (y1+y2)/2 };
         pp->SetAttLine({GetFillColor() > 0 ? GetFillColor() : (Color_t) kBlack, 1, 2});
         pp->DrawPolyLine(5, arrx, arry);
      }
   };

   switch (event) {

   case kArrowKeyPress:
   case kButton1Down:

      oldX1 = GetX1();
      oldY1 = GetY1();
      oldX2 = GetX2();
      oldY2 = GetY2();
      hasOld = kTRUE;

      // No break !!!

   case kMouseMotion:

      px1 = parent.XtoAbsPixel(parent.XtoPad(GetX1()));
      py1 = parent.YtoAbsPixel(parent.YtoPad(GetY1()));
      px2 = parent.XtoAbsPixel(parent.XtoPad(GetX2()));
      py2 = parent.YtoAbsPixel(parent.YtoPad(GetY2()));
      if (px1 > px2)
         std::swap(px1, px2);
      if (py1 < py2)
         std::swap(py1, py2);

      px1p = parent.XtoAbsPixel(parent.GetX1()) + parent.GetBorderSize();
      py1p = parent.YtoAbsPixel(parent.GetY1()) - parent.GetBorderSize();
      px2p = parent.XtoAbsPixel(parent.GetX2()) - parent.GetBorderSize();
      py2p = parent.YtoAbsPixel(parent.GetY2()) + parent.GetBorderSize();
      if (px1p > px2p)
         std::swap(px1p, px2p);
      if (py1p < py2p)
         std::swap(py1p, py2p);

      if ((TMath::Abs(px-(px1+px2)/2) < kMaxDiff) && (TMath::Abs(py - py2) < kMaxDiff)) { // top edge
         mode = pTop;
         parent.SetCursor(kTopSide);
      } else if ((TMath::Abs(px-(px1+px2)/2) < kMaxDiff) && (TMath::Abs(py - py1) < kMaxDiff)) { // bottom edge
         mode = pBot;
         parent.SetCursor(kBottomSide);
      } else if ((TMath::Abs(py-(py1+py2)/2) < kMaxDiff) && (TMath::Abs(px - px1) < kMaxDiff)) { // left edge
         mode = pL;
         parent.SetCursor(kLeftSide);
      } else if ((TMath::Abs(py-(py1+py2)/2) < kMaxDiff) && (TMath::Abs(px - px2) < kMaxDiff)) { // right edge
         mode = pR;
         parent.SetCursor(kRightSide);
      } else if (IsInside(parent.PadtoX(parent.AbsPixeltoX(px)), parent.PadtoY(parent.AbsPixeltoY(py)))) {
         mode = pINSIDE;
         dpx1 = px - px1; // cursor position relative to top-left corner
         dpy2 = py - py2;
         parent.SetCursor(event == kButton1Down ? kMove : kCross);
      } else {
         mode = pNone;
         parent.SetCursor(kCross);
      }

      fResizing = mode == pTop || mode == pL || mode == pR || mode == pBot;

      if ((!opaque && mode == pINSIDE) || (!ropaque && fResizing))
         paint_or_set(true);

      break;

   case kArrowKeyRelease:
   case kButton1Motion:
      switch (mode) {
         case pNone: return;
         case pTop:
            if (!ropaque) paint_or_set(kTRUE);
            py2 = TMath::Max(py2p, TMath::Min(py, py1 - kMinSize));
            paint_or_set(!ropaque);
            break;
         case pBot:
            if (!ropaque) paint_or_set(kTRUE);
            py1 = TMath::Min(py1p, TMath::Max(py, py2 + kMinSize));
            paint_or_set(!ropaque);
            break;
         case pL:
            if (!ropaque) paint_or_set(kTRUE);
            px1 = TMath::Max(px1p, TMath::Min(px, px2 - kMinSize));
            paint_or_set(!ropaque);
            break;
         case pR:
            if (!ropaque) paint_or_set(kTRUE);
            px2 = TMath::Min(px2p, TMath::Max(px, px1 + kMinSize));
            paint_or_set(!ropaque);
            break;
         case pINSIDE:
            if (!opaque) paint_or_set(kTRUE);
            px2 += px - dpx1 - px1;
            px1 = px - dpx1;
            py1 += py - dpy2 - py2;
            py2 = py - dpy2;
            if (px1 < px1p) { px2 += px1p - px1; px1 = px1p; }
            if (px2 > px2p) { px1 -= px2 - px2p; px2 = px2p; }
            if (py1 > py1p) { py2 -= py1 - py1p; py1 = py1p; }
            if (py2 < py2p) { py1 += py2p - py2; py2 = py2p; }
            paint_or_set(!opaque);
            break;
      }

      if ((mode == pINSIDE && opaque) || (fResizing && ropaque)) {
         switch(mode) {
            case pINSIDE: parent.ShowGuidelines(this, event, 'i', true); break;
            case pL: parent.ShowGuidelines(this, event, 'l', true); break;
            case pR: parent.ShowGuidelines(this, event, 'r', true); break;
            case pTop: parent.ShowGuidelines(this, event, 't', true); break;
            case pBot: parent.ShowGuidelines(this, event, 'b', true); break;
            default: break; // not involved
         }
         parent.Modified(kTRUE);
      }

      break;

   case kButton1Up:

      if (opaque || ropaque)
         parent.ShowGuidelines(this, event);

      if (gROOT->IsEscaped()) {
         gROOT->SetEscape(kFALSE);
         if (opaque && (mode != pNone)) {
            if (hasOld) {
               SetX1(oldX1);
               SetY1(oldY1);
               SetX2(oldX2);
               SetY2(oldY2);
            }
            hasOld = kFALSE;
            mode = pNone;
            fResizing = kFALSE;
            parent.ModifiedUpdate();
         }
         break;
      }

      if ((!opaque && mode == pINSIDE) || (!ropaque && fResizing))
         paint_or_set(kFALSE);

      if (mode != pNone)
         parent.Modified(kTRUE);

      mode = pNone;
      fResizing = kFALSE;
      hasOld = kFALSE;

      break;

   case kButton1Locate:
      // Sergey: code is never used, has to be removed in ROOT7
      ExecuteEvent(kButton1Down, px, py);

      while (true) {
         px = py = 0;
         event = parent.GetCanvasImp()->RequestLocator(px, py);

         ExecuteEvent(kButton1Motion, px, py);

         if (event != -1) {                     // button is released
            ExecuteEvent(kButton1Up, px, py);
            return;
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return 1 if the point (x,y) is inside the polygon defined by
/// the diamond 0 otherwise.

Int_t TDiamond::IsInside(Double_t x, Double_t y) const
{

   Double_t xd[4], yd[4];

   xd[0] = fX1;
   yd[0] = (fY2 + fY1) / 2.;
   xd[1] = (fX2 + fX1) / 2.;
   yd[1] = fY1;
   xd[2] = fX2;
   yd[2] = yd[0];
   xd[3] = xd[1];
   yd[3] = fY2;

   return (Int_t)TMath::IsInside(x, y, 4, xd, yd);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint this diamond with its current attributes.

void TDiamond::Paint(Option_t *)
{
   if (!gPad) return;
   Double_t x[7],y[7],depx,depy;
   Double_t x1 = fX1;
   Double_t y1 = fY1;
   Double_t x2 = fX2;
   Double_t y2 = fY2;
   Int_t fillstyle = GetFillStyle();
   Int_t fillcolor = GetFillColor();
   Int_t linecolor = GetLineColor();
   if (fBorderSize) {
      Double_t wy = gPad->PixeltoY(0) - gPad->PixeltoY(fBorderSize);
      Double_t wx = gPad->PixeltoX(fBorderSize) - gPad->PixeltoX(0);
      // Draw the frame top right
      if (y2-y1>x2-x1) {
         depx = wx;
         depy = 0;
         }
      else if (y2-y1<x2-x1) {
         depx = 0;
         depy = -wy;
         }
      else {
         depx = wx;
         depy = -wy;
      }
      x[0] = x[2] = (x1+x2)/2+depx;
      x[1] = x2+depx;
      x[3] = x1+depx;
      y[0] = y2+depy;
      y[2] = y1+depy;
      y[1] = y[3] =(y1+y2)/2+depy;
      x[4] = x[0]; y[4] = y[0];
      SetFillStyle(fillstyle);
      SetFillColor(linecolor);
      TAttFill::Modify();  //Change fill area attributes only if necessary
      gPad->PaintFillArea(4,x,y);
   }
   x[0] = x[2] = (x1+x2)/2;
   x[1] = x2;
   x[3] = x1;
   y[0] = y2;
   y[2] = y1;
   y[1] = y[3] = (y1+y2)/2;
   x[4] = x[0]; y[4] =y[0];
   SetLineColor(linecolor);
   SetFillColor(fillcolor);
   TAttLine::Modify();  //Change line attributes only if necessary
   TAttFill::Modify();  //Change fill area attributes only if necessary
   gPad->PaintFillArea(4,x,y);
   gPad->PaintPolyLine(5,x,y);

   // Paint list of primitives (test,etc)
   PaintPrimitives(kDiamond);
}

////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out.

void TDiamond::SavePrimitive(std::ostream &out, Option_t *option)
{
   SavePrimitiveConstructor(out, Class(), "diamond", TString::Format("%g, %g, %g, %g", fX1, fY1, fX2, fY2));

   SaveFillAttributes(out, "diamond", -1, -1);
   SaveLineAttributes(out, "diamond", 1, 1, 1);
   SaveTextAttributes(out, "diamond", 11, 0, 1, 62, 0.05);

   SaveLines(out, "diamond", kTRUE);

   SavePrimitiveDraw(out, "diamond", option);
}
