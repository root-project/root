// @(#)root/graf:$Name:  $:$Id: TDiamond.cxx,v 1.5 2002/05/18 08:21:59 brun Exp $
// Author: Rene Brun   22/06/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdlib.h>

#include "Riostream.h"
#include "TROOT.h"
#include "TDiamond.h"
#include "TVirtualPad.h"
#include "TVirtualX.h"
#include "TClass.h"
#include "TMath.h"


ClassImp(TDiamond)

//______________________________________________________________________________
//
// A diamond is defined by :
//   - Its central left coordinates x1,y1
//   - Its top central coordinates x2,y2
//
// A diamond has line attributes (see TAttLine)
//   and fill area attributes (see TAttFill).
//
// Like for the class TPaveText, a TDiamond may have one or more line(s)
// of text inside.
//Begin_Html
/*
<img src="gif/diamond.gif">
*/
//End_Html
//

//______________________________________________________________________________
TDiamond::TDiamond(): TPaveText()
{
//*-*-*-*-*-*-*-*-*-*-*Diamond default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===========================

}

//______________________________________________________________________________
TDiamond::TDiamond(Double_t x1, Double_t y1,Double_t x2, Double_t  y2)
     :TPaveText(x1,y1,x2,y2)
{
//*-*-*-*-*-*-*-*-*-*-*Diamond standard constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ============================
}

//______________________________________________________________________________
TDiamond::~TDiamond()
{
//*-*-*-*-*-*-*-*-*-*-*Diamond destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==================

}

//______________________________________________________________________________
TDiamond::TDiamond(const TDiamond &diamond) : TPaveText()
{
   TBuffer b(TBuffer::kWrite);
   TDiamond *p = (TDiamond*)(&diamond);
   p->Streamer(b);
   b.SetReadMode();
   b.SetBufferOffset(0);
   Streamer(b);
}

//______________________________________________________________________________
Int_t TDiamond::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*Compute distance from point px,py to a diamond*-*-*-*-*-*
//*-*              ==============================================
//  Compute the closest distance of approach from point px,py to the
//  edges of this diamond.
//  The distance is computed in pixels units.
//

   return TPaveText::DistancetoPrimitive(px,py);
}

//______________________________________________________________________________
void TDiamond::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this diamond with its current attributes*-*-*-*-*-*
//*-*                  =============================================

   AppendPad(option);

}

//______________________________________________________________________________
void TDiamond::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//  This member function is called when a Diamond object is clicked.
//
//  If the mouse is clicked inside the diamond, the diamond is moved.
//
//  If the mouse is clicked on the 4 tops (L,R,T,B), the diamond is rscaled.
//
//
//                             T
//                        +---------+
//                        |   / \   |
//                        |  /   \  |
//                        | /     \ |
//                       L|/ inside\|R
//                        |\       /|
//                        | \     / |
//                        |  \   /  |
//                        |   \ /   |
//                        +---------+
//                             B
//

   const Int_t kMaxDiff = 5;
   const Int_t kMinSize = 20;

   static Int_t px1, px2, py1, py2, pxl, pyl, pxt, pyt, pxold, pyold;
   static Int_t px1p, px2p, py1p, py2p;
   static Int_t Tx,Ty,Lx,Ly,Rx,Ry,Bx,By;
   static Double_t X1c,X2c,X3c,X4c;
   static Bool_t T, L, R, B, INSIDE;
   static Int_t i,x[5], y[5];
   Int_t  wx, wy;
   TVirtualPad  *parent;
   Bool_t doing_again = kFALSE;
   Bool_t opaque  = gPad->OpaqueMoving();
   Bool_t ropaque = gPad->OpaqueResizing();

   if (!gPad->IsEditable()) return;

   parent = gPad;

again:

   switch (event) {

   case kButton1Down:

      gVirtualX->SetLineColor(-1);
      TAttLine::Modify();  //Change line attributes only if necessary
      if (GetFillColor())
         gVirtualX->SetLineColor(GetFillColor());
      else
         gVirtualX->SetLineColor(1);
      gVirtualX->SetLineWidth(2);

      // No break !!!

   case kMouseMotion:

      px1 = gPad->XtoAbsPixel(GetX1());
      py1 = gPad->YtoAbsPixel(GetY1());
      px2 = gPad->XtoAbsPixel(GetX2());
      py2 = gPad->YtoAbsPixel(GetY2());

      if (px1 < px2) {
         pxl = px1;
         pxt = px2;
      } else {
         pxl = px2;
         pxt = px1;
      }
      if (py1 < py2) {
         pyl = py1;
         pyt = py2;
      } else {
         pyl = py2;
         pyt = py1;
      }

      px1p = parent->XtoAbsPixel(parent->GetX1()) + parent->GetBorderSize();
      py1p = parent->YtoAbsPixel(parent->GetY1()) - parent->GetBorderSize();
      px2p = parent->XtoAbsPixel(parent->GetX2()) - parent->GetBorderSize();
      py2p = parent->YtoAbsPixel(parent->GetY2()) + parent->GetBorderSize();

      Tx = Bx = (pxl+pxt)/2;
      Ly = Ry = (pyl+pyt)/2;
      Ty = pyl;
      By = pyt;
      Lx = pxl;
      Rx = pxt;

      T = L = R = B = INSIDE = kFALSE;

      if ((TMath::Abs(px-(pxl+pxt)/2) < kMaxDiff) &&
          (TMath::Abs(py - pyl) < kMaxDiff)) {             // top edge
         pxold = pxl; pyold = pyl; T = kTRUE;
         gPad->SetCursor(kTopSide);
      }

      if ((TMath::Abs(px-(pxl+pxt)/2) < kMaxDiff) &&
          (TMath::Abs(py - pyt) < kMaxDiff)) {             // bottom edge
         pxold = pxt; pyold = pyt; B = kTRUE;
         gPad->SetCursor(kBottomSide);
      }

      if ((TMath::Abs(py-(pyl+pyt)/2) < kMaxDiff) &&
          (TMath::Abs(px - pxl) < kMaxDiff)) {             // left edge
         pxold = pxl; pyold = pyl; L = kTRUE;
         gPad->SetCursor(kLeftSide);
      }

      if ((TMath::Abs(py-(pyl+pyt)/2) < kMaxDiff) &&
          (TMath::Abs(px - pxt) < kMaxDiff)) {             // right edge
          pxold = pxt; pyold = pyt; R = kTRUE;
          gPad->SetCursor(kRightSide);
      }

      X1c = (py-Ty)*(Tx-Lx)/(Ty-Ly)+Tx;
      X2c = (py-Ty)*(Rx-Tx)/(Ry-Ty)+Tx;
      X3c = (py-Ry)*(Rx-Bx)/(Ry-By)+Rx;
      X4c = (py-By)*(Bx-Lx)/(By-Ly)+Bx;

      if (px > X1c+kMaxDiff && px < X2c-kMaxDiff &&
          px > X4c+kMaxDiff && px < X3c-kMaxDiff) {    // inside box
         pxold = px; pyold = py; INSIDE = kTRUE;
         if (event == kButton1Down)
            gPad->SetCursor(kMove);
         else
            gPad->SetCursor(kCross);
      }

      fResizing = kFALSE;
      if (T || L || R || B)
         fResizing = kTRUE;

      if (!T && !L && !R && !B && !INSIDE)
         gPad->SetCursor(kCross);

      break;

   case kButton1Motion:

      wx = wy = 0;
      x[0] = x[2] = x[4] = (px1+px2)/2;
      x[1] = px2;
      x[3] = px1;
      y[0] = y[4] = py1;
      y[2] = py2;
      y[1] = y[3] = (py1+py2)/2;
      if (T) {
         for (i=0;i<4;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
         py2 += py - pyold;
         if (py2 > py1-kMinSize) { py2 = py1-kMinSize; wy = py2; }
         if (py2 < py2p) { py2 = py2p; wy = py2; }
         y[2] = py2;
         y[1] = y[3] = (py1+py2)/2;
         for (i=0;i<4;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
      }
      if (B) {
         for (i=0;i<4;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
         py1 += py - pyold;
         if (py1 < py2+kMinSize) { py1 = py2+kMinSize; wy = py1; }
         if (py1 > py1p) { py1 = py1p; wy = py1; }
         y[0] = y[4] = py1;
         y[1] = y[3] = (py1+py2)/2;
         for (i=0;i<4;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
      }
      if (L) {
         for (i=0;i<4;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
         px1 += px - pxold;
         if (px1 > px2-kMinSize) { px1 = px2-kMinSize; wx = px1; }
         if (px1 < px1p) { px1 = px1p; wx = px1; }
         x[3] = px1;
         x[0] = x[2] = x[4] = (px1+px2)/2;
         for (i=0;i<4;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
      }
      if (R) {
         for (i=0;i<4;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
         px2 += px - pxold;
         if (px2 < px1+kMinSize) { px2 = px1+kMinSize; wx = px2; }
         if (px2 > px2p) { px2 = px2p; wx = px2; }
         x[1] = px2;
          x[0] = x[2] = x[4] = (px1+px2)/2;
         for (i=0;i<4;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
      }
      if (INSIDE) {
         for (i=0;i<4;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
         Int_t dx = px - pxold;
         Int_t dy = py - pyold;
         px1 += dx; py1 += dy; px2 += dx; py2 += dy;
         if (px1 < px1p) { dx = px1p - px1; px1 += dx; px2 += dx; wx = px+dx; }
         if (px2 > px2p) { dx = px2 - px2p; px1 -= dx; px2 -= dx; wx = px-dx; }
         if (py1 > py1p) { dy = py1 - py1p; py1 -= dy; py2 -= dy; wy = py-dy; }
         if (py2 < py2p) { dy = py2p - py2; py1 += dy; py2 += dy; wy = py+dy; }
         x[0] = x[2] = x[4] = (px1+px2)/2;
         x[1] = px2;
         x[3] = px1;
         y[0] = y[4] = py1;
         y[2] = py2;
         y[1] = y[3] = (py1+py2)/2;
         for (i=0;i<4;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
      }

      if (wx || wy) {
         if (wx) px = wx;
         if (wy) py = wy;
         gVirtualX->Warp(px, py);
      }

      pxold = px;
      pyold = py;

      if ((INSIDE && opaque) || (fResizing && ropaque)) {
         event = kButton1Up;
         doing_again = kTRUE;
         goto again;
      }

      break;

   case kButton1Up:

      if (T || B || L || R || INSIDE) {
         fX1 = gPad->AbsPixeltoX(px1);
         fY1 = gPad->AbsPixeltoY(py1);
         fX2 = gPad->AbsPixeltoX(px2);
         fY2 = gPad->AbsPixeltoY(py2);
      }

      if (INSIDE) {
         // if it was not a pad that was moved then it must have been
         // a box or something like that so we have to redraw the pad
         if (parent == gPad) gPad->Modified(kTRUE);
         if (!doing_again) gPad->SetCursor(kCross);
      }

      if (T || L || R || B)
         gPad->Modified(kTRUE);

      gVirtualX->SetLineColor(-1);
      gVirtualX->SetLineWidth(-1);

      break;

   case kButton1Locate:

      ExecuteEvent(kButton1Down, px, py);

      while (1) {
         px = py = 0;
         event = gVirtualX->RequestLocator(1, 1, px, py);

         ExecuteEvent(kButton1Motion, px, py);

         if (event != -1) {                     // button is released
            ExecuteEvent(kButton1Up, px, py);
            return;
         }
      }
   }
}

//______________________________________________________________________________
void TDiamond::Paint(Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*Paint this diamond with its current attributes*-*-*-*-*
//*-*                  ==============================================

   Double_t x[7],y[7],Depx,Depy;
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
     //*-*- Draw the frame top right
     if (y2-y1>x2-x1) {
        Depx = wx;
        Depy = 0;
        }
     else if (y2-y1<x2-x1) {
        Depx = 0;
        Depy = -wy;
        }
     else {
        Depx = wx;
        Depy = -wy;
     }
     x[0] = x[2] = (x1+x2)/2+Depx;
     x[1] = x2+Depx;
     x[3] = x1+Depx;
     y[0] = y2+Depy;
     y[2] = y1+Depy;
     y[1] = y[3] =(y1+y2)/2+Depy;
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

//  Paint list of primitives (test,etc)
   PaintPrimitives(kDiamond);
}

//______________________________________________________________________________
void TDiamond::SavePrimitive(ofstream &out, Option_t *)
{
    // Save primitive as a C++ statement(s) on output stream out

   if (gROOT->ClassSaved(TDiamond::Class())) {
       out<<"   ";
   } else {
       out<<"   TDiamond *";
   }
   out<<"diamond = new TDiamond("<<fX1<<","<<fY1<<","<<fX2<<","<<fY2<<");"<<endl;

   SaveFillAttributes(out,"diamond",0,1001);
   SaveLineAttributes(out,"diamond",1,1,1);
   SaveTextAttributes(out,"diamond",11,0,1,62,1);

   SaveLines(out,"diamond");
   out<<"   diamond->Draw();"<<endl;
}
