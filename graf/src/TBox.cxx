// @(#)root/graf:$Name:  $:$Id: TBox.cxx,v 1.6 2001/06/05 14:51:07 rdm Exp $
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdlib.h>
#include <fstream.h>
#include <iostream.h>

#include "TROOT.h"
#include "TBox.h"
#include "TVirtualPad.h"
#include "TVirtualX.h"
#include "TClass.h"
#include "TMath.h"


ClassImp(TBox)

//______________________________________________________________________________
//
// A box is defined by :
//   - Its bottom left coordinates x1,y1
//   - Its top right coordinates x2,y2
//
// A box has line attributes (see TAttLine)
//   and fill area attributes (see TAttFill).
//
//Begin_Html
/*
<img src="gif/tg_classtree.gif">
*/
//End_Html
//

//______________________________________________________________________________
TBox::TBox(): TObject(), TAttLine(), TAttFill()
{
//*-*-*-*-*-*-*-*-*-*-*Box default constructor-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =======================

   fTip = 0;
}

//______________________________________________________________________________
TBox::TBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
     : TObject(), TAttLine(), TAttFill()
{
//*-*-*-*-*-*-*-*-*-*-*Box standard constructor-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================
   fX1=x1; fY1=y1; fX2=x2; fY2=y2;
   fResizing = kFALSE;
   fTip = 0;
}

//______________________________________________________________________________
TBox::~TBox()
{
//*-*-*-*-*-*-*-*-*-*-*Box destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============

   if (gPad) {
      gPad->CloseToolTip(fTip);
      gPad->DeleteToolTip(fTip);
   }
}

//______________________________________________________________________________
TBox::TBox(const TBox &box)
{
   ((TBox&)box).Copy(*this);
}

//______________________________________________________________________________
void TBox::Copy(TObject &obj)
{
//*-*-*-*-*-*-*-*-*-*-*Copy a Box*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==========

   TObject::Copy(obj);
   TAttLine::Copy(((TBox&)obj));
   TAttFill::Copy(((TBox&)obj));
   ((TBox&)obj).fX1 = fX1;
   ((TBox&)obj).fY1 = fY1;
   ((TBox&)obj).fX2 = fX2;
   ((TBox&)obj).fY2 = fY2;
   ((TBox&)obj).fResizing = fResizing;
   ((TBox&)obj).fTip = 0;   //FIXME
}

//______________________________________________________________________________
Int_t TBox::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Compute distance from point px,py to a box*-*-*-*-*-*
//*-*                  ==========================================
//  Compute the closest distance of approach from point px,py to the
//  edges of this box.
//  The distance is computed in pixels units.
//

   Int_t pxl, pyl, pxt, pyt;
   Int_t px1 = gPad->XtoAbsPixel(fX1);
   Int_t py1 = gPad->YtoAbsPixel(fY1);
   Int_t px2 = gPad->XtoAbsPixel(fX2);
   Int_t py2 = gPad->YtoAbsPixel(fY2);
   if (px1 < px2) {pxl = px1; pxt = px2;}
   else           {pxl = px2; pxt = px1;}
   if (py1 < py2) {pyl = py1; pyt = py2;}
   else           {pyl = py2; pyt = py1;}

//*-*- Are we inside the box?
//*-*  ======================
   if ( (px > pxl && px < pxt) && (py > pyl && py < pyt) ) {
      if (GetFillStyle()) return 0;  //*-* if box is filled
   }

//*-*- Are we on the edges?
//*-*  ====================
   Int_t dxl = TMath::Abs(px - pxl);
   if (py < pyl) dxl += pyl - py; if (py > pyt) dxl += py - pyt;
   Int_t dxt = TMath::Abs(px - pxt);
   if (py < pyl) dxt += pyl - py; if (py > pyt) dxt += py - pyt;
   Int_t dyl = TMath::Abs(py - pyl);
   if (px < pxl) dyl += pxl - px; if (px > pxt) dyl += px - pxt;
   Int_t dyt = TMath::Abs(py - pyt);
   if (px < pxl) dyt += pxl - px; if (px > pxt) dyt += px - pxt;

   Int_t distance = dxl;
   if (dxt < distance) distance = dxt;
   if (dyl < distance) distance = dyl;
   if (dyt < distance) distance = dyt;

   return distance - Int_t(0.5*fLineWidth);
}

//______________________________________________________________________________
void TBox::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this box with its current attributes*-*-*-*-*-*-*
//*-*                  =========================================

   AppendPad(option);

}

//______________________________________________________________________________
void TBox::DrawBox(Double_t x1, Double_t y1,Double_t x2, Double_t  y2)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this box with new coordinates*-*-*-*-*-*-*-*-*-*
//*-*                  ==================================
   TBox *newbox = new TBox(x1,y1,x2,y2);
   TAttLine::Copy(*newbox);
   TAttFill::Copy(*newbox);
   newbox->SetBit(kCanDelete);
   newbox->AppendPad();
}

//______________________________________________________________________________
void TBox::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//  This member function is called when a BOX/WBOX/PAD object is clicked.
//
//  If the mouse is clicked in one of the 4 corners of the box (PA,PB,PC,PD)
//  the box is resized with the rubber rectangle.
//
//  If the mouse is clicked inside the box, the box is moved.
//
//  If the mouse is clicked on the 4 edges (L,R,T,B), the box is rscaled
//  parallel to this edge (same as Motif window manager).
//
//    PA                    T                       PB
//     +--------------------------------------------+
//     |                                            |
//     |                                            |
//     |                                            |
//    L|                  INSIDE                    |R
//     |                                            |
//     |                                            |
//     |                                            |
//     |                                            |
//     +--------------------------------------------+
//    PD                    B                      PC
//
//  Note that this function is duplicated on purpose by TPad::ExecuteEvent.
//  If somebody modifies this function, may be similar changes should also
//  be applied to TPad::ExecuteEvent.

   if (!gPad) return;
   if (!gPad->IsEditable() && event != kMouseEnter) return;

   const Int_t kMaxDiff = 5;
   const Int_t kMinSize = 20;

   static Int_t px1, px2, py1, py2, pxl, pyl, pxt, pyt, pxold, pyold;
   static Int_t px1p, px2p, py1p, py2p, pxlp, pylp, pxtp, pytp;
   static Bool_t PA, PB, PC, PD, T, L, R, B, INSIDE;
   Int_t  wx, wy;
   TVirtualPad  *parent = gPad;
   Bool_t doing_again = kFALSE;
   Bool_t opaque  = gPad->OpaqueMoving();
   Bool_t ropaque = gPad->OpaqueResizing();

   HideToolTip(event);

again:

   switch (event) {

   case kMouseEnter:
      if (fTip)
         gPad->ResetToolTip(fTip);
      break;

   case kButton1Double:
      px1 = -1; //used by kButton1Up
      break;

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

      if (px1p < px2p) {
         pxlp = px1p;
         pxtp = px2p;
      } else {
         pxlp = px2p;
         pxtp = px1p;
      }
      if (py1p < py2p) {
         pylp = py1p;
         pytp = py2p;
      } else {
         pylp = py2p;
         pytp = py1p;
      }

      PA = PB = PC = PD = T = L = R = B = INSIDE = kFALSE;

                                                         // case PA
      if (TMath::Abs(px - pxl) <= kMaxDiff && TMath::Abs(py - pyl) <= kMaxDiff) {
         pxold = pxl; pyold = pyl; PA = kTRUE;
         gPad->SetCursor(kTopLeft);
      }
                                                         // case PB
      if (TMath::Abs(px - pxt) <= kMaxDiff && TMath::Abs(py - pyl) <= kMaxDiff) {
         pxold = pxt; pyold = pyl; PB = kTRUE;
         gPad->SetCursor(kTopRight);
      }
                                                         // case PC
      if (TMath::Abs(px - pxt) <= kMaxDiff && TMath::Abs(py - pyt) <= kMaxDiff) {
         pxold = pxt; pyold = pyt; PC = kTRUE;
         gPad->SetCursor(kBottomRight);
      }
                                                         // case PD
      if (TMath::Abs(px - pxl) <= kMaxDiff && TMath::Abs(py - pyt) <= kMaxDiff) {
         pxold = pxl; pyold = pyt; PD = kTRUE;
         gPad->SetCursor(kBottomLeft);
      }

      if ((px > pxl+kMaxDiff && px < pxt-kMaxDiff) &&
          TMath::Abs(py - pyl) < kMaxDiff) {             // top edge
         pxold = pxl; pyold = pyl; T = kTRUE;
         gPad->SetCursor(kTopSide);
      }

      if ((px > pxl+kMaxDiff && px < pxt-kMaxDiff) &&
          TMath::Abs(py - pyt) < kMaxDiff) {             // bottom edge
         pxold = pxt; pyold = pyt; B = kTRUE;
         gPad->SetCursor(kBottomSide);
      }

      if ((py > pyl+kMaxDiff && py < pyt-kMaxDiff) &&
          TMath::Abs(px - pxl) < kMaxDiff) {             // left edge
         pxold = pxl; pyold = pyl; L = kTRUE;
         gPad->SetCursor(kLeftSide);
      }

      if ((py > pyl+kMaxDiff && py < pyt-kMaxDiff) &&
          TMath::Abs(px - pxt) < kMaxDiff) {             // right edge
          pxold = pxt; pyold = pyt; R = kTRUE;
          gPad->SetCursor(kRightSide);
      }

      if ((px > pxl+kMaxDiff && px < pxt-kMaxDiff) &&
          (py > pyl+kMaxDiff && py < pyt-kMaxDiff)) {    // inside box
         pxold = px; pyold = py; INSIDE = kTRUE;
         if (event == kButton1Down)
            gPad->SetCursor(kMove);
         else
            gPad->SetCursor(kCross);
      }

      fResizing = kFALSE;
      if (PA || PB || PC || PD || T || L || R || B)
         fResizing = kTRUE;

      if (!PA && !PB && !PC && !PD && !T && !L && !R && !B && !INSIDE)
         gPad->SetCursor(kCross);

      break;

   case kButton1Motion:

      wx = wy = 0;

      if (PA) {
         if (!ropaque) gVirtualX->DrawBox(pxold, pyt, pxt, pyold, TVirtualX::kHollow);  // draw the old box
         if (px > pxt-kMinSize) { px = pxt-kMinSize; wx = px; }
         if (py > pyt-kMinSize) { py = pyt-kMinSize; wy = py; }
         if (px < pxlp) { px = pxlp; wx = px; }
         if (py < pylp) { py = pylp; wy = py; }
         if (!ropaque) gVirtualX->DrawBox(px   , pyt, pxt, py,    TVirtualX::kHollow);  // draw the new box
      }
      if (PB) {
         if (!ropaque) gVirtualX->DrawBox(pxl  , pyt, pxold, pyold, TVirtualX::kHollow);
         if (px < pxl+kMinSize) { px = pxl+kMinSize; wx = px; }
         if (py > pyt-kMinSize) { py = pyt-kMinSize; wy = py; }
         if (px > pxtp) { px = pxtp; wx = px; }
         if (py < pylp) { py = pylp; wy = py; }
         if (!ropaque) gVirtualX->DrawBox(pxl  , pyt, px ,  py,    TVirtualX::kHollow);
      }
      if (PC) {
         if (!ropaque) gVirtualX->DrawBox(pxl  , pyl, pxold, pyold, TVirtualX::kHollow);
         if (px < pxl+kMinSize) { px = pxl+kMinSize; wx = px; }
         if (py < pyl+kMinSize) { py = pyl+kMinSize; wy = py; }
         if (px > pxtp) { px = pxtp; wx = px; }
         if (py > pytp) { py = pytp; wy = py; }
         if (!ropaque) gVirtualX->DrawBox(pxl  , pyl, px ,   py,    TVirtualX::kHollow);
      }
      if (PD) {
         if (!ropaque) gVirtualX->DrawBox(pxold, pyold, pxt, pyl, TVirtualX::kHollow);
         if (px > pxt-kMinSize) { px = pxt-kMinSize; wx = px; }
         if (py < pyl+kMinSize) { py = pyl+kMinSize; wy = py; }
         if (px < pxlp) { px = pxlp; wx = px; }
         if (py > pytp) { py = pytp; wy = py; }
         if (!ropaque) gVirtualX->DrawBox(px   , py ,   pxt, pyl, TVirtualX::kHollow);
      }
      if (T) {
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         py2 += py - pyold;
         if (py2 > py1-kMinSize) { py2 = py1-kMinSize; wy = py2; }
         if (py2 < py2p) { py2 = py2p; wy = py2; }
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      }
      if (B) {
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         py1 += py - pyold;
         if (py1 < py2+kMinSize) { py1 = py2+kMinSize; wy = py1; }
         if (py1 > py1p) { py1 = py1p; wy = py1; }
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      }
      if (L) {
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         px1 += px - pxold;
         if (px1 > px2-kMinSize) { px1 = px2-kMinSize; wx = px1; }
         if (px1 < px1p) { px1 = px1p; wx = px1; }
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      }
      if (R) {
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         px2 += px - pxold;
         if (px2 < px1+kMinSize) { px2 = px1+kMinSize; wx = px2; }
         if (px2 > px2p) { px2 = px2p; wx = px2; }
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      }
      if (INSIDE) {
         if (!opaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);  // draw the old box
         Int_t dx = px - pxold;
         Int_t dy = py - pyold;
         px1 += dx; py1 += dy; px2 += dx; py2 += dy;
         if (px1 < px1p) { dx = px1p - px1; px1 += dx; px2 += dx; wx = px+dx; }
         if (px2 > px2p) { dx = px2 - px2p; px1 -= dx; px2 -= dx; wx = px-dx; }
         if (py1 > py1p) { dy = py1 - py1p; py1 -= dy; py2 -= dy; wy = py-dy; }
         if (py2 < py2p) { dy = py2p - py2; py1 += dy; py2 += dy; wy = py+dy; }
         if (!opaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);  // draw the new box
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

      if (px1 < 0 ) break;
      if (PA) {
         fX1 = gPad->AbsPixeltoX(pxold);
         fY1 = gPad->AbsPixeltoY(pyt);
         fX2 = gPad->AbsPixeltoX(pxt);
         fY2 = gPad->AbsPixeltoY(pyold);
      }
      if (PB) {
         fX1 = gPad->AbsPixeltoX(pxl);
         fY1 = gPad->AbsPixeltoY(pyt);
         fX2 = gPad->AbsPixeltoX(pxold);
         fY2 = gPad->AbsPixeltoY(pyold);
      }
      if (PC) {
         fX1 = gPad->AbsPixeltoX(pxl);
         fY1 = gPad->AbsPixeltoY(pyold);
         fX2 = gPad->AbsPixeltoX(pxold);
         fY2 = gPad->AbsPixeltoY(pyl);
      }
      if (PD) {
         fX1 = gPad->AbsPixeltoX(pxold);
         fY1 = gPad->AbsPixeltoY(pyold);
         fX2 = gPad->AbsPixeltoX(pxt);
         fY2 = gPad->AbsPixeltoY(pyl);
      }
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

      if (PA || PB || PC || PD || T || L || R || B)
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
void TBox::HideToolTip(Int_t event)
{
   // Hide tool tip depending on the event type. Typically tool tips
   // are hidden when event is not a kMouseEnter and not a kMouseMotion
   // event.

   if (event != kMouseEnter && event != kMouseMotion && fTip && gPad)
      gPad->CloseToolTip(fTip);
}

//______________________________________________________________________________
void TBox::ls(Option_t *) const
{
//*-*-*-*-*-*-*-*-*-*-*-*List this box with its attributes*-*-*-*-*-*-*-*-*
//*-*                    =================================
   TROOT::IndentLevel();
   printf("%s  X1= %f Y1=%f X2=%f Y2=%f\n",IsA()->GetName(),fX1,fY1,fX2,fY2);
}

//______________________________________________________________________________
void TBox::Paint(Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*Paint this box with its current attributes*-*-*-*-*-*-*
//*-*                  ==========================================
   PaintBox(fX1,fY1,fX2,fY2);
}

//______________________________________________________________________________
void TBox::PaintBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this box with new coordinates*-*-*-*-*-*-*-*-*-*
//*-*                  ==================================

   TAttLine::Modify();  //Change line attributes only if necessary
   TAttFill::Modify();  //Change fill area attributes only if necessary

   gPad->PaintBox(x1,y1,x2,y2);
}

//______________________________________________________________________________
void TBox::Print(Option_t *) const
{
//*-*-*-*-*-*-*-*-*-*-*Dump this box with its attributes*-*-*-*-*-*-*-*-*-*
//*-*                  =================================

   printf("%s  X1=%f Y1=%f X2=%f Y2=%f",IsA()->GetName(),fX1,fY1,fX2,fY2);
   if (GetLineColor() != 1) printf(" Color=%d",GetLineColor());
   if (GetLineStyle() != 1) printf(" Style=%d",GetLineStyle());
   if (GetLineWidth() != 1) printf(" Width=%d",GetLineWidth());
   if (GetFillColor() != 0) printf(" FillColor=%d",GetFillColor());
   if (GetFillStyle() != 0) printf(" FillStyle=%d",GetFillStyle());
   printf("\n");
}


//______________________________________________________________________________
void TBox::SavePrimitive(ofstream &out, Option_t *)
{
    // Save primitive as a C++ statement(s) on output stream out

   if (gROOT->ClassSaved(TBox::Class())) {
       out<<"   ";
   } else {
       out<<"   TBox *";
   }
   out<<"box = new TBox("<<fX1<<","<<fY1<<","<<fX2<<","<<fY2<<");"<<endl;

   SaveFillAttributes(out,"box",0,1001);
   SaveLineAttributes(out,"box",1,1,1);

   out<<"   box->Draw();"<<endl;
}

//______________________________________________________________________________
void TBox::SetToolTipText(const char *text, Long_t delayms)
{
   // Set tool tip text associated with this box. The delay is in
   // milliseconds (minimum 250). To remove tool tip call method with
   // text = 0.

   if (!gPad) {
      Warning("SetToolTipText", "a canvas must exist before setting the tool tip text");
      return;
   }

   if (fTip) {
      gPad->DeleteToolTip(fTip);
      fTip = 0;
   }

   if (text && strlen(text))
      fTip = gPad->CreateToolTip(this, text, delayms);
}

//______________________________________________________________________________
void TBox::Streamer(TBuffer &R__b)
{
   // Stream an object of class TBox.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TBox::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TObject::Streamer(R__b);
      TAttLine::Streamer(R__b);
      TAttFill::Streamer(R__b);
      Float_t x1,y1,x2,y2;
      R__b >> x1; fX1 = x1;
      R__b >> y1; fY1 = y1;
      R__b >> x2; fX2 = x2;
      R__b >> y2; fY2 = y2;
      R__b.CheckByteCount(R__s, R__c, TBox::IsA());
      //====end of old versions

   } else {
      TBox::Class()->WriteBuffer(R__b,this);
   }
}
