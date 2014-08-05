// @(#)root/graf:$Id$
// Author: Rene Brun   12/12/94

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
#include "TBox.h"
#include "TVirtualPad.h"
#include "TVirtualX.h"
#include "TClass.h"
#include "TMath.h"
#include "TPoint.h"


ClassImp(TBox)


//______________________________________________________________________________
/* Begin_Html
<center><h2>Box class</h2></center>
A box is defined by :
<ul>
<li> Its bottom left coordinates x1,y1
<li> Its top right coordinates x2,y2
</ul>
A box has line attributes (see TAttLine) and fill area attributes
(see TAttFill).
End_Html */


//______________________________________________________________________________
TBox::TBox(): TObject(), TAttLine(), TAttFill()
{
   // Box default constructor.

   fTip = 0;
   fX1       = 0.;
   fY1       = 0.;
   fX2       = 0.;
   fY2       = 0.;
   fResizing = kTRUE;
}


//______________________________________________________________________________
TBox::TBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
     : TObject(), TAttLine(), TAttFill()
{
   // Box standard constructor.

   if (x2 >= x1) {fX1  =x1; fX2 = x2;}
   else          {fX1 = x2; fX2 = x1;}
   if (y2 >= y1) {fY1  =y1; fY2 = y2;}
   else          {fY1 = y2; fY2 = y1;}
   fResizing = kFALSE;
   fTip = 0;
}


//______________________________________________________________________________
TBox::~TBox()
{
   // Box destructor.

   if (fTip && gPad) {
      gPad->CloseToolTip(fTip);
      gPad->DeleteToolTip(fTip);
   }
}


//______________________________________________________________________________
TBox::TBox(const TBox &box) : TObject(box), TAttLine(box), TAttFill(box), TAttBBox2D(box)
{
   // Box copy constructor.

   fX1       = 0.;
   fY1       = 0.;
   fX2       = 0.;
   fY2       = 0.;
   fResizing = kTRUE;
   ((TBox&)box).TBox::Copy(*this);
}


//______________________________________________________________________________
TBox& TBox::operator=(const TBox& b)
{
   // Assignment operator.

   if(this!=&b) {
      TObject::operator=(b);
      TAttLine::operator=(b);
      TAttFill::operator=(b);
      fTip=b.fTip;
      fX1=b.fX1;
      fY1=b.fY1;
      fX2=b.fX2;
      fY2=b.fY2;
      fResizing=b.fResizing;
   }
   return *this;
}


//______________________________________________________________________________
void TBox::Copy(TObject &obj) const
{
   // Copy a Box.

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
   // Compute distance from point px,py to a box.
   //
   // Compute the closest distance of approach from point px,py to the
   // edges of this box.
   // The distance is computed in pixels units.
   //
   // In case of a filled box the distance returned is 0 if the point
   // (px,py) is inside the box, and is huge if the point is outside.

   Int_t pxl, pyl, pxt, pyt;
   Int_t px1 = gPad->XtoAbsPixel(fX1);
   Int_t py1 = gPad->YtoAbsPixel(fY1);
   Int_t px2 = gPad->XtoAbsPixel(fX2);
   Int_t py2 = gPad->YtoAbsPixel(fY2);
   if (px1 < px2) {pxl = px1; pxt = px2;}
   else           {pxl = px2; pxt = px1;}
   if (py1 < py2) {pyl = py1; pyt = py2;}
   else           {pyl = py2; pyt = py1;}

   // Are we inside the box?
   if (GetFillStyle()) {
      if ( (px >= pxl && px <= pxt) && (py >= pyl && py <= pyt) ) return 0;
      else return 9999;
   }

   // Are we on the edges?
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
   // Draw this box with its current attributes.
   // if the box has no fill style (ie fill style=0), the box contour is drawn
   // if the box has a fill style, the box contour is not drawn by default.
   // to force the contour to be drawn, specify option "l"

   AppendPad(option);

}


//______________________________________________________________________________
TBox *TBox::DrawBox(Double_t x1, Double_t y1,Double_t x2, Double_t  y2)
{
   // Draw this box with new coordinates.

   TBox *newbox = new TBox(x1,y1,x2,y2);
   TAttLine::Copy(*newbox);
   TAttFill::Copy(*newbox);
   newbox->SetBit(kCanDelete);
   newbox->AppendPad();
   return newbox;
}


//______________________________________________________________________________
void TBox::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute action corresponding to one event.
   //
   //  This member function is called when a BOX/WBOX/PAD object is clicked.
   //
   //  If the mouse is clicked in one of the 4 corners of the box (pA,pB,pC,pD)
   //  the box is resized with the rubber rectangle.
   //
   //  If the mouse is clicked inside the box, the box is moved.
   //
   //  If the mouse is clicked on the 4 edges (pL,pR,pTop,pBot), the box is
   //  rescaled parallel to this edge (same as Motif window manager).
   //
   //    pA                 pTop                       pB
   //     +--------------------------------------------+
   //     |                                            |
   //     |                                            |
   //     |                                            |
   //   pL|                  pINSIDE                   |pR
   //     |                                            |
   //     |                                            |
   //     |                                            |
   //     |                                            |
   //     +--------------------------------------------+
   //    pD                 pBot                      pC
   //
   //  Note that this function is duplicated on purpose by TPad::ExecuteEvent.
   //  If somebody modifies this function, may be similar changes should also
   //  be applied to TPad::ExecuteEvent.

   if (!gPad) return;
   if (!gPad->IsEditable() && event != kMouseEnter) return;

   if (TestBit(kCannotMove)) return;

   const Int_t kMaxDiff = 7;
   const Int_t kMinSize = 20;

   static Int_t px1, px2, py1, py2, pxl, pyl, pxt, pyt, pxold, pyold;
   static Int_t px1p, px2p, py1p, py2p, pxlp, pylp, pxtp, pytp;
   static Double_t oldX1, oldY1, oldX2, oldY2;
   static Bool_t pA, pB, pC, pD, pTop, pL, pR, pBot, pINSIDE;
   Int_t  wx, wy;
   TVirtualPad  *parent = gPad;
   Bool_t opaque  = gPad->OpaqueMoving();
   Bool_t ropaque = gPad->OpaqueResizing();

   HideToolTip(event);

   switch (event) {

   case kMouseEnter:
      if (fTip) gPad->ResetToolTip(fTip);
      break;

   case kButton1Double:
      px1 = -1; //used by kButton1Up
      break;

   case kArrowKeyPress:
   case kButton1Down:

      oldX1 = fX1;
      oldY1 = fY1;
      oldX2 = fX2;
      oldY2 = fY2;
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

      pA = pB = pC = pD = pTop = pL = pR = pBot = pINSIDE = kFALSE;

                                                         // case pA
      if (TMath::Abs(px - pxl) <= kMaxDiff && TMath::Abs(py - pyl) <= kMaxDiff) {
         pxold = pxl; pyold = pyl; pA = kTRUE;
         gPad->SetCursor(kTopLeft);
      }
                                                         // case pB
      if (TMath::Abs(px - pxt) <= kMaxDiff && TMath::Abs(py - pyl) <= kMaxDiff) {
         pxold = pxt; pyold = pyl; pB = kTRUE;
         gPad->SetCursor(kTopRight);
      }
                                                         // case pC
      if (TMath::Abs(px - pxt) <= kMaxDiff && TMath::Abs(py - pyt) <= kMaxDiff) {
         pxold = pxt; pyold = pyt; pC = kTRUE;
         gPad->SetCursor(kBottomRight);
      }
                                                         // case pD
      if (TMath::Abs(px - pxl) <= kMaxDiff && TMath::Abs(py - pyt) <= kMaxDiff) {
         pxold = pxl; pyold = pyt; pD = kTRUE;
         gPad->SetCursor(kBottomLeft);
      }

      if ((px > pxl+kMaxDiff && px < pxt-kMaxDiff) &&
          TMath::Abs(py - pyl) < kMaxDiff) {             // top edge
         pxold = pxl; pyold = pyl; pTop = kTRUE;
         gPad->SetCursor(kTopSide);
      }

      if ((px > pxl+kMaxDiff && px < pxt-kMaxDiff) &&
          TMath::Abs(py - pyt) < kMaxDiff) {             // bottom edge
         pxold = pxt; pyold = pyt; pBot = kTRUE;
         gPad->SetCursor(kBottomSide);
      }

      if ((py > pyl+kMaxDiff && py < pyt-kMaxDiff) &&
          TMath::Abs(px - pxl) < kMaxDiff) {             // left edge
         pxold = pxl; pyold = pyl; pL = kTRUE;
         gPad->SetCursor(kLeftSide);
      }

      if ((py > pyl+kMaxDiff && py < pyt-kMaxDiff) &&
          TMath::Abs(px - pxt) < kMaxDiff) {             // right edge
         pxold = pxt; pyold = pyt; pR = kTRUE;
         gPad->SetCursor(kRightSide);
      }

      if ((px > pxl+kMaxDiff && px < pxt-kMaxDiff) &&
          (py > pyl+kMaxDiff && py < pyt-kMaxDiff)) {    // inside box
         pxold = px; pyold = py; pINSIDE = kTRUE;
         if (event == kButton1Down)
            gPad->SetCursor(kMove);
         else
            gPad->SetCursor(kCross);
      }

      fResizing = kFALSE;
      if (pA || pB || pC || pD || pTop || pL || pR || pBot)
         fResizing = kTRUE;

      if (!pA && !pB && !pC && !pD && !pTop && !pL && !pR && !pBot && !pINSIDE)
         gPad->SetCursor(kCross);

      break;

   case kArrowKeyRelease:
   case kButton1Motion:

      wx = wy = 0;

      if (pA) {
         if (!ropaque) gVirtualX->DrawBox(pxold, pyt, pxt, pyold, TVirtualX::kHollow);  // draw the old box
         if (px > pxt-kMinSize) { px = pxt-kMinSize; wx = px; }
         if (py > pyt-kMinSize) { py = pyt-kMinSize; wy = py; }
         if (px < pxlp) { px = pxlp; wx = px; }
         if (py < pylp) { py = pylp; wy = py; }
         if (!ropaque) gVirtualX->DrawBox(px   , pyt, pxt, py,    TVirtualX::kHollow);  // draw the new box
      }
      if (pB) {
         if (!ropaque) gVirtualX->DrawBox(pxl  , pyt, pxold, pyold, TVirtualX::kHollow);
         if (px < pxl+kMinSize) { px = pxl+kMinSize; wx = px; }
         if (py > pyt-kMinSize) { py = pyt-kMinSize; wy = py; }
         if (px > pxtp) { px = pxtp; wx = px; }
         if (py < pylp) { py = pylp; wy = py; }
         if (!ropaque) gVirtualX->DrawBox(pxl  , pyt, px ,  py,    TVirtualX::kHollow);
      }
      if (pC) {
         if (!ropaque) gVirtualX->DrawBox(pxl  , pyl, pxold, pyold, TVirtualX::kHollow);
         if (px < pxl+kMinSize) { px = pxl+kMinSize; wx = px; }
         if (py < pyl+kMinSize) { py = pyl+kMinSize; wy = py; }
         if (px > pxtp) { px = pxtp; wx = px; }
         if (py > pytp) { py = pytp; wy = py; }
         if (!ropaque) gVirtualX->DrawBox(pxl  , pyl, px ,   py,    TVirtualX::kHollow);
      }
      if (pD) {
         if (!ropaque) gVirtualX->DrawBox(pxold, pyold, pxt, pyl, TVirtualX::kHollow);
         if (px > pxt-kMinSize) { px = pxt-kMinSize; wx = px; }
         if (py < pyl+kMinSize) { py = pyl+kMinSize; wy = py; }
         if (px < pxlp) { px = pxlp; wx = px; }
         if (py > pytp) { py = pytp; wy = py; }
         if (!ropaque) gVirtualX->DrawBox(px   , py ,   pxt, pyl, TVirtualX::kHollow);
      }
      if (pTop) {
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         py2 += py - pyold;
         if (py2 > py1-kMinSize) { py2 = py1-kMinSize; wy = py2; }
         if (py2 < py2p) { py2 = py2p; wy = py2; }
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      }
      if (pBot) {
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         py1 += py - pyold;
         if (py1 < py2+kMinSize) { py1 = py2+kMinSize; wy = py1; }
         if (py1 > py1p) { py1 = py1p; wy = py1; }
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      }
      if (pL) {
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         px1 += px - pxold;
         if (px1 > px2-kMinSize) { px1 = px2-kMinSize; wx = px1; }
         if (px1 < px1p) { px1 = px1p; wx = px1; }
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      }
      if (pR) {
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         px2 += px - pxold;
         if (px2 < px1+kMinSize) { px2 = px1+kMinSize; wx = px2; }
         if (px2 > px2p) { px2 = px2p; wx = px2; }
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      }
      if (pINSIDE) {
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


      if ((pINSIDE && opaque) || (fResizing && ropaque)) {
         if (pA) {
            fX1 = gPad->AbsPixeltoX(pxold);
            fY1 = gPad->AbsPixeltoY(pyt);
            fX2 = gPad->AbsPixeltoX(pxt);
            fY2 = gPad->AbsPixeltoY(pyold);
         }
         if (pB) {
            fX1 = gPad->AbsPixeltoX(pxl);
            fY1 = gPad->AbsPixeltoY(pyt);
            fX2 = gPad->AbsPixeltoX(pxold);
            fY2 = gPad->AbsPixeltoY(pyold);
         }
         if (pC) {
            fX1 = gPad->AbsPixeltoX(pxl);
            fY1 = gPad->AbsPixeltoY(pyold);
            fX2 = gPad->AbsPixeltoX(pxold);
            fY2 = gPad->AbsPixeltoY(pyl);
         }
         if (pD) {
            fX1 = gPad->AbsPixeltoX(pxold);
            fY1 = gPad->AbsPixeltoY(pyold);
            fX2 = gPad->AbsPixeltoX(pxt);
            fY2 = gPad->AbsPixeltoY(pyl);
         }
         if (pTop || pBot || pL || pR || pINSIDE) {
            fX1 = gPad->AbsPixeltoX(px1);
            fY1 = gPad->AbsPixeltoY(py1);
            fX2 = gPad->AbsPixeltoX(px2);
            fY2 = gPad->AbsPixeltoY(py2);
         }
         if (pINSIDE) gPad->ShowGuidelines(this, event, 'i', true);
         if (pTop) gPad->ShowGuidelines(this, event, 't', true);
         if (pBot) gPad->ShowGuidelines(this, event, 'b', true);
         if (pL) gPad->ShowGuidelines(this, event, 'l', true);
         if (pR) gPad->ShowGuidelines(this, event, 'r', true);
         if (pA) gPad->ShowGuidelines(this, event, '1', true);
         if (pB) gPad->ShowGuidelines(this, event, '2', true);
         if (pC) gPad->ShowGuidelines(this, event, '3', true);
         if (pD) gPad->ShowGuidelines(this, event, '4', true);
         gPad->Modified(kTRUE);
      }

      break;

   case kButton1Up:
      if (gROOT->IsEscaped()) {
         gROOT->SetEscape(kFALSE);
         if (opaque) {
            this->SetX1(oldX1);
            this->SetY1(oldY1);
            this->SetX2(oldX2);
            this->SetY2(oldY2);
            gPad->Modified(kTRUE);
            gPad->Update();
         }
         break;
      }

      if (opaque || ropaque) {
         gPad->ShowGuidelines(this, event);
      } else {
         if (px1 < 0 ) break;
         if (pA) {
            fX1 = gPad->AbsPixeltoX(pxold);
            fY1 = gPad->AbsPixeltoY(pyt);
            fX2 = gPad->AbsPixeltoX(pxt);
            fY2 = gPad->AbsPixeltoY(pyold);
         }
         if (pB) {
            fX1 = gPad->AbsPixeltoX(pxl);
            fY1 = gPad->AbsPixeltoY(pyt);
            fX2 = gPad->AbsPixeltoX(pxold);
            fY2 = gPad->AbsPixeltoY(pyold);
         }
         if (pC) {
            fX1 = gPad->AbsPixeltoX(pxl);
            fY1 = gPad->AbsPixeltoY(pyold);
            fX2 = gPad->AbsPixeltoX(pxold);
            fY2 = gPad->AbsPixeltoY(pyl);
         }
         if (pD) {
            fX1 = gPad->AbsPixeltoX(pxold);
            fY1 = gPad->AbsPixeltoY(pyold);
            fX2 = gPad->AbsPixeltoX(pxt);
            fY2 = gPad->AbsPixeltoY(pyl);
         }
         if (pTop || pBot || pL || pR || pINSIDE) {
            fX1 = gPad->AbsPixeltoX(px1);
            fY1 = gPad->AbsPixeltoY(py1);
            fX2 = gPad->AbsPixeltoX(px2);
            fY2 = gPad->AbsPixeltoY(py2);
         }

         if (pINSIDE) {
            // if it was not a pad that was moved then it must have been
            // a box or something like that so we have to redraw the pad
           if (parent == gPad) gPad->Modified(kTRUE);
         }
      }

      if (pA || pB || pC || pD || pTop || pL || pR || pBot) gPad->Modified(kTRUE);

      if (!opaque) {
         gVirtualX->SetLineColor(-1);
         gVirtualX->SetLineWidth(-1);
      }

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
Int_t TBox::IsInside(Double_t x, Double_t y) const
{
   // Function which returns 1 if point x,y lies inside the box, 0 otherwise.

   if (x < fX1 || x > fX2) return 0;
   if (y < fY1 || y > fY2) return 0;
   return 1;
}


//______________________________________________________________________________
void TBox::ls(Option_t *) const
{
   // List this box with its attributes.

   TROOT::IndentLevel();
   printf("%s  X1= %f Y1=%f X2=%f Y2=%f\n",IsA()->GetName(),fX1,fY1,fX2,fY2);
}


//______________________________________________________________________________
void TBox::Paint(Option_t *option)
{
   // Paint this box with its current attributes.

   PaintBox(gPad->XtoPad(fX1),gPad->YtoPad(fY1),gPad->XtoPad(fX2),gPad->YtoPad(fY2),option);
}


//______________________________________________________________________________
void TBox::PaintBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Option_t *option)
{
   // Draw this box with new coordinates.

   TAttLine::Modify();  //Change line attributes only if necessary
   TAttFill::Modify();  //Change fill area attributes only if necessary

   if (option) {
      TString opt = option;
      opt.ToLower();
      if (opt.Contains("l")) gPad->PaintBox(x1,y1,x2,y2,"l");
      else                   gPad->PaintBox(x1,y1,x2,y2);
   } else {
      gPad->PaintBox(x1,y1,x2,y2);
   }
}


//______________________________________________________________________________
void TBox::Print(Option_t *) const
{
   // Dump this box with its attributes.

   printf("%s  X1=%f Y1=%f X2=%f Y2=%f",IsA()->GetName(),fX1,fY1,fX2,fY2);
   if (GetLineColor() != 1) printf(" Color=%d",GetLineColor());
   if (GetLineStyle() != 1) printf(" Style=%d",GetLineStyle());
   if (GetLineWidth() != 1) printf(" Width=%d",GetLineWidth());
   if (GetFillColor() != 0) printf(" FillColor=%d",GetFillColor());
   if (GetFillStyle() != 0) printf(" FillStyle=%d",GetFillStyle());
   printf("\n");
}


//______________________________________________________________________________
void TBox::SavePrimitive(std::ostream &out, Option_t * /*= ""*/)
{
   // Save primitive as a C++ statement(s) on output stream out

   if (gROOT->ClassSaved(TBox::Class())) {
      out<<"   ";
   } else {
      out<<"   TBox *";
   }
   out<<"box = new TBox("<<fX1<<","<<fY1<<","<<fX2<<","<<fY2<<");"<<std::endl;

   SaveFillAttributes(out,"box",0,1001);
   SaveLineAttributes(out,"box",1,1,1);

   out<<"   box->Draw();"<<std::endl;
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
         R__b.ReadClassBuffer(TBox::Class(), this, R__v, R__s, R__c);
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
      R__b.WriteClassBuffer(TBox::Class(),this);
   }
}

//______________________________________________________________________________
Rectangle_t TBox::GetBBox()
{
   // Return the "bounding Box" of the Box

   Rectangle_t BBox;
   Int_t px1, py1, px2, py2;
   px1 = gPad->XtoPixel(fX1);
   px2 = gPad->XtoPixel(fX2);
   py1 = gPad->YtoPixel(fY1);
   py2 = gPad->YtoPixel(fY2);

   Int_t tmp;
   if (px1>px2) { tmp = px1; px1 = px2; px2 = tmp;}
   if (py1>py2) { tmp = py1; py1 = py2; py2 = tmp;}

   BBox.fX = px1;
   BBox.fY = py1;
   BBox.fWidth = px2-px1;
   BBox.fHeight = py2-py1;

   return (BBox);
}


//______________________________________________________________________________
TPoint TBox::GetBBoxCenter()
{
   // Return the center of the Box as TPoint in pixels

   TPoint p;
   p.SetX(gPad->XtoPixel(TMath::Min(fX1,fX2)+0.5*(TMath::Max(fX1, fX2)-TMath::Min(fX1, fX2))));
   p.SetY(gPad->YtoPixel(TMath::Min(fY1,fY2)+0.5*(TMath::Max(fY1, fY2)-TMath::Min(fY1, fY2))));
   return(p);
}


//______________________________________________________________________________
void TBox::SetBBoxCenter(const TPoint &p)
{
   // Set center of the Box

   Double_t w = TMath::Max(fX1, fX2)-TMath::Min(fX1, fX2);
   Double_t h = TMath::Max(fY1, fY2)-TMath::Min(fY1, fY2);
   if (fX2>fX1) {
      this->SetX1(gPad->PixeltoX(p.GetX())-0.5*w);
      this->SetX2(gPad->PixeltoX(p.GetX())+0.5*w);
   }
   else {
      this->SetX2(gPad->PixeltoX(p.GetX())-0.5*w);
      this->SetX1(gPad->PixeltoX(p.GetX())+0.5*w);
   }
   if (fY2>fY1) {
      this->SetY1(gPad->PixeltoY(p.GetY()-gPad->VtoPixel(0))-0.5*h);
      this->SetY2(gPad->PixeltoY(p.GetY()-gPad->VtoPixel(0))+0.5*h);
   }
   else {
      this->SetY2(gPad->PixeltoY(p.GetY()-gPad->VtoPixel(0))-0.5*h);
      this->SetY1(gPad->PixeltoY(p.GetY()-gPad->VtoPixel(0))+0.5*h);
   }
}


//______________________________________________________________________________
void TBox::SetBBoxCenterX(const Int_t x)
{
   // Set X coordinate of the center of the Box

   Double_t w = TMath::Max(fX1, fX2)-TMath::Min(fX1, fX2);
   if (fX2>fX1) {
      this->SetX1(gPad->PixeltoX(x)-0.5*w);
      this->SetX2(gPad->PixeltoX(x)+0.5*w);
   }
   else {
      this->SetX2(gPad->PixeltoX(x)-0.5*w);
      this->SetX1(gPad->PixeltoX(x)+0.5*w);
   }
}


//______________________________________________________________________________
void TBox::SetBBoxCenterY(const Int_t y)
{
   // Set Y coordinate of the center of the Box

   Double_t h = TMath::Max(fY1, fY2)-TMath::Min(fY1, fY2);
   if (fY2>fY1) {
      this->SetY1(gPad->PixeltoY(y-gPad->VtoPixel(0))-0.5*h);
      this->SetY2(gPad->PixeltoY(y-gPad->VtoPixel(0))+0.5*h);
   }
   else {
      this->SetY2(gPad->PixeltoY(y-gPad->VtoPixel(0))-0.5*h);
      this->SetY1(gPad->PixeltoY(y-gPad->VtoPixel(0))+0.5*h);
   }
}

//______________________________________________________________________________
void TBox::SetBBoxX1(const Int_t x)
{
   // Set lefthandside of BoundingBox to a value
   // (resize in x direction on left)
   fX1 = gPad->PixeltoX(x);
}

//______________________________________________________________________________
void TBox::SetBBoxX2(const Int_t x)
{
   // Set righthandside of BoundingBox to a value
   // (resize in x direction on right)
   fX2 = gPad->PixeltoX(x);
}

//_______________________________________________________________________________
void TBox::SetBBoxY1(const Int_t y)
{
   // Set top of BoundingBox to a value (resize in y direction on top)
   fY2 = gPad->PixeltoY(y - gPad->VtoPixel(0));
}

//_______________________________________________________________________________
void TBox::SetBBoxY2(const Int_t y)
{
   // Set bottom of BoundingBox to a value
   // (resize in y direction on bottom)
   fY1 = gPad->PixeltoY(y - gPad->VtoPixel(0));
}
