// @(#)root/graf:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <cstdlib>

#include <iostream>
#include "TROOT.h"
#include "TBuffer.h"
#include "TBox.h"
#include "TVirtualPad.h"
#include "TVirtualPadPainter.h"
#include "TCanvasImp.h"
#include "TClass.h"
#include "TMath.h"
#include "TPoint.h"


/** \class TBox
\ingroup BasicGraphics

Create a Box.

A box is defined by :

- Its bottom left coordinates x1,y1
- Its top right coordinates x2,y2

A box has line attributes (see TAttLine) and fill area attributes (see TAttFill).
*/

////////////////////////////////////////////////////////////////////////////////
/// Box default constructor.

TBox::TBox(): TObject(), TAttLine(), TAttFill()
{
   fTip      = nullptr;
   fX1       = 0.;
   fY1       = 0.;
   fX2       = 0.;
   fY2       = 0.;
   fResizing = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Box standard constructor.

TBox::TBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
     : TObject(), TAttLine(), TAttFill()
{
   if (x2 >= x1) {fX1  =x1; fX2 = x2;}
   else          {fX1 = x2; fX2 = x1;}
   if (y2 >= y1) {fY1  =y1; fY2 = y2;}
   else          {fY1 = y2; fY2 = y1;}
   fResizing = kFALSE;
   fTip = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Box destructor.

TBox::~TBox()
{
   if (fTip && gPad) {
      gPad->CloseToolTip(fTip);
      gPad->DeleteToolTip(fTip);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Box copy constructor.

TBox::TBox(const TBox &box) : TObject(box), TAttLine(box), TAttFill(box), TAttBBox2D(box)
{
   fX1       = 0.;
   fY1       = 0.;
   fX2       = 0.;
   fY2       = 0.;
   fResizing = kTRUE;
   ((TBox&)box).TBox::Copy(*this);
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.

TBox& TBox::operator=(const TBox& b)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Copy a Box.

void TBox::Copy(TObject &obj) const
{
   TObject::Copy(obj);
   TAttLine::Copy(((TBox&)obj));
   TAttFill::Copy(((TBox&)obj));
   ((TBox&)obj).fX1 = fX1;
   ((TBox&)obj).fY1 = fY1;
   ((TBox&)obj).fX2 = fX2;
   ((TBox&)obj).fY2 = fY2;
   ((TBox&)obj).fResizing = fResizing;
   ((TBox&)obj).fTip = nullptr;   //FIXME
}

////////////////////////////////////////////////////////////////////////////////
/// Compute distance from point px,py to a box.
///
/// Compute the closest distance of approach from point px,py to the
/// edges of this box.
/// The distance is computed in pixels units.
///
/// In case of a filled box the distance returned is 0 if the point
/// (px,py) is inside the box, and is huge if the point is outside.

Int_t TBox::DistancetoPrimitive(Int_t px, Int_t py)
{
   if (!gPad) return 9999;
   Int_t pxl, pyl, pxt, pyt;
   Int_t px1 = gPad->XtoAbsPixel(fX1);
   Int_t py1 = gPad->YtoAbsPixel(fY1);
   Int_t px2 = gPad->XtoAbsPixel(fX2);
   Int_t py2 = gPad->YtoAbsPixel(fY2);

   Bool_t isBox = !(InheritsFrom("TPave") || InheritsFrom("TWbox"));

   if (isBox) {
      if (gPad->GetLogx()) {
        if (fX1>0) px1 = gPad->XtoAbsPixel(TMath::Log10(fX1));
        if (fX2>0) px2 = gPad->XtoAbsPixel(TMath::Log10(fX2));
      }
      if (gPad->GetLogy()) {
        if (fY1>0) py1 = gPad->YtoAbsPixel(TMath::Log10(fY1));
        if (fY2>0) py2 = gPad->YtoAbsPixel(TMath::Log10(fY2));
      }
   }

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
   if (py < pyl) dxl += pyl - py;
   if (py > pyt) dxl += py - pyt;
   Int_t dxt = TMath::Abs(px - pxt);
   if (py < pyl) dxt += pyl - py;
   if (py > pyt) dxt += py - pyt;
   Int_t dyl = TMath::Abs(py - pyl);
   if (px < pxl) dyl += pxl - px;
   if (px > pxt) dyl += px - pxt;
   Int_t dyt = TMath::Abs(py - pyt);
   if (px < pxl) dyt += pxl - px;
   if (px > pxt) dyt += px - pxt;

   Int_t distance = dxl;
   if (dxt < distance) distance = dxt;
   if (dyl < distance) distance = dyl;
   if (dyt < distance) distance = dyt;

   return distance - Int_t(0.5*fLineWidth);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this box with its current attributes.
/// if the box has no fill style (ie fill style=0), the box contour is drawn
/// if the box has a fill style, the box contour is not drawn by default.
/// to force the contour to be drawn, specify option "l"

void TBox::Draw(Option_t *option)
{
   AppendPad(option);

}

////////////////////////////////////////////////////////////////////////////////
/// Draw this box with new coordinates.

TBox *TBox::DrawBox(Double_t x1, Double_t y1,Double_t x2, Double_t  y2)
{
   TBox *newbox = new TBox(x1,y1,x2,y2);
   TAttLine::Copy(*newbox);
   TAttFill::Copy(*newbox);
   newbox->SetBit(kCanDelete);
   newbox->AppendPad();
   return newbox;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute action corresponding to one event.
///
///  This member function is called when a BOX/WBOX/PAD object is clicked.
///
///  If the mouse is clicked in one of the 4 corners of the box (pA,pB,pC,pD)
///  the box is resized with the rubber rectangle.
///
///  If the mouse is clicked inside the box, the box is moved.
///
///  If the mouse is clicked on the 4 edges (pL,pR,pTop,pBot), the box is
///  rescaled parallel to this edge (same as Motif window manager).
///
///  Note that this function is duplicated on purpose by TPad::ExecuteEvent.
///  If somebody modifies this function, may be similar changes should also
///  be applied to TPad::ExecuteEvent.

void TBox::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   if (TestBit(kCannotMove)) return;

   if (!gPad) return;

   auto &parent = *gPad;

   if (!parent.IsEditable() && event != kMouseEnter) return;

   Bool_t isBox = !(InheritsFrom("TPave") || InheritsFrom("TWbox"));

   constexpr Int_t kMaxDiff = 7;
   constexpr Int_t kMinSize = 20;

   static Int_t px1, px2, py1, py2, dpx1, dpy2;
   static Int_t px1p, px2p, py1p, py2p;
   static Double_t oldX1, oldY1, oldX2, oldY2;
   static Bool_t hasOld = kFALSE;
   static enum { pNone, pA, pB, pC, pD, pTop, pL, pR, pBot, pINSIDE } mode = pNone;
   static bool firstPaint = kFALSE;
   Bool_t opaque  = parent.OpaqueMoving();
   Bool_t ropaque = parent.OpaqueResizing();

   // convert to user coordinates and either paint ot set back
   auto paint_or_set = [&parent,isBox,this](Bool_t paint)
   {
      auto x1 = parent.AbsPixeltoX(px1);
      auto y1 = parent.AbsPixeltoY(py1);
      auto x2 = parent.AbsPixeltoX(px2);
      auto y2 = parent.AbsPixeltoY(py2);
      if (!paint) {
         if (isBox) {
            x1 = parent.PadtoX(x1);
            x2 = parent.PadtoX(x2);
            y1 = parent.PadtoY(y1);
            y2 = parent.PadtoY(y2);
         }
         SetX1(x1);
         SetY1(y1);
         SetX2(x2);
         SetY2(y2);
      } else if (firstPaint) {
         firstPaint = kFALSE;
      } else {
         auto pp = parent.GetPainter();
         pp->SetAttLine({GetFillColor() > 0 ? GetFillColor() : (Color_t) 1, GetLineStyle(), 2});
         pp->DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kHollow);
      }
   };

   HideToolTip(event);

   switch (event) {

   case kMouseEnter:
      if (fTip) parent.ResetToolTip(fTip);
      break;

   case kArrowKeyPress:
   case kButton1Down:

      oldX1 = fX1;
      oldY1 = fY1;
      oldX2 = fX2;
      oldY2 = fY2;
      hasOld = kTRUE;

      // No break !!!

   case kMouseMotion:

      px1 = parent.XtoAbsPixel(isBox ? parent.XtoPad(GetX1()) : GetX1());
      py1 = parent.YtoAbsPixel(isBox ? parent.YtoPad(GetY1()) : GetY1());
      px2 = parent.XtoAbsPixel(isBox ? parent.XtoPad(GetX2()) : GetX2());
      py2 = parent.YtoAbsPixel(isBox ? parent.YtoPad(GetY2()) : GetY2());
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

      mode = pNone;

      if (TMath::Abs(px - px1) <= kMaxDiff && TMath::Abs(py - py2) <= kMaxDiff) {
         mode = pA;
         parent.SetCursor(kTopLeft);
      } else if (TMath::Abs(px - px2) <= kMaxDiff && TMath::Abs(py - py2) <= kMaxDiff) {
         mode = pB;
         parent.SetCursor(kTopRight);
      } else if (TMath::Abs(px - px2) <= kMaxDiff && TMath::Abs(py - py1) <= kMaxDiff) {
         mode = pC;
         parent.SetCursor(kBottomRight);
      } else if (TMath::Abs(px - px1) <= kMaxDiff && TMath::Abs(py - py1) <= kMaxDiff) {
         mode = pD;
         parent.SetCursor(kBottomLeft);
      } else if ((px > px1 + kMaxDiff && px < px2 - kMaxDiff) && TMath::Abs(py - py2) < kMaxDiff) {
         mode = pTop;
         parent.SetCursor(kTopSide);
      } else if ((px > px1 + kMaxDiff && px < px2 - kMaxDiff) && TMath::Abs(py - py1) < kMaxDiff) {
         mode = pBot;
         parent.SetCursor(kBottomSide);
      } else if ((py > py2 + kMaxDiff && py < py1 - kMaxDiff) && TMath::Abs(px - px1) < kMaxDiff) {
         mode = pL;
         parent.SetCursor(kLeftSide);
      } else if ((py > py2 + kMaxDiff && py < py1 - kMaxDiff) && TMath::Abs(px - px2) < kMaxDiff) {
         mode = pR;
         parent.SetCursor(kRightSide);
      } else if ((px > px1+kMaxDiff && px < px2-kMaxDiff) && (py > py2+kMaxDiff && py < py1-kMaxDiff)) {
         dpx1 = px - px1; // cursor position relative to top-left corner
         dpy2 = py - py2;
         mode = pINSIDE;
         if (event == kButton1Down)
            parent.SetCursor(kMove);
         else
            parent.SetCursor(kCross);
      }

      fResizing = (mode != pNone) && (mode != pINSIDE);
      firstPaint = kTRUE;
      if (mode == pNone)
         parent.SetCursor(kCross);

      break;

   case kArrowKeyRelease:
   case kButton1Motion:

      switch (mode) {
      case pNone:
         return;
      case pA:
         if (!ropaque) paint_or_set(kTRUE);
         px1 = TMath::Max(px1p, TMath::Min(px, px2 - kMinSize));
         py2 = TMath::Max(py2p, TMath::Min(py, py1 - kMinSize));
         paint_or_set(!ropaque);
         break;
      case pB:
         if (!ropaque) paint_or_set(kTRUE);
         px2 = TMath::Min(px2p, TMath::Max(px, px1 + kMinSize));
         py2 = TMath::Max(py2p, TMath::Min(py, py1 - kMinSize));
         paint_or_set(!ropaque);
         break;
      case pC:
         if (!ropaque) paint_or_set(kTRUE);
         px2 = TMath::Min(px2p, TMath::Max(px, px1 + kMinSize));
         py1 = TMath::Min(py1p, TMath::Max(py, py2 + kMinSize));
         paint_or_set(!ropaque);
         break;
      case pD:
         if (!ropaque) paint_or_set(kTRUE);
         px1 = TMath::Max(px1p, TMath::Min(px, px2 - kMinSize));
         py1 = TMath::Min(py1p, TMath::Max(py, py2 + kMinSize));
         paint_or_set(!ropaque);
         break;
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
            case pTop: parent.ShowGuidelines(this, event, 't', true); break;
            case pBot: parent.ShowGuidelines(this, event, 'b', true); break;
            case pL: parent.ShowGuidelines(this, event, 'l', true); break;
            case pR: parent.ShowGuidelines(this, event, 'r', true); break;
            case pA: parent.ShowGuidelines(this, event, '1', true); break;
            case pB: parent.ShowGuidelines(this, event, '2', true); break;
            case pC: parent.ShowGuidelines(this, event, '3', true); break;
            case pD: parent.ShowGuidelines(this, event, '4', true); break;
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
/// Hide tool tip depending on the event type. Typically tool tips
/// are hidden when event is not a kMouseEnter and not a kMouseMotion
/// event.

void TBox::HideToolTip(Int_t event)
{
   if (event != kMouseEnter && event != kMouseMotion && fTip && gPad)
      gPad->CloseToolTip(fTip);
}

////////////////////////////////////////////////////////////////////////////////
/// Function which returns 1 if point x,y lies inside the box, 0 otherwise.

Int_t TBox::IsInside(Double_t x, Double_t y) const
{
   if (x < fX1 || x > fX2) return 0;
   if (y < fY1 || y > fY2) return 0;
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// List this box with its attributes.

void TBox::ls(Option_t *) const
{
   TROOT::IndentLevel();
   printf("%s  X1= %f Y1=%f X2=%f Y2=%f\n",IsA()->GetName(),fX1,fY1,fX2,fY2);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint this box with its current attributes.

void TBox::Paint(Option_t *option)
{
   if(gPad) PaintBox(gPad->XtoPad(fX1),gPad->YtoPad(fY1),gPad->XtoPad(fX2),gPad->YtoPad(fY2),option);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this box with new coordinates.

void TBox::PaintBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Option_t *option)
{
   if (!gPad) return;

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

////////////////////////////////////////////////////////////////////////////////
/// Dump this box with its attributes.

void TBox::Print(Option_t *) const
{
   printf("%s  X1=%f Y1=%f X2=%f Y2=%f",IsA()->GetName(),fX1,fY1,fX2,fY2);
   if (GetLineColor() != 1) printf(" Color=%d",GetLineColor());
   if (GetLineStyle() != 1) printf(" Style=%d",GetLineStyle());
   if (GetLineWidth() != 1) printf(" Width=%d",GetLineWidth());
   if (GetFillColor() != 0) printf(" FillColor=%d",GetFillColor());
   if (GetFillStyle() != 0) printf(" FillStyle=%d",GetFillStyle());
   printf("\n");
}

////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TBox::SavePrimitive(std::ostream &out, Option_t *option)
{
   SavePrimitiveConstructor(out, Class(), "box", TString::Format("%g, %g, %g, %g", fX1, fY1, fX2, fY2));

   SaveFillAttributes(out, "box", -1, -1);
   SaveLineAttributes(out, "box", 1, 1, 1);

   SavePrimitiveDraw(out, "box", option);
}

////////////////////////////////////////////////////////////////////////////////
/// Set tool tip text associated with this box. The delay is in
/// milliseconds (minimum 250). To remove tool tip call method with
/// text = 0.

void TBox::SetToolTipText(const char *text, Long_t delayms)
{
   if (!gPad) {
      Warning("SetToolTipText", "a canvas must exist before setting the tool tip text");
      return;
   }

   if (fTip) {
      gPad->DeleteToolTip(fTip);
      fTip = nullptr;
   }

   if (text && strlen(text))
      fTip = gPad->CreateToolTip(this, text, delayms);
}

////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class TBox.

void TBox::Streamer(TBuffer &R__b)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Return the "bounding Box" of the Box

Rectangle_t TBox::GetBBox()
{
   Rectangle_t BBox{0,0,0,0};
   if (gPad) {
      Int_t px1 = gPad->XtoPixel(fX1);
      Int_t px2 = gPad->XtoPixel(fX2);
      Int_t py1 = gPad->YtoPixel(fY1);
      Int_t py2 = gPad->YtoPixel(fY2);

      if (px1 > px2)
         std::swap(px1, px2);
      if (py1 > py2)
         std::swap(py1, py2);

      BBox.fX = px1;
      BBox.fY = py1;
      BBox.fWidth = px2 - px1;
      BBox.fHeight = py2 - py1;
   }

   return BBox;
}

////////////////////////////////////////////////////////////////////////////////
/// Set X coordinate of the center of the Box

void TBox::SetBBoxCenterX(const Int_t x)
{
   Double_t w2 = 0.5* (fX2 - fX1);
   Double_t midx = GetXCoord(x);
   SetX1(midx - w2);
   SetX2(midx + w2);
}

////////////////////////////////////////////////////////////////////////////////
/// Set Y coordinate of the center of the Box

void TBox::SetBBoxCenterY(const Int_t y)
{
   Double_t h2 = 0.5 * (fY2 - fY1);
   Double_t midy = GetYCoord(y);
   SetY1(midy - h2);
   SetY2(midy + h2);
}

////////////////////////////////////////////////////////////////////////////////
/// Set left hand side of BoundingBox to a value
/// (resize in x direction on left)

void TBox::SetBBoxX1(const Int_t x)
{
   SetX1(GetXCoord(x));
}

////////////////////////////////////////////////////////////////////////////////
/// Set right hand side of BoundingBox to a value
/// (resize in x direction on right)

void TBox::SetBBoxX2(const Int_t x)
{
   SetX2(GetXCoord(x));
}

////////////////////////////////////////////////////////////////////////////////
/// Set top of BoundingBox to a value (resize in y direction on top)

void TBox::SetBBoxY1(const Int_t y)
{
   SetY2(GetYCoord(y));
}

////////////////////////////////////////////////////////////////////////////////
/// Set bottom of BoundingBox to a value
/// (resize in y direction on bottom)

void TBox::SetBBoxY2(const Int_t y)
{
   SetY1(GetYCoord(y));
}
