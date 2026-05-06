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
#include "TLine.h"
#include "TVirtualPad.h"
#include "TClass.h"
#include "TVirtualPadPainter.h"
#include "TCanvasImp.h"
#include "TMath.h"
#include "TPoint.h"


/** \class TLine
\ingroup BasicGraphics

Use the TLine constructor to create a simple line.

~~~ {.cpp}
   TLine(Double_t x1,Double_t y1,Double_t x2,Double_t y2)
~~~

`x1`, `y1`, `x2`, `y2` are the coordinates of the first and the second point.

_**Example**_:

~~~ {.cpp}
   root[] l = new TLine(0.2,0.2,0.8,0.3)
   root[] l->Draw()
~~~
*/


////////////////////////////////////////////////////////////////////////////////
/// Line normal constructor.

TLine::TLine(Double_t x1, Double_t y1, Double_t x2, Double_t  y2)
      :TObject(), TAttLine()
{
   fX1=x1; fY1=y1; fX2=x2; fY2=y2;
}


////////////////////////////////////////////////////////////////////////////////
/// Line copy constructor.

TLine::TLine(const TLine &line) : TObject(line), TAttLine(line), TAttBBox2D(line)
{
   line.TLine::Copy(*this);
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator

TLine &TLine::operator=(const TLine &src)
{
   src.TLine::Copy(*this);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy this line to line.

void TLine::Copy(TObject &obj) const
{
   TObject::Copy(obj);
   TAttLine::Copy(((TLine&)obj));
   ((TLine&)obj).fX1 = fX1;
   ((TLine&)obj).fY1 = fY1;
   ((TLine&)obj).fX2 = fX2;
   ((TLine&)obj).fY2 = fY2;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute distance from point px,py to a line.

Int_t TLine::DistancetoPrimitive(Int_t px, Int_t py)
{
   if (!gPad) return 9999;
   if (!TestBit(kLineNDC)) return DistancetoLine(px,py,gPad->XtoPad(fX1),gPad->YtoPad(fY1),gPad->XtoPad(fX2),gPad->YtoPad(fY2));
   Double_t x1 = gPad->GetX1() + fX1*(gPad->GetX2()-gPad->GetX1());
   Double_t y1 = gPad->GetY1() + fY1*(gPad->GetY2()-gPad->GetY1());
   Double_t x2 = gPad->GetX1() + fX2*(gPad->GetX2()-gPad->GetX1());
   Double_t y2 = gPad->GetY1() + fY2*(gPad->GetY2()-gPad->GetY1());
   return DistancetoLine(px,py,x1,y1,x2,y2);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this line with new coordinates.

TLine *TLine::DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t  y2)
{
   TLine *newline = new TLine(x1, y1, x2, y2);
   TAttLine::Copy(*newline);
   newline->SetBit(kCanDelete);
   newline->AppendPad();
   return newline;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this line with new coordinates in NDC.

TLine *TLine::DrawLineNDC(Double_t x1, Double_t y1, Double_t x2, Double_t  y2)
{
   TLine *newline = DrawLine(x1, y1, x2, y2);
   newline->SetBit(kLineNDC);
   return newline;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute action corresponding to one event.
///  This member function is called when a line is clicked with the locator
///
///  If Left button clicked on one of the line end points, this point
///     follows the cursor until button is released.
///
///  if Middle button clicked, the line is moved parallel to itself
///     until the button is released.

void TLine::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   if (!gPad || !gPad->IsEditable()) return;

   constexpr Int_t kMaxDiff = 20;
   static Int_t px1,px2,py1,py2,pxold,pyold;
   static Double_t oldX1, oldY1, oldX2, oldY2;
   static Int_t selectPoint;

   auto &parent = *gPad;

   Bool_t opaque  = parent.OpaqueMoving();

   auto action = [this, &parent](Int_t code, Int_t _x1, Int_t _y1, Int_t _x2 = 0, Int_t _y2 = 0) {
      Double_t x1, y1, x2, y2;

      Bool_t isndc = TestBit(kLineNDC);

      if (isndc) {
         x1 = (1. * _x1 / parent.GetWw() - parent.GetAbsXlowNDC()) / parent.GetAbsWNDC();
         y1 = ((1 - 1. * _y1 / parent.GetWh()) - parent.GetAbsYlowNDC()) / parent.GetAbsHNDC();
         x2 = (1. * _x2 / parent.GetWw() - parent.GetAbsXlowNDC()) / parent.GetAbsWNDC();
         y2 = ((1 - 1. * _y2 / parent.GetWh()) - parent.GetAbsYlowNDC()) / parent.GetAbsHNDC();
      } else {
         x1 = parent.AbsPixeltoX(_x1);
         y1 = parent.AbsPixeltoY(_y1);
         x2 = parent.AbsPixeltoX(_x2);
         y2 = parent.AbsPixeltoY(_y2);
      }
      if (code == 0) {
         auto pp = parent.GetPainter();
         pp->SetAttLine(*this);
         if (isndc)
            pp->DrawLineNDC(x1, y1, x2, y2);
         else
            pp->DrawLine(x1, y1, x2, y2);
      } else {
         if (!isndc) {
            x1 = parent.PadtoX(x1);
            x2 = parent.PadtoX(x2);
            y1 = parent.PadtoY(y1);
            y2 = parent.PadtoY(y2);
         }

         if (code & 1) {
            SetX1(x1);
            SetY1(y1);
         }
         if (code & 2) {
            SetX2(x2);
            SetY2(y2);
         }
         if (TestBit(kVertical)) {
            if (code & 1)
               SetX2(GetX1());
            else
               SetX1(GetX2());
         }
         if (TestBit(kHorizontal)) {
            if (code & 1)
               SetY2(GetY1());
            else
               SetY1(GetY2());
         }
      }
   };

   switch (event) {

   case kArrowKeyPress:
   case kButton1Down:
      oldX1 = GetX1();
      oldY1 = GetY1();
      oldX2 = GetX2();
      oldY2 = GetY2();

      // No break !!!

   case kMouseMotion:

      if (TestBit(kLineNDC)) {
         px1 = parent.UtoAbsPixel(GetX1());
         py1 = parent.VtoAbsPixel(GetY1());
         px2 = parent.UtoAbsPixel(GetX2());
         py2 = parent.VtoAbsPixel(GetY2());
      } else {
         px1 = parent.XtoAbsPixel(parent.XtoPad(GetX1()));
         py1 = parent.YtoAbsPixel(parent.YtoPad(GetY1()));
         px2 = parent.XtoAbsPixel(parent.XtoPad(GetX2()));
         py2 = parent.YtoAbsPixel(parent.YtoPad(GetY2()));
      }

      //simply take sum of pixels differences
      if (abs(px1 - px) + abs(py1 - py) < kMaxDiff) { //*-*================>OK take point number 1
         selectPoint = 1;
         parent.SetCursor(kPointer);
      } else if (abs(px2 - px) + abs(py2 - py) < kMaxDiff) { //*-*================>OK take point number 2
         selectPoint = 2;
         parent.SetCursor(kPointer);
      } else {
         selectPoint = 3;
         pxold = px;
         pyold = py;
         parent.SetCursor(kMove);
      }

      break;

   case kArrowKeyRelease:
   case kButton1Motion:
      if (!opaque)
         action(0, px1, py1, px2, py2);
      if (selectPoint == 1) {
         px1 = px;
         py1 = py;
      } else if (selectPoint == 2) {
         px2 = px;
         py2 = py;
      } else if (selectPoint == 3) {
         px1 += px - pxold;
         py1 += py - pyold;
         px2 += px - pxold;
         py2 += py  -pyold;
         pxold = px;
         pyold = py;
      }
      action(!opaque ? 0 : selectPoint, px1, py1, px2, py2);
      if (opaque) {
         if (selectPoint == 1) {
            //check in which corner the BBox is edited
            if (GetX1() > GetX2())
               parent.ShowGuidelines(this, event, GetY1() > GetY2() ? '2' : '3', true);
            else
               parent.ShowGuidelines(this, event, GetY1() > GetY2() ? '1' : '4', true);
         } else if (selectPoint == 2) {
            //check in which corner the BBox is edited
            if (GetX1() > GetX2())
               parent.ShowGuidelines(this, event, GetY1() > GetY2() ? '4' : '1', true);
            else
               parent.ShowGuidelines(this, event, GetY1() > GetY2() ? '3' : '2', true);
         } else if (selectPoint == 3) {
            parent.ShowGuidelines(this, event, 'i', true);
         }
         parent.ModifiedUpdate();
      }
      break;

   case kButton1Up:

      if (gROOT->IsEscaped()) {
         gROOT->SetEscape(kFALSE);
         if (opaque) {
            SetX1(oldX1);
            SetY1(oldY1);
            SetX2(oldX2);
            SetY2(oldY2);
            parent.ShowGuidelines(this, event);
            parent.ModifiedUpdate();
         }
         break;
      }
      if (opaque) {
         parent.ShowGuidelines(this, event);
      } else {
         action(selectPoint, px1, py1, px2, py2);
         parent.ModifiedUpdate();
      }
      selectPoint = 0;
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
/// Get the slope of this TLine

Double_t TLine::GetSlope() const
{
   Double_t m = 0;
   if (fX2 == fX1) {
      Error("GetSlope", "This line is vertical. The slope in undefined");
      m = TMath::Infinity();
   } else {
      m = (fY2-fY1)/(fX2-fX1);
   }
   return m;
}

////////////////////////////////////////////////////////////////////////////////
/// Get the Y-Intercept of this TLine

Double_t TLine::GetYIntercept() const
{
   Double_t b = 0;
   if (fX2 == fX1) {
      Error("GetYIntercept", "This line is vertical. The Y-Intercept in undefined");
      b = TMath::Infinity();
   } else {
      b = (fY1*fX2-fY2*fX1)/(fX2-fX1);
   }
   return b;
}

////////////////////////////////////////////////////////////////////////////////
/// List this line with its attributes.

void TLine::ls(Option_t *) const
{
   TROOT::IndentLevel();
   printf("%s  X1=%f Y1=%f X2=%f Y2=%f\n",IsA()->GetName(),fX1,fY1,fX2,fY2);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint this line with its current attributes.

void TLine::Paint(Option_t *)
{
   if (!gPad) return;
   if (TestBit(kLineNDC)) PaintLineNDC(fX1,fY1,fX2,fY2);
   else                   PaintLine(gPad->XtoPad(fX1),gPad->YtoPad(fY1),gPad->XtoPad(fX2),gPad->YtoPad(fY2));
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this line with new coordinates.

void TLine::PaintLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   if (!gPad) return;
   TAttLine::Modify();  //Change line attributes only if necessary
   gPad->PaintLine(x1,y1,x2,y2);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this line with new coordinates in NDC.

void TLine::PaintLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2)
{
   if (!gPad) return;
   TAttLine::Modify();  //Change line attributes only if necessary
   gPad->PaintLineNDC(u1,v1,u2,v2);
}

////////////////////////////////////////////////////////////////////////////////
/// Dump this line with its attributes.

void TLine::Print(Option_t *) const
{
   printf("%s  X1=%f Y1=%f X2=%f Y2=%f",IsA()->GetName(),fX1,fY1,fX2,fY2);
   if (GetLineColor() != 1) printf(" Color=%d",GetLineColor());
   if (GetLineStyle() != 1) printf(" Style=%d",GetLineStyle());
   if (GetLineWidth() != 1) printf(" Width=%d",GetLineWidth());
   printf("\n");
}

////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TLine::SavePrimitive(std::ostream &out, Option_t *option)
{
   SavePrimitiveConstructor(out, Class(), "line", TString::Format("%g, %g, %g, %g", fX1, fY1, fX2, fY2), kFALSE);

   SaveLineAttributes(out, "line", 1, 1, 1);

   if (TestBit(kLineNDC))
      out << "   line->SetNDC();\n";

   if (TestBit(kVertical))
      out << "   line->SetBit(TLine::kVertical);\n";

   if (TestBit(kHorizontal))
      out << "   line->SetBit(TLine::kHorizontal);\n";

   SavePrimitiveDraw(out, "line", option);
}

////////////////////////////////////////////////////////////////////////////////
/// Check whether this line is to be drawn horizontally.

Bool_t TLine::IsHorizontal()
{
   return TestBit(kHorizontal);
}

////////////////////////////////////////////////////////////////////////////////
/// Check whether this line is to be drawn vertically.

Bool_t TLine::IsVertical()
{
   return TestBit(kVertical);
}

////////////////////////////////////////////////////////////////////////////////
/// Set NDC mode on if isNDC = kTRUE, off otherwise

void TLine::SetNDC(Bool_t isNDC)
{
   SetBit(kLineNDC, isNDC);
}

////////////////////////////////////////////////////////////////////////////////
/// Force the line to be drawn horizontally.
/// Makes fY2 equal to fY1. The line length is kept.
/// TArrow and TGaxis also get this function by inheritance.

void TLine::SetHorizontal(Bool_t set /*= kTRUE*/)
{
   SetBit(kHorizontal, set);
   if (set && gPad) {
      SetVertical(kFALSE);
      Int_t px1 = gPad->XtoAbsPixel(fX1);
      Int_t px2 = gPad->XtoAbsPixel(fX2);
      Int_t py1 = gPad->YtoAbsPixel(fY1);
      Int_t py2 = gPad->YtoAbsPixel(fY2);
      Int_t l   = Int_t(TMath::Sqrt((px2-px1)*(px2-px1)+(py2-py1)*(py2-py1)));
      if (fX2 >= fX1) fX2 = gPad->AbsPixeltoX(px1+l);
      else            fX2 = gPad->AbsPixeltoX(px1-l);
      fY2 = fY1;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Force the line to be drawn vertically.
/// Makes fX2 equal to fX1. The line length is kept.
/// TArrow and TGaxis also get this function by inheritance.

void TLine::SetVertical(Bool_t set /*= kTRUE*/)
{
   SetBit(kVertical, set);
   if (set && gPad) {
      SetHorizontal(kFALSE);
      Int_t px1 = gPad->XtoAbsPixel(fX1);
      Int_t px2 = gPad->XtoAbsPixel(fX2);
      Int_t py1 = gPad->YtoAbsPixel(fY1);
      Int_t py2 = gPad->YtoAbsPixel(fY2);
      Int_t l   = Int_t(TMath::Sqrt((px2-px1)*(px2-px1)+(py2-py1)*(py2-py1)));
      if (fY2 >= fY1) fY2 = gPad->AbsPixeltoY(py1-l);
      else            fY2 = gPad->AbsPixeltoY(py1+l);
      fX2 = fX1;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class TLine.

void TLine::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TLine::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TObject::Streamer(R__b);
      TAttLine::Streamer(R__b);
      Float_t x1,y1,x2,y2;
      R__b >> x1; fX1 = x1;
      R__b >> y1; fY1 = y1;
      R__b >> x2; fX2 = x2;
      R__b >> y2; fY2 = y2;
      //====end of old versions

   } else {
      R__b.WriteClassBuffer(TLine::Class(),this);
   }
}
////////////////////////////////////////////////////////////////////////////////
/// Return the bounding Box of the Line

Rectangle_t TLine::GetBBox()
{
   Rectangle_t BBox{0, 0, 0, 0};
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
/// Set X coordinate of the center of the BoundingBox

void TLine::SetBBoxCenterX(const Int_t x)
{
   Double_t w2 = 0.5 * (fX2 - fX1);
   Double_t midx = GetXCoord(x, TestBit(kLineNDC));
   SetX1(midx - w2);
   SetX2(midx + w2);
}

////////////////////////////////////////////////////////////////////////////////
/// Set Y coordinate of the center of the BoundingBox

void TLine::SetBBoxCenterY(const Int_t y)
{
   Double_t h2 = 0.5 * (fY2 - fY1);
   Double_t midy = GetYCoord(y, TestBit(kLineNDC));
   SetY1(midy - h2);
   SetY2(midy + h2);
}

////////////////////////////////////////////////////////////////////////////////
/// Set left hand side of BoundingBox to a value
/// (resize in x direction on left)

void TLine::SetBBoxX1(const Int_t x)
{
   auto xx = GetXCoord(x, TestBit(kLineNDC));
   if (fX2 > fX1)
      SetX1(xx);
   else
      SetX2(xx);
}

////////////////////////////////////////////////////////////////////////////////
/// Set right hand side of BoundingBox to a value
/// (resize in x direction on right)

void TLine::SetBBoxX2(const Int_t x)
{
   auto xx = GetXCoord(x, TestBit(kLineNDC));
   if (fX2 > fX1)
      SetX2(xx);
   else
      SetX1(xx);
}

////////////////////////////////////////////////////////////////////////////////
/// Set top of BoundingBox to a value (resize in y direction on top)

void TLine::SetBBoxY1(const Int_t y)
{
   auto yy = GetYCoord(y, TestBit(kLineNDC));
   if (fY2 > fY1)
      SetY2(yy);
   else
      SetY1(yy);
}

////////////////////////////////////////////////////////////////////////////////
/// Set bottom of BoundingBox to a value
/// (resize in y direction on bottom)

void TLine::SetBBoxY2(const Int_t y)
{
   auto yy = GetYCoord(y, TestBit(kLineNDC));
   if (fY2 > fY1)
      SetY1(yy);
   else
      SetY2(yy);
}
