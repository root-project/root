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
#include "TROOT.h"
#include "TDiamond.h"
#include "TVirtualPad.h"
#include "TBoxInteractive.h"
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

TDiamond::TDiamond(const TDiamond &diamond) : TPaveText(diamond)
{
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

class TDiamondInteractive : public TBoxInteractive {
   public:
      using TBoxInteractive::TBoxInteractive;

      void PaintOutline(TVirtualPad &parent) override
      {
         Double_t xd[5] = { (newX1 + newX2) / 2, newX1, (newX1 + newX2) / 2, newX2, (newX1 + newX2) / 2 };
         Double_t yd[5] = { newY2, (newY1 + newY2)/2, newY1, (newY1 + newY2)/2, newY2 };

         // "i" is interactive painting, "diamond" is id
         parent.PaintPolyLine(5, xd, yd, "idiamond");
      }
};


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

   auto inter = dynamic_cast<TDiamondInteractive *>(parent.Interactive(this));

   auto setNewValues = [&inter, this]() {
      SetX1(inter->newX1);
      SetX2(inter->newX2);
      SetY1(inter->newY1);
      SetY2(inter->newY2);
   };

   switch (event) {

   case kArrowKeyPress:
   case kButton1Down:

      inter = new TDiamondInteractive(kFALSE, GetX1(), GetY1(), GetX2(), GetY2());
      parent.Interactive(this, inter);

      // No break !!!

   case kMouseMotion: {

      TDiamondInteractive dummy(kFALSE);
      if (!inter) inter = &dummy;
      inter->CalcPixelCoord(parent, GetX1(), GetY1(), GetX2(), GetY2());

      if (!inter->SelectDiamondCorner(px, py)) {
         // refuse interactive changes
         parent.Interactive();
      } else {
         inter->SetCursor(parent, event == kButton1Down);
         fResizing = inter->IsResizing() && (event != kMouseMotion);
      }

      break;
   }

   case kArrowKeyRelease:
   case kButton1Motion: {

      if (!inter)
         return;

      if (!inter->ProcessMouseMove(parent, px, py))
         return;

      inter->ApplyChanges(parent);

      if (inter->IsOpaque(parent)) {
         setNewValues();
         parent.ShowGuidelines(this, event, inter->GetGuideChar(), true);
         parent.Modified(kTRUE);
      }

      break;
   }

   case kButton1Up:

      if (inter && inter->IsOpaque(parent))
         parent.ShowGuidelines(this, event);

      if (gROOT->IsEscaped()) {
         gROOT->SetEscape(kFALSE);
         if (inter && inter->IsOpaque(parent)) {
            SetX1(inter->oldX1);
            SetY1(inter->oldY1);
            SetX2(inter->oldX2);
            SetY2(inter->oldY2);
         }
      } else if (inter && !inter->IsOpaque(parent) && (inter->newX1 != inter->newX2)) {
         setNewValues();
      }

      parent.Modified();
      parent.Interactive(); // delete interactive object
      fResizing = kFALSE;

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
