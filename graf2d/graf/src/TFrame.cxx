// @(#)root/graf:$Id$
// Author: Rene Brun   31/10/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "TVirtualPad.h"
#include "TFrame.h"
#include "TStyle.h"

ClassImp(TFrame)

//______________________________________________________________________________
//
// a TFrame is a TWbox for drawing histogram frames.
//

//______________________________________________________________________________
TFrame::TFrame(): TWbox()
{
   // Frame default constructor.

}

//______________________________________________________________________________
TFrame::TFrame(Double_t x1, Double_t y1,Double_t x2, Double_t  y2)
       :TWbox(x1,y1,x2,y2)
{
   // Frame normal constructor.

}

//______________________________________________________________________________
TFrame::TFrame(const TFrame &frame) : TWbox(frame)
{
   // Frame copy constructor.

   ((TFrame&)frame).Copy(*this);
}

//______________________________________________________________________________
TFrame::~TFrame()
{
   // Frame default destructor.

}

//______________________________________________________________________________
void TFrame::Copy(TObject &frame) const
{
   // Copy this frame to frame.

   TWbox::Copy(frame);
}

//______________________________________________________________________________
void TFrame::Draw(Option_t *option)
{
   // Draw this frame with its current attributes.

   AppendPad(option);
}

//______________________________________________________________________________
void TFrame::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute action corresponding to one event.
   //
   //  This member function is called when a TFrame object is clicked.

   if (!gPad->IsEditable()) return;

   TWbox::ExecuteEvent(event, px, py);

   if (event != kButton1Up) return;
   // update pad margins
   Double_t xmin         = gPad->GetUxmin();
   Double_t xmax         = gPad->GetUxmax();
   Double_t ymin         = gPad->GetUymin();
   Double_t ymax         = gPad->GetUymax();
   Double_t dx           = xmax-xmin;
   Double_t dy           = ymax-ymin;
   Double_t leftMargin   = (fX1-gPad->GetX1())/(gPad->GetX2()-gPad->GetX1());
   Double_t topMargin    = (gPad->GetY2()-fY2)/(gPad->GetY2()-gPad->GetY1());
   Double_t rightMargin  = (gPad->GetX2()-fX2)/(gPad->GetX2()-gPad->GetX1());
   Double_t bottomMargin = (fY1-gPad->GetY1())/(gPad->GetY2()-gPad->GetY1());
   // margin may get very small negative values
   if (leftMargin   < 0) leftMargin   = 0;
   if (topMargin    < 0) topMargin    = 0;
   if (rightMargin  < 0) rightMargin  = 0;
   if (bottomMargin < 0) bottomMargin = 0;
   gPad->SetLeftMargin(leftMargin);
   gPad->SetRightMargin(rightMargin);
   gPad->SetBottomMargin(bottomMargin);
   gPad->SetTopMargin(topMargin);
   Double_t dxr  = dx/(1 - gPad->GetLeftMargin() - gPad->GetRightMargin());
   Double_t dyr  = dy/(1 - gPad->GetBottomMargin() - gPad->GetTopMargin());

   // Range() could change the size of the pad pixmap and therefore should
   // be called before the other paint routines
   gPad->Range(xmin - dxr*gPad->GetLeftMargin(),
                      ymin - dyr*gPad->GetBottomMargin(),
                      xmax + dxr*gPad->GetRightMargin(),
                      ymax + dyr*gPad->GetTopMargin());
   gPad->RangeAxis(xmin, ymin, xmax, ymax);
   fX1 = xmin;
   fY1 = ymin;
   fX2 = xmax;
   fY2 = ymax;
}

//______________________________________________________________________________
void TFrame::Paint(Option_t *option)
{
   // Paint this wbox with its current attributes.
   const TPickerStackGuard stackGuard(this);

   if (!gPad->PadInHighlightMode() || (gPad->PadInHighlightMode() && this == gPad->GetSelected())) {
      TWbox::Paint(option);

      gPad->PaintBox(fX1,fY1,fX2,fY2,"s");
   }

}

//______________________________________________________________________________
void TFrame::Pop()
{
   // Do not pop frame's, if allowed they would cover the picture they frame.
}

//______________________________________________________________________________
void TFrame::SavePrimitive(ostream &, Option_t * /*= ""*/)
{
    // Save primitive as a C++ statement(s) on output stream out

}

//______________________________________________________________________________
void TFrame::UseCurrentStyle()
{
   // Replace current frame attributes by current style.

   if (gStyle->IsReading()) {
      SetFillColor(gStyle->GetFrameFillColor());
      SetLineColor(gStyle->GetFrameLineColor());
      SetFillStyle(gStyle->GetFrameFillStyle());
      SetLineStyle(gStyle->GetFrameLineStyle());
      SetLineWidth(gStyle->GetFrameLineWidth());
      SetBorderSize(gStyle->GetFrameBorderSize());
      SetBorderMode(gStyle->GetFrameBorderMode());
   } else {
      gStyle->SetFrameFillColor(GetFillColor());
      gStyle->SetFrameLineColor(GetLineColor());
      gStyle->SetFrameFillStyle(GetFillStyle());
      gStyle->SetFrameLineStyle(GetLineStyle());
      gStyle->SetFrameLineWidth(GetLineWidth());
      gStyle->SetFrameBorderSize(GetBorderSize());
      gStyle->SetFrameBorderMode(GetBorderMode());
   }
}
