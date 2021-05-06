/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2020-02-20
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Sergey Linev <s.linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/RFrameTitle.hxx"
#include "ROOT/RFrame.hxx"
#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RStyle.hxx"
#include "ROOT/RLine.hxx"
#include "ROOT/RPad.hxx"

// macro must be here while cling is not capable to load
// library automatically for outlined function see ROOT-10336
R__LOAD_LIBRARY(libROOTHistDraw)

using namespace ROOT::Experimental;

void draw_frame()
{
   // Create the histogram.
   RAxisConfig xaxis("x", 10, 0.1, 110.);
   RAxisConfig yaxis("y", {0., 1., 2., 3., 10.});
   auto pHist = std::make_shared<RH2D>(xaxis, yaxis);

   // Fill a few points.
   pHist->Fill({0.11, 1.02});
   pHist->Fill({5.54, 3.02});
   pHist->Fill({20.98, 1.02});
   pHist->Fill({20.90, 1.02});
   pHist->Fill({1.75, -0.02});

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Canvas Title");

   // Make divisions
   auto subpads = canvas->Divide(2,1);

   // configure RFrame with direct API calls
   auto frame1 = subpads[0][0]->GetOrCreateFrame();
   frame1->AttrFill().SetColor(RColor::kBlue);
   frame1->AttrBorder().SetColor(RColor::kRed);
   frame1->AttrBorder().SetWidth(3);
   frame1->Margins().SetTop(0.3_normal);
   frame1->Margins().SetBottom(0.1_normal);
   frame1->Margins().SetLeft(0.2_normal);
   frame1->Margins().SetRight(0.2_normal);

   frame1->AttrX().AttrLine().SetColor(RColor::kGreen);
   frame1->AttrY().AttrLine().SetColor(RColor::kBlue);

   frame1->AttrX().SetLog(2.);
   frame1->AttrX().SetZoom(2.,80.);
   frame1->AttrY().SetZoom(2,8);

   subpads[0][0]->Draw<RFrameTitle>("Frame1 title")->SetMargin(0.01_normal).SetHeight(0.05_normal);

   auto draw1 = subpads[0][0]->Draw(pHist);

   // create frame before drawing histograms
   auto frame2 = subpads[1][0]->GetOrCreateFrame();

   auto draw2 = subpads[1][0]->Draw(pHist);

   subpads[1][0]->Draw<RFrameTitle>("Frame2 with margins set via CSS");

   // draw line under the frame with line width 3
   auto line1 = subpads[1][0]->Draw<RLine>(RPadPos(.1_normal, .1_normal), RPadPos(.9_normal , .1_normal));
   line1->SetOnFrame(false);
   line1->AttrLine().SetWidth(3);

   // draw line in the frame, allowed to set user coordinate
   auto line2 = subpads[1][0]->Draw<RLine>(RPadPos(20_user, 1.5_user), RPadPos(80_user, 8_user));
   //line2->SetOnFrame(true); // configured via CSS "onframe"

   // draw line in the frame, but disable default cutting by the frame borders
   auto line3 = subpads[1][0]->Draw<RLine>(RPadPos(20_user, 8_user), RPadPos(80_user, 1.5_user));
   //line3->SetOnFrame(true); // configured via CSS "onframe"
   line3->SetCutByFrame(false);

   auto style = RStyle::Parse("frame { margin_left: 0.1; margin_right: 0.1; margin_all: 0.2; x_line_color_name: blue; y_line_color: green; } "
                              "title { margin: 0.02; height: 0.1; text_size: 20; }"
                              "line { onframe: true; }");

   subpads[1][0]->UseStyle(style);

   canvas->Show();

   RDirectory::Heap().Add("custom_style", style); // required to keep style alive
}
