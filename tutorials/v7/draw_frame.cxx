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
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RStyle.hxx"

// macro must be here while cling is not capable to load
// library automatically for outlined function see ROOT-10336
R__LOAD_LIBRARY(libROOTHistDraw)

using namespace ROOT::Experimental;

auto style = std::make_shared<RStyle>(); // keep here to avoid destroy when leaving function scope

void draw_frame()
{
   // Create the histogram.
   RAxisConfig xaxis("x", 10, 0., 1.);
   RAxisConfig yaxis("y", {0., 1., 2., 3., 10.});
   auto pHist = std::make_shared<RH2D>(xaxis, yaxis);

   // Fill a few points.
   pHist->Fill({0.01, 1.02});
   pHist->Fill({0.54, 3.02});
   pHist->Fill({0.98, 1.02});
   pHist->Fill({1.90, 1.02});
   pHist->Fill({0.75, -0.02});

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

   auto draw1 = subpads[0][0]->Draw(pHist);
   draw1->AttrLine().SetColor(RColor::kRed);

   // create frame before drawing histograms
   auto frame2 = subpads[1][0]->GetOrCreateFrame();

   auto draw2 = subpads[1][0]->Draw(pHist);
   draw2->AttrLine().SetColor(RColor::kBlue).SetWidth(12);

   style->ParseString("frame { margin_left: 0.3; margin_right: 0.3; x_line_color_name: blue; y_line_color_name: green; }");

   canvas->UseStyle(style);

   canvas->Show();
}
