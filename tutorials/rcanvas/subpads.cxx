/// \file
/// \ingroup tutorial_rcanvas
///
/// This ROOT 7 example demonstrates how to create a ROOT 7 canvas (RCanvas),
/// divide on sub-pads and draw histograms there
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2018-03-13
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
/// \author Sergey Linev

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/RPad.hxx"
#include "ROOT/RStyle.hxx"
#include "ROOT/RDirectory.hxx"
#include "TRandom.h"

// macro must be here while cling is not capable to load
// library automatically for outlined function see ROOT-10336
R__LOAD_LIBRARY(libROOTHistDraw)

void subpads()
{
  using namespace ROOT::Experimental;

  // Create the histogram.
  RAxisConfig xaxis(25, 0., 10.);
  auto pHist1 = std::make_shared<RH1D>(xaxis);
  auto pHist2 = std::make_shared<RH1D>(xaxis);
  auto pHist3 = std::make_shared<RH1D>(xaxis);


  for (int n=0;n<1000;n++) {
     pHist1->Fill(gRandom->Gaus(3., 0.8));
     pHist2->Fill(gRandom->Gaus(5., 1.));
     pHist3->Fill(gRandom->Gaus(7., 1.2));
  }

  // Create a canvas to be displayed.
  auto canvas = RCanvas::Create("Sub-sub pads example");

  // Divide canvas on sub-pads

  auto subpads = canvas->Divide(2,2);

  subpads[0][0]->Draw(pHist1)->line.color = RColor::kRed;
  subpads[1][0]->Draw(pHist2)->line.color = RColor::kBlue;
  subpads[0][1]->Draw(pHist3)->line.color = RColor::kGreen;

  // Divide sub-pad on sub-sub-pads
  auto subsubpads = subpads[1][1]->Divide(2,2);

  subsubpads[0][0]->Draw(pHist1)->line.color = RColor::kBlue;
  subsubpads[1][0]->Draw(pHist2)->line.color = RColor::kGreen;
  subsubpads[0][1]->Draw(pHist3)->line.color = RColor::kRed;

   auto style = RStyle::Parse(
        "frame {"              // select type frame for RFrame
        "   gridX: true;"      // enable grid drawing
        "   gridY: true;"
        "   ticksX: 2;"        // enable ticks drawing on both sides
        "   ticksY: 2;"
        "   x_labels_size: 0.05;" // below 1 is scaling factor for pad height
        "   y_labels_size: 20;"   // just a font size in pixel
        "   y_labels_color: green;"  // and name labels color
        "}");
  canvas->UseStyle(style);

  canvas->SetSize(1200, 600);
  canvas->Show();

  RDirectory::Heap().Add("subpads_style", style); // required to keep style alive
}
