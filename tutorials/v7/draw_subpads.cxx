/// \file
/// \ingroup tutorial_v7
//
/// This ROOT 7 example demonstrates how to create a ROOT 7 canvas (RCanvas),
/// divide on subpads and draw histograms there
///
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
#include "TRandom.h"

void draw_subpads()
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
  auto canvas = RCanvas::Create("Canvas Title");

  // Divide canvas on sub-pads

  auto subpads = canvas->Divide(2,2);

  subpads[0][0]->Draw(pHist1)->AttrLine().SetColor(RColor::kRed);
  subpads[1][0]->Draw(pHist2)->AttrLine().SetColor(RColor::kBlue);
  subpads[0][1]->Draw(pHist3)->AttrLine().SetColor(RColor::kGreen);

  // Divide pad on sub-sub-pads
  auto subsubpads = subpads[1][1]->Divide(2,2);

  subsubpads[0][0]->Draw(pHist1)->AttrLine().SetColor(RColor::kBlue);
  subsubpads[1][0]->Draw(pHist2)->AttrLine().SetColor(RColor::kGreen);
  subsubpads[0][1]->Draw(pHist3)->AttrLine().SetColor(RColor::kRed);

  canvas->Show();
}
