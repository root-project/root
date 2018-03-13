/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2018-03-13
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
/// \author Sergey Linev

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/THist.hxx"
#include "ROOT/TCanvas.hxx"
#include "ROOT/TPad.hxx"

void draw_subpads() {
  using namespace ROOT;

  // Create the histogram.
  Experimental::TAxisConfig xaxis(10, 0., 10.);
  auto pHist1 = std::make_shared<Experimental::TH1D>(xaxis);
  auto pHist2 = std::make_shared<Experimental::TH1D>(xaxis);
  auto pHist3 = std::make_shared<Experimental::TH1D>(xaxis);

  // Fill a few points.
  pHist1->Fill(1);
  pHist1->Fill(2);
  pHist1->Fill(2);
  pHist1->Fill(3);

  pHist2->Fill(5);
  pHist2->Fill(6);
  pHist2->Fill(6);
  pHist2->Fill(7);

  pHist3->Fill(4);
  pHist3->Fill(5);
  pHist3->Fill(5);
  pHist3->Fill(6);

  // Create a canvas to be displayed.
  auto canvas = Experimental::TCanvas::Create("Canvas Title");

  // Divide canvas on sub-pads

  auto subpads = canvas->Divide(2,2);

  subpads[0][0]->Draw(pHist1)->SetLineColor(Experimental::TColor::kRed);
  subpads[1][0]->Draw(pHist2)->SetLineColor(Experimental::TColor::kBlue);
  subpads[0][1]->Draw(pHist3)->SetLineColor(Experimental::TColor::kGreen);

  // Divide pad on sub-sub-pads
  auto subsubpads = subpads[1][1]->Divide(2,2);

  subsubpads[0][0]->Draw(pHist1)->SetLineColor(Experimental::TColor::kBlue);
  subsubpads[1][0]->Draw(pHist2)->SetLineColor(Experimental::TColor::kGreen);
  subsubpads[0][1]->Draw(pHist3)->SetLineColor(Experimental::TColor::kRed);

  canvas->Show();
}
