/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2015-03-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
/// \author Axel Naumann <axel@cern.ch>

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHist.hxx"
#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/RFitPanel.hxx"
#include "ROOT/RDirectory.hxx"

using namespace ROOT::Experimental;

void fitpanel() {

   RAxisConfig xaxis(10, 0., 10.);
   // Create the histogram.
   auto pHist = std::make_shared<RH1D>(xaxis);

   // Fill a few points.
   pHist->Fill(1);
   pHist->Fill(2);
   pHist->Fill(2);
   pHist->Fill(3);

   auto canvas = RCanvas::Create("RCanvas with histogram");
   canvas->Draw(pHist); //->SetLineColor(RColor::kRed);

   canvas->Show();
   canvas->Update(); // need to ensure canvas is drawn

   auto panel = std::make_shared<RFitPanel>("FitPanel Title");

   RDirectory::Heap().Add("fitpanel", panel);
   RDirectory::Heap().Add("firsthisto", pHist);

   // TODO: how combine there methods together
   // here std::shread_ptr<> on both sides

   panel->AssignCanvas(canvas);
   panel->AssignHistogram(pHist);

   canvas->AddPanel(panel);
}

