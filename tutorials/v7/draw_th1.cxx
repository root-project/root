/// \file
/// \ingroup tutorial_v7
///
/// This macro generates a small V7 TH1D, fills it and draw it in a V7 canvas.
/// The canvas is display in the web browser and the corresponding png picture
/// is generated.
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

R__LOAD_LIBRARY(libROOTGpadv7);

#include "ROOT/RHist.hxx"
#include "ROOT/RCanvas.hxx"

void draw_th1() {
   using namespace ROOT;

   // Create the histogram.
   Experimental::RAxisConfig xaxis(10, 0., 10.);
   auto pHist = std::make_shared<Experimental::RH1D>(xaxis);
   auto pHist2 = std::make_shared<Experimental::RH1D>(xaxis);

   // Fill a few points.
   pHist->Fill(1);
   pHist->Fill(2);
   pHist->Fill(2);
   pHist->Fill(3);

   pHist2->Fill(5);
   pHist2->Fill(6);
   pHist2->Fill(6);
   pHist2->Fill(7);

   // Create a canvas to be displayed.
   auto canvas = Experimental::RCanvas::Create("Canvas Title");
   canvas->Draw(pHist)->SetLineColor(Experimental::RColor::kRed);
   canvas->Draw(pHist2)->SetLineColor(Experimental::RColor::kBlue);

   canvas->Show();
   // Saving to PNG doesn't work reliably in batch yet:
   if (!gROOT->IsWebDisplayBatch())
      canvas->SaveAs("th1.png");
}
