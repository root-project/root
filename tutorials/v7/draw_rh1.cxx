/// \file
/// \ingroup tutorial_v7
///
/// This macro generates a small V7 TH1D, fills it and draw it in a V7 canvas.
/// The canvas is display in the web browser
///
/// \macro_code
///
/// \date 2015-03-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
/// \author Axel Naumann <axel@cern.ch>

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RFrameTitle.hxx"
#include "ROOT/RCanvas.hxx"
#include "TRandom.h"

// macro must be here while cling is not capable to load
// library automatically for outlined function see ROOT-10336
R__LOAD_LIBRARY(libROOTHistDraw)

using namespace ROOT::Experimental;

void draw_rh1()
{
   // Create the histogram.
   RAxisConfig xaxis(25, 0., 10.);
   auto pHist1 = std::make_shared<RH1D>(xaxis);
   auto pHist2 = std::make_shared<RH1D>(xaxis);

   for (int n=0;n<1000;n++) {
      pHist1->Fill(gRandom->Gaus(3,0.8));
      pHist2->Fill(gRandom->Gaus(7,1.2));
   }

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Canvas Title");

   // histograms colors
   auto col1 = RColor::kRed, col2 = RColor::kBlue;

   // Divide canvas on 2x3 sub-pads to show different draw options
   auto subpads = canvas->Divide(2,3);

   // default draw option
   subpads[0][0]->Draw<RFrameTitle>("Default \"hist\" draw options");
   subpads[0][0]->Draw(pHist1)->AttrLine().SetColor(col1).SetWidth(2);
   subpads[0][0]->Draw(pHist2)->AttrLine().SetColor(col2).SetWidth(4);

   // errors draw options
   subpads[1][0]->Draw<RFrameTitle>("Errors draw options");
   subpads[1][0]->Draw(pHist1)->Error(1).AttrLine().SetColor(col1);
   subpads[1][0]->Draw(pHist2)->Error(4).AttrFill().SetColor(col2).SetStyle(3003);

   // text and marker draw options
   subpads[0][1]->Draw<RFrameTitle>("Text and marker draw options");
   subpads[0][1]->Draw(pHist1)->Text(true).AttrText().SetColor(col1);
   subpads[0][1]->Draw(pHist2)->Marker().AttrMarker().SetColor(col2).SetStyle(30).SetSize(1.5);


   canvas->SetSize(1000, 700);
   canvas->Show();
}
