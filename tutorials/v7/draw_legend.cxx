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
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/RLegend.hxx"
#include "TRandom.h"

using namespace ROOT::Experimental;

void draw_legend()
{
   // Create the histogram.
   RAxisConfig xaxis(25, 0., 10.);
   auto pHist = std::make_shared<RH1D>(xaxis);
   auto pHist2 = std::make_shared<RH1D>(xaxis);

   for (int n=0;n<1000;n++) {
      pHist->Fill(gRandom->Gaus(3,0.8));
      pHist2->Fill(gRandom->Gaus(7,1.2));
   }

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Canvas Title");
   auto draw1 = canvas->Draw(pHist);
   draw1->AttrLine().SetColor(RColor::kRed).SetWidth(2);

   auto draw2 = canvas->Draw(pHist2);
   draw2->AttrLine().SetColor(RColor::kBlue).SetWidth(4);
   
   auto legend = canvas->Draw<RLegend>(RPadPos(0.5_normal, 0.6_normal), RPadPos(0.9_normal,0.9_normal), "Legend title");
   legend->AttrBox().Fill().SetStyle(5).SetColor(RColor::kWhite);
   legend->AttrBox().Border().SetWidth(2).SetColor(RColor::kRed);
   legend->AddEntry(draw1, "histo1").SetLine("line_");
   legend->AddEntry(draw2, "histo2").SetLine("line_");

   canvas->Show();
}
