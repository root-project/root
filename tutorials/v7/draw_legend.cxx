/// \file
/// \ingroup tutorial_v7
///
/// This macro generates two TH1D objects and build RLegend
/// In addition use of auto colors are shown
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2019-10-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
/// \author Sergey Linev <s.linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RPaletteDrawable.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/RLegend.hxx"
#include "TRandom.h"

// macro must be here while cling is not capable to load
// library automatically for outlined function see ROOT-10336
R__LOAD_LIBRARY(libROOTHistDraw)
R__LOAD_LIBRARY(libROOTGraphicsPrimitives)

using namespace ROOT::Experimental;

void draw_legend()
{
   // Create the histograms.
   RAxisConfig xaxis(25, 0., 10.);
   auto pHist = std::make_shared<RH1D>(xaxis);
   auto pHist2 = std::make_shared<RH1D>(xaxis);

   for (int n=0;n<1000;n++) {
      pHist->Fill(gRandom->Gaus(3,0.8));
      pHist2->Fill(gRandom->Gaus(7,1.2));
   }

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Canvas Title");

   // add palette to canvas, it will not be seen on the canvas but used for colors
   canvas->Draw<RPaletteDrawable>(RPalette({{0., RColor::kWhite}, {.3, RColor::kRed}, {.7, RColor::kBlue}, {1., RColor::kBlack}}), false);

   // draw first histogram
   auto draw1 = canvas->Draw(pHist);
   draw1->SetLineWidth(2).SetLineColor(0.3); // should be red color

   // draw second histogram
   auto draw2 = canvas->Draw(pHist2);
   draw2->SetLineWidth(4).SetLineColor(0.7); // should be blue color

   auto legend = canvas->Draw<RLegend>("Legend title");
   legend->SetFillStyle(5).SetFillColor(RColor::kWhite);
   legend->SetLineWidth(2).SetLineColor(RColor::kRed);
   legend->AddEntry(draw1, "histo1");
   legend->AddEntry(draw2, "histo2");

   auto entry = legend->AddEntry("test");
   entry->SetLine(true);
   entry->SetLineColor(RColor::kGreen).SetLineWidth(5);
   entry->SetFill(true);
   entry->SetFillColor(RColor::kBlue).SetFillStyle(3004);
   entry->SetMarker(true);
   entry->SetMarkerColor(RColor::kRed).SetMarkerSize(3).SetMarkerStyle(28);

   canvas->SetSize(1000, 700);
   canvas->Show();
}
