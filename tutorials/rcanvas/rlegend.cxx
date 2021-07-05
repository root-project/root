/// \file
/// \ingroup tutorial_rcanvas
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

void rlegend()
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
   auto canvas = RCanvas::Create("RLegend example");

   // add palette to canvas, it will not be seen on the canvas but used for colors
   canvas->Draw<RPaletteDrawable>(RPalette({{0., RColor::kWhite}, {.3, RColor::kRed}, {.7, RColor::kBlue}, {1., RColor::kBlack}}), false);

   // draw first histogram
   auto draw1 = canvas->Draw(pHist);
   draw1->line.width = 2.f;
   draw1->line.color = .3f; // should be red color

   // draw second histogram
   auto draw2 = canvas->Draw(pHist2);
   draw2->line.width = 4.f;
   draw2->line.color = .7f; // should be blue color

   auto legend = canvas->Draw<RLegend>("Legend title");
   legend->fill.style = 5;
   legend->fill.color = RColor::kWhite;
   legend->border.color = RColor::kRed;
   legend->border.width = 2;
   legend->AddEntry(draw1, "histo1");
   legend->AddEntry(draw2, "histo2");

   legend->AddEntry("test").SetAttrLine(RAttrLine(RColor::kGreen, 5., 1))
                           .SetAttrFill(RAttrFill(RColor::kBlue, 3004))
                           .SetAttrMarker(RAttrMarker(RColor::kRed, 3., RAttrMarker::kOpenCross));

   canvas->SetSize(1000, 700);
   canvas->Show();
}
