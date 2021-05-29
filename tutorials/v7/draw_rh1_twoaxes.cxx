/// \file
/// \ingroup tutorial_v7
///
/// This macro generates two RH1D, fills them and draw in RCanvas.
/// Second histogram uses enables SecondY() draw option to draw separate Y axis on right side
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2021-05-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
/// \author Sergey Linev <s.linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHist.hxx"
#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RFrameTitle.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/RPad.hxx"
#include "TRandom.h"

// macro must be here while cling is not capable to load
// library automatically for outlined function see ROOT-10336
R__LOAD_LIBRARY(libROOTHistDraw)

using namespace ROOT::Experimental;

void draw_rh1_twoaxes()
{
   // Create the histogram.
   RAxisConfig xaxis(25, 0., 10.);
   auto pHist1 = std::make_shared<RH1D>(xaxis);
   auto pHist2 = std::make_shared<RH1D>(xaxis);

   for (int n=0;n<1000;n++)
      pHist1->Fill(gRandom->Gaus(3,0.8));

   for (int n=0;n<3000;n++)
      pHist2->Fill(gRandom->Gaus(7,1.2));

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("RH1 with two Y scales");

   // histograms colors
   auto col1 = RColor::kRed, col2 = RColor::kBlue;

   // default draw option
   canvas->Draw<RFrameTitle>("Two independent Y axes for histograms");
   canvas->Draw(pHist1)->AttrLine().SetColor(col1).SetWidth(2);
   canvas->Draw(pHist2)->SecondY().AttrLine().SetColor(col2).SetWidth(4);

   canvas->GetFrame()->AttrY().SetTicksColor(col1);
   canvas->GetFrame()->AttrY2().SetTicksColor(col2);

   canvas->SetSize(800, 600);
   canvas->Show();
}
