/// \file
/// \ingroup tutorial_v7
///
/// This macro generates a small V7 TH2D, fills it with random values and
/// draw it in a V7 canvas, using configured web browser
///
/// \macro_code
///
/// \date 2020-03-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
/// \author Sergey Linev <s.linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/RFrameTitle.hxx"
#include "ROOT/RPaletteDrawable.hxx"
#include "ROOT/RFrame.hxx"
#include "TRandom.h"

// macro must be here while cling is not capable to load
// library automatically for outlined function see ROOT-10336
R__LOAD_LIBRARY(libROOTHistDraw)

using namespace ROOT::Experimental;

void draw_rh2_colz()
{
   // Create the histogram.
   RAxisConfig xaxis("x", 20, 0., 10.);
   RAxisConfig yaxis("y", 20, 0., 10.);
   auto pHist = std::make_shared<RH2D>(xaxis, yaxis);

   for (int n=0;n<10000;n++)
      pHist->Fill({gRandom->Gaus(5.,2.), gRandom->Gaus(5.,2.)});

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Canvas Title");

   // should we made special style for frame with palette?
   canvas->GetOrCreateFrame()->Margins().SetRight(0.2_normal);

   canvas->Draw<RFrameTitle>("2D histogram with color palette");

   canvas->Draw<RPaletteDrawable>(RPalette::GetPalette(), true);

   auto draw1 = canvas->Draw(pHist);

   canvas->Show("1000x700");
}
