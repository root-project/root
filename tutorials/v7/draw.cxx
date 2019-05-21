/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2015-03-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Axel Naumann <axel@cern.ch>

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Rtypes.h"

R__LOAD_LIBRARY(libROOTHistDraw);

#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/TDirectory.hxx"

void draw()
{
   using namespace ROOT::Experimental;

   // Create the histogram.
   RAxisConfig xaxis("x", 10, 0., 1.);
   RAxisConfig yaxis("y", {0., 1., 2., 3., 10.});
   auto pHist = std::make_shared<RH2D>(xaxis, yaxis);

   // Fill a few points.
   pHist->Fill({0.01, 1.02});
   pHist->Fill({0.54, 3.02});
   pHist->Fill({0.98, 1.02});
   pHist->Fill({1.90, 1.02});
   pHist->Fill({0.75, -0.02});

   // Register the histogram with ROOT: now it lives even after draw() ends.
   ROOT::Experimental::TDirectory::Heap().Add("hist", pHist);

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Canvas Title");
   auto pOpts = canvas->Draw(pHist);
   pOpts->SetLine({RColor::kRed});
   pOpts->Line().SetColor();

   RH2D other = *pHist;
   auto pOptsOther = canvas->Draw(other);
   *pOptsOther = *pOpts;
   pOptsOther->Line().SetColor(RColor::kBlue);
   auto lineAttrs = pOptsOther->Line();
   lineAttrs.SetWidth(12);
   pOpts->Line() = lineAttrs;

   RAttrLineRef lineAttr1 = opts.Line();
   lineAttr1.SetColor(RColor::kRed);
   RAttrLine lineAttrs2;
   lineAttrs2 = lineAttrs1;
   lineAttrs2.SetWidth(3);
   ops.Line().GetWidth() == ?


   TH3D third = *pHist;
   canvas->Draw(third, *pOpts);

   canvas->Show();
}
