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
#include "TRandom.h"

// macro must be here while cling is not capable to load
// library automatically for outlined function see ROOT-10336
R__LOAD_LIBRARY(libROOTHistDraw)

using namespace ROOT::Experimental;

void draw_rh3()
{
   // Create the histogram.
   RAxisConfig xaxis("x", 10, -5., 5.);
   RAxisConfig yaxis("y", 10, -5., 5.);
   RAxisConfig zaxis("z", 10, -5., 5.);
   auto pHist = std::make_shared<RH3D>(xaxis, yaxis, zaxis);

   for (int n=0;n<10000;n++)
      pHist->Fill({gRandom->Gaus(0.,2.), gRandom->Gaus(0.,2.), gRandom->Gaus(0.,2.)});

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Canvas Title");

   canvas->Draw<RFrameTitle>("3D histogram drawing");

   auto draw = canvas->Draw(pHist);
   draw->SetColor(RColor::kBlue); // use color in some draw options
   draw->SetLineColor(RColor::kRed);
   draw->Scatter(); // scatter plot
   draw->Sphere(1); // draw spheres 0 - default, 1 - with colors
   draw->Color(); // draw colored boxes
   draw->Box(0); // draw boxes 0 - default, 1 - with colors, 2 - without lines

   canvas->SetSize(1000, 700);
   canvas->Show();
}
