/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2017-06-02
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Axel Naumann <axel@cern.ch>

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

R__LOAD_LIBRARY(libGpad);

#include <ROOT/TObjectDrawable.hxx>
#include <ROOT/TCanvas.hxx>
#include <TGraph.h>

// Show how to display v6 objects in a v7 TCanvas.

void draw_v6()
{
   using namespace ROOT;

   static constexpr int npoints = 4;
   double x[npoints] = {0., 1., 2., 3.};
   double y[npoints] = {.1, .2, .3, .4};
   auto gr = std::make_shared<TGraph>(npoints, x, y);
   auto canvas = Experimental::TCanvas::Create("v7 TCanvas showing a v6 TGraph");
   canvas->Draw(gr);

   canvas->Show();

   canvas->SaveAs("draw.png"); // only .svg and .png are supported for the moment, asynchron
}
