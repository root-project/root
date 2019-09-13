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

#include <ROOT/RObjectDrawable.hxx>
#include <ROOT/RCanvas.hxx>
#include <TGraph.h>

#include <iostream>

// Show how to display v6 objects in a v7 RCanvas.

void draw_v6()
{
   using namespace ROOT::Experimental;

   static constexpr int npoints = 10;
   double x[npoints] = { 0., 1., 2., 3., 4., 5., 6., 7., 8., 9. };
   double y[npoints] = { .1, .2, .3, .4, .3, .2, .1, .2, .3, .4 };
   auto gr = std::make_shared<TGraph>(npoints, x, y);
   auto canvas = RCanvas::Create("v7 RCanvas showing a v6 TGraph");
   canvas->Draw<RObjectDrawable>(gr, "AL");

   // canvas->Show("opera");   // one could specify program name which should show canvas (like chromium or firefox)
   // canvas->Show("/usr/bin/chromium --app=$url &"); // one could use $url parameter, which replaced with canvas URL

   canvas->Show(); // new window in default browser should popup and async update will be triggered

   // synchronous, wait until painting is finished
   canvas->Update(false,
                  [](bool res) { std::cout << "First Update done = " << (res ? "true" : "false") << std::endl; });

   canvas->Modified(); // invalidate canvas and force repainting with next Update()

   // call Update again, should return immediately if canvas was not modified
   canvas->Update(false,
                  [](bool res) { std::cout << "Second Update done = " << (res ? "true" : "false") << std::endl; });

   // request to create PNG file in asynchronous mode and specify lambda function as callback
   // when request processed by the client, callback invoked with result value
   // canvas->SaveAs("draw.png", true,
   //               [](bool res) { std::cout << "Producing PNG done res = " << (res ? "true" : "false") << std::endl; });

   // this function executed in synchronous mode (async = false is default),
   // previous file saving will be completed at this point as well
   // canvas->SaveAs("draw.svg"); // synchronous
}
