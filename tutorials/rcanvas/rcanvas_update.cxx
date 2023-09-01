/// \file
/// \ingroup tutorial_rcanvas
///
/// This macro shows how ROOT RCanvas::Update method is working.
/// One can do sync and/or async depending how important is that graphics is updated before next action will be performed
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2021-07-05
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Sergey Linev <s.linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/TObjectDrawable.hxx>
#include <ROOT/RCanvas.hxx>
#include "TGraph.h"

#include <iostream>

using namespace ROOT::Experimental;

void rcanvas_update()
{
   static constexpr int npoints = 10;
   double x[npoints] = { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10. };
   double y[npoints] = { .1, .2, .3, .2, .1, .2, .3, .2, .1, .2 };
   auto gr = new TGraph(npoints, x, y);

   auto canvas = RCanvas::Create("Demo of RCanvas update");

   canvas->Draw<TObjectDrawable>(gr, "AL");

   // new window in web browser should popup and async update will be triggered
   canvas->Show();

   // synchronous, wait until drawing is really finished
   canvas->Update(false, [](bool res) { std::cout << "First sync update done = " << (res ? "true" : "false") << std::endl; });

   // modify TGraph making different form
   gr->SetPoint(1, 1., .3);
   gr->SetPoint(3, 3., .1);
   gr->SetPoint(5, 5., .3);
   gr->SetPoint(7, 7., .1);
   gr->SetPoint(9, 9., .3);

   // invalidate canvas and force repainting with next Update()
   canvas->Modified();

   // call Update again, return before actual drawing will be performed by the browser
   canvas->Update(true, [](bool res) { std::cout << "Second async update done = " << (res ? "true" : "false") << std::endl; });

   std::cout << "This message appear normally before second async update" << std::endl;
}
