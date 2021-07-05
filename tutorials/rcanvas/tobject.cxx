/// \file
/// \ingroup tutorial_rcanvas
///
/// This macro shows how ROOT objects like TH1, TH2, TGraph can be drawn in RCanvas.
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2017-06-02
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \authors Axel Naumann <axel@cern.ch>, Sergey Linev <s.linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/TObjectDrawable.hxx>
#include <ROOT/RPad.hxx>
#include <ROOT/RCanvas.hxx>
#include "TH1.h"
#include "TH2.h"
#include "TGraph.h"
#include "TMath.h"
#include "TStyle.h"

#include <iostream>

using namespace ROOT::Experimental;

auto v6_style = RStyle::Parse("tgraph { line_width: 3; line_color: red; }");

void tobject()
{
   static constexpr int npoints = 10;
   static constexpr int nth1points = 100;
   static constexpr int nth2points = 40;

   double x[npoints] = { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10. };
   double y[npoints] = { .1, .2, .3, .4, .3, .2, .1, .2, .3, .4 };
   auto gr = new TGraph(npoints, x, y);

   // create normal object to be able draw it once
   auto th1 = new TH1I("gaus", "Example of TH1", nth1points, -5, 5);
   // it is recommended to set directory to nullptr, but it is also automatically done in TObjectDrawable
   // th1->SetDirectory(nullptr);
   th1->FillRandom("gaus", 5000);

   // use std::shared_ptr<TH2I> to let draw same histogram twice with different draw options
   auto th2 = std::make_shared<TH2I>("gaus2", "Example of TH2", nth2points, -5, 5, nth2points, -5, 5);
   // is is highly recommended to set directory to nullptr to avoid ownership conflicts
   th2->SetDirectory(nullptr);
   for (int n=0;n<nth2points;++n) {
      for (int k=0;k<nth2points;++k) {
         double x = 10.*n/nth2points-5.;
         double y = 10.*k/nth2points-5.;
         th2->SetBinContent(th2->GetBin(n+1, k+1), (int) (1000*TMath::Gaus(x)*TMath::Gaus(y)));
      }
   }

   gStyle->SetPalette(kRainBow);

   auto canvas = RCanvas::Create("RCanvas showing a v6 objects");

   // add gStyle object, will be applied on JSROOT side
   // set on the canvas before any other object is drawn
   canvas->Draw<TObjectDrawable>(TObjectDrawable::kStyle);

   // add ROOT colors, required when they are changed from default values
   canvas->Draw<TObjectDrawable>(TObjectDrawable::kColors);

   // copy custom palette to canvas, will be used for col drawings
   // style object does not include color settings
   canvas->Draw<TObjectDrawable>(TObjectDrawable::kPalette);

   // Divide canvas on 2x2 sub-pads to show different draw options
   auto subpads = canvas->Divide(2,2);

   // draw graph with "AL" option, drawable take over object ownership
   subpads[0][0]->Draw<TObjectDrawable>(gr, "AL");

   // one can change basic attributes via v7 classes, value will be replaced on client side
   auto drawth1 = subpads[0][1]->Draw<TObjectDrawable>(th1);
   drawth1->line = RAttrLine(RColor::kBlue, 3., 2);

   subpads[1][0]->Draw<TObjectDrawable>(th2, "colz");

   // show same object again, but with other draw options
   subpads[1][1]->Draw<TObjectDrawable>(th2, "lego2");

   // add style, here used to configure TGraph attributes, evaluated only on client side
   canvas->UseStyle(v6_style);

   // new window in web browser should popup and async update will be triggered
   canvas->Show();

   // synchronous, wait until drawing is really finished
   canvas->Update(false, [](bool res) { std::cout << "First sync update done = " << (res ? "true" : "false") << std::endl; });

   // invalidate canvas and force repainting with next Update()
   canvas->Modified();

   // call Update again, should return immediately if canvas was not modified
   canvas->Update(true, [](bool res) { std::cout << "Second async update done = " << (res ? "true" : "false") << std::endl; });

   std::cout << "This message appear normally before second async update" << std::endl;

   // create SVG file
   // canvas->SaveAs("tobject.svg");

   // create PNG file
   // canvas->SaveAs("tobject.png");
}
