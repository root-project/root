/// \file
/// \ingroup tutorial_v7
///
/// This ROOT 7 example shows how to use symlog scale on RAxis
/// See discussion on forum https://root-forum.cern.ch/t/symlog-scale-for-plotting/ for more details
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2021-05-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Sergey Linev <s.linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RCanvas.hxx"
#include "ROOT/RFrameTitle.hxx"
#include "ROOT/RFrame.hxx"
#include "ROOT/RMarker.hxx"

using namespace ROOT::Experimental;

auto symlog_style = RStyle::Parse("frame { margins_left: 0.1; }"
                                  "marker { onframe: true; clipping: true; }"
                                  ".group1 { marker_style: 8; marker_color: blue; }"
                                  ".group2 { marker_style: 8; marker_color: orange; }");

void draw_symlog()
{
   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Drawing with symlog");

   auto pads   = canvas->Divide(1, 3);

   // first pad with linear scales
   auto frame1 = pads[0][0]->AddFrame();
   frame1->SetDrawAxes(true);
   frame1->x.min = -40;
   frame1->x.max = 1040;
   frame1->x.title = "x linear";
   frame1->x.title.SetCenter();
   frame1->y.log = 10.;
   frame1->y.min = 1;
   frame1->y.max = 1e4;
   frame1->y.title = "y log";
   frame1->y.title.SetCenter();
   pads[0][0]->Draw<RFrameTitle>("linear scale")->SetMargin(0.01_normal).SetHeight(0.1_normal);

   // second pad with log scales, negative values missing
   auto frame2 = pads[0][1]->AddFrame();
   frame2->SetDrawAxes(true);
   frame2->x.log = 10.;
   frame2->x.min = 0.05;
   frame2->x.max = 1.2e3;
   frame2->x.title = "x log";
   frame2->x.title.SetCenter();
   frame2->y.log = 10.;
   frame2->y.min = 1;
   frame2->y.max = 1e4;
   frame2->y.title = "y log";
   frame2->y.title.SetCenter();
   pads[0][1]->Draw<RFrameTitle>("log scale, missing points")->SetMargin(0.01_normal).SetHeight(0.1_normal);

   // third pad with symlog scales
   auto frame3 = pads[0][2]->AddFrame();
   frame3->SetDrawAxes(true);
   // configure synlog scale with 10 for linear range, rest will be logarithmic, including negative
   frame3->x.symlog = 10.; // boundary inside which linear scale will be used
   frame3->x.min = -10;
   frame3->x.max = 1.2e3;
   frame3->x.title = "x symlog";
   frame3->x.title.SetCenter();
   frame3->y.log = 10.;
   frame3->y.min = 1;
   frame3->y.max = 1e4;
   frame3->y.title = "y log";
   frame3->y.title.SetCenter();
   pads[0][2]->Draw<RFrameTitle>("symlog scale")->SetMargin(0.01_normal).SetHeight(0.1_normal);

   for (int n=0;n<100;n++) {
      auto x1 = TMath::Power(10, gRandom->Uniform(1, 2.9));
      auto y1 = TMath::Power(10, gRandom->Uniform(1, 1.9));

      if (n % 27 == 0) { x1 = 100; y1 *= 100; }

      RPadPos pm{RPadLength::User(x1), RPadLength::User(y1)};
      pads[0][0]->Draw<RMarker>(pm)->SetCssClass("group1");
      pads[0][1]->Draw<RMarker>(pm)->SetCssClass("group1");
      pads[0][2]->Draw<RMarker>(pm)->SetCssClass("group1");
   }

   for (int n=0;n<75;n++) {
      auto x2 = gRandom->Uniform(-5., 5.);
      auto y2 = TMath::Power(10, gRandom->Uniform(2.2, 3.7));

      RPadPos pm{RPadLength::User(x2), RPadLength::User(y2)};
      pads[0][0]->Draw<RMarker>(pm)->SetCssClass("group2");
      pads[0][1]->Draw<RMarker>(pm)->SetCssClass("group2");
      pads[0][2]->Draw<RMarker>(pm)->SetCssClass("group2");
   }

   canvas->UseStyle(symlog_style);

   canvas->SetSize(500, 900);

   canvas->Show();
}
