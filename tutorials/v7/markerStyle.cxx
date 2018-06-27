/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2018-03-18
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Iliana Betsou

#include "ROOT/RCanvas.hxx"
#include "ROOT/RText.hxx"
#include "ROOT/RMarker.hxx"
#include "ROOT/RPadPos.hxx"

void markerStyle() {
   using namespace ROOT;
   using namespace ROOT::Experimental;

   auto canvas = Experimental::RCanvas::Create("Canvas Title");
   double num = 0.3;

   double x = 0;
   double dx = 1/16.0;
   for (Int_t i=1;i<16;i++) {
      x += dx;

      RPadPos pt1(RPadLength::Normal(x), .12_normal);
      auto ot1 = canvas->Draw(Experimental::RText(pt1, Form("%d", i)));
      RPadPos pm1(RPadLength::Normal(x), .25_normal);
      auto om1 = canvas->Draw(Experimental::RMarker(pm1));
      om1->SetMarkerStyle(i);
      om1->SetMarkerSize(2.5);

      RPadPos pt2(RPadLength::Normal(x), .42_normal);
      auto ot2 = canvas->Draw(Experimental::RText(pt2, Form("%d", i+19)));
      RPadPos pm2(RPadLength::Normal(x), .55_normal);
      auto om2 = canvas->Draw(Experimental::RMarker(pm2));
      om2->SetMarkerStyle(i+19);
      om2->SetMarkerSize(2.5);

      RPadPos pt3(RPadLength::Normal(x), .72_normal);
      auto ot3 = canvas->Draw(Experimental::RText(pt3, Form("%d", i+34)));
      RPadPos pm3(RPadLength::Normal(x), .85_normal);
      auto om3 = canvas->Draw(Experimental::RMarker(pm3));
      om3->SetMarkerStyle(i+34);
      om3->SetMarkerSize(2.5);
   }

   canvas->Show();
}