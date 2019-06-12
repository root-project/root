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
#include <string>

void markerStyle() {
   using namespace ROOT::Experimental;

   auto canvas = RCanvas::Create("Canvas Title");
   double num = 0.3;

   double x = 0;
   double dx = 1/16.0;
   for (Int_t i=1;i<16;i++) {
      x += dx;

      RPadPos pt1(RPadLength::Normal(x), .12_normal);
      auto ot1 = canvas->Draw(RText(pt1, std::to_string(i)));
      RPadPos pm1(RPadLength::Normal(x), .25_normal);
      auto om1 = canvas->Draw(RMarker(pm1));
      om1->SetStyle(i).SetSize(2.5);

      RPadPos pt2(RPadLength::Normal(x), .42_normal);
      auto ot2 = canvas->Draw(RText(pt2, std::to_string(i+19)));
      RPadPos pm2(RPadLength::Normal(x), .55_normal);
      auto om2 = canvas->Draw(RMarker(pm2));
      om2->SetStyle(i+19).SetSize(2.5);

      RPadPos pt3(RPadLength::Normal(x), .72_normal);
      auto ot3 = canvas->Draw(RText(pt3, std::to_string(i+34)));
      RPadPos pm3(RPadLength::Normal(x), .85_normal);
      auto om3 = canvas->Draw(RMarker(pm3));
      om3->SetStyle(i+34).SetSize(2.5);
   }

   canvas->Show();
}
