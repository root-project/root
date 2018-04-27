/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2018-03-18
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Iliana Betsou

#include "ROOT/TCanvas.hxx"
#include "ROOT/TColor.hxx"
#include "ROOT/TText.hxx"
#include "ROOT/TMarker.hxx"
#include "ROOT/TPadPos.hxx"

void markerStyle() {
   using namespace ROOT;
   using namespace ROOT::Experimental;

   auto canvas = Experimental::TCanvas::Create("Canvas Title");
   double num = 0.3;

   double x = 0;
   double dx = 1/16.0;
   for (Int_t i=1;i<16;i++) {
      x += dx;

      TPadPos pt1(TPadLength::Normal(x), .12_normal);
      auto ot1 = canvas->Draw(Experimental::TText(pt1, Form("%d", i)));
      TPadPos pm1(TPadLength::Normal(x), .25_normal);
      auto om1 = canvas->Draw(Experimental::TMarker(pm1));
      om1->SetMarkerStyle(i);
      om1->SetMarkerSize(2.5);

      TPadPos pt2(TPadLength::Normal(x), .42_normal);
      auto ot2 = canvas->Draw(Experimental::TText(pt2, Form("%d", i+19)));
      TPadPos pm2(TPadLength::Normal(x), .55_normal);
      auto om2 = canvas->Draw(Experimental::TMarker(pm2));
      om2->SetMarkerStyle(i+19);
      om2->SetMarkerSize(2.5);

      TPadPos pt3(TPadLength::Normal(x), .72_normal);
      auto ot3 = canvas->Draw(Experimental::TText(pt3, Form("%d", i+34)));
      TPadPos pm3(TPadLength::Normal(x), .85_normal);
      auto om3 = canvas->Draw(Experimental::TMarker(pm3));
      om3->SetMarkerStyle(i+34);
      om3->SetMarkerSize(2.5);
   }

   canvas->Show();
}