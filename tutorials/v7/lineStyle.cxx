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
#include "ROOT/TLine.hxx"
#include "ROOT/TPadPos.hxx"

void lineStyle() {
   using namespace ROOT;
   using namespace ROOT::Experimental;

   auto canvas = Experimental::TCanvas::Create("Canvas Title");
   double num = 0.3;

   for (int i=10; i>0; i--){
      num = num + 0.02;

      TPadPos pt(.3_normal, TPadLength::Normal(num));
      auto optts = canvas->Draw(Experimental::TText(pt, Form("%d", i)));
      optts->SetTextSize(13);
      optts->SetTextAlign(32);
      optts->SetTextFont(52);

      TPadPos pl1(.32_normal, TPadLength::Normal(num));
      TPadPos pl2(.8_normal , TPadLength::Normal(num));
      auto optls = canvas->Draw(Experimental::TLine(pl1, pl2));
      optls->SetLineStyle(i);
   }

   canvas->Show();
}
