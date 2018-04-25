/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2018-03-18
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \authors Olivier couet, Iliana Betsou

R__LOAD_LIBRARY(libROOTGpadv7);

#include "ROOT/TCanvas.hxx"
#include "ROOT/TColor.hxx"
#include "ROOT/TLine.hxx"
#include <ROOT/TPadPos.hxx>

void line()
{
   using namespace ROOT;
   using namespace ROOT::Experimental;

   // Create a canvas to be displayed.
   auto canvas = Experimental::TCanvas::Create("Canvas Title");

   TPadPos p1,p2;

   p1.fHoriz = 0._normal;
   p1.fVert  = 0._normal;

   for (double i = 0; i < 360; i+=1) {
      double ang = i * TMath::Pi() / 180;
      double x   = 0.3*TMath::Cos(ang) + 0.5;
      double y   = 0.3*TMath::Sin(ang) + 0.5;

      p2.fHoriz = TPadLength::Normal(x);
      p2.fVert  = TPadLength::Normal(y);

      auto line = std::make_shared<Experimental::TLine>(p1 , p2);

      auto col = Experimental::TColor(0.0025*i, 0, 0);
      line->GetOptions().SetLineColor(col);
      line->GetOptions().SetLineWidth(1);

      canvas->Draw(line);
    }

   canvas->Draw(Experimental::TLine({0.0_normal, 0.0_normal}, {1.0_normal,1.0_normal}));
   canvas->Draw(Experimental::TLine({0.1_normal, 0.1_normal}, {0.9_normal,0.1_normal}));
   canvas->Draw(Experimental::TLine({0.9_normal, 0.1_normal}, {0.9_normal,0.9_normal}));
   canvas->Draw(Experimental::TLine({0.9_normal, 0.9_normal}, {0.1_normal,0.9_normal}));
   canvas->Draw(Experimental::TLine({0.1_normal, 0.1_normal}, {0.1_normal,0.9_normal}));
   canvas->Draw(Experimental::TLine({0.0_normal, 1.0_normal}, {1.0_normal,0.0_normal}));

   canvas->Show();
}
