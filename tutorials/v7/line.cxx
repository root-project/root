/// \file
/// \ingroup tutorial_v7
///
/// This ROOT 7 example demonstrates how to create a ROOT 7 canvas (RCanvas) and
/// draw ROOT 7 lines in it (RLine). It generates a set of lines using the
/// "normal" coordinates' system and changing the line color linearly from black
/// to red.
///
/// \macro_image (line.png)
/// \macro_code
///
/// \date 2018-03-18
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \authors Olivier couet, Iliana Betsou

#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/RLine.hxx"
#include <ROOT/RPadPos.hxx>
#include "TMath.h"

void line()
{
   using namespace ROOT::Experimental;

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Canvas Title");

   for (double i = 0; i < 360; i+=1) {
      double angle = i * TMath::Pi() / 180;
      RPadPos p(0.3_normal*TMath::Cos(angle) + 0.5_normal,
                 0.3_normal*TMath::Sin(angle) + 0.5_normal);
      auto opts = canvas->Draw(RLine({0.5_normal, 0.5_normal} , p));
      RColor col(0.0025*i, 0, 0);
      opts->SetColor(col);
    }

   canvas->Draw(RLine({0.0_normal, 0.0_normal}, {1.0_normal,1.0_normal}));
   canvas->Draw(RLine({0.1_normal, 0.1_normal}, {0.9_normal,0.1_normal}));
   canvas->Draw(RLine({0.9_normal, 0.1_normal}, {0.9_normal,0.9_normal}));
   canvas->Draw(RLine({0.9_normal, 0.9_normal}, {0.1_normal,0.9_normal}));
   canvas->Draw(RLine({0.1_normal, 0.1_normal}, {0.1_normal,0.9_normal}));
   canvas->Draw(RLine({0.0_normal, 1.0_normal}, {1.0_normal,0.0_normal}));

  // canvas->Show();
   canvas->SaveAs("line.png");
}
