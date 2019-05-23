/// \file
/// \ingroup tutorial_v7
///
/// This ROOT 7 example demonstrates how to create a ROOT 7 canvas (RCanvas) and
/// draw ROOT 7 boxes in it (RBox). It generates a set of boxes using the
/// "normal" coordinates' system.
///
/// \macro_image
/// \macro_code
///
/// \date 2018-10-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Olivier couet

#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/RBox.hxx"
#include <ROOT/RPadPos.hxx>
#include "TMath.h"

void box()
{
   using namespace ROOT::Experimental;

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Canvas Title");

   auto OptsBox1 = canvas->Draw(RBox({0.1_normal, 0.3_normal}, {0.3_normal,0.6_normal}));
   RColor Color1(1., 0., 0., 0.5); // 50% opaque
   RColor Color2(0., 0., 1., 0.3); // 30% opaque

   OptsBox1->Box().Border().SetColor(Color1).SetWidth(5);
   //OptsBox1->Fill().SetColor(Color2);

   auto OptsBox2 = canvas->Draw(RBox({0.4_normal, 0.2_normal}, {0.6_normal,0.7_normal}));
   OptsBox2->Box().Border().SetColor(Color2).SetStyle(2).SetWidth(3);
   //OptsBox2->Fill().SetStyle(0);

   auto OptsBox3 = canvas->Draw(RBox({0.7_normal, 0.4_normal}, {0.9_normal,0.6_normal}));
   //OptsBox3->SetFillStyle(0);
   //OptsBox3->SetRoundWidth(50);
   //OptsBox3->SetRoundHeight(50);
   OptsBox3->Box().Border().SetWidth(3);

   auto OptsBox4 = canvas->Draw(RBox({0.7_normal, 0.7_normal}, {0.9_normal,0.9_normal}));
   //OptsBox4->SetFillStyle(0);
   //OptsBox4->SetRoundWidth(50);
   //OptsBox4->SetRoundHeight(25);
   OptsBox4->Box().Border().SetWidth(3);

   auto OptsBox5 = canvas->Draw(RBox({0.7_normal, 0.1_normal}, {0.9_normal,0.3_normal}));
   //OptsBox5->SetFillStyle(0);
   //OptsBox5->SetRoundWidth(25);
   //OptsBox5->SetRoundHeight(50);
   OptsBox5->Box().Border().SetWidth(3);

   canvas->Show();
}
