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

   auto Box1 = canvas->Draw<RBox>(RPadPos(0.1_normal, 0.3_normal), RPadPos(0.3_normal,0.6_normal));
   RColor Color1(255, 0, 0, 0.5); // 50% opaque
   RColor Color2(0, 0, 255, 0.3); // 30% opaque

   Box1->AttrBox().Border().SetColor(Color1).SetWidth(5);
   Box1->AttrBox().Fill().SetColor(RColor::kRed);
   //OptsBox1->Fill().SetColor(Color2);

   auto Box2 = canvas->Draw<RBox>(RPadPos(0.4_normal, 0.2_normal), RPadPos(0.6_normal,0.7_normal));
   Box2->AttrBox().Border().SetColor(Color2).SetStyle(2).SetWidth(3);
   Box2->AttrBox().Fill().SetColor(RColor::kGreen);
   //OptsBox2->Fill().SetStyle(0);

   auto Box3 = canvas->Draw<RBox>(RPadPos(0.7_normal, 0.4_normal), RPadPos(0.9_normal,0.6_normal));
   //OptsBox3->SetFillStyle(0);
   //OptsBox3->SetRoundWidth(50);
   //OptsBox3->SetRoundHeight(50);
   Box3->AttrBox().Border().SetWidth(3);
   Box3->AttrBox().Fill().SetColor(RColor::kBlue);

   auto Box4 = canvas->Draw<RBox>(RPadPos(0.7_normal, 0.7_normal), RPadPos(0.9_normal,0.9_normal));
   //OptsBox4->SetFillStyle(0);
   //OptsBox4->SetRoundWidth(50);
   //OptsBox4->SetRoundHeight(25);
   Box4->AttrBox().Border().SetWidth(3);

   auto Box5 = canvas->Draw<RBox>(RPadPos(0.7_normal, 0.1_normal), RPadPos(0.9_normal,0.3_normal));
   //OptsBox5->SetFillStyle(0);
   //OptsBox5->SetRoundWidth(25);
   //OptsBox5->SetRoundHeight(50);
   Box5->AttrBox().Border().SetWidth(3);

   canvas->Show();
}
